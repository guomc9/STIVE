import sys
sys.path.append('.')
import argparse
import datetime
import logging
import inspect
from typing import Dict, Optional
from omegaconf import OmegaConf
import json
import yaml
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import shutil
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import PeftModel
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.models import UNet3DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from einops import rearrange, repeat
from stive.pipelines.pipeline_ptp_ttv import PtpTextToVideoSDPipeline
from stive.models.concepts_clip import ConceptsCLIPTextModel, ConceptsCLIPTokenizer
from stive.data.dataset import VideoSliceEditPromptDataset
from stive.utils.ddim_utils import ddim_inversion
from stive.utils.pta_utils import save_gif_mp4_folder_type
from stive.utils.save_utils import save_videos_grid, save_video, save_images
from stive.utils.cache_latents_utils import encode_videos_latents
from stive.utils.textual_inversion_utils import add_concepts_embeddings, update_concepts_embedding
from stive.prompt_attention.attention_util import AttentionStore, make_controller
from stive.prompt_attention.attention_register import register_attention_control
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
check_min_version("0.28.0")

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def main(
    pretrained_t2v_model_path: str,
    pretrained_lora_model_path: str, 
    pretrained_concepts_model_path: str,
    output_dir: str,
    checkpoints_dir: str, 
    validation_data: Dict,
    inference_conf: Dict, 
    ptp_conf: Dict, 
    mixed_precision: Optional[str] = "fp16",
    seed: Optional[int] = None, 
    **extra_args,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        mixed_precision=mixed_precision
    )
    
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(checkpoints_dir, 'inferences'), exist_ok=True)
        output_dir = checkpoints_dir

    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder="scheduler")
    concepts_text_encoder = ConceptsCLIPTextModel.from_pretrained(pretrained_concepts_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_t2v_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_t2v_model_path, subfolder="text_encoder")
    scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder='scheduler')
    scheduler.set_timesteps(inference_conf.num_inv_steps)
    concepts_text_encoder.requires_grad_(False)
    add_concepts_embeddings(tokenizer, text_encoder, concept_tokens=concepts_text_encoder.concepts_list, concept_embeddings=concepts_text_encoder.concepts_embedder.weight.detach().clone())
    del concepts_text_encoder
    torch.cuda.empty_cache()
    
    vae = AutoencoderKL.from_pretrained(pretrained_t2v_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_lora_model_path, subfolder="unet")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    lora_unet = PeftModel.from_pretrained(unet, os.path.join(pretrained_lora_model_path, 'lora'))
    lora_unet.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    lora_unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    lora_unet.eval()
    text_encoder.eval()
    torch.cuda.empty_cache()
    validation_pipeline = PtpTextToVideoSDPipeline(disk_store=False, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=lora_unet, scheduler=scheduler)
    validation_pipeline.enable_vae_slicing()
    
    
    with open(os.path.join(pretrained_lora_model_path, 'config.yaml'), 'r', encoding='utf-8') as fp:
        conf = yaml.safe_load(fp)
        num_slice = conf['train_data']['num_frames']
    
    source = validation_data['source']
    source_prompt = validation_data['source_prompt']
    sample_stride = validation_data['sample_stride']
    edit_prompts = validation_data['edit_prompts']
    width = validation_data['width']
    height = validation_data['height']
    latents = encode_videos_latents([source], height=height, width=width)
    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator.manual_seed(seed)
        
    lora_unet, text_encoder, vae = accelerator.prepare(
        lora_unet, text_encoder, vae
    )
    for i, edit_prompt in enumerate(edit_prompts):
        val_dataset = VideoSliceEditPromptDataset(
            latents=latents, 
            source_prompt=source_prompt, 
            edit_prompt=edit_prompt, 
            num_slice=num_slice, 
            sample_stride=sample_stride, 
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1
        )

        val_dataloader = accelerator.prepare(val_dataloader)
        
        with torch.no_grad(), accelerator.autocast():
            samples = []
            attention_all = []
            prompts_samples = {}
            for _, batch in enumerate(val_dataloader):
                source_text_embeddings = validation_pipeline._encode_prompt(
                    prompt=batch['source_prompts'], 
                    device=accelerator.device, 
                    num_images_per_prompt=1, 
                    do_classifier_free_guidance=True, 
                    negative_prompt=None
                )
                print(f"batch['latents'].shape: {batch['latents'].shape}")
                
                source_init_latents = validation_pipeline.prepare_ddim_source_latents_with_latents(
                    init_latents=batch['latents'].to(accelerator.device, dtype=weight_dtype), 
                    text_embeddings=source_text_embeddings.to(accelerator.device, dtype=weight_dtype), 
                    prompt=batch['source_prompts'], 
                    store_attention=ptp_conf['use_inversion_attention'], 
                    LOW_RESOURCE=True, 
                    generator=generator, 
                    save_path=output_dir
                )[-1]
                
                torch.cuda.empty_cache()
                gc.collect()
                print(f'source_init_latents.shape: {source_init_latents.shape}')
            
                target_prompts = batch['prompts'][0]                            # [T]
                source_prompts = batch['source_prompts'][0]                     # [T]
                blend_word = [[ptp_conf['blend_words']['sources'][i]], [ptp_conf['blend_words']['targets'][i]]]
                
                edited_output = validation_pipeline.ptp_replace_edit(
                    latents=source_init_latents, 
                    source_prompt=source_prompts, 
                    target_prompt=target_prompts, 
                    num_inference_steps=inference_conf["num_inference_steps"], 
                    is_replace_controller=ptp_conf.get('is_replace_controller', True), 
                    cross_replace_steps=ptp_conf.get('cross_replace_steps', 0.5), 
                    self_replace_steps=ptp_conf.get('self_replace_steps', 0.5), 
                    blend_words=blend_word, 
                    equilizer_params=ptp_conf.get('eq_params', None), 
                    use_inversion_attention = ptp_conf.get('use_inversion_attention', None), 
                    blend_th = ptp_conf.get('blend_th', (0.3, 0.3)), 
                    blend_self_attention = ptp_conf.get('blend_self_attention', None), 
                    blend_latents=ptp_conf.get('blend_latents', None), 
                    save_path=output_dir, 
                    save_self_attention=ptp_conf.get('save_self_attention', True), 
                    guidance_scale=inference_conf["guidance_scale"], 
                    generator=generator, 
                    disk_store = ptp_conf.get('disk_store', False), 
                )
                samples = edited_output['edited_frames'].detach().cpu()
                attention_output = edited_output['attention_output']
                validation_pipeline.clear_store_controller()
                torch.cuda.empty_cache()
                gc.collect()
                save_path = os.path.join(output_dir, f"attn_prob")
                os.makedirs(save_path, exist_ok=True)
                save_gif_mp4_folder_type(attention_output, os.path.join(save_path, f'{target_prompts}-attn_prob.gif'))
                
                for p, s in zip([target_prompts], samples):
                    if p in prompts_samples:
                        pre_s = prompts_samples[p]
                        s = torch.cat([pre_s, s], dim=0)   # [F, C, H, W]
                        prompts_samples[p] = s
                        print(f's.shape: {s.shape}')
                        save_path = f'{output_dir}/inferences/{p}-{s.shape[0]}-frames.gif'
                        save_video(s, save_path, rescale=False)
                        logger.info(f"Saved {p} to {save_path}")
                    else:
                        prompts_samples[p] = s
                        print(f's.shape: {s.shape}')
                        save_path = f'{output_dir}/inferences/{p}-{s.shape[0]}-frames.gif'
                        save_video(s, save_path, rescale=False)
                        logger.info(f"Saved {p} to {save_path}")
                
            if accelerator.is_main_process:
                prompts_samples = accelerator.gather(prompts_samples)
                for prompt, sample in prompts_samples.items():
                    save_path = f'{output_dir}/inferences/{prompt}.gif'
                    save_video(sample, save_path, rescale=False)
                    logger.info(f"Saved {prompt} to {save_path}")
            gc.collect()
            torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/concepts_clip/vehicles.yaml")
    args = parser.parse_args()
    
    main(**OmegaConf.load(args.config))