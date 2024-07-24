import argparse
import datetime
import logging
import inspect
import os
from typing import Dict, Optional
from omegaconf import OmegaConf
import json
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
from stive.pipelines.pipeline_ptp_ttv import PtpTextToVideoSDPipeline
from stive.models.concepts_clip import ConceptsCLIPTextModel, ConceptsCLIPTokenizer
from stive.data.dataset import VideoEditPromptsDataset
from stive.utils.ddim_utils import ddim_inversion
from stive.utils.pta_utils import save_gif_mp4_folder_type
from stive.utils.save_utils import save_videos_grid, save_video, save_images
from einops import rearrange, repeat
from stive.utils.textual_inversion_utils import add_concepts_embeddings, update_concepts_embedding
from stive.prompt_attention.attention_util import AttentionStore, make_controller
from stive.prompt_attention.attention_register import register_attention_control

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
check_min_version("0.10.0.dev0")

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

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())

def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])

def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, gradient_checkpointing, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)
            
        if gradient_checkpointing:
            unet._set_gradient_checkpointing(True)

    except Exception as e:
        print(f"Could not enable memory efficient attention for xformers or Torch 2.0: {e}.")

def main(
    pretrained_t2v_model_path: str,
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
    unet = UNet3DConditionModel.from_pretrained(pretrained_t2v_model_path, subfolder="unet")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, gradient_checkpointing, unet)

    val_dataset = VideoEditPromptsDataset(**validation_data)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1
    )
    validation_pipeline = PtpTextToVideoSDPipeline(disk_store=False, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    # validation_pipeline = TextToVideoSDPipeline.from_pretrained(
    #     pretrained_model_name_or_path=pretrained_t2v_model_path, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
    # )
    validation_pipeline.enable_vae_slicing()
    

    unet, text_encoder, vae, val_dataloader = accelerator.prepare(
        unet, text_encoder, vae, val_dataloader
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    unet.eval()
    text_encoder.eval()
    torch.cuda.empty_cache()
    source = val_dataset.get_source()
    
    with torch.no_grad(), accelerator.autocast():
        source_text_embeddings = validation_pipeline._encode_prompt(
            prompt=source['source_prompts'], 
            device=accelerator.device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True, 
            negative_prompt=None
        )
        
        all_step_source_latents = validation_pipeline.prepare_ddim_source_latents(
            frames=source['frames'].to(accelerator.device, dtype=weight_dtype), 
            text_embeddings=source_text_embeddings.to(accelerator.device, dtype=weight_dtype), 
            prompt=source['source_prompts'], 
            store_attention=ptp_conf['use_inversion_attention'], 
            LOW_RESOURCE=True, 
            save_path=output_dir
        )
        source_init_latents = all_step_source_latents[-1]
        
        generator = torch.Generator(device=accelerator.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        samples = []
        attention_all = []
        for i, batch in enumerate(val_dataloader):
            pixel_values = batch["frames"].to(weight_dtype)                 # [1, F, C, H, W]
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
                save_self_attention = ptp_conf.get('save_self_attention', True), 
                guidance_scale=inference_conf["guidance_scale"], 
                generator=generator, 
                disk_store = ptp_conf.get('disk_store', False), 
            )
            samples = edited_output['edited_frames']
            attention_output = edited_output['attention_output']
            
            save_path = os.path.join(output_dir, f"attn_prob")
            os.makedirs(save_path, exist_ok=True)
            save_gif_mp4_folder_type(attention_output, os.path.join(save_path, f'{target_prompts}-attn_prob.gif'))
            
            for p, s in zip(target_prompts, samples):
                prompts_samples[p] = s
            
        if accelerator.is_main_process:
            samples = []
            prompts_samples = accelerator.gather(prompts_samples)
            for prompt, sample in prompts_samples.items():
                save_path = f'{output_dir}/inferences/{prompt}.gif'
                save_video(sample, save_path, rescale=False)
                logger.info(f"Saved {prompt} to {save_path}")
        
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/concepts_clip/vehicles.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))