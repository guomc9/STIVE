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
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines import TextToVideoSDPipeline
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from stive.models.concepts_clip import ConceptsCLIPTextModel, ConceptsCLIPTokenizer
from stive.models.unet_3d_condition import UNet3DConditionModel
from stive.data.dataset import VideoPromptTupleDataset, VideoEditPromptsDataset
from stive.utils.ddim_utils import ddim_inversion
from stive.utils.save_utils import save_videos_grid, save_video, save_images
from einops import rearrange, repeat
from stive.utils.textual_inversion_utils import add_concepts_embeddings, update_concepts_embedding
import os
import wandb
import subprocess
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
    train_data: Dict,
    validation_data: Dict,
    inference_conf: Dict, 
    lora_conf: Dict, 
    validation_steps: int = 200,
    batch_size: int = 1,
    warmup_epoch_rate: float = 0.1, 
    num_train_epoch: int = 200,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler_type: str = "constant",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 200,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = True, 
    seed: Optional[int] = None, 
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )
    
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(checkpoints_dir, 'inferences'), exist_ok=True)
        output_dir = checkpoints_dir

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder="scheduler")
    concepts_text_encoder = ConceptsCLIPTextModel.from_pretrained(pretrained_concepts_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_t2v_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_t2v_model_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    concepts_text_encoder.requires_grad_(False)
    add_concepts_embeddings(tokenizer, text_encoder, concept_tokens=concepts_text_encoder.concepts_list, concept_embeddings=concepts_text_encoder.concepts_embedder.weight.detach().clone())
    del concepts_text_encoder
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(pretrained_t2v_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_t2v_model_path, subfolder="unet")
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, gradient_checkpointing, unet)
    unet.to(dtype=weight_dtype)
    lora_unet = PeftModel.from_pretrained(unet, os.path.join(checkpoints_dir, 'lora'))
    validation_pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_t2v_model_path, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=lora_unet, 
    )
    validation_pipeline.enable_vae_slicing()
    val_dataset = VideoEditPromptsDataset(**validation_data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(inference_conf.num_inv_steps)

    lora_unet, vae, text_encoder, val_dataloader = accelerator.prepare(
        lora_unet, vae, text_encoder, val_dataloader
    )

    text_encoder.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    unet.eval()
    text_encoder.eval()

    torch.cuda.empty_cache()
    with torch.no_grad(), accelerator.autocast():
        generator = torch.Generator(device=accelerator.device)
        if seed is not None:
            generator.manual_seed(seed)
        prompts_samples = {}
        for _, batch in enumerate(val_dataloader):
            pixel_values = batch["frames"].to(weight_dtype)
            prompts = batch['prompts']
            
            video_length = pixel_values.shape[1]
            pixel_values = rearrange(pixel_values, "b f h w c -> (b f) c h w")

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            latents = latents * 0.18215
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            ddim_inv_latent = None
            
            if inference_conf.use_inv_latent:
                ddim_inv_latent = ddim_inversion(
                    validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                    num_inv_steps=inference_conf.num_inv_steps, prompt="")[-1].to(weight_dtype)

            samples = validation_pipeline(
                prompt=prompts, 
                generator=generator, 
                latents=ddim_inv_latent, 
                num_inference_steps=inference_conf['num_inference_steps'], 
                guidance_scale=inference_conf['guidance_scale'], 
                num_frames=validation_data['num_frames'], 
                output_type='pt', 
            ).frames
            
            for p, s in zip(prompts, samples):
                prompts_samples[p] = s
            
        if accelerator.is_main_process:
            samples = []
            prompts_samples = accelerator.gather(prompts_samples)
            for prompt, sample in prompts_samples.items():
                save_video(sample, f"{output_dir}/inferences/{prompt}.gif", rescale=False)
                samples.append(sample)
            samples = torch.stack(samples)
            save_path = f"{output_dir}/inferences/global.gif"
            save_videos_grid(samples, save_path, rescale=False)
            logger.info(f"Saved samples to {save_path}")

    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/stive/vehicles.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
