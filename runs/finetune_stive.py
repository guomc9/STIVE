import sys
sys.path.append('.')
import argparse
import datetime
import logging
import inspect
import re
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
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines import TextToVideoSDPipeline
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models import UNet3DConditionModel
from stive.models.concepts_clip import ConceptsCLIPTextModel, ConceptsCLIPTokenizer
from stive.data.dataset import VideoEditPromptsDataset, LatentPromptTupleCacheDataset
from stive.utils.ddim_utils import ddim_inversion
from stive.utils.save_utils import save_videos_grid, save_video, save_images
from stive.utils.textual_inversion_utils import add_concepts_embeddings, update_concepts_embedding
from stive.utils.cache_latents_utils import encode_videos_latents
from einops import rearrange, repeat
import wandb
import subprocess
import os
import gc
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

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, now)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    return output_dir

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
    extra_trainable_modules = None, 
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb"
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="stive-unet-fine-tune", 
            config={'train_data': train_data, 'inference_conf': inference_conf, 'validation_data': validation_data},
            init_kwargs={"wandb": {"name": f'{os.path.basename(checkpoints_dir)}-{datetime.now().strftime("%Y.%m.%d.%H-%M-%S")}'}}
        )
    
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_t2v_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_t2v_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_t2v_model_path, subfolder="vae")
    if pretrained_concepts_model_path is not None and os.path.exists(pretrained_concepts_model_path):
        concepts_text_encoder = ConceptsCLIPTextModel.from_pretrained(pretrained_concepts_model_path, subfolder="text_encoder")
        add_concepts_embeddings(tokenizer, text_encoder, concept_tokens=concepts_text_encoder.concepts_list, concept_embeddings=concepts_text_encoder.concepts_embedder.weight.detach().clone())
        del concepts_text_encoder
        gc.collect()
        torch.cuda.empty_cache()
    unet = UNet3DConditionModel.from_pretrained(pretrained_t2v_model_path, subfolder="unet")
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, gradient_checkpointing, unet)
    unet.to(dtype=weight_dtype)
    lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
    target_modules = []
    patterns = lora_conf['target_modules']
    for name, param in unet.named_parameters():
        for pattern in patterns:
            if re.match(pattern, name):
                target_modules.append(name.rstrip('.weight').rstrip('.bias'))
    lora_conf['target_modules'] = target_modules
    lora_conf = LoraConfig(**lora_conf)
    lora_unet = get_peft_model(model=unet, peft_config=lora_conf)
    
    lora_unet.print_trainable_parameters()
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    lora_unet.requires_grad_(False)
    for name, param in lora_unet.named_parameters():
        if 'lora' in name:
            print(f'lora trainable param: {name}')
            param.requires_grad = True
            param.data = param.data.float()
        elif extra_trainable_modules is not None:
            for extra_trainable_module in extra_trainable_modules:
                if re.match(extra_trainable_module, name):
                    print(f'extra trainable param: {name}')
                    param.requires_grad = True
                    param.data = param.data.float()
            
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * batch_size * accelerator.num_processes
        )

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optim_params = [p for p in lora_unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(
        optim_params, 
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    
    latents = encode_videos_latents(video_paths=train_data['sources'], height=train_data['height'], width=train_data['width'])
    train_data = OmegaConf.to_container(train_data, resolve=True)
    train_data.pop('sources')
    train_data['latents'] = latents
    train_dataset = LatentPromptTupleCacheDataset(**train_data)
    
    val_dataset = VideoEditPromptsDataset(**validation_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    validation_pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_t2v_model_path, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=lora_unet, 
    )
    validation_pipeline.enable_vae_slicing()
    
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(inference_conf.num_inv_steps)

    num_train_steps = (num_train_epoch * len(train_dataloader)) // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_epoch_rate * num_train_steps, 
        num_training_steps=num_train_steps, 
    )
    
    lora_unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        lora_unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    vae.eval()
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, num_train_epoch):
        lora_unet.train()
        train_loss = 0.0
        optimizer.zero_grad()
        with tqdm(train_dataloader) as progress_bar:
            for step, batch in enumerate(progress_bar):
                with accelerator.autocast():
                    with accelerator.accumulate(lora_unet):
                        # pixel_values = batch["frames"].to(weight_dtype)
                        # prompts = batch['prompts']
                        
                        # video_length = pixel_values.shape[1]
                        # pixel_values = rearrange(pixel_values, "b f h w c -> (b f) c h w")

                        # latents = vae.encode(pixel_values).latent_dist.sample()
                        # latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                        # latents = latents * 0.18215
                        latents = batch["latents"]
                        prompts = batch['prompts']
                        video_length = latents.shape[1]
                        
                        latents = rearrange(latents, "b f c h w -> b c f h w")

                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()

                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        tokens = tokenizer(prompts, return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)

                        encoder_hidden_states = text_encoder(tokens.input_ids.to(latents.device))[0]
                        
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        model_pred = lora_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()

                        train_loss += avg_loss.item() / gradient_accumulation_steps

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(optim_params, max_grad_norm)
                            for param in optim_params:
                                if param.grad is None:
                                    print(f"Epoch {epoch}, Parameter: {param}, Gradient: {param.grad}, Requires_grad: {param.requires_grad}")

            
                            if (global_step + 1) % gradient_accumulation_steps == 0:
                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()
                                
                                accelerator.log({"train_loss": train_loss}, step=global_step)
                                train_loss = 0.0

                        logs = {"Epoch": f'{epoch}/{num_train_epoch}', "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                        progress_bar.set_postfix(**logs)
                        global_step += 1
                
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    with torch.no_grad(), accelerator.autocast():
                        torch.cuda.empty_cache()
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
                                save_video(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif", rescale=False)
                                accelerator.log({f"sample/{prompt}": wandb.Video((255 * sample).to(torch.uint8).detach().cpu().numpy())}, step=global_step)
                                samples.append(sample)
                            samples = torch.stack(samples)
                            save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                            save_videos_grid(samples, save_path, rescale=False)
                            logger.info(f"Saved samples to {save_path}")

                        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lora_unet = accelerator.unwrap_model(lora_unet)
        save_path = os.path.join(output_dir, 'lora')
        lora_unet.save_pretrained(save_path)
        save_path = os.path.join(output_dir, 'unet')
        unet = lora_unet.unload()
        unet.save_pretrained(save_path)
        logger.info(f"Saved Lora UNet3DConditionModel to {save_path}")

    accelerator.end_training()
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    shutil.copytree(output_dir, checkpoints_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/concepts_clip/vehicles.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
