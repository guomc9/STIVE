import sys
sys.path.append('.')
import argparse
import datetime
import logging
import inspect
from typing import Dict, Optional
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import shutil
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.pipelines import TextToVideoSDPipeline
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from stive.models.unet import UNet3DConditionModel
from stive.models.attention import reset_sparse_casual_processors as unregister_attention_control
from stive.models.concepts_clip import ConceptsCLIPTextModel
from stive.data.dataset import VideoEditPromptsDataset as VideoPromptValDataset, LatentPromptDataset
from stive.utils.ddim_utils import ddim_inversion
from stive.utils.save_utils import save_videos_grid, save_video
from stive.utils.textual_inversion_utils import add_concepts_embeddings
from stive.prompt_attention.attention_register import register_attention_control
from stive.prompt_attention.attention_store import StepAttentionSupervisor
from einops import rearrange
import os
import gc
import re
import wandb
import random
from peft import PeftModel
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


def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, now)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    return output_dir

import numpy as np
import cv2
from einops import rearrange
import stive.prompt_attention.ptp_utils as ptp_utils

def collect_cross_attention(attns_dict, prompts, video_length):
    b = len(prompts)
    f = video_length
    out = {prompt:{} for prompt in prompts}
    for key, attns in attns_dict.items():    # [B * F, Q, K]
        q = attns.shape[1]
        h = w = int(np.sqrt(q))
        attns = rearrange(attns, '(b f) (h w) k -> b f h w k', b=b, f=f, h=h, w=w)
        
        for j in range(b):
            tokens = ['SOT']
            prompt = prompts[j]
            out[prompt][key] = {}
            tokens.extend(prompt.split(' '))
            tokens.append('EOT')
            prompt_frames = []
            for i, token in enumerate(tokens):
                frames = attns[j, ..., i]                                           # [F, H, W]
                token_frames = []
                for k in range(len(frames)):
                    frame = frames[k]                                               # [H, W]
                    frame = 255 * frame / frame.max()                               
                    frame = frame.unsqueeze(-1).expand(*frame.shape, 3)
                    frame = frame.numpy().astype(np.uint8)
                    frame = cv2.resize(frame, (256, 256))
                    frame = ptp_utils.text_under_image(frame, token)                # [H, W, 3]
                    token_frames.append(frame)                                      # [F, H, W, 3]
                token_frames = np.stack(token_frames)                               # [F, H, W, 3]
                prompt_frames.append(token_frames)
            prompt_frames = np.concatenate(prompt_frames, axis=-2)                  # [F, H, T * W, 3]
            out[prompt][key] = prompt_frames

    return out

def collect_cross_attention_mask(masks_dict):
    ret = {}
    for key, masks in masks_dict.items():
        np_masks = []
        for i in range(len(masks)):
            mask = masks[i]
            mask = 255 * masks[i].unsqueeze(-1).expand(*mask.shape, 3)               # [F, H, W, 3]
            mask = mask.numpy().astype(np.uint8)
            mask = cv2.resize(mask, (256, 256))
            np_masks.append(mask)
        np_masks = np.stack(np_masks)
        ret[key] = np_masks
    return ret

from stive.utils.save_utils import save_video
def log_cross_attention(accelerator, prompt_attn_dict, save_path, step):
    os.makedirs(save_path, exist_ok=True)
    for prompt, unet_attns in prompt_attn_dict.items():
        for key, unet_attn in unet_attns.items():
            os.makedirs(os.path.join(save_path, str(step), prompt), exist_ok=True)
            unet_attn = rearrange(torch.from_numpy(unet_attn), 'f h w c -> f c h w')  # [F, C, H, T * W]
            save_video(unet_attn, path=f'{os.path.join(save_path, str(step), prompt, key)}.gif', rescale=False, to_uint8=False)
            accelerator.log({f"{key}/{prompt}": wandb.Video(unet_attn.to(torch.uint8).detach().cpu().numpy())}, step=step)
    return

def log_cross_attention_mask(accelerator, masks_dict, save_path, step, log_prefix=''):
    os.makedirs(save_path, exist_ok=True)
    for key, masks in masks_dict.items():
        masks = rearrange(masks, 'f h w c -> f c h w')            # [F, C, H, W]
        save_video(torch.from_numpy(masks), path=f'{os.path.join(save_path, str(step), key)}.gif', rescale=False, to_uint8=False)
        accelerator.log({f"{log_prefix}{key}": wandb.Video(masks)}, step=step)
    return



def main(
    pretrained_sd_model_path: str,
    output_dir: str,
    checkpoints_dir: str, 
    train_data: Dict,
    validation_data: Dict,
    inference_conf: Dict, 
    lora_conf: Dict, 
    pretrained_concepts_model_path: str = None, 
    cam_loss_type: str = 'mae', 
    sub_sot: bool = True, 
    enable_scam_loss: bool = False, 
    scam_weight: float = 1.0e-1, 
    scam_only_neg: bool = False, 
    enable_tcam_loss: bool = False, 
    tcam_weight: float = 5.0e-2, 
    tcam_only_neg: bool = False, 
    cam_loss_reduction: str = 'mean', 
    attn_check_steps: int = 10, 
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
    extra_trainable_modules=None, 
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb"
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="stive-sd-unet-fine-tune", 
            config={'train_data': train_data, 'inference_conf': inference_conf, 'validation_data': validation_data},
            init_kwargs={"wandb": {"name": f'{os.path.basename(checkpoints_dir)}-{datetime.datetime.now().strftime("%Y.%m.%d.%H-%M-%S")}'}}
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
    
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_sd_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_sd_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_sd_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_sd_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_sd_model_path, subfolder="unet")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet._set_gradient_checkpointing(gradient_checkpointing)
    
    if pretrained_concepts_model_path is not None and os.path.exists(pretrained_concepts_model_path):
        concepts_text_encoder = ConceptsCLIPTextModel.from_pretrained(pretrained_concepts_model_path, subfolder="text_encoder")
        add_concepts_embeddings(tokenizer, text_encoder, concept_tokens=concepts_text_encoder.concepts_list, concept_embeddings=concepts_text_encoder.concepts_embedder.weight.detach().clone())
        del concepts_text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    unet.to(dtype=weight_dtype)
    lora_unet = unet
    if lora_conf is not None:
        lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
        target_modules = []
        patterns = lora_conf['target_modules']
        if patterns is not None and len(patterns) > 0:
            for name, param in unet.named_parameters():
                for pattern in patterns:
                    if re.match(pattern, name):
                        target_modules.append(name.rstrip('.weight').rstrip('.bias'))
            if len(target_modules) > 0:
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


    if not enable_scam_loss and not enable_tcam_loss:
        unregister_attention_control(unet)

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

    replace_indices = validation_data['replace_indices']
    validation_data.pop('replace_indices')
    replace_indices = [torch.as_tensor(replace_inds, device=accelerator.device, dtype=torch.int64) for replace_inds in replace_indices]
    train_dataset = LatentPromptDataset(**train_data)
    val_dataset = VideoPromptValDataset(**validation_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    validation_pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_sd_model_path, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=lora_unet, 
    )
    validation_pipeline.enable_vae_slicing()
    
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_sd_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(inference_conf.num_inv_steps, device=accelerator.device)

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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    
    if enable_scam_loss or enable_tcam_loss:
        supervisor = StepAttentionSupervisor()
        register_attention_control(lora_unet, supervisor, only_cross=True, replace_attn_prob=False, self_to_st_attn=True)
    
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
        train_mse_loss = 0.0
        train_scam_loss = 0.0
        train_tcam_loss = 0.0
        optimizer.zero_grad()
        with tqdm(train_dataloader) as progress_bar:
            for step, batch in enumerate(progress_bar):
                with accelerator.autocast():
                    with accelerator.accumulate(lora_unet):
                        latents = batch["latents"]          # [B, F, C, H, W]
                        prompts = batch['prompts']
                        masks = batch['masks']              # [B, F, 1, H, W]
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
                        torch.cuda.empty_cache()
                        
                        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        scam_loss = 0.0
                        tcam_loss = 0.0
                        if enable_scam_loss and replace_indices is not None:
                            scam_loss, masks_dict = supervisor.get_cross_attn_mask_loss(mask=masks, target_indices=replace_indices, sub_sot=sub_sot, only_neg=scam_only_neg, loss_type=cam_loss_type, reduction=cam_loss_reduction)
                            if global_step % attn_check_steps == 0:
                                save_path = os.path.join(output_dir, 'source-cross-attn')
                                prompt_attn_dict = collect_cross_attention(supervisor.get_mean_head_attns(), prompts, video_length=video_length)
                                log_cross_attention(accelerator, prompt_attn_dict, save_path, step=global_step)
                                save_path = os.path.join(output_dir, 'source-cross-attn-mask')
                                masks_dict = collect_cross_attention_mask(masks_dict)
                                log_cross_attention_mask(accelerator, masks_dict, save_path, step=global_step, log_prefix='source-')
                            supervisor.reset()
                            torch.cuda.empty_cache()
                            
                        if enable_tcam_loss and batch.get('target_latents') is not None and batch.get('target_masks') is not None:
                            target_prompts = batch['target_prompts']                # [B]
                            target_tokens = tokenizer(target_prompts, return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
                            target_encoder_hidden_states = text_encoder(target_tokens.input_ids.to(latents.device))[0]
                            if replace_indices is not None:
                                timesteps = ddim_inv_scheduler.timesteps
                                t = timesteps[random.randint(0, len(timesteps)-1)].long()
                                target_latents = batch['target_latents']            # [B, F, C, H, W]
                                target_masks = batch['target_masks']                # [B, F, 1, H, W]
                                target_latents = rearrange(target_latents, "b f c h w -> b c f h w")
                                lora_unet(target_latents, t, target_encoder_hidden_states)
                                tcam_loss, masks_dict = supervisor.get_cross_attn_mask_loss(mask=target_masks, target_indices=replace_indices, sub_sot=sub_sot, only_neg=tcam_only_neg, loss_type=cam_loss_type, reduction=cam_loss_reduction)
                                if global_step % attn_check_steps == 0:
                                    save_path = os.path.join(output_dir, 'target-cross-attn')
                                    prompt_attn_dict = collect_cross_attention(supervisor.get_mean_head_attns(), target_prompts, video_length=video_length)
                                    log_cross_attention(accelerator, prompt_attn_dict, save_path, step=global_step)
                                    save_path = os.path.join(output_dir, 'target-cross-attn-mask')
                                    masks_dict = collect_cross_attention_mask(masks_dict)
                                    log_cross_attention_mask(accelerator, masks_dict, save_path, step=global_step, log_prefix='target-')
                                    
                            supervisor.reset()
                            torch.cuda.empty_cache()
                            
                        loss = mse_loss + scam_weight * scam_loss + tcam_weight * tcam_loss
                        
                        avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                        mse_loss = accelerator.gather(mse_loss.repeat(batch_size)).mean()
                        train_loss += avg_loss.item() / gradient_accumulation_steps
                        train_mse_loss += mse_loss.item() / gradient_accumulation_steps

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
                                if enable_scam_loss:
                                    scam_loss = accelerator.gather(scam_loss.repeat(batch_size)).mean()
                                    train_scam_loss += scam_loss.item() / gradient_accumulation_steps
                                    accelerator.log({"scam_loss": train_scam_loss}, step=global_step)
                                    
                                if enable_tcam_loss:
                                    tcam_loss = accelerator.gather(tcam_loss.repeat(batch_size)).mean()
                                    train_tcam_loss += tcam_loss.item() / gradient_accumulation_steps
                                    accelerator.log({"tcam_loss": train_tcam_loss}, step=global_step)
                                    
                                accelerator.log({"train_loss": train_loss, 'mse_loss': train_mse_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
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
                    if enable_scam_loss and enable_tcam_loss:
                        unregister_attention_control(lora_unet)

                    with torch.no_grad(), accelerator.autocast():
                        generator = torch.Generator(device=latents.device)
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
                            
                    if enable_scam_loss and enable_tcam_loss:
                        supervisor.reset()
                        register_attention_control(lora_unet, supervisor, only_cross=True, replace_attn_prob=False)
                        
                    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lora_unet = accelerator.unwrap_model(lora_unet)
        if isinstance(lora_unet, PeftModel):
            save_path = os.path.join(output_dir, 'lora')
            lora_unet.save_pretrained(save_path)
            unet = lora_unet.unload()
            unet.save_pretrained(save_path, safe_serialization=False)
        elif isinstance(lora_unet, UNet3DConditionModel):
            save_path = os.path.join(output_dir, 'unet')
            lora_unet.save_pretrained(save_path, safe_serialization=False)
        logger.info(f"Saved Lora UNet3DConditionModel to {save_path}")

    accelerator.end_training()
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    shutil.copytree(output_dir, checkpoints_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sd_concepts/lambo.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
