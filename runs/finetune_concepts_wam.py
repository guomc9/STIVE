import sys
sys.path.append('.')
import argparse
import datetime
import logging
import inspect
from typing import Dict, Optional, List
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
from tqdm.auto import tqdm
from diffusers.models import UNet3DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from stive.models.concepts_clip import ConceptsCLIPTextModel, ConceptsCLIPTokenizer
from stive.data.dataset import VideoPromptTupleDataset as VideoPromptValDataset, LatentPromptDataset
from stive.utils.ddim_utils import ddim_inversion
from stive.utils.save_utils import save_videos_grid, save_video, save_images
from stive.utils.textual_inversion_utils import add_concepts_embeddings, update_concepts_embedding, init_concepts_embedding
from stive.prompt_attention.attention_register import register_attention_control
from stive.prompt_attention.attention_store import StepAttentionSupervisor
from einops import rearrange, repeat
import os
import wandb
import random
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

unregister_attention_control = handle_memory_attention

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, now)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    return output_dir

from PIL import Image
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
    pretrained_t2v_model_path: str,
    output_dir: str,
    checkpoints_dir: str, 
    train_data: Dict,
    validation_data: Dict,
    inference_conf: Dict, 
    concepts: List[str] = None, 
    pseudo_words: List[str] = None, 
    concepts_num_embedding: int = 1, 
    retain_position_embedding: bool = True, 
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
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb"
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="stive-concepts-fine-tune", 
            config={'train_data': train_data, 'inference_conf': inference_conf, 'validation_data': validation_data},
            init_kwargs={"wandb": {"name": f'{os.path.basename(checkpoints_dir)}-{datetime.datetime.now().strftime("%Y.%m.%d.%H-%M-%S")}'}}
        )
    
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)
    concepts = OmegaConf.to_container(concepts, resolve=True)
    
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder="scheduler")
    tokenizer = ConceptsCLIPTokenizer.from_pretrained_clip(pretrained_t2v_model_path, subfolder="tokenizer", concepts_list=concepts, concepts_num_embedding=concepts_num_embedding)
    text_encoder = ConceptsCLIPTextModel.from_pretrained_clip(pretrained_t2v_model_path, subfolder="text_encoder", concepts_list=concepts, concepts_num_embedding=concepts_num_embedding, retain_position_embedding=retain_position_embedding)
    vae = AutoencoderKL.from_pretrained(pretrained_t2v_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_t2v_model_path, subfolder="unet")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.concepts_embedder.requires_grad_(True)
    unet.requires_grad_(False)
    if not enable_scam_loss and not enable_tcam_loss:
        handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, gradient_checkpointing, unet)

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

    optimizer = optimizer_cls(
        [text_encoder.concepts_embedder.weight],
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    
    train_dataset = LatentPromptDataset(**train_data)
    val_dataset = VideoPromptValDataset(**validation_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    clip_tokenizer = CLIPTokenizer.from_pretrained(pretrained_t2v_model_path, subfolder="tokenizer")
    clip_text_encoder = CLIPTextModel.from_pretrained(pretrained_t2v_model_path, subfolder="text_encoder")
    clip_text_encoder.requires_grad_(False)
    if pseudo_words is not None:
        init_concepts_embedding(tokenizer=clip_tokenizer, text_encoder=clip_text_encoder, pseudo_tokens=pseudo_words, concept_tokens=concepts, concept_text_encoder=text_encoder)
        text_encoder.concepts_embedder.requires_grad_(True)
    add_concepts_embeddings(clip_tokenizer, clip_text_encoder, concept_tokens=concepts, concept_embeddings=text_encoder.concepts_embedder.weight.detach().clone())
    
    validation_pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_t2v_model_path, vae=vae, text_encoder=clip_text_encoder, tokenizer=clip_tokenizer, unet=unet, 
    )
    validation_pipeline.enable_vae_slicing()
    
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_t2v_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(inference_conf.num_inv_steps, device=accelerator.device)

    num_train_steps = (num_train_epoch * len(train_dataloader)) // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_epoch_rate * num_train_steps, 
        num_training_steps=num_train_steps, 
    )
    
    text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(dtype=weight_dtype)
    text_encoder.concepts_embedder.to(dtype=torch.float)
    clip_text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    unet.eval()
    clip_text_encoder.eval()
    
    if enable_scam_loss or enable_tcam_loss:
        supervisor = StepAttentionSupervisor()
        register_attention_control(unet, supervisor, only_cross=True, replace_attn_prob=False)
    
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    logger.info(f"  Concepts list = {concepts}")
    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, num_train_epoch):
        text_encoder.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_scam_loss = 0.0
        train_tcam_loss = 0.0
        optimizer.zero_grad()
        with tqdm(train_dataloader) as progress_bar:
            for step, batch in enumerate(progress_bar):
                with accelerator.autocast():
                    with accelerator.accumulate(text_encoder):
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

                        # if hasattr(tokens, 'replace_indices') and hasattr(tokens, 'concept_indices'):
                        #     encoder_hidden_states = text_encoder(tokens.input_ids.to(latents.device), tokens.replace_indices, tokens.concept_indices)[0]
                        # else:
                        #     encoder_hidden_states = text_encoder(tokens.input_ids.to(latents.device))[0]
                        encoder_hidden_states = text_encoder(tokens.input_ids.to(latents.device), tokens.replace_indices, tokens.concept_indices)[0]
                        
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        torch.cuda.empty_cache()
                        
                        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        scam_loss = 0.0
                        tcam_loss = 0.0
                        if enable_scam_loss and tokens.replace_indices is not None:
                            scam_loss, masks_dict = supervisor.get_cross_attn_mask_loss(mask=masks, target_indices=tokens.replace_indices, sub_sot=sub_sot, only_neg=scam_only_neg, loss_type=cam_loss_type, reduction=cam_loss_reduction)
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
                            target_prompts = batch['target_prompts']            # [B]
                            target_tokens = tokenizer(target_prompts, return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
                            target_encoder_hidden_states = encoder_hidden_states = text_encoder(target_tokens.input_ids.to(latents.device), target_tokens.replace_indices, target_tokens.concept_indices)[0]
                            if target_tokens.replace_indices is not None:
                                timesteps = ddim_inv_scheduler.timesteps
                                t = timesteps[random.randint(0, len(timesteps)-1)].long()
                                target_latents = batch['target_latents']            # [B, F, C, H, W]
                                target_masks = batch['target_masks']                # [B, F, 1, H, W]
                                target_latents = rearrange(target_latents, "b f c h w -> b c f h w")
                                unet(target_latents, t, target_encoder_hidden_states)
                                tcam_loss, masks_dict = supervisor.get_cross_attn_mask_loss(mask=target_masks, target_indices=tokens.replace_indices, sub_sot=sub_sot, only_neg=tcam_only_neg, loss_type=cam_loss_type, reduction=cam_loss_reduction)
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
                            accelerator.clip_grad_norm_(text_encoder.parameters(), max_grad_norm)
                            for name, module in text_encoder.named_modules():
                                if name.endswith(tuple(['concepts_embedder'])):
                                    for param in module.parameters():
                                        if param.grad is None:
                                            print(f"Epoch {epoch}, Parameter: {name}, Gradient: {param.grad}, Requires_grad: {param.requires_grad}")

            
                            if (global_step + 1) % gradient_accumulation_steps == 0:
                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()
                                if enable_scam_loss and scam_weight > 1e-6:
                                    scam_loss = accelerator.gather(scam_loss.repeat(batch_size)).mean()
                                    train_scam_loss += scam_loss.item() / gradient_accumulation_steps
                                    accelerator.log({"scam_loss": train_scam_loss}, step=global_step)
                                    
                                if enable_tcam_loss and tcam_weight > 1e-6:
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
                        unregister_attention_control(enable_xformers_memory_efficient_attention, enable_torch_2_attn, gradient_checkpointing, unet)
                    with torch.no_grad(), accelerator.autocast():
                        update_concepts_embedding(clip_tokenizer, clip_text_encoder, concept_tokens=concepts, concept_embeddings=text_encoder.concepts_embedder.weight.detach().clone())
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
                        supervisor = StepAttentionSupervisor()
                        register_attention_control(unet, supervisor, only_cross=True, replace_attn_prob=False)
                        
                    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        text_encoder = accelerator.unwrap_model(text_encoder)
        text_encoder.save_pretrained(os.path.join(output_dir, 'text_encoder'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    accelerator.end_training()
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    shutil.copytree(output_dir, checkpoints_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/concepts_clip/vehicles.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
