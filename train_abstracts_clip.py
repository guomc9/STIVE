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
import shutil
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from stive.models.abstracts_clip import AbstractsCLIPTextModel, AbstractsCLIPTokenizer
from sd.models.unet import UNet3DConditionModel
from sd.data.dataset import TGVEDataset, TGVEValDataset
from sd.pipelines.pipeline_stable_diffusion_vid import StableDiffusionVidPipeline

from sd.util import save_videos_grid, save_video, pad_to_tensor, ddim_inversion
from einops import rearrange, repeat
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    checkpoints_dir: str, 
    train_data: Dict,
    validation_data: Dict,
    inference_conf: Dict, 
    validation_steps: int = 200,
    abstracts_file: str = None, 
    abstracts_num_embedding: int = 1, 
    retain_position_embedding: bool = True, 
    attention_mask_one_hot: bool = False, 
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
    seed: Optional[int] = None, 
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    accelerator.state.ddp_kwargs = {'find_unused_parameters': True}
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        # os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if abstracts_file is None and os.path.exists(abstracts_file):
        raise Exception(f"abstracts_file is None or not exists.")
    else:
        with open(abstracts_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            abstracts_list = data['abstracts']

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    tokenizer = AbstractsCLIPTokenizer.from_pretrained_clip(pretrained_model_path, subfolder="tokenizer", abstracts_list=abstracts_list, abstracts_num_embedding=abstracts_num_embedding)
    text_encoder = AbstractsCLIPTextModel.from_pretrained_clip(pretrained_model_path, subfolder="text_encoder", abstracts_list=abstracts_list, abstracts_num_embedding=abstracts_num_embedding, retain_position_embedding=retain_position_embedding)
    
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.abstracts_embedder.requires_grad_(True)
    unet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
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
        list(unet.parameters()) + list(text_encoder.parameters()),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    
    # Get the training dataset
    train_dataset = TGVEDataset(**train_data)
    val_dataset = TGVEValDataset(**validation_data)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )

    # Get the validation pipeline
    validation_pipeline = StableDiffusionVidPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )

    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(inference_conf.num_inv_steps)

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


    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    if accelerator.is_main_process:
        accelerator.init_trackers("abstracts_clip-fine-tune")

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
        unet.train()
        text_encoder.train()
        train_loss = 0.0
        with tqdm(train_dataloader) as progress_bar:
            for step, batch in enumerate(progress_bar):
                with accelerator.accumulate(text_encoder):
                    # pixel_values = batch["frames"].to(weight_dtype)
                    pixel_values = batch["frames"]
                    # pixel_values = batch["frames"]
                    frame_masks = batch['frame_masks']              # [B, A, F, H, W, 1]
                    frame_masks = repeat(frame_masks, 'b a f h w c -> b a n f h w c', n=text_encoder.abstracts_num_embedding)
                    frame_masks = rearrange(frame_masks, 'b a n f h w c -> b (a n) f h w c')
                    print(frame_masks.shape)
                    video_length = pixel_values.shape[1]
                    pixel_values = rearrange(pixel_values, "b f h w c -> (b f) c h w")

                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each video
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    tokens = tokenizer(batch['prompts'], return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)

                    if hasattr(tokens, 'replace_indices') and hasattr(tokens, 'abstract_indices'):
                        encoder_hidden_states = text_encoder(tokens.input_ids.to(latents.device), tokens.replace_indices, tokens.abstract_indices)[0]
                    else:
                        encoder_hidden_states = text_encoder(tokens.input_ids.to(latents.device))[0]

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                    # Predict the noise residual and compute loss
                    replace_masks = torch.zeros((bsz, encoder_hidden_states.shape[1]), device=encoder_hidden_states.device, dtype=torch.bool)                   # [B, T]
                    print(f'frame_masks.shape: {frame_masks.shape}')
                    pad_replace_indices = pad_to_tensor(tokens.replace_indices, max_length=frame_masks.shape[1])                                                # [B, A]
                    print(f'pad_replace_indices: {pad_replace_indices}')
                    replace_masks[torch.arange(pad_replace_indices.shape[0]).unsqueeze(1), pad_replace_indices] = True      # [B, T]

                    replace_masks = replace_masks.to(encoder_hidden_states.device)
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, attention_mask=frame_masks, attention_mask_one_hot=attention_mask_one_hot, replace_masks=replace_masks).sample
                    
                    # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, attention_mask=None, replace_masks=None).sample
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                    train_loss += avg_loss.item() / gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(text_encoder.parameters(), max_grad_norm)
                        for name, module in text_encoder.named_modules():
                            if name.endswith(tuple(['abstracts_embedder'])):
                                for param in module.parameters():
                                    if param.grad is None:
                                        print(f"Epoch {epoch}, Parameter: {name}, Gradient: {param.grad}, Requires_grad: {param.requires_grad}")

        
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
                    with torch.no_grad():
                        unet.eval()
                        text_encoder.eval()
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)
                        prompts_samples = {}
                        for _, batch in enumerate(val_dataloader):
                            pixel_values = batch["frames"].to(weight_dtype)
                            prompts = batch['prompts']
                            # depth_values = batch["depth_values"].to(weight_dtype)
                            print(f'val prompts: {prompts}')
                            video_length = pixel_values.shape[1]
                            pixel_values = rearrange(pixel_values, "b f h w c -> (b f) c h w")
                            # depth_values = rearrange(depth_values, "b f c h w -> b c f h w")

                            latents = vae.encode(pixel_values).latent_dist.sample()
                            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                            latents = latents * 0.18215
                            
                            noise = torch.randn_like(latents)
                            bsz = latents.shape[0]
                            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                            timesteps = timesteps.long()

                            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                            # token_ids = tokenizer(batch['prompts'], return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)['input_ids'].to(latents.device)
                            # encoder_hidden_states = text_encoder(token_ids)
                            ddim_inv_latent = None
                            
                            if inference_conf.use_inv_latent:
                                # inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                                ddim_inv_latent = ddim_inversion(
                                    validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=inference_conf.num_inv_steps, prompt="")[-1].to(weight_dtype)
                                # torch.save(ddim_inv_latent, inv_latents_path)

                            samples = validation_pipeline(prompts, video_length=video_length, generator=generator, latents=ddim_inv_latent, 
                                                        **validation_data, **inference_conf).videos
                            
                            for p, s in zip(prompts, samples):
                                prompts_samples[p] = s
                            
                        if accelerator.is_main_process:
                            samples = []
                            prompts_samples = accelerator.gather(prompts_samples)
                            for prompt, sample in prompts_samples.items():
                                save_video(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                                samples.append(sample)
                            samples = torch.stack(samples)
                            save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                            save_videos_grid(samples, save_path)
                            logger.info(f"Saved samples to {save_path}")


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        text_encoder = accelerator.unwrap_model(text_encoder)
        text_encoder.save_pretrained(os.path.join(output_dir, 'text_encoder'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    accelerator.end_training()
    os.makedirs(checkpoints_dir, exist_ok=True)
    shutil.copytree(output_dir, checkpoints_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tive.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
