import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.optim.lr_scheduler import MultiStepLR
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from tive.data.dataset import TGVEDataset, TGVEValDataset
from diffusers.models import AutoencoderKL
from einops import rearrange
from tive.util import save_videos_grid, save_video
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    num_temp_up_blocks: int = 1, 
    num_temp_down_blocks: int = 1, 
    trainable_modules: Tuple[str] = (
        "temp_up_blocks", 
        "temp_down_blocks"
    ),
    batch_size: int = 1,
    num_train_epoch: int = 200,
    learning_rate: float = 3e-5,
    rec_weight: float = 1,
    kl_weight: float = 1e-3, 
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    seed: Optional[int] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load VAE
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", num_temp_down_blocks=num_temp_down_blocks, num_temp_up_blocks=num_temp_up_blocks)
    # torch.nn.MSELoss()
    vae.requires_grad_(False)

    # if gradient_checkpointing:
        # vae.enable_gradient_checkpointing()

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
        vae.parameters(),
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

    num_train_steps = (num_train_epoch * len(train_dataloader)) // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=num_train_steps * gradient_accumulation_steps,
    )
    # milestones = [30, 80]
    # gamma = 0.1
    # scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for m_name, module in vae.named_modules():
        if m_name.endswith(tuple(trainable_modules)):
            for p_name, params in module.named_parameters():
                print(f'CHECK {m_name}.{p_name}.requires_grad: {params.requires_grad}, dtype: {params.dtype}')

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(num_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("tive-vae3d-fine-tune")

    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, num_train_epochs):
        vae.train()
        train_loss = 0.0
        train_kl_loss = 0.0
        train_rec_loss = 0.0
        with tqdm(train_dataloader) as progress_bar:
            for step, batch in enumerate(progress_bar):
                with accelerator.accumulate(vae):
                    # Convert videos to latent space
                    source = batch["videos"].to(weight_dtype)
                    video_length = source.shape[1]
                    
                    source = rearrange(source, "b f h w c-> (b f) c h w")
                    with accelerator.autocast():
                        z = vae.encode(source).latent_dist.sample()

                        z =  rearrange(z, '(b f) c h w -> b f c h w')
                        inp = torch.arange(0, video_length, step=2, device=z.device)

                        rec_loss = F.mse_loss(pred.float(), source.float(), reduction="mean")
                        kl_loss = mid.latent_dist_td.kl().mean()
                        loss = rec_weight * rec_loss + kl_weight * kl_loss
                        
                        avg_rec_loss = accelerator.gather(rec_loss.repeat(batch_size)).mean()
                        avg_kl_loss = accelerator.gather(kl_loss.repeat(batch_size)).mean()
                        avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()

                        train_loss += avg_loss.item() / gradient_accumulation_steps
                        train_kl_loss += avg_kl_loss.item() / gradient_accumulation_steps
                        train_rec_loss += avg_rec_loss.item() / gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        for m_name, module in vae.named_modules():
                            if m_name.endswith(tuple(trainable_modules)):
                                for p_name, param in module.named_parameters():
                                    if param.grad is None:
                                        print(f"Epoch: {epoch}, Module: {m_name}, Parameter: {p_name}, Gradient: {param.grad}")
                        accelerator.clip_grad_norm_(vae.parameters(), max_grad_norm)

                        if (global_step + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    if global_step % validation_steps == 0:
                        vae.eval()
                        if accelerator.is_main_process:
                            samples = []
                            names = []
                            generator = torch.Generator(device=accelerator.device)
                            generator.manual_seed(seed)
                            with torch.no_grad():
                                for _, batch in enumerate(val_dataloader):
                                    source = batch["videos"].to(weight_dtype)
                                    prompt = batch["prompts"]
                                    video_length = source.shape[1]
                                    source = rearrange(source, "b f h w c-> (b f) c h w")
                                    pred = vae(source, video_length, sample_posterior=True).sample
                                    pred = rearrange(pred, "(b f) c h w -> b c f h w", f=video_length)
                                    samples.append(pred)
                                    names.extend(prompt)
                                samples = accelerator.gather(samples)
                                names = accelerator.gather(names)
                                samples = torch.cat(samples)
                                for name, sample in zip(names, samples):
                                    save_path = f"{output_dir}/samples/sample-{global_step}/{name}.gif"
                                    save_video(sample, save_path)

                                save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                                save_videos_grid(samples, save_path)
                                logger.info(f"Saved samples to {save_path}")

                logs = {"Epoch": f'{epoch}/{num_train_epoch}', "Step": f'{global_step}/{num_train_steps}', "step_loss": loss.detach().item(), "rec_loss": rec_loss.detach().item(), "kl_loss": kl_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= num_train_steps:
                    break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/vae3d/loveu-tgve.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
