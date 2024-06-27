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

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from tive.data.dataset import TGVEDataset
from tive.models.vae_3d import AutoencoderKL3D
from diffusers.models.vae import AutoencoderKL
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
    eval_data: Dict,
    st_insertion_up_block_ids: Tuple[int] = (0), 
    batch_size: int = 1,
    mixed_precision: Optional[str] = "fp16",
    seed: Optional[int] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
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
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load VAE
    # vae = AutoencoderKL3D.from_pretrained_2d(pretrained_model_path, subfolder="vae", st_insertion_up_block_ids=st_insertion_up_block_ids)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    vae.requires_grad_(False)

    # Get the training dataset
    eval_dataset = TGVEDataset(**eval_data)

    # DataLoaders creation:
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size
    )

    # Prepare everything with our `accelerator`.
    vae, eval_dataloader = accelerator.prepare(
        vae, eval_dataloader
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.eval()
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(seed)

        with torch.no_grad():
            with tqdm(eval_dataloader) as progress_bar:
                for _, batch in enumerate(progress_bar):
                    names = []
                    source = batch["videos"]
                    prompt = batch["prompts"]
                    video_length = source.shape[1]
                    source = rearrange(source, "b f h w c-> (b f) c h w")
                    # pred = vae(source, video_length, sample_posterior=False).sample
                    pred = vae(source, sample_posterior=True).sample
                    sources = rearrange(source, "(b f) c h w -> b c f h w", f=video_length)
                    samples = rearrange(pred, "(b f) c h w-> b c f h w", f=video_length)
                    names.extend(prompt)
                    
                    for name, sample, source in zip(names, samples, sources):
                        save_path = f"{output_dir}/pred/{name}.gif"
                        save_video(sample, save_path, rescale=True)
                        logger.info(f"Saved samples to {save_path}")
                        save_path = f"{output_dir}/gt/{name}.gif"
                        save_video(source, save_path, rescale=True)
                        logger.info(f"Saved sources to {save_path}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/vae3d/loveu-tgve.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
