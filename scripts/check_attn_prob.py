import sys
import argparse
import datetime
import logging
import os
import torch
import torch.nn.functional as F
from einops import rearrange
from diffusers import AutoencoderKL, DDIMScheduler, UNet3DConditionModel, TextToVideoSDPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import numpy as np
import cv2
import random
import imageio
from PIL import Image
from stive.prompt_attention.attention_register import register_attention_control
from stive.prompt_attention.attention_store import StepAttentionSupervisor

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def collect_cross_attention(attns_dict, prompts, video_length):
    b = len(prompts)
    f = video_length
    out = {prompt: {} for prompt in prompts}
    for key, attns in attns_dict.items():
        q = attns.shape[1]
        h = w = int(np.sqrt(q))
        attns = rearrange(attns, '(b f) (h w) k -> b f h w k', b=b, f=f, h=h, w=w)
        for j in range(b):
            tokens = ['SOT'] + prompts[j].split(' ') + ['EOT']
            prompt_frames = []
            for i, token in enumerate(tokens):
                frames = attns[j, ..., i]
                token_frames = [cv2.resize((255 * frame / frame.max()).numpy().astype(np.uint8), (256, 256)) for frame in frames]
                prompt_frames.append(np.stack(token_frames))
            prompt_frames = np.concatenate(prompt_frames, axis=-2)
            out[prompt][key] = prompt_frames
    return out

def save_attention_videos(prompt_attn_dict, save_path):
    os.makedirs(save_path, exist_ok=True)
    for prompt, unet_attns in prompt_attn_dict.items():
        for key, unet_attn in unet_attns.items():
            unet_attn = rearrange(torch.from_numpy(unet_attn), 'f h w c -> f c h w')
            save_video(unet_attn, f'{save_path}/{prompt}_{key}.gif')

def save_video(frames, path):
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).astype(np.uint8)
    imageio.mimsave(path, frames, fps=5)

def main(video_path, prompt):
    accelerator = Accelerator()
    create_logging(logging, logger, accelerator)

    video_frames = [Image.open(frame) for frame in video_path]
    video_length = len(video_frames)
    prompts = [prompt]

    pretrained_model_path = "checkpoints/zeroscope_v2_576w"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path)
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path)
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path)

    pipeline = TextToVideoSDPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    pipeline.enable_vae_slicing()

    supervisor = StepAttentionSupervisor()
    register_attention_control(unet, supervisor, only_cross=True, replace_attn_prob=False)

    generator = torch.Generator(device="cuda")

    with torch.no_grad():
        sample_videos = pipeline(prompt=prompts, generator=generator)

    attn_dict = supervisor.get_mean_head_attns()
    prompt_attn_dict = collect_cross_attention(attn_dict, prompts, video_length)
    save_path = "./attention_videos"
    save_attention_videos(prompt_attn_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args.video_path, args.prompt)