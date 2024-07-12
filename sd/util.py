# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/util.py
import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
from tqdm import tqdm
from einops import rearrange
import decord
decord.bridge.set_bridge('torch')

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = ((x + 1.0) / 2.0).clamp(0, 1)  # -1,1 -> 0,1
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_video(video: torch.Tensor, path: str, rescale=False, fps=8):
    video = rearrange(video, "c t h w -> t h w c")
    if rescale:
        video = (video / 2 + 0.5)
    video = video.clamp(0, 1)
    video = (video * 255).cpu().numpy().astype(np.uint8)
    frames = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(video.shape[0]):
        frames.append(video[i])

    imageio.mimsave(path, frames, fps=fps)

def pad_to_tensor(list_of_lists, padding_value=0, max_length=None):
    if max_length is None:
        max_length = max(len(sublist) for sublist in list_of_lists)
    padded_list = [sublist + [padding_value] * (max_length - len(sublist)) for sublist in list_of_lists]
    tensor = torch.from_numpy(np.array(padded_list))
    
    return tensor

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    if hasattr(uncond_input, 'replace_indices') and hasattr(uncond_input, 'abstract_indices'):    
        # !!!!! Abstracts BRANCH
        uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device), uncond_input.replace_indices, uncond_input.abstract_indices)[0]
    else:
        uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device), )[0]

    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    if hasattr(uncond_input, 'replace_indices') and hasattr(uncond_input, 'abstract_indices'):
        # !!!!! Abstracts BRANCH
        text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device), text_input.replace_indices, text_input.abstract_indices)[0]
    else:
        text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    f = latent.shape[2]
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)

    all_latent = [latent]
    latent = latent.clone().detach()

    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

