import torch
import numpy as np
import cv2
from einops import rearrange
from diffusers import AutoencoderKL

PRETRAINED_T2V_CHECKPOINT_PATH = 'checkpoints/zeroscope_v2_576w'

def encode_video_to_latents(video_path, vae, height, width, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (torch.from_numpy(frame).float() / 255.0 * 2) - 1
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
    
    cap.release()
    chunk = 8
    frames = torch.stack(frames).unsqueeze(0).to(vae.device)
    with torch.no_grad():
        video_length = frames.shape[1]
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')
        latents = []
        for i in range(0, frames.shape[0], chunk):
            latents.append(vae.encode(frames[i:i+chunk]).latent_dist.sample() * 0.18215)
        latents = torch.cat(latents, dim=0)
        latents = rearrange(latents, '(b f) c h w -> b f c h w', f=video_length).squeeze(0)
    return latents.detach().cpu()


def encode_videos_latents(video_paths, height=512, width=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = AutoencoderKL.from_pretrained(PRETRAINED_T2V_CHECKPOINT_PATH, subfolder='vae').to(device)
    vae.eval()
    latents = []
    for video_path in video_paths:
        if video_path.endswith('.mp4'):
            print(f'Encoding {video_path}')
            latents.append(encode_video_to_latents(video_path, vae, height, width, device))
    del vae
    torch.cuda.empty_cache()
    return latents