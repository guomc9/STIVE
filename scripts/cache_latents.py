import os
import argparse
import torch
from diffusers import AutoencoderKL
from accelerate.utils import set_seed
import cv2
from einops import rearrange

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True)
    parser.add_argument('-H', '--height', type=int, default=512)
    parser.add_argument('-W', '--width', type=int, default=512)
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-o', '--save_directory', type=str, required=False, default=None)
    
    args = parser.parse_args()
    base_dir = args.directory
    video_dir = os.path.join(base_dir, 'videos')
    save_dir = args.save_directory
    seed = args.seed
    if seed is not None:
        set_seed(seed)
        
    if save_dir is None:
        save_dir = os.path.join(base_dir, 'latents')
    
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = AutoencoderKL.from_pretrained(PRETRAINED_T2V_CHECKPOINT_PATH, subfolder='vae').to(device)
    vae.eval()

    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            print(f'Encoding {video_path}')
            latents = encode_video_to_latents(video_path, vae, args.height, args.width, device)
            latent_file_path = os.path.join(save_dir, f"{os.path.splitext(video_file)[0]}.pt")
            torch.save(latents, latent_file_path)
            print(f'Saved latents to {latent_file_path}')