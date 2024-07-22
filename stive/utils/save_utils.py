import os
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b t c h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = ((x + 1.0) / 2.0).clamp(0, 1)  # -1,1 -> 0,1、
        else:
            x = x.clamp(0, 1)
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_images(images, rescale=True, save_path=None):
    """
    Store a batch of images in a single tensor by arranging them horizontally.
    
    Args:
        images (torch.Tensor): Input image tensor of shape [B, C, H, W].
        rescale_images (bool): Whether to rescale the images from [-1, 1] to [0, 1].
        save_path (str): Path to save the arranged image. If None, the image won't be saved.
        
    Returns:
        torch.Tensor: A single image tensor with shape [C, H, W * B] arranged horizontally.
    """
    if rescale:
        images = (images + 1) / 2
    
    B, C, H, W = images.shape
    # Reshape and permute to arrange images horizontally
    horizontal_image = images.permute(1, 2, 3, 0).reshape(C, H, W * B)
    
    if save_path is not None:
        # Convert tensor to numpy array and save using PIL
        numpy_image = horizontal_image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        pil_image = Image.fromarray((numpy_image * 255).astype(np.uint8))
        pil_image.save(save_path)
    
    return horizontal_image

def save_video(video: torch.Tensor, path: str, rescale=False, fps=8):
    video = rearrange(video, "t c h w -> t h w c")
    if rescale:
        video = (video / 2 + 0.5)
    video = video.clamp(0, 1)
    video = (video * 255).cpu().numpy().astype(np.uint8)
    frames = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(video.shape[0]):
        frames.append(video[i])

    imageio.mimsave(path, frames, fps=fps)