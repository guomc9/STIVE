import numpy as np
import datetime
import imageio
from typing import Sequence
from PIL import Image, ImageDraw, ImageFont

def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

def save_images_as_mp4(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:

    writer_edit = imageio.get_writer(
        save_path,
        fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()
    
def save_gif_mp4_folder_type(images, save_path, save_gif=True):

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]
    elif isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(i.cpu().clone()[0]) for i in images]
    save_path_mp4 = save_path.replace('gif', 'mp4')
    save_path_folder = save_path.replace('.gif', '')
    if save_gif: save_images_as_gif(images, save_path)
    save_images_as_mp4(images, save_path_mp4)
    save_images_as_folder(images, save_path_folder)