import os
import numpy as np
import datetime
import imageio
from typing import Sequence
from PIL import Image, ImageDraw, ImageFont
import cv2
from torchvision import transforms

def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG")

FONT_URL = "https://raw.github.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf"
FONT_PATH = "./docs/OpenSans-Regular.ttf"


def pad(image: Image.Image, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)) -> Image.Image:
    new_image = Image.new(image.mode, (image.width + right + left, image.height + top + bottom), color)
    new_image.paste(image, (left, top))
    return new_image


def download_font_opensans(path=FONT_PATH):
    font_url = FONT_URL
    response = requests.get(font_url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)


def annotate_image_with_font(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    image_w = image.width
    _, _, text_w, text_h = font.getbbox(text)
    line_size = math.floor(len(text) * image_w / text_w)

    lines = textwrap.wrap(text, width=line_size)
    padding = text_h * len(lines)
    image = pad(image, top=padding + 3)

    ImageDraw.Draw(image).text((0, 0), "\n".join(lines), fill=(0, 0, 0), font=font)
    return image


def annotate_image(image: Image.Image, text: str, font_size: int = 15):
    if not os.path.isfile(FONT_PATH):
        download_font_opensans()
    font = ImageFont.truetype(FONT_PATH, size=font_size)
    return annotate_image_with_font(image=image, text=text, font=font)


def save_images_as_gif(
    images: Sequence[Image.Image],
    save_path: str,
    loop=0,
    duration=100,
    optimize=False,
) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        loop=loop,
        duration=duration,
    )

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



def save_images_as_folder(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for index, image in enumerate(images):
        init_image = image
        if len(np.array(init_image).shape) == 3:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.png"), np.array(init_image)[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.png"), np.array(init_image))


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
    
    
import torch
import numpy as np
def load_masks(video_path, sample_stride=1, num_frames=None, height=None, width=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = range(0, frame_count, sample_stride)
    
    for idx in sample_indices:
        if num_frames is not None and len(frames) >= num_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            break
        if height is not None and width is not None:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return (torch.from_numpy(np.asarray(frames)) / 255.).mean(dim=-1, keepdim=True)