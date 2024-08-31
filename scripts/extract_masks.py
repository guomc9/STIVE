import json
import os
import cv2
import torch
import argparse
import numpy as np
import pickle
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from einops import rearrange
import imageio


def load_video_frames(video_path, height=None, width=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        if height is not None and width is not None:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def save_video(video: torch.Tensor, path: str, rescale=False, fps=8):
    video = rearrange(video, "c t h w -> t h w c")
    if rescale:
        video = (video / 2 + 0.5).clamp(0, 1)
    video = video.cpu().numpy().astype(np.uint8)
    frames = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(video.shape[0]):
        frames.append(video[i])

    imageio.mimsave(path, frames, fps=fps)


def draw_boxes(image, boxes):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    return image


def create_unet_mask(boxes, frame_size):
    height, width = frame_size
    unet_mask = np.zeros((height, width), dtype=np.uint8)
    
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        unet_mask[y_min:y_max, x_min:x_max] = 255
    
    return unet_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, help='Path to the video file', required=True)
    parser.add_argument('-t', '--thresh', type=float, help='Threshold for object detection', required=False, default=0.05)
    parser.add_argument('-s', '--str', type=str, help='Target string for object detection', required=True)
    args = parser.parse_args()

    video_path = args.video_path
    thresh = args.thresh
    target_str = args.str

    height = 512
    width = 512
    check_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'check')
    os.makedirs(check_dir, exist_ok=True)
    masks_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'masks')
    os.makedirs(check_dir, exist_ok=True)
    processor = OwlViTProcessor.from_pretrained("./checkpoints/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("./checkpoints/owlvit-base-patch16").to(device)
    owl_size = (768, 768)
    patch_size = 16

    frames = load_video_frames(video_path)
    boxed_frames = []
    frame_masks = []
    for frame_index, frame in enumerate(frames):
        owl_frame = cv2.resize(frame, owl_size)
        inputs = processor(text=[target_str], images=owl_frame, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([owl_frame.shape[:2]]).to(device)
        results = processor.post_process_object_detection(outputs=outputs, threshold=thresh, target_sizes=target_sizes)
        
        for i in range(len(results)):
            boxes = results[i]["boxes"].cpu()
            frame_with_boxes = draw_boxes(owl_frame.copy(), boxes)
            boxed_frames.append(frame_with_boxes)

            frame_mask = create_unet_mask(boxes, owl_size)
            frame_masks.append(frame_mask)

    boxed_frames = np.array(boxed_frames)
    frame_masks = np.expand_dims(frame_masks, axis=-1)          # [F, H, W, 1]
    frame_masks = np.array(frame_masks)

    frame_masks = np.repeat(frame_masks, 3, axis=-1)            # [F, H, W, 3]

    boxed_frames = rearrange(boxed_frames, "t h w c -> c t h w")
    frame_masks = rearrange(frame_masks, "t h w c -> c t h w")

    boxed_frames_save_path = os.path.join(check_dir, f"{os.path.basename(video_path).split('.')[0]}_{target_str}_boxed_frame.mp4")
    save_video(torch.from_numpy(boxed_frames), boxed_frames_save_path)
    print(f'Saved boxed frames to {boxed_frames_save_path}.')

    frame_masks_save_path = os.path.join(masks_dir, f"{os.path.basename(video_path).split('.')[0]}_{target_str}_frame_masks.mp4")
    save_video(torch.from_numpy(frame_masks), frame_masks_save_path)
    print(f'Saved frame masks to {frame_masks_save_path}.')

if __name__ == "__main__":
    main()