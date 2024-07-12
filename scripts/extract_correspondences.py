import json
import os
import cv2
import torch
import argparse
import numpy as np
import pickle
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from einops import rearrange
import imageio

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_video_frames(video_path, height=None, width=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        if height is not None and width is not None:
            frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_LANCZOS4)
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

def get_patch_indices(box, image_size, patch_size):
    x_min, y_min, x_max, y_max = box
    img_h, img_w = image_size
    patch_h, patch_w = patch_size

    patch_x_min = max(int(x_min // patch_w), 0)
    patch_y_min = max(int(y_min // patch_h), 0)
    patch_x_max = min(int(x_max // patch_w), img_w // patch_w - 1)
    patch_y_max = min(int(y_max // patch_h), img_h // patch_h - 1)

    patch_indices = []
    for y in range(patch_y_min, patch_y_max + 1):
        for x in range(patch_x_min, patch_x_max + 1):
            patch_indices.append((y, x))

    return patch_indices

def draw_boxes(image, boxes):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    return image

def create_mask_matrix(patches, height, width, patch_size):
    mask = np.zeros((height // patch_size, width // patch_size), dtype=np.uint8)
    for (y, x) in patches:
        mask[y, x] = 255
    return mask

def create_unet_mask(boxes, frame_size):
    height, width = frame_size
    unet_mask = np.zeros((height, width), dtype=np.uint8)
    
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        unet_mask[y_min:y_max, x_min:x_max] = 255
    
    return unet_mask


def pool(input_matrix, kernel_size=3, stride=3):
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    weight = torch.ones(1, 1, kernel_size, kernel_size)
    convolved = torch.nn.functional.conv2d(input_tensor, weight, stride=stride)
    convolved[convolved > 0] = 255
    output_tensor = convolved.int()
    return output_tensor.squeeze().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='Path to the dataset directory', required=True)
    parser.add_argument('-t', '--thresh', type=float, help='Threshold for object detection', required=False, default=0.01)
    parser.add_argument('-u', '--unet_size_list', type=list, help='', required=False, default=[64, ])
    # parser.add_argument('-H', '--height', type=int, help='Resize frame height', required=False, default=224)
    # parser.add_argument('-W', '--width', type=int, help='Resize frame width', required=False, default=224)
    # parser.add_argument('-p', '--patch_size', type=int, help='Patch size', required=False, default=16)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    thresh = args.thresh
    # height = args.height
    # width = args.width
    # patch_size = args.patch_size
    height = 512
    width = 512
    abstracts_path = os.path.join(dataset_dir, 'abstracts.json')
    pseudo_labels_path = os.path.join(dataset_dir, 'pseudo_labels.json')
    check_dir = os.path.join(dataset_dir, 'check')
    pkl_path = os.path.join(dataset_dir, 'video_abstract_patches.pkl')
    os.makedirs(check_dir, exist_ok=True)

    abstracts_data = load_json(abstracts_path)
    pseudo_labels_data = load_json(pseudo_labels_path)

    processor = OwlViTProcessor.from_pretrained("./checkpoints/google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("./checkpoints/google/owlvit-base-patch16")
    owl_size = (768, 768)
    patch_size = 16
    all_patches = {}

    for video_name, abstract_words in abstracts_data['video_abstracts'].items():
        video_path = os.path.join(dataset_dir, f"videos/{video_name}.mp4")
        # frames = load_video_frames(video_path, height=owl_size[0], width=owl_size[1])
        frames = load_video_frames(video_path)
        
        for word in abstract_words:
            if word in pseudo_labels_data['pseudo_labels']:
                pseudo_label = pseudo_labels_data['pseudo_labels'][word]
                masks = []
                resized_frames = []
                boxed_frames = []
                frame_masks = []
                all_patches[word] = {}
                
                for frame_index, frame in enumerate(frames):
                    owl_frame = cv2.resize(frame, owl_size)
                    inputs = processor(text=[pseudo_label], images=owl_frame, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)

                    target_sizes = torch.Tensor([owl_frame.shape[:2]])
                    results = processor.post_process_object_detection(outputs=outputs, threshold=thresh, target_sizes=target_sizes)

                    patches = []
                    for i in range(len(results)):
                        boxes = results[i]["boxes"]
                        for box in boxes:
                            patch_indices = get_patch_indices(box.tolist(), owl_frame.shape[:2], (patch_size, patch_size))
                            patches.extend(patch_indices)

                    frame_with_boxes = draw_boxes(owl_frame.copy(), boxes)

                    boxed_frames.append(frame_with_boxes)
                    mask = create_mask_matrix(patches, owl_size[0], owl_size[1], patch_size)
                    mask = pool(mask)
                    masks.append(mask)
                    resized_frame = frame
                    if frame.shape[0] != height or frame.shape[1] != width:
                        resized_frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_LANCZOS4)
                    resized_frames.append(resized_frame)

                    frame_mask = create_unet_mask(boxes, owl_size)
                    frame_masks.append(frame_mask)
                

                masks = np.array(masks)
                masks = np.expand_dims(masks, axis=-1)                      # [F, H, W, 1]
                resized_frames = np.array(resized_frames)
                boxed_frames = np.array(boxed_frames)
                frame_masks = np.expand_dims(frame_masks, axis=-1)          # [F, H, W, 1]
                frame_masks = np.array(frame_masks)

                all_patches[word]['video_name'] = video_name   
                all_patches[word]['frame_masks'] = frame_masks  # [F, H, W, 1]
                all_patches[word]['frames'] = resized_frames    # [F, H, W, C]
                all_patches[word]['masks'] = masks              # [F, H, W, 1]
                masks = np.repeat(masks, 3, axis=-1)                        # [F, H, W, 3]
                frame_masks = np.repeat(frame_masks, 3, axis=-1)            # [F, H, W, 3]

                
                boxed_frames = rearrange(boxed_frames, "t h w c -> c t h w")
                resized_frames = rearrange(resized_frames, "t h w c -> c t h w")
                masks = rearrange(masks, "t h w c -> c t h w")
                boxed_frames_save_path = os.path.join(check_dir, 'boxed_frames', f"{video_name}_{word}_boxed_frame.gif")
                save_video(torch.from_numpy(boxed_frames), boxed_frames_save_path)
                print(f'save to {boxed_frames_save_path}.')
                frames_save_path = os.path.join(check_dir, 'frames', f"{video_name}_{word}_frames.gif")
                save_video(torch.from_numpy(resized_frames), frames_save_path)
                print(f'save to {frames_save_path}.')
                masks_save_path = os.path.join(check_dir, 'masks', f"{video_name}_{word}_masks.gif")
                save_video(torch.from_numpy(masks), masks_save_path)
                print(f'save to {masks_save_path}.')

                frame_masks = rearrange(frame_masks, "t h w c -> c t h w")
                frame_masks_save_path = os.path.join(check_dir, 'frame_masks', f"{video_name}_{word}_frame_masks.gif")
                save_video(torch.from_numpy(frame_masks), frame_masks_save_path)
                print(f'save to {frame_masks_save_path}.')


                
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_patches, f)
    print(f'save pkl to {pkl_path}.')

if __name__ == "__main__":
    main()