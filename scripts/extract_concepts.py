import os
import cv2
import numpy as np
from tqdm import tqdm

def find_mp4_file(directory):
    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            return os.path.join(directory, file)
    return None

def process_concept_directory(concept_path):
    masks_dir = os.path.join(concept_path, 'masks')
    video_dir = os.path.join(concept_path, 'videos')
    
    if not os.path.exists(masks_dir) or not os.path.exists(video_dir):
        print(f"Required directories not found in {concept_path}")
        return
    
    mask_video = find_mp4_file(masks_dir)
    main_video = find_mp4_file(video_dir)
    
    if not mask_video or not main_video:
        print(f"MP4 files not found in {concept_path}")
        return
    
    concepts_dir = os.path.join(concept_path, 'concepts')
    os.makedirs(concepts_dir, exist_ok=True)
    
    mask_cap = cv2.VideoCapture(mask_video)
    main_cap = cv2.VideoCapture(main_video)
    
    frame_count = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_num in tqdm(range(frame_count), desc=f"Processing {concept_path}"):
        ret_mask, mask_frame = mask_cap.read()
        ret_main, main_frame = main_cap.read()
        
        if not ret_mask or not ret_main:
            break
        
        # Resize mask to 512x512
        mask_resized = cv2.resize(mask_frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        main_frame = cv2.resize(main_frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        # Convert mask to grayscale and binary
        mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        
        # Apply mask
        masked_frame = cv2.bitwise_and(main_frame, main_frame, mask=mask_binary)
        
        # Find bounding box
        y, x = np.where(mask_binary > 0)
        if len(y) == 0 or len(x) == 0:
            continue
        top, bottom, left, right = y.min(), y.max(), x.min(), x.max()
        
        # Crop image
        cropped_frame = masked_frame[top:bottom+1, left:right+1]
        
        # Resize to 512x512
        resized_frame = cv2.resize(cropped_frame, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Save result
        output_path = os.path.join(concepts_dir, f"concept_{frame_num:04d}.png")
        cv2.imwrite(output_path, resized_frame)
    
    mask_cap.release()
    main_cap.release()

def process_root_directory(root_dir):
    for concept_dir in os.listdir(root_dir):
        concept_path = os.path.join(root_dir, concept_dir)
        if os.path.isdir(concept_path):
            process_concept_directory(concept_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process concept videos and create concept images.")
    parser.add_argument('-r', "--root_dir", help="Root directory containing concept video folders.")
    
    args = parser.parse_args()
    
    process_root_directory(args.root_dir)