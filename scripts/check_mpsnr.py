import cv2
import numpy as np
import argparse

def load_video(path, target_size=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if target_size:
            frame = cv2.resize(frame, dsize=target_size, interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32) / 255.0

def calculate_masked_psnr(video1, video2, mask):
    print(mask.shape)
    print(np.sum(mask))
    mask = np.where(mask > 0.5, 1, 0)
    # print(np.sum(mask) / (6 * 512 * 512 * 3))
    # Calculate MSE only for masked region
    mse = (np.sum((mask * (video1 - video2) ** 2), axis=(1, 2, 3)) / np.sum(mask, axis=(1, 2, 3))).mean()
    
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    
    return psnr

def process_videos(video1_path, video2_path, mask_path):
    # Load videos and mask
    video1 = load_video(video1_path, target_size=(512, 512))
    video2 = load_video(video2_path, target_size=(512, 512))
    mask = load_video(mask_path, target_size=(512, 512))

    # Ensure all videos have the same number of frames
    min_frames = min(len(video1), len(video2), len(mask))
    video1 = video1[:min_frames]
    video2 = video2[:min_frames]
    mask = mask[:min_frames]
    print(mask.shape)
    psnr = calculate_masked_psnr(video1, video2, 1 - mask)

    return psnr

def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR for masked regions of two videos")
    parser.add_argument("-v1", "--video1", required=True, help="Path to the first video")
    parser.add_argument("-v2", "--video2", required=True, help="Path to the second video")
    parser.add_argument("-m", "--mask", required=True, help="Path to the mask video")

    args = parser.parse_args()

    psnr = process_videos(args.video1, args.video2, args.mask)
    print(f"PSNR for masked regions: {psnr:.2f} dB")

if __name__ == "__main__":
    main()