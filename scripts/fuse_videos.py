import cv2
import os
import torch
import numpy as np
import argparse

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))
        frames.append(frame)
    cap.release()
    return np.array(frames)

def write_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(os.path.join(output_path, 'fuse.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

def fuse_videos(video1_path, video2_path, output_path, weight1=0.5, weight2=0.5):
    frames1 = read_video(video1_path)
    frames2 = read_video(video2_path)
    assert frames1.shape == frames2.shape, "The two videos must have the same shape"
    
    frames1_tensor = torch.from_numpy(frames1).float()
    frames2_tensor = torch.from_numpy(frames2).float()
    
    fused_frames_tensor = weight1 * frames1_tensor + weight2 * frames2_tensor
    fused_frames_tensor = fused_frames_tensor / (weight1 + weight2)  # Normalize the weights

    fused_frames = fused_frames_tensor.byte().numpy()
    
    cap = cv2.VideoCapture(video1_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    write_video(fused_frames, output_path, fps)

def main():
    parser = argparse.ArgumentParser(description="Fuse two videos with given weights")
    parser.add_argument('-s', "--video1_path", type=str, help="Path to the first video")
    parser.add_argument('-t', "--video2_path", type=str, help="Path to the second video")
    parser.add_argument('-o', "--output_path", type=str, help="Path to save the fused video")
    parser.add_argument('-w', "--weight1", type=float, default=0.5, help="Weight for the first video (default: 0.5)")

    args = parser.parse_args()

    fuse_videos(args.video1_path, args.video2_path, args.output_path, args.weight1, 1 - args.weight1)

if __name__ == "__main__":
    main()