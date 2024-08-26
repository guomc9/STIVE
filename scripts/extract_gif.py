import argparse
from moviepy.editor import VideoFileClip
import imageio

def extract_frames(video_path, frame_count, frame_interval, output_path):
    video = VideoFileClip(video_path)
    duration = video.duration
    fps = video.fps
    
    # Extract frames
    frames = []
    for i in range(frame_count):
        frame_time = i * frame_interval / fps
        if frame_time < duration:
            frame = video.get_frame(frame_time)
            frames.append(frame)
    
    # Save frames as GIF
    imageio.mimsave(output_path, frames, format='GIF', fps=8)

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 to GIF with frame extraction.')
    parser.add_argument('input_video', type=str, help='Path to the input MP4 video file.')
    parser.add_argument('frame_count', type=int, help='Number of frames to extract.')
    parser.add_argument('frame_interval', type=int, help='Interval between frames in frame count.')
    parser.add_argument('output_path', type=str, help='Path to save the output GIF file.')
    
    args = parser.parse_args()
    
    extract_frames(args.input_video, args.frame_count, args.frame_interval, args.output_path)

if __name__ == '__main__':
    main()
