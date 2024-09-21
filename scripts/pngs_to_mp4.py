#!/usr/bin/env python3

import os
import argparse
from moviepy.editor import ImageSequenceClip

def create_mp4_from_pngs(image_folder, output_filename, fps):
    # Get list of PNG files
    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.lower().endswith('.png')]
    image_files.sort()  # Sort files by name
    print(len(image_files))

    if not image_files:
        print('No PNG images found in the specified directory.')
        return

    # Create video clip from images
    clip = ImageSequenceClip(image_files, fps=fps)
    # Write the video file
    clip.write_videofile(output_filename, codec='libx264')
    print(f'Video successfully saved as {output_filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine PNG images into an MP4 video.')
    parser.add_argument('-i', '--image_folder', help='Folder containing PNG images')
    parser.add_argument('-o', '--output', default='output.mp4',
                        help='Output MP4 filename (default: output.mp4)')
    parser.add_argument('-f', '--fps', type=int, default=8,
                        help='Frames per second (default: 24)')

    args = parser.parse_args()
    create_mp4_from_pngs(args.image_folder, args.output, args.fps)