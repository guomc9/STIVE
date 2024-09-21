#!/usr/bin/env python3

import os
import subprocess
import argparse

def convert_gif_to_mp4(gif_path, mp4_path):
    """Converts a single GIF file to MP4 format using ffmpeg."""
    # Command to convert GIF to MP4 using ffmpeg
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', gif_path,  # Input file
        '-movflags', 'faststart',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even
        mp4_path  # Output file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'Converted: {gif_path} --> {mp4_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error converting {gif_path}:', e)

def process_directory(root_dir):
    """Recursively finds and converts all GIF files in the given directory."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.gif'):
                gif_path = os.path.join(dirpath, filename)
                mp4_filename = os.path.splitext(filename)[0] + '.mp4'
                mp4_path = os.path.join(dirpath, mp4_filename)
                convert_gif_to_mp4(gif_path, mp4_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recursively convert all GIF files in a directory to MP4.')
    parser.add_argument('-d', '--directory', help='Root directory to search for GIF files')
    args = parser.parse_args()
    process_directory(args.directory)