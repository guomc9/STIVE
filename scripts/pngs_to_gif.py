#!/usr/bin/env python3

import os
from PIL import Image
import argparse

def create_gif(input_dir, output_filename, duration):
    # Get all PNG files in the input directory
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    files.sort()  # Sort the files by name, adjust sorting if necessary

    if not files:
        print('No PNG images found in the specified directory.')
        return

    # Open images and ensure they have the same mode and size
    images = []
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        img = Image.open(filepath)
        images.append(img)

    # Convert all images to RGBA mode
    images = [img.convert('RGB') for img in images]
    # # Resize images to match the size of the first image
    # width, height = images[0].size
    # images = [img.resize((width, height)) for img in images]
    print(len(images))
    # Save as GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        # loop=0,  # Set loop to 0 for infinite loop
    )
    print(f'Successfully saved GIF: {output_filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine PNG images into a GIF.')
    parser.add_argument('-i', '--input_dir', help='Input directory containing PNG images')
    parser.add_argument('-o', '--output', default='output.gif', help='Output GIF filename (default: output.gif)')
    parser.add_argument('-d', '--duration', type=int, default=100, help='Duration between frames in milliseconds (default: 100)')

    args = parser.parse_args()
    create_gif(args.input_dir, args.output, args.duration)