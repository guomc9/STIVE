import os
import argparse
from PIL import Image

def gif_to_png(gif_path, output_dir=None, verbose=False):
    parent_dir = os.path.dirname(gif_path)
    base_name = os.path.splitext(os.path.basename(gif_path))[0]
    
    if output_dir:
        output_folder = os.path.join(output_dir, base_name)
    else:
        output_folder = os.path.join(parent_dir, base_name)
    
    os.makedirs(output_folder, exist_ok=True)
    
    with Image.open(gif_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            rgb_img = img.convert('RGB')
            output_path = os.path.join(output_folder, f"frame_{i:03d}.png")
            rgb_img.save(output_path, 'PNG')
    
    if verbose:
        print(f"Processed: {gif_path}")

def process_directory(directory, output_dir=None, verbose=False):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.gif'):
                gif_path = os.path.join(root, file)
                gif_to_png(gif_path, output_dir, verbose)

def main():
    parser = argparse.ArgumentParser(description="Convert GIF files to PNG sequences.")
    parser.add_argument("-i", "--directory", help="Directory containing GIF files to process")
    parser.add_argument("-o", "--output", help="Output directory for PNG sequences (default: same as input)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    if args.verbose:
        print(f"Processing directory: {args.directory}")
    
    process_directory(args.directory, args.output, args.verbose)
    
    if args.verbose:
        print("Processing complete")

if __name__ == "__main__":
    main()