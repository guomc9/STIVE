import os
import argparse
import imageio

def process_gif(file_path, fps=8):
    """
    Read GIF file, set it to loop infinitely, and overwrite the original file
    :param file_path: Path to the GIF file
    :param fps: Frames per second for the output GIF (default: 8)
    """
    try:
        # Read the input GIF file
        reader = imageio.get_reader(file_path)

        # Extract all frames
        frames = [frame for frame in reader]

        # Save and overwrite the original GIF file with loop=0 for infinite looping
        imageio.mimsave(file_path, frames, fps=fps, loop=0)

        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory, fps=8):
    """
    Recursively process all GIF files in the given directory and its subdirectories
    :param directory: Path to the directory to process
    :param fps: Frames per second for the output GIFs (default: 8)
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.gif'):
                file_path = os.path.join(root, file)
                process_gif(file_path, fps)

def main():
    parser = argparse.ArgumentParser(description='Process all GIF files in a directory to make them loop infinitely')
    parser.add_argument('-d', '--directory', help='Directory path containing GIF files')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for the output GIFs (default: 8)')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    process_directory(args.directory, args.fps)

if __name__ == "__main__":
    main()