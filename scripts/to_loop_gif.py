import imageio
import argparse
import os

def process_gif(input_file, output_file, fps=8):
    """
    Read input GIF file and save as a new infinitely looping GIF file with specified FPS
    :param input_file: Path to input GIF file
    :param output_file: Path to output GIF file
    :param fps: Frames per second for the output GIF (default: 8)
    """
    try:
        # Read the input GIF file
        reader = imageio.get_reader(input_file)

        # Extract all frames
        frames = [frame for frame in reader]

        # Save as a new GIF file with loop=0 for infinite looping and specified fps
        imageio.mimsave(output_file, frames, fps=fps, loop=0)

        print(f"Processing completed: {output_file}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process GIF file to make it loop infinitely with specified FPS')
    parser.add_argument('-i', '--input', help='Input GIF file path')
    parser.add_argument('-o', '--output', help='Output GIF file path')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for the output GIF (default: 8)')

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output if args.output else os.path.splitext(input_file)[0] + '_loop.gif'

    process_gif(input_file, output_file, args.fps)

if __name__ == "__main__":
    main()