import os
import argparse
import subprocess
from pathlib import Path

def convert_mp4_to_gif(input_file, output_file, fps=10):
    """
    Convert MP4 file to looping GIF using ffmpeg
    :param input_file: Path to input MP4 file
    :param output_file: Path to output GIF file
    :param fps: Frames per second for the output GIF (default: 10)
    """
    try:
        command = [
            'ffmpeg',
            '-i', str(input_file),
            '-vf', f'fps={fps},scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
            '-loop', '0',
            str(output_file)
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Converted: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e}")

def process_directory(root_dir, fps=10):
    """
    Process all subdirectories, find 'videos' directories, and convert MP4 files to GIFs
    :param root_dir: Root directory to start searching from
    :param fps: Frames per second for the output GIFs (default: 10)
    """
    root_path = Path(root_dir)
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            videos_dir = subdir / 'videos'
            if videos_dir.is_dir():
                for mp4_file in videos_dir.glob('*.mp4'):
                    gif_file = mp4_file.with_suffix('.gif')
                    convert_mp4_to_gif(mp4_file, gif_file, fps)

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 files in videos subdirectories to looping GIFs')
    parser.add_argument('-d', '--directory', help='Root directory path to search for videos subdirectories')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for the output GIFs (default: 10)')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    process_directory(args.directory, args.fps)

if __name__ == "__main__":
    main()