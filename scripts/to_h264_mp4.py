import subprocess
import argparse
import os

def convert_to_h264(input_file, output_file, crf=23, preset='medium', audio_bitrate='128k'):
    """
    Convert input MP4 file to H.264-encoded MP4 file
    :param input_file: Path to input file
    :param output_file: Path to output file
    :param crf: Constant Rate Factor (0-51), default 23
    :param preset: Encoding speed preset, default 'medium'
    :param audio_bitrate: Audio bitrate, default '128k'
    """
    try:
        command = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-c:a', 'aac',
            '-b:a', audio_bitrate,
            output_file
        ]

        subprocess.run(command, check=True)
        print(f"Conversion completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 file to H.264-encoded MP4 file')
    parser.add_argument('-i', '--input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--crf', type=int, default=23, help='Constant Rate Factor (0-51, default: 23)')
    parser.add_argument('--preset', default='medium', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], help='Encoding speed preset (default: medium)')
    parser.add_argument('--audio-bitrate', default='128k', help='Audio bitrate (default: 128k)')

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output if args.output else os.path.splitext(input_file)[0] + '_h264.mp4'

    convert_to_h264(input_file, output_file, args.crf, args.preset, args.audio_bitrate)

if __name__ == "__main__":
    main()