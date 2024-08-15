import cv2
import numpy as np
import imageio

HEIGHT, WIDTH = 512, 512

def read_gif_with_cv2(gif_path):
    cap = cv2.VideoCapture(gif_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames

def concat_gifs(gif_files, output_file):
    gifs = [read_gif_with_cv2(gif) for gif in gif_files]
    num_frames = len(gifs[0])

    for gif in gifs:
        if len(gif) != num_frames:
            raise ValueError("All GIFs must have the same number of frames.")

    concatenated_frames = []
    for i in range(num_frames):
        frames = [gif[i] for gif in gifs]
        concatenated_frame = np.concatenate(frames, axis=1)
        concatenated_frames.append(concatenated_frame)

    imageio.mimsave(output_file, concatenated_frames, duration=0.1)

# concat_gifs([
#     ".assets/jeep/car-turn.gif", 
#     ".assets/jeep/to-lambo.gif", 
#     ".assets/jeep/to-bmw.gif", 
#     ".assets/jeep/to-ferrari.gif", 
#     ".assets/jeep/to-cybertruck.gif"
#     ], ".assets/jeep/concat.gif")

concat_gifs([
    ".assets/jeep-unet-full-supvis/car-turn.gif", 
    ".assets/jeep-unet-full-supvis/to-lambo.gif", 
    ".assets/jeep-unet-full-supvis/to-bmw.gif", 
    ".assets/jeep-unet-full-supvis/to-ferrari.gif", 
    ".assets/jeep-unet-full-supvis/to-cybertruck.gif"
    ], ".assets/jeep-unet-full-supvis/concat.gif")