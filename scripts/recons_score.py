import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_ssim(gif_path, mp4_path, height, width):
    gif = cv2.VideoCapture(gif_path)
    mp4 = cv2.VideoCapture(mp4_path)
    
    gif_frame_count = int(gif.get(cv2.CAP_PROP_FRAME_COUNT))
    mp4_frame_count = int(mp4.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = min(gif_frame_count, mp4_frame_count)
    
    psnr_values = []
    ssim_values = []
    
    for i in range(frame_count):
        success_gif, frame_gif = gif.read()
        success_mp4, frame_mp4 = mp4.read()
        
        if not success_gif or not success_mp4:
            print(f"Frame {i+1}: Could not read frame, stopping.")
            break

        frame_gif = cv2.resize(frame_gif, (width, height))
        frame_mp4 = cv2.resize(frame_mp4, (width, height))
        
        frame_gif = cv2.cvtColor(frame_gif, cv2.COLOR_BGR2RGB)
        frame_mp4 = cv2.cvtColor(frame_mp4, cv2.COLOR_BGR2RGB)
        
        psnr_value = psnr(frame_gif, frame_mp4)
        psnr_values.append(psnr_value)
        
        # 设置较小的窗口大小并指定通道轴
        ssim_value, _ = ssim(frame_gif, frame_mp4, win_size=3, channel_axis=2, full=True)
        ssim_values.append(ssim_value)
        
        print(f"Frame {i+1}: PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}")
    
    gif.release()
    mp4.release()
    
    return psnr_values, ssim_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR and SSIM between corresponding frames of a GIF and MP4 video.")
    parser.add_argument('-g', '--gif_path', type=str, required=True, help='Path to the GIF file')
    parser.add_argument('-m', '--mp4_path', type=str, required=True, help='Path to the MP4 file')
    parser.add_argument('-H', '--height', type=int, required=False, help='Height to resize the frames to', default=512)
    parser.add_argument('-W', '--width', type=int, required=False, help='Width to resize the frames to', default=512)
    
    args = parser.parse_args()
    
    psnr_values, ssim_values = calculate_psnr_ssim(args.gif_path, args.mp4_path, args.height, args.width)
    
    print(f"Average PSNR: {np.mean(psnr_values):.2f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")