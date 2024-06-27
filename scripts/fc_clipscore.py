import clip
import torch
from PIL import Image, ImageSequence
import argparse

def compute_cosine_similarities(frames, model, preprocess, device):
    # 将所有帧转换为模型可以处理的格式
    frame_tensors = [preprocess(frame).unsqueeze(0).to(device) for frame in frames]

    # 使用CLIP模型编码所有帧
    with torch.no_grad():
        features = [model.encode_image(frame_tensor) for frame_tensor in frame_tensors]

    # 计算并存储每对帧之间的余弦相似度
    num_frames = len(features)
    similarities = []
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            cosine_similarity = torch.nn.functional.cosine_similarity(features[i], features[j])
            similarities.append((i, j, cosine_similarity.item()))
    return similarities

def process_gif(gif_path):
    # 加载GIF文件中的所有帧
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    
    # 计算余弦相似度
    similarities = compute_cosine_similarities(frames, model, preprocess, device)
    
    # 打印结果
    scores = []
    for i, j, sim in similarities:
        scores.append(sim)
        print(f"Cosine similarity between frame {i} and frame {j}: {sim}")

    print(f"Frame consistency CLIP-score: {sum(scores) / len(scores)}")

def main():
    parser = argparse.ArgumentParser(description="Compute CLIP cosine similarities between all pairs of frames in a GIF.")
    parser.add_argument('-g', '--gif', required=True, help="Path to the GIF file")
    
    args = parser.parse_args()
    
    process_gif(args.gif)

if __name__ == "__main__":
    main()