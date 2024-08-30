import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
import argparse
from itertools import combinations

def calculate_frame_similarity(model, processor, frame1, frame2):
    # Process the frames
    inputs = processor(images=[frame1, frame2], return_tensors="pt", padding=True)
    
    # Get image features
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(outputs[0].unsqueeze(0), outputs[1].unsqueeze(0))
    return similarity.item()

def compute_frame_consistency_score(video_path, model, processor):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    
    cap.release()
    
    # Calculate similarities between all pairs of frames
    similarities = []
    for frame1, frame2 in combinations(frames, 2):
        similarity = calculate_frame_similarity(model, processor, frame1, frame2)
        similarities.append(similarity)
    
    # Calculate average frame consistency score
    avg_score = np.mean(similarities)
    return avg_score

def main():
    parser = argparse.ArgumentParser(description="Calculate Frame Consistency Score for a video")
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to the video file")

    args = parser.parse_args()

    # Load pre-trained CLIP model
    model = CLIPModel.from_pretrained("checkpoints/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("checkpoints/clip-vit-large-patch14")

    # Compute and print the average Frame Consistency Score
    avg_consistency_score = compute_frame_consistency_score(args.video, model, processor)
    print(f"{args.video} - Average Frame Consistency Score: {avg_consistency_score}")

if __name__ == "__main__":
    main()