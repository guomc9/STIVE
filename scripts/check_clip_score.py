import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
from torchvision import transforms
import numpy as np

def calculate_clip_score(model, processor, image, text):
    # Process the image and text
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    
    # Get image and text features
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    # Calculate similarity score
    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    return similarity.item()

def compute_video_clip_score(video_path, text, model, processor):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    frame_scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        
        # Calculate ClipScore for the current frame
        score = calculate_clip_score(model, processor, frame, text)
        frame_scores.append(score)
    
    cap.release()
    
    # Calculate average ClipScore
    avg_score = np.mean(frame_scores)
    return avg_score


if __name__ == '__main__':
    model = CLIPModel.from_pretrained("checkpoints/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("checkpoints/clip-vit-large-patch14")

    video_path = "path/to/your/video.mp4"
    text = "A person playing tennis"

    avg_clip_score = compute_video_clip_score(video_path, text, model, processor)
    print(f"{video_path} - {text} - Average ClipScore: {avg_clip_score}")