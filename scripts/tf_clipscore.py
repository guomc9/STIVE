import clip
import torch
from PIL import Image, ImageSequence
import argparse
import os

def load_gif_frames(gif_path):
    """ Load all frames from a GIF file """
    with Image.open(gif_path) as gif:
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    return frames

def compute_cosine_similarity_image_text(frames, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    
    # Tokenize text
    text = clip.tokenize([text]).to(device)
    
    similarities = []
    
    # Process each frame
    for frame in frames:
        image = preprocess(frame).unsqueeze(0).to(device)

        # Encode image and text
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        # Compute cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        similarities.append(cosine_similarity.item())
    
    return similarities

def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity between frames of a GIF and text.")
    parser.add_argument("-g", "--gif", required=True, help="Path to the GIF file")

    args = parser.parse_args()

    # Extract filename without extension as text
    base_text = os.path.splitext(os.path.basename(args.gif))[0]
    print(base_text)

    # Load GIF frames
    frames = load_gif_frames(args.gif)

    # Compute similarities for each frame
    similarities = compute_cosine_similarity_image_text(frames, base_text)
    
    # # Print similarities
    # for i, similarity in enumerate(similarities):
    #     print(f"Frame {i}: Cosine Similarity = {similarity}")
    
    print(f"Text faithfulness CLIP-score: {sum(similarities) / len(similarities)}")

if __name__ == "__main__":
    main()