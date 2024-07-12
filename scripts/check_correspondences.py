import os
import sys
sys.path.append('.')
import imageio
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from tive.models.abstracts_clip import AbstractsCLIPTokenizer, AbstractsCLIPTextModel, abstracts_contrastive_loss
import cv2
import argparse
from einops import rearrange
import json
import pickle
from accelerate.utils import set_seed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def load_abstracts_patches(pickle_path):
        with open(pickle_path, 'rb') as f:
            patches_data = pickle.load(f)

        return patches_data

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def save_video(video: torch.Tensor, path: str, rescale=False, fps=8):
    video = rearrange(video, "c t h w -> t h w c")
    if rescale:
        video = (video / 2 + 0.5).clamp(0, 1)
    video = (video * 255).cpu().numpy().astype(np.uint8)
    frames = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(video.shape[0]):
        frames.append(video[i])

    imageio.mimsave(path, frames, fps=fps)


def main(pickle_path, abstract_clip_path, output_path):
    video_abstracts_patches = load_abstracts_patches(pickle_path)
    
    tokenizer = AbstractsCLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path=abstract_clip_path, 
        subfolder = 'tokenizer',
        local_files_only=True
    )
    text_encoder = AbstractsCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path=abstract_clip_path, 
        subfolder = 'text_encoder',
        local_files_only=True
    )

    clip = CLIPModel.from_pretrained(
        pretrained_model_name_or_path="checkpoints/openai/clip-vit-large-patch14", 
        local_files_only=True
    )
    processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path="checkpoints/openai/clip-vit-large-patch14", 
        local_files_only=True
    )

    text_encoder.requires_grad_(False)
    clip.requires_grad_(False)

    text_encoder.to(device)
    clip.to(device)
    
    # text_encoder.train()
    text_encoder.eval()
    # clip.eval()

    for video_name, video_abstracts in video_abstracts_patches.items():
        for abstract, values in video_abstracts.items():
            frames = values['frames'] 
            masks = torch.from_numpy(values['masks'] / 255).to(dtype=torch.bool)
            masks = rearrange(masks, 'f h w -> f (h w)').to(device)
            # batch_size = len(frames)
            batch_size = 6
            images = frames[torch.arange(0, batch_size)]
            masks = masks[torch.arange(0, batch_size)]
            inputs = processor(images=images, return_tensors="pt", padding=True)

            tokens = tokenizer(['$' + abstract] * batch_size, prefix='a photo of', return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
            with torch.no_grad():
                text_feat = text_encoder(tokens.input_ids.to(device), tokens.replace_indices, tokens.abstract_indices)[1]
                text_feat = clip.text_projection(text_feat)
                patch_feat = clip.vision_model(pixel_values=inputs['pixel_values'].to(device))[0]
                patch_feat = clip.visual_projection(patch_feat)
                image_feat = patch_feat[:, 0, :]

            # similarity_matrix = torch.cosine_similarity(patch_feat[:, 1:, :].unsqueeze(2), text_feat.unsqueeze(1).unsqueeze(1), dim=-1).squeeze(dim=-1)
            similarity_matrix = torch.nn.functional.normalize(patch_feat[:, 1:, :], dim=-1) @ torch.nn.functional.normalize(text_feat, dim=-1).unsqueeze(-1)
            similarity_matrix = similarity_matrix.squeeze(-1)               # [B * F, P]
            print(f'similarity_matrix.shape: {similarity_matrix.shape}')
            print(f'masks.shape: {masks.shape}')
            align_loss = -(similarity_matrix * masks.float()).mean(dim=0).sum() \
                + (similarity_matrix * (~masks).float()).mean(dim=0).sum()
            
            print(f'{video_name}-{abstract} align_loss: {align_loss}')

            pos_similarity = (similarity_matrix * masks.float()).mean(dim=0).sum().item()
            neg_similarity = (similarity_matrix * (~masks).float()).mean(dim=0).sum().item()
            similarity_matrix = similarity_matrix.unsqueeze(-1)
            similarity_matrix = rearrange(similarity_matrix, ' f (h w) c -> c f h w', h=16).repeat(3, 1, 1 ,1)
            similarity_matrix = similarity_matrix.repeat_interleave(14, dim=-2).repeat_interleave(14, dim=-1)

            print(f'{video_name}-{abstract} pos_similarity: {pos_similarity} neg_similarity: {neg_similarity}')
            save_dir = os.path.join(output_path, abstract_clip_path)
            os.makedirs(save_dir, exist_ok=True)

            save_video(similarity_matrix, f'{save_dir}/{video_name}-{abstract}-sim.gif', rescale=True)
    print(abstracts_contrastive_loss(abstracts_tokenizer=tokenizer, abstracts_text_encoder=text_encoder, prefix='a photo of'))
    info = {}
    info['abstract_clip_path'] = abstract_clip_path
    info['pickle_path'] = pickle_path
    with open(f'{save_dir}/info.json', "w+") as f:
        json.dump(info, f)


if __name__ == '__main__':
    set_seed(33)
    # video_path = "data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4"
    # video_path = "data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4"

    # text = "A time lapse video of $CHAIRLIFTS moving up and down with a snowy mountain and a blue sky in the background."
    # text = "An image of $CHAIRLIFTS"
    # text = "An image of sky"
    # text = "A man does kettlebell exercises on a beach with a warm $SUNSET on the sky in the background."
    # abstract_clip_path = ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"
    parser = argparse.ArgumentParser(description="Process video and text with AbstractsCLIP models")
    parser.add_argument('-p', "--pickle_path", type=str, help="Path to the pickle file")
    parser.add_argument('-a', "--abstract_clip_path", type=str, help="Path to the AbstractsCLIP model")
    parser.add_argument('-o', "--output_path", type=str, default='.output/')

    args = parser.parse_args()

    main(args.pickle_path, args.abstract_clip_path, args.output_path)