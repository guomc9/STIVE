import sys
sys.path.append('.')
import argparse
import torch
from transformers import CLIPModel
from tive.models.abstracts_clip import AbstractsCLIPTokenizer, AbstractsCLIPTextModel

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def main(src, tgt, abstract_clip_path):
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
    
    clip.requires_grad_(False)
    clip.to(device)
    clip.eval()
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    text_encoder.eval()

    src_tokens = tokenizer([src], return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
    tgt_tokens = tokenizer([tgt], return_tensors="pt", max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
    with torch.no_grad():
        src_text_feat = text_encoder(src_tokens.input_ids.to(device), src_tokens.replace_indices, src_tokens.abstract_indices)[1]
        tgt_text_feat = text_encoder(tgt_tokens.input_ids.to(device), tgt_tokens.replace_indices, tgt_tokens.abstract_indices)[1]
        print(src_text_feat.shape)
        print(tgt_text_feat.shape)
        # src_text_feat = clip.text_projection(src_text_feat)
        # tgt_text_feat = clip.text_projection(tgt_text_feat)
        # print(src_text_feat.shape)
        # print(tgt_text_feat.shape)

    similarity = torch.cosine_similarity(src_text_feat, tgt_text_feat, dim=-1)
    avg_similarity = similarity.mean().item()

    print(f'avg_similarity: {avg_similarity}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video and text with AbstractsCLIP models")
    parser.add_argument('-s', "--text1", type=str, help="Source text description for the video")
    parser.add_argument('-t', "--text2", type=str, help="Target text description for the video")
    parser.add_argument('-a', "--abstract_clip_path", type=str, help="Path to the AbstractsCLIP model")

    args = parser.parse_args()
    main(args.text1, args.text2, args.abstract_clip_path)