import torch
from transformers import CLIPTextModel, CLIPTokenizer


def add_concepts_embeddings(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, concept_tokens: list, concept_embeddings: torch.Tensor):
    assert len(concept_tokens) == concept_embeddings.shape[0]
    tokenizer.add_tokens(concept_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(concept_tokens)
    token_ids_and_embeddings = zip(token_ids, concept_embeddings)
    with torch.no_grad():
        for i, (token_id, embedding) in enumerate(token_ids_and_embeddings):
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding
    

def update_concepts_embedding(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, concept_tokens: list, concept_embeddings: torch.Tensor):
    token_ids = tokenizer.convert_tokens_to_ids(concept_tokens)
    token_ids_and_embeddings = zip(token_ids, concept_embeddings)
    with torch.no_grad():
        for i, (token_id, embedding) in enumerate(token_ids_and_embeddings):
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            
