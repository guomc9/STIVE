import torch
from stive.models.concepts_clip import ConceptsCLIPTextModel
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
            
def init_concepts_embedding(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, pseudo_tokens: list, concept_tokens: list, concept_text_encoder: ConceptsCLIPTextModel):
    assert len(pseudo_tokens) == len(concept_tokens)
    for token, concept_token in zip(pseudo_tokens, concept_tokens):
        print(f'token: {token}')
        idx = tokenizer.encode(token)
        cidx = concept_text_encoder.concepts_list.index(concept_token)
        concept_text_encoder.concepts_embedder.weight.data[cidx] = text_encoder.text_model.embeddings(torch.as_tensor([idx[1]]))