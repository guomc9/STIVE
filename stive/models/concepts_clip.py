import os
import json
import torch
import random
from torch import nn
from typing import Optional
from transformers import CLIPTextModel, CLIPTokenizer, CLIPConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

class ConceptsCLIPTokenizerOutput:
    def __init__(self, input_ids, replace_indices, concept_indices, prefix_texts):
        self.input_ids = input_ids
        self.replace_indices = replace_indices
        self.concept_indices = concept_indices
        self.prefix_texts = prefix_texts


class ConceptsCLIPTokenizer(CLIPTokenizer):
    def __init__(
            self, 
            concepts_num_embedding : int = 1, 
            concepts_list : list[str]=None, 
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concepts_list = concepts_list
        self.concepts_num_embedding = concepts_num_embedding

    def __call__(self, text, enable_prefix: bool=True, **kwargs):
        # Process single text or multiple texts
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Initialize lists to store results
        all_token_ids = []
        all_replace_indices = []
        all_concept_indices = []

        # Process each text individually
        prefix_texts = []
        for text in texts:
            replace_indices = []
            concept_indices = []
            tokens = text.split()
            normal_tokens = []
            # Replace concepts with placeholders and record indices
            cnt = 0
            is_concept_text = False
            for i, token in enumerate(tokens):
                punc = None
                if token.endswith(('.', ',')):
                    punc = token[-1]
                    token = token[:-1]
                if token.startswith('$') and token in self.concepts_list:
                    index = self.concepts_list.index(token)
                    start_replace_index = i + cnt * (self.concepts_num_embedding - 1) + 1
                    end_replace_index = start_replace_index + self.concepts_num_embedding
                    replace_indices.extend(list(range(start_replace_index, end_replace_index)))
                    concept_indices.extend(list(range(index * self.concepts_num_embedding, (index + 1) * self.concepts_num_embedding)))
                    normal_tokens.extend(['*'] * self.concepts_num_embedding)
                    is_concept_text = True
                    cnt += 1
                else:
                    normal_tokens.append(token)
                if punc is not None:
                    normal_tokens.append(punc)

            # Convert modified tokens back to text
            modified_text = ' '.join(normal_tokens)
            if is_concept_text and enable_prefix:
                prefix = random.choice(imagenet_templates_small)
                prefix_length = len(prefix.split())
                prefix_text = prefix.format(text)
                replace_indices = [inds + prefix_length for inds in replace_indices]
                prefix_texts.append(prefix_text)

            # Encode the text using the parent class method
            encoding = super().__call__(modified_text, **kwargs)

            all_token_ids.append(encoding['input_ids'])
            all_replace_indices.append(replace_indices)
            all_concept_indices.append(concept_indices)

        # print(f'modified_text: {modified_text}')
        # print(f'all_replace_indices: {all_replace_indices}')
        # print(f'all_concept_indices: {all_concept_indices}')

        all_token_ids = torch.stack(all_token_ids)

        return ConceptsCLIPTokenizerOutput(all_token_ids, all_replace_indices, all_concept_indices, prefix_texts)

    @classmethod
    def from_pretrained_clip(cls, pretrained_model_name_or_path, *model_args, concepts_list:list[str]=None, concepts_num_embedding:int=1,**kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, *model_args, **kwargs)
        model.concepts_list = concepts_list
        model.concepts_num_embedding = concepts_num_embedding
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        subfolder = kwargs.get('subfolder')

        model = super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, *model_args, **kwargs)
        
        if subfolder is not None:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
        
        concepts_path = os.path.join(pretrained_model_name_or_path, "concepts.json")
        if os.path.exists(concepts_path):
            with open(concepts_path, 'r', encoding='utf-8') as f:
                concepts = json.load(f)
                model.concepts_list = concepts['concepts_list']
                model.concepts_num_embedding = concepts['concepts_num_embedding']

        return model

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        
        concepts_path = os.path.join(save_directory, "concepts.json")
        with open(concepts_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'concepts_list': self.concepts_list, 
                    'concepts_num_embedding': self.concepts_num_embedding
                }, 
                f, ensure_ascii=False, indent=2
            )


class ConceptsCLIPTextModel(CLIPTextModel):
    def __init__(
            self, 
            config: CLIPConfig, 
            embedding_dim : int = 768, 
            concepts_num_embedding : int = 1, 
            concepts_list : list[str]=None, 
            retain_position_embedding: bool = True, 
            *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.concepts_list = concepts_list
        self.concepts_num_embedding = concepts_num_embedding
        self.embedding_dim = embedding_dim
        self.retain_position_embedding = retain_position_embedding
        if concepts_list is not None:
            self.concepts_embedder = nn.Embedding(num_embeddings=len(self.concepts_list) * self.concepts_num_embedding, embedding_dim=self.embedding_dim)

    def __call__(
            self, 
            input_ids: Optional[torch.Tensor] = None,
            replace_indices: list=None, 
            concept_indices: list=None, 
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # print(f'input_ids: {input_ids}')
        # print(f'replace_indices: {replace_indices}')
        # print(f'concept_indices: {concept_indices}')
        # print(f'concepts_list: {self.concepts_list}')
        # print(f'concepts_embedder: {self.concepts_embedder.weight.data}')
        
        assert input_ids is not None and len(input_ids) > 0, f"Invaild input_ids, input_ids is None or len(input_ids) == 0"
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)

        if isinstance(input_ids, torch.Tensor) is False:
            if isinstance(input_ids, list) and isinstance(input_ids[0], torch.Tensor):
                input_ids = torch.stack(input_ids)
            else:
                input_ids = torch.as_tensor(input_ids)
        
        if self.concepts_list is not None and replace_indices is not None and concept_indices is not None:
            assert len(replace_indices) == len(concept_indices), f"len(replace_indices){len(replace_indices)} is not equal to len(concept_indices){len(concept_indices)}"
            for i in range(len(replace_indices)):
                replace_idx = torch.as_tensor(replace_indices[i], dtype=torch.long, device=input_ids.device)
                concept_idx = torch.as_tensor(concept_indices[i], dtype=torch.long, device=input_ids.device)
                if self.retain_position_embedding:
                    hidden_states[i, replace_idx] = 0
                    hidden_states[i, replace_idx] += self.concepts_embedder(concept_idx) + self.text_model.embeddings.position_embedding(replace_idx)
                else:
                    hidden_states[i, replace_idx] = 0
                    hidden_states[i, replace_idx] += self.concepts_embedder(concept_idx)
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        if self.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @classmethod
    def from_pretrained_clip(cls, 
            pretrained_model_name_or_path, 
            *model_args, 
            concepts_list:list[str]=None, 
            concepts_num_embedding : int = 1, 
            retain_position_embedding: bool = True, 
            **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = CLIPConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        model.concepts_list = concepts_list
        model.concepts_num_embedding = concepts_num_embedding
        model.retain_position_embedding = retain_position_embedding
        if concepts_list is not None:
            model.embedding_dim = config.hidden_size
            model.concepts_num_embedding = concepts_num_embedding
            model.concepts_embedder = nn.Embedding(num_embeddings=len(model.concepts_list) * model.concepts_num_embedding, embedding_dim=model.embedding_dim)

        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        subfolder = kwargs.get('subfolder')
        
        config = kwargs.pop('config', None)

        if config is None:
            config = CLIPConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if subfolder is not None:
            concepts_path = os.path.join(pretrained_model_name_or_path, subfolder, 'concepts.json')
            concepts_weight_path = os.path.join(pretrained_model_name_or_path, subfolder, 'concepts_embedder.pth')
        else:
            concepts_path = os.path.join(pretrained_model_name_or_path, 'concepts.json')
            concepts_weight_path = os.path.join(pretrained_model_name_or_path, 'concepts_embedder.pth')

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        if os.path.exists(concepts_path):
            with open(concepts_path, 'r', encoding='utf-8') as f:
                concepts = json.load(f)
                model.concepts_list = concepts['concepts_list']
                model.embedding_dim = config.hidden_size
                model.concepts_num_embedding = concepts['concepts_num_embedding']
                model.retain_position_embedding = concepts['retain_position_embedding']
                model.concepts_embedder = nn.Embedding(num_embeddings=len(model.concepts_list) * model.concepts_num_embedding, embedding_dim=model.embedding_dim)

        model.concepts_embedder.load_state_dict(torch.load(concepts_weight_path))
        print('ConceptsCLIPTextModel.concepts_embedder weights loaded.')
        
        return model

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        concepts_embedder_path = os.path.join(save_directory, "concepts_embedder.pth")

        torch.save(self.concepts_embedder.state_dict(), concepts_embedder_path)

        concepts_path = os.path.join(save_directory, "concepts.json")
        with open(concepts_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'concepts_list': self.concepts_list, 
                    'concepts_num_embedding': self.concepts_num_embedding, 
                    'retain_position_embedding': self.retain_position_embedding
                }, 
                f, ensure_ascii=False, indent=2
            )
            