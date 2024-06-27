import torch
from torch import nn
from typing import Optional
from transformers import CLIPTextModel, CLIPTokenizer, CLIPConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
class AbstractsCLIPTokenizerOutput:
    def __init__(self, input_ids, replace_indices, abstract_indices):
        self.input_ids = input_ids
        self.replace_indices = replace_indices
        self.abstract_indices = abstract_indices


class AbstractsCLIPTokenizer(CLIPTokenizer):
    def __init__(
            self, 
            abstracts_num_embedding : int = 1, 
            abstracts_list : list[str]=None, 
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abstracts_list = abstracts_list
        self.abstracts_num_embedding = abstracts_num_embedding

    def __call__(self, text, **kwargs):
        # Process single text or multiple texts
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Initialize lists to store results
        all_token_ids = []
        all_replace_indices = []
        all_abstract_indices = []

        # Process each text individually
        for text in texts:
            replace_indices = []
            abstract_indices = []
            tokens = text.split()
            normal_tokens = []

            # Replace abstracts with placeholders and record indices
            cnt = 0
            for i, token in enumerate(tokens):
                punc = None
                if token.endswith(('.', ',')):
                    punc = token[-1]
                    token = token[:-1]
                if token.startswith('$') and token[1:] in self.abstracts_list:
                    index = self.abstracts_list.index(token[1:])
                    start_replace_index = i + cnt * (self.abstracts_num_embedding - 1) + 1
                    end_replace_index = start_replace_index + self.abstracts_num_embedding
                    replace_indices.extend(list(range(start_replace_index, end_replace_index)))
                    abstract_indices.extend(list(range(index * self.abstracts_num_embedding, (index + 1) * self.abstracts_num_embedding)))
                    normal_tokens.extend(['*'] * self.abstracts_num_embedding)
                    cnt += 1
                else:
                    normal_tokens.append(token)
                if punc is not None:
                    normal_tokens.append(punc)

            # Convert modified tokens back to text
            modified_text = ' '.join(normal_tokens)
            
            # Encode the text using the parent class method
            encoding = super().__call__(modified_text, **kwargs)

            all_token_ids.append(encoding['input_ids'])
            all_replace_indices.append(replace_indices)
            all_abstract_indices.append(abstract_indices)

        all_token_ids = torch.stack(all_token_ids)

        return AbstractsCLIPTokenizerOutput(all_token_ids, all_replace_indices, all_abstract_indices)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, abstracts_list:list[str]=None, abstracts_num_embedding:int=1,**kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, *model_args, **kwargs)
        model.abstracts_list = abstracts_list
        model.abstracts_num_embedding = abstracts_num_embedding
        return model


# class AbstractsCLIPTextModel(CLIPTextModel):
#     def __init__(
#             self, 
#             config: CLIPConfig, 
#             embedding_dim : int = 768, 
#             abstracts_num_embedding : int = 1, 
#             abstracts_list : list[str]=None, 
#             *args, **kwargs):
#         super().__init__(config, *args, **kwargs)
#         if abstracts_list is not None:
#             self.abstracts_list = abstracts_list
#             self.abstracts_num_embedding = abstracts_num_embedding
#             self.embedding_dim = embedding_dim
#             self.abstracts_embedder = nn.Embedding(num_embeddings=len(self.abstracts_list) * self.abstracts_num_embedding, embedding_dim=self.embedding_dim)
            
#     def __call__(
#             self, 
#             input_ids: Optional[torch.Tensor] = None,
#             replace_indices: list=None, 
#             abstract_indices: list=None, 
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.Tensor] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ):
#         assert input_ids is not None and len(input_ids) > 0, f"Invaild input_ids, input_ids is None or len(input_ids) == 0"
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         if input_ids is None:
#             raise ValueError("You have to specify input_ids")
        
#         input_shape = input_ids.size()
#         input_ids = input_ids.view(-1, input_shape[-1])

#         hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)

#         if isinstance(input_ids, torch.Tensor) is False:
#             if isinstance(input_ids, list) and isinstance(input_ids[0], torch.Tensor):
#                 input_ids = torch.stack(input_ids)
#             else:
#                 input_ids = torch.as_tensor(input_ids)
        
#         if self.abstracts_list is not None and replace_indices is not None and abstract_indices is not None:
#             assert len(replace_indices) == len(abstract_indices), f"len(replace_indices){len(replace_indices)} is not equal to len(abstract_indices){len(abstract_indices)}"
#             for i in range(len(replace_indices)):
#                 replace_idx = torch.as_tensor(replace_indices[i], dtype=torch.long, device=input_ids.device)
#                 abstract_idx = torch.as_tensor(abstract_indices[i], dtype=torch.long, device=input_ids.device)
#                 hidden_states[i, replace_idx] = self.abstracts_embedder(abstract_idx)
        
#         # CLIP's text model uses causal mask, prepare it here.
#         # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
#         causal_attention_mask = _create_4d_causal_attention_mask(
#             input_shape, hidden_states.dtype, device=hidden_states.device
#         )
#         # expand attention_mask
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

#         encoder_outputs = self.text_model.encoder(
#             inputs_embeds=hidden_states,
#             attention_mask=attention_mask,
#             causal_attention_mask=causal_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         last_hidden_state = encoder_outputs[0]
#         last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

#         if self.text_model.eos_token_id == 2:
#             # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
#             # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
#             # ------------------------------------------------------------
#             # text_embeds.shape = [batch_size, sequence_length, transformer.width]
#             # take features from the eot embedding (eot_token is the highest number in each sequence)
#             # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
#             pooled_output = last_hidden_state[
#                 torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
#                 input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
#             ]
#         else:
#             # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
#             pooled_output = last_hidden_state[
#                 torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
#                 # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
#                 (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
#                 .int()
#                 .argmax(dim=-1),
#             ]

#         if not return_dict:
#             return (last_hidden_state, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPooling(
#             last_hidden_state=last_hidden_state,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *model_args, abstracts_list:list[str]=None, abstracts_num_embedding : int = 1, **kwargs):
#         config = kwargs.pop('config', None)
#         if config is None:
#             config = CLIPConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
#         model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        
#         model.abstracts_list = abstracts_list
#         if abstracts_list is not None:
#             model.embedding_dim = config.hidden_size
#             model.abstracts_num_embedding = abstracts_num_embedding
#             model.abstracts_embedder = nn.Embedding(num_embeddings=len(model.abstracts_list) * model.abstracts_num_embedding, embedding_dim=model.embedding_dim)
#         return model



class AbstractsCLIPTextModel(CLIPTextModel):
    def __init__(
            self, 
            config: CLIPConfig, 
            embedding_dim : int = 768, 
            abstracts_num_embedding : int = 1, 
            abstracts_list : list[str]=None, 
            *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if abstracts_list is not None:
            self.abstracts_list = abstracts_list
            self.abstracts_num_embedding = abstracts_num_embedding
            self.embedding_dim = embedding_dim
            self.abstracts_embedder = nn.Embedding(num_embeddings=len(self.abstracts_list) * self.abstracts_num_embedding, embedding_dim=self.embedding_dim)
            
    def __call__(
            self, 
            input_ids: Optional[torch.Tensor] = None,
            replace_indices: list=None, 
            abstract_indices: list=None, 
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        assert input_ids is not None and len(input_ids) > 0, f"Invaild input_ids, input_ids is None or len(input_ids) == 0"
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)

        if isinstance(input_ids, torch.Tensor) is False:
            if isinstance(input_ids, list) and isinstance(input_ids[0], torch.Tensor):
                input_ids = torch.stack(input_ids)
            else:
                input_ids = torch.as_tensor(input_ids)
        
        if self.abstracts_list is not None and replace_indices is not None and abstract_indices is not None:
            assert len(replace_indices) == len(abstract_indices), f"len(replace_indices){len(replace_indices)} is not equal to len(abstract_indices){len(abstract_indices)}"
            for i in range(len(replace_indices)):
                replace_idx = torch.as_tensor(replace_indices[i], dtype=torch.long, device=input_ids.device)
                abstract_idx = torch.as_tensor(abstract_indices[i], dtype=torch.long, device=input_ids.device)
                hidden_states[i, replace_idx] = self.abstracts_embedder(abstract_idx)
        
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

        # if self.abstracts_list is not None and replace_indices is not None and abstract_indices is not None:
        #     assert len(replace_indices) == len(abstract_indices), f"len(replace_indices){len(replace_indices)} is not equal to len(abstract_indices){len(abstract_indices)}"
        #     for i in range(len(replace_indices)):
        #         replace_idx = torch.as_tensor(replace_indices[i], dtype=torch.long, device=input_ids.device)
        #         abstract_idx = torch.as_tensor(abstract_indices[i], dtype=torch.long, device=input_ids.device)
        #         last_hidden_state[i, replace_idx] = self.abstracts_embedder(abstract_idx)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, abstracts_list:list[str]=None, abstracts_num_embedding : int = 1, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = CLIPConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        
        model.abstracts_list = abstracts_list
        if abstracts_list is not None:
            model.embedding_dim = config.hidden_size
            model.abstracts_num_embedding = abstracts_num_embedding
            model.abstracts_embedder = nn.Embedding(num_embeddings=len(model.abstracts_list) * model.abstracts_num_embedding, embedding_dim=model.embedding_dim)
        return model
