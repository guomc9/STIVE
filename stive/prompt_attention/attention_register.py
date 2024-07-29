"""
register the attention controller into the UNet of stable diffusion
Build a customized attention function `_attention'
Replace the original attention function with `forward' and `spatial_temporal_forward' in attention_controlled_forward function
Most of spatial_temporal_forward is directly copy from `video_diffusion/models/attention.py'
TODO FIXME: merge redundant code with attention.py
"""
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import XFormersAttnProcessor
from einops import rearrange
import datetime
from peft import PeftModel
from ..utils.save_utils import save_video

def register_attention_control(model, controller):
    "Connect a model with a controller"
    def attention_controlled_processor(store, place_in_unet):
        
        def reshape_temporal_heads_to_batch_dim(tensor, head_size):
            tensor = rearrange(tensor, " b h s t -> (b h) s t ", h = head_size)
            return tensor

        def reshape_batch_dim_to_temporal_heads(tensor, head_size):
            tensor = rearrange(tensor, "(b h) s t -> b h s t", h = head_size)
            return tensor    
        
        def reshape_batch_dim_to_heads(tensor, head_size):
            batch_size, seq_len, dim = tensor.shape
            return tensor.reshape(batch_size // head_size, head_size, seq_len, dim).transpose(1, 2).reshape(batch_size // head_size, seq_len, dim * head_size)
        
        class AttnWithProbProcessor:
            r"""
            Processor for implementing attention with attention probability.
            """
            def __init__(self):
                pass

            
            def __call__(
                self,
                attn: Attention,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                temb: Optional[torch.Tensor] = None,
                *args,
                **kwargs,
            ) -> torch.Tensor:
                is_cross = encoder_hidden_states is not None
                residual = hidden_states
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    # scaled_dot_product_attention expects attention_mask shape to be
                    # (batch, heads, source_length, target_length)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)    # [B, H, Q, C]
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)        # [B, H, K, C]
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)    # [B, H, K, C]
                
                query = reshape_temporal_heads_to_batch_dim(query, attn.heads)
                key = reshape_temporal_heads_to_batch_dim(key, attn.heads)
                value = reshape_temporal_heads_to_batch_dim(value, attn.heads)
                
                attention_probs = torch.baddbmm(
                    torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                    query,
                    key.transpose(-1, -2),
                    beta=0,
                    alpha=attn.scale,
                )

                if attention_mask is not None:
                    attention_probs = attention_probs + attention_mask

                attention_probs = attention_probs.softmax(dim=-1)

                attention_probs = attention_probs.to(value.dtype)   # [B * H, Q, K]
                
                attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_probs, attn.heads), is_cross, place_in_unet) # [B, H, Q, K]
        
                
                attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs, attn.heads)
                
                hidden_states = torch.bmm(attention_probs, value)   # [B * H, N, C]

                hidden_states = reshape_batch_dim_to_heads(hidden_states, attn.heads)
                hidden_states = hidden_states.to(query.dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states
        
        if store:
            return AttnWithProbProcessor()
        else:
            return XFormersAttnProcessor()

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    
    def register_recr(net_, count, place_in_unet, begin_store, down_tsfm_count, up_tsfm_count):
        if net_[1].__class__.__name__ == 'Attention':
            store = (place_in_unet == 'down' and down_tsfm_count > begin_store) or (place_in_unet == 'up' and up_tsfm_count < half_unet_len - begin_store) or (place_in_unet == 'mid')
            net_[1].processor = attention_controlled_processor(store, place_in_unet)
            return count + 1
        elif hasattr(net_[1], 'children'):
            for net in net_[1].named_children():
                if net[1].__class__.__name__ != 'TransformerTemporalModel':
                    if net[1].__class__.__name__ == 'CrossAttnDownBlock3D':
                        down_tsfm_count += 1
                        count = register_recr(net, count, place_in_unet, begin_store, down_tsfm_count, up_tsfm_count)
                    elif net[1].__class__.__name__ == 'CrossAttnUpBlock3D':
                        up_tsfm_count += 1
                        count = register_recr(net, count, place_in_unet, begin_store, down_tsfm_count, up_tsfm_count)
                    else:
                        count = register_recr(net, count, place_in_unet, begin_store, down_tsfm_count, up_tsfm_count)

        return count
    
    half_unet_len = 4
    begin_store = 1
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    
    print(f'unet is an instance of {type(model.unet)}')
    if isinstance(model.unet, PeftModel):
        sub_nets = model.unet.base_model.model.named_children()
    else:
        sub_nets = model.unet.named_children()
        
        
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down", begin_store=begin_store, down_tsfm_count=0, up_tsfm_count=0)
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up", begin_store=begin_store, down_tsfm_count=0, up_tsfm_count=0)
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid", begin_store=begin_store, down_tsfm_count=0, up_tsfm_count=0)
    print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count
