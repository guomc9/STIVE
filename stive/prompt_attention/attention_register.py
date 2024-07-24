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

def register_attention_control(model, controller):
    "Connect a model with a controller"
    def attention_controlled_processor(place_in_unet):
            
        class AttnWithProbProcessor2_0:
            r"""
            Processor for implementing scaled dot-product attention with attention score (enabled by default if you're using PyTorch 2.0).
            """

            def __init__(self):
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnWithScoreProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

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
                if len(args) > 0 or kwargs.get("scale", None) is not None:
                    deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                    deprecate("scale", "1.0.0", deprecation_message)
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
                
                attention_scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / np.sqrt(head_dim)

                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask

                # if self.upcast_softmax:
                #     attention_scores = attention_scores.float()

                attention_probs = attention_scores.softmax(dim=-1)

                # print(f'attention_probs.dtype: {attention_probs.dtype}')
                # print(f'value.dtype: {value.dtype}')
                # cast back to the original dtype
                attention_probs = attention_probs.to(value.dtype)

                # START OF CORE FUNCTION
                # Record during inversion and edit the attention probs during editing
                controller(attention_probs, is_cross=is_cross, place_in_unet=place_in_unet)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
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
            
        return AttnWithProbProcessor2_0()


    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    
    def register_recr(net_, count, place_in_unet):
        # print(f'net_[1].__class__.__name__: {net_[1].__class__.__name__}')
        if net_[1].__class__.__name__ == 'Attention':
            # print(f'net_: {net_[0], net_[1]}')
            net_[1].processor = attention_controlled_processor(place_in_unet)
            return count + 1
        elif hasattr(net_[1], 'children'):
            for net in net_[1].named_children():
                # if net[0] !='temp_attentions':
                if net[1].__class__.__name__ != 'TransformerTemporalModel':
                    
                    count = register_recr(net, count, place_in_unet)

        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for name, module in model.unet.named_children():
        print(f"Module Name: {name}, Module: {module}")
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid")
    print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count
