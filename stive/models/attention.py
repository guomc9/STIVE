# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, condition=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)
            
        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length, 
                condition=condition
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        num_frame_sample: int = 12,
        primary_K: int = 16,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # IF-Attn
        self.attn1 = IFAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )

        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        
        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn1._use_memory_efficient_attention_xformers = False
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers


    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, condition=None):
        if condition is not None:
            condition = rearrange(condition, 'b c f h w -> (b f) (h w) c')
            print(f'hidden_states.shape: {hidden_states.shape}, condition.shape: {condition.shape}')

        # IF-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask, video_length=video_length, condition=condition) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length, condition=condition) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )
            # hidden_states = (
            #     self.attn2(
            #         norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, video_length=video_length
            #     )
            #     + hidden_states
            # )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        
        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
    
    def get_if_mask(self, num_frame, num_res, device):
        repeat_indices = torch.arange(0, num_frame, device=device) // (num_frame // self.num_frame_sample)

        mask = torch.zeros((self.num_frame_sample, self.num_frame_sample, self.primary_K*num_res), dtype=torch.float32, device=device)

        row = torch.arange(0, self.num_frame_sample, device=device)
        col = torch.arange(0, self.num_frame_sample, device=device)

        mask[row, col] = 1
        mask = mask.reshape(self.num_frame_sample, self.num_frame_sample*self.primary_K, num_res)
        mask = torch.cumsum(mask, dim=0)
        mask = 1e9 * (mask - 1)

        mask = mask[repeat_indices]

        return mask.transpose(-1, -2)


class IFAttention(nn.Module):
    r"""
    InterFrame attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        primary_K: int = 128, 
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None
        
        self.primary_lin = nn.Linear(query_dim, query_dim, bias=True)
        self.primary_K = primary_K
        self.comp_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.comp_k = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        self.comp_v = nn.Linear(cross_attention_dim, inner_dim, bias=True)

        nn.init.zeros_(self.primary_lin.weight.data)
        nn.init.zeros_(self.primary_lin.bias.data)

        self.comp_out = nn.ModuleList([])
        self.comp_out.append(nn.Linear(inner_dim, query_dim, bias=True))
        self.comp_out.append(nn.Dropout(dropout))
        
        nn.init.zeros_(self.comp_out[0].weight.data)
        nn.init.zeros_(self.comp_out[0].bias.data)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def copy_weights(self):
        self.comp_q.weight = self.to_q.weight.clone()
        self.comp_k.weight = self.to_k.weight.clone()
        self.comp_v.weight = self.to_v.weight.clone()

        self.comp_q.bias = self.to_q.bias.clone()
        self.comp_k.bias = self.to_k.bias.clone()
        self.comp_v.bias = self.to_v.bias.clone()
        
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, condition=None):
        batch_size, sequence_length, dim = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)


        # condition : [B*F, N, C]
        # hidden_states : [B*F, N, D]
        # pre_hidden_states: [B*F, N, D]
        # diff = abs(hidden_states - pre_hidden_states) : [B*F, N, D]
        # mean diff: [B*F, N]
        # topK diff: [B*F, K] ==> topk_index
        # primary_hidden_states: [B*F, K, D]
        pre_hidden_states = torch.zeros_like(hidden_states)
        pre_hidden_states[1:] = hidden_states[:-1]
        pre_hidden_states[0] = hidden_states[0]
        diff = hidden_states - pre_hidden_states
        diff_score = diff.abs().mean(dim=-1)
        comp_ratio = 0.5
        min_comp = 64
        primary_K = int(comp_ratio * diff_score.shape[-1])
        primary_K = min_comp if primary_K < min_comp else primary_K

        _, topk_indices = torch.topk(diff_score, primary_K, dim=-1)
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, dim)
        primary_hidden_states = torch.gather(hidden_states, 1, topk_indices)
        primary_diff = torch.gather(diff, 1, topk_indices)

        primary_hidden_states = self.primary_lin(primary_diff) + primary_hidden_states
        
        query = self.comp_q(primary_hidden_states)              # [B * F, K, D]
        key = self.comp_k(pre_hidden_states)                    # [B * F, N, D]
        value = self.comp_v(pre_hidden_states)                  # [B * F, N, D]
        
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self._use_memory_efficient_attention_xformers:
            comp_hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            comp_hidden_states = comp_hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                comp_hidden_states = self._attention(query, key, value, attention_mask)
            else:
                comp_hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        comp_hidden_states = self.comp_out[0](comp_hidden_states)
        comp_hidden_states = self.comp_out[1](comp_hidden_states)
        
        comp_hidden_states = rearrange(comp_hidden_states, "(b f) k c -> b f k c", f=video_length)
        comp_hidden_states = torch.cumsum(comp_hidden_states, dim=1, dtype=comp_hidden_states.dtype)
        # comp_hidden_states = torch.cumprod(comp_hidden_states, dim=1, dtype=comp_hidden_states.dtype)
        comp_hidden_states = rearrange(comp_hidden_states, "b f k c -> (b f) k c")
        # print(f'comp_hidden_states.dtype: {comp_hidden_states.dtype}')
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        former_frame_index = torch.arange(video_length, device=hidden_states.device) - 1
        former_frame_index[0] = 0
        # first_frame_index = torch.zeros((video_length), dtype=torch.long, device=hidden_states.device)
        # # mid_frame_index = former_frame_index // 2
        # # mid_frame_index_2 = former_frame_index // 4
        # # mid_frame_index_3 = 3 * former_frame_index // 4

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = key[:, former_frame_index]
        # key = key[:, first_frame_index]
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        # # key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index], key[:, mid_frame_index], key[:, mid_frame_index_2], key[:, mid_frame_index_3]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = value[:, former_frame_index]
        # value = value[:, first_frame_index]
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        # # value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index], value[:, mid_frame_index], value[:, mid_frame_index_2], value[:, mid_frame_index_3]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        
        
        # print(f'hidden_states.dtype: {hidden_states.dtype}')
        hidden_states.scatter_add_(1, topk_indices, comp_hidden_states)
        # condition : [B*F, N, C]
        # hidden_states = hidden_states + condition
        return hidden_states

        # batch_size, sequence_length, _ = hidden_states.shape

        # encoder_hidden_states = encoder_hidden_states

        # if self.group_norm is not None:
        #     hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query = self.to_q(hidden_states)
        # dim = query.shape[-1]
        # query = self.reshape_heads_to_batch_dim(query)

        # if self.added_kv_proj_dim is not None:
        #     raise NotImplementedError

        # encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # key = self.to_k(encoder_hidden_states)
        # value = self.to_v(encoder_hidden_states)

        # former_frame_index = torch.arange(video_length) - 1
        # former_frame_index[0] = 0
        # mid_frame_index = former_frame_index // 2
        # mid_frame_index_2 = former_frame_index // 4
        # mid_frame_index_3 = 3 * former_frame_index // 4

        # key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        # key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index], key[:, mid_frame_index], key[:, mid_frame_index_2], key[:, mid_frame_index_3]], dim=2)
        # key = rearrange(key, "b f d c -> (b f) d c")

        # value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        # value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index], value[:, mid_frame_index], value[:, mid_frame_index_2], value[:, mid_frame_index_3]], dim=2)
        # value = rearrange(value, "b f d c -> (b f) d c")

        # key = self.reshape_heads_to_batch_dim(key)
        # value = self.reshape_heads_to_batch_dim(value)

        # if attention_mask is not None:
        #     if attention_mask.shape[-1] != query.shape[1]:
        #         target_length = query.shape[1]
        #         attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        #         attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # # attention, what we cannot get enough of
        # if self._use_memory_efficient_attention_xformers:
        #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #         hidden_states = self._attention(query, key, value, attention_mask)
        #     else:
        #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # # linear proj
        # hidden_states = self.to_out[0](hidden_states)

        # # dropout
        # hidden_states = self.to_out[1](hidden_states)
        # return hidden_states

class FCAttention(nn.Module):
    r"""
    Frame Condition attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        primary_K: int = 128, 
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.primary_lin = nn.Linear(query_dim, query_dim, bias=True)
        self.primary_K = primary_K
        self.comp_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.comp_k = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        self.comp_v = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        # # nn.init.kaiming_normal_(self.comp_q.weight.data)
        # # nn.init.kaiming_normal_(self.comp_k.weight.data)
        # # nn.init.kaiming_normal_(self.comp_v.weight.data)

        nn.init.zeros_(self.primary_lin.weight.data)
        nn.init.zeros_(self.primary_lin.bias.data)

        # nn.init.zeros_(self.comp_q.bias.data)
        # nn.init.zeros_(self.comp_k.bias.data)
        # nn.init.zeros_(self.comp_v.bias.data)

        # nn.init.zeros_(self.comp_q.weight.data)
        # nn.init.zeros_(self.comp_k.weight.data)
        # nn.init.zeros_(self.comp_v.weight.data)
        self.comp_out = nn.ModuleList([])
        self.comp_out.append(nn.Linear(inner_dim, query_dim, bias=True))
        self.comp_out.append(nn.Dropout(dropout))
        
        nn.init.zeros_(self.comp_out[0].weight.data)
        nn.init.zeros_(self.comp_out[0].bias.data)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def copy_weights(self):
        self.comp_q.weight = self.to_q.weight.clone()
        self.comp_k.weight = self.to_k.weight.clone()
        self.comp_v.weight = self.to_v.weight.clone()

        self.comp_q.bias = self.to_q.bias.clone()
        self.comp_k.bias = self.to_k.bias.clone()
        self.comp_v.bias = self.to_v.bias.clone()
        
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, dim = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # hidden_states : [B*F, N, D]
        # pre_hidden_states: [B*F, N, D]
        # diff = abs(hidden_states - pre_hidden_states) : [B*F, N, D]
        # mean diff: [B*F, N]
        # topK diff: [B*F, K] ==> topk_index
        # primary_hidden_states: [B*F, K, D]
        pre_hidden_states = torch.zeros_like(hidden_states)
        pre_hidden_states[1:] = hidden_states[:-1]
        pre_hidden_states[0] = hidden_states[0]
        diff = hidden_states - pre_hidden_states
        diff_score = diff.abs().mean(dim=-1)
        comp_ratio = 0.5
        min_comp = 64
        primary_K = int(comp_ratio * diff_score.shape[-1])
        primary_K = min_comp if primary_K < min_comp else primary_K

        _, topk_indices = torch.topk(diff_score, primary_K, dim=-1)
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, dim)
        primary_hidden_states = torch.gather(hidden_states, 1, topk_indices)
        primary_diff = torch.gather(diff, 1, topk_indices)

        primary_hidden_states = self.primary_lin(primary_diff) + primary_hidden_states
        
        query = self.comp_q(primary_hidden_states)                  # [B * F, K, D]
        key = self.comp_k(encoder_hidden_states)                    # [B * F, N, D]
        value = self.comp_v(encoder_hidden_states)                  # [B * F, N, D]
        
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self._use_memory_efficient_attention_xformers:
            comp_hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            comp_hidden_states = comp_hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                comp_hidden_states = self._attention(query, key, value, attention_mask)
            else:
                comp_hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        comp_hidden_states = self.comp_out[0](comp_hidden_states)
        comp_hidden_states = self.comp_out[1](comp_hidden_states)
        
        comp_hidden_states = rearrange(comp_hidden_states, "(b f) k c -> b f k c", f=video_length)
        comp_hidden_states = torch.cumsum(comp_hidden_states, dim=1, dtype=comp_hidden_states.dtype)
        comp_hidden_states = rearrange(comp_hidden_states, "b f k c -> (b f) k c")
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        
        hidden_states.scatter_add_(1, topk_indices, comp_hidden_states)
        return hidden_states
