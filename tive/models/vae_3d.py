# Adapted from: https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py

import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from einops import rearrange
import torch.utils

@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: torch.FloatTensor

@dataclass
class AutoencoderKL3DOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"
    latent_dist_td: "DiagonalGaussianDistribution"
    td_seq: "torch.Tensor"

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
        num_temp_down_blocks=1
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.temp_down_blocks = nn.ModuleList([])
        for i in range(num_temp_down_blocks):
            self.temp_down_blocks.append(TinyTemporalDownBlock(hidden_channels=block_out_channels[-1]))

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        
        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x, video_length:int):
        sample = x
        sample = self.conv_in(sample)

        # spatial down
        for down_block in self.down_blocks:
            sample = down_block(sample)
            
        # temporal down
        sample_td = sample
        td_seq_len = video_length
        for _, temp_down_block in enumerate(self.temp_down_blocks):
            sample_td, td_seq_len = temp_down_block(sample_td, td_seq_len)

        td_seq = torch.arange(0, video_length, step=video_length // td_seq_len, device=sample.device)[:td_seq_len]

        sample = rearrange(sample, '(b f) c h w -> b f c h w', f=video_length)
        sample_td = rearrange(sample_td, '(b f) c h w -> b f c h w', f=td_seq_len)
        sample_td = sample_td + sample[:, td_seq, ...]
        sample = rearrange(sample, 'b f c h w -> (b f) c h w')
        sample_td = rearrange(sample_td, 'b f c h w -> (b f) c h w')

        # middle
        sample = self.mid_block(sample)
        sample_td = self.mid_block(sample_td)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        sample_td = self.conv_norm_out(sample_td)
        sample_td = self.conv_act(sample_td)
        sample_td = self.conv_out(sample_td)

        return sample, sample_td, td_seq


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        num_temp_up_blocks=1
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        self.temp_up_blocks = nn.ModuleList([])
        for i in range(num_temp_up_blocks):
            self.temp_up_blocks.append(TinyTemporalUpBlock(hidden_channels=block_out_channels[-1]))

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z, video_length=None, z_td=None, td_seq_len:int=None):
        sample = self.conv_in(z)

        # middle
        sample = self.mid_block(sample)

        # temporal up
        if z_td is not None and td_seq_len is not None:
            sample_td = self.conv_in(z_td)
            sample_td = self.mid_block(sample_td)

            for _, temp_up_block in enumerate(self.temp_up_blocks):
                sample_td, td_seq_len = temp_up_block(sample_td, td_seq_len, video_length)
            sample = sample + sample_td

        # spatial up
        for _, up_block in enumerate(self.up_blocks):
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        return sample

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        device = self.parameters.device
        sample_device = "cpu" if device.type == "mps" else device
        sample = torch.randn(self.mean.shape, generator=generator, device=sample_device)
        # make sure sample is on the same device as the parameters and has same dtype
        sample = sample.to(device=device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean

class TinyTemporalDownBlock(nn.Module):
    def __init__(
            self, 
            hidden_channels: int, 
            temporal_kernel: int = 3, 
            num_groups: int = 8
            ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels, affine=True)
        self.nonlinearity = nn.SiLU()
        self.temporal_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=temporal_kernel, padding=1, stride=2)

        # nn.init.xavier_normal_(self.temporal_conv.weight)
        # nn.init.xavier_normal_(self.temporal_conv.bias)
        nn.init.zeros_(self.temporal_conv.weight)
        nn.init.zeros_(self.temporal_conv.bias)

    def _forward_tp_conv(self, hidden_states, video_length: int):
        height, width = hidden_states.shape[2:]

        # [B * F, C, H, W] -> [B, C, F, H, W]
        hidden_states = rearrange(hidden_states, '(b f) c h w -> (b h w) c f', f=video_length)

        hidden_states = self.temporal_conv(hidden_states)
        
        td_seq_len = hidden_states.shape[-1]

        # [B, C, F, H, W] -> [B * F, C, H, W]
        hidden_states = rearrange(hidden_states, '(b h w) c f -> (b f) c h w', h=height, w=width)

        return hidden_states, td_seq_len

    def forward(self, hidden_states, video_length: int):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, td_seq_len = self._forward_tp_conv(hidden_states, video_length)
        # hidden_states, td_seq_len = torch.utils.checkpoint.checkpoint(self._forward_tp_conv, hidden_states, video_length, use_reentrant=True)

        return hidden_states, td_seq_len


class TinyTemporalUpBlock(nn.Module):
    def __init__(
            self, 
            hidden_channels: int, 
            temporal_kernel: int = 3, 
            num_groups: int = 8
            ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels, affine=True)
        self.nonlinearity = nn.SiLU()
        self.temporal_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=temporal_kernel, padding=1)
        nn.init.zeros_(self.temporal_conv.weight)
        nn.init.zeros_(self.temporal_conv.bias)
        
        # nn.init.xavier_normal_(self.temporal_conv.weight)
        # nn.init.xavier_normal_(self.temporal_conv.bias)

    def _forward_tp_conv(self, hidden_states, video_length: int, target_length: int = None):
        height, width = hidden_states.shape[2:]

        # [B * F, C, H, W] -> [B * H * W, C, F]
        hidden_states = rearrange(hidden_states, '(b f) c h w -> (b h w) c f', f=video_length)
        if target_length is None:
            hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2, mode='nearest')
        else:
            hidden_states = nn.functional.interpolate(hidden_states, size=target_length, mode='nearest')
        hidden_states = self.temporal_conv(hidden_states)
        
        td_seq_len = hidden_states.shape[-1]

        # [B * H * W, C, F] -> [B * F, C, H, W]
        hidden_states = rearrange(hidden_states, '(b h w) c f -> (b f) c h w', h=height, w=width)

        return hidden_states, td_seq_len

    def forward(self, hidden_states, video_length: int, target_length: int = None):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, td_seq_len = self._forward_tp_conv(hidden_states, video_length, target_length)
        # hidden_states, td_seq_len = torch.utils.checkpoint.checkpoint(self._forward_tp_conv, hidden_states, video_length, target_length, use_reentrant=True)

        return hidden_states, td_seq_len


class AutoencoderKL3D(ModelMixin, ConfigMixin):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        num_temp_down_blocks=1, 
        num_temp_up_blocks=1
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            num_temp_down_blocks=num_temp_down_blocks
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            num_temp_up_blocks=num_temp_up_blocks
        )

        self.quant_conv = torch.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)
        self.use_slicing = False

    def encode(self, x: torch.FloatTensor, video_length: int, return_dict: bool = True) -> AutoencoderKL3DOutput | dict:
        h, h_td, td_seq = self.encoder(x, video_length)
        moments = self.quant_conv(h)
        
        moments_td = self.quant_conv(h_td)
        posterior = DiagonalGaussianDistribution(moments)

        posterior_td = DiagonalGaussianDistribution(moments_td)

        if not return_dict:
            return (posterior, posterior_td, td_seq)

        return AutoencoderKL3DOutput(latent_dist=posterior, latent_dist_td=posterior_td, td_seq=td_seq)

    def _decode(self, z: torch.FloatTensor, z_td: torch.FloatTensor=None, td_seq_len: int=None, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        if z_td is not None and td_seq_len is not None:
            z_td = self.post_quant_conv(z_td)
            
        dec = self.decoder(z=z, z_td=z_td, td_seq_len=td_seq_len)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def decode(self, z: torch.FloatTensor, video_length: int, z_td: torch.FloatTensor=None, td_seq_len: int=None, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if z_td is not None and td_seq_len is not None:
            if self.use_slicing and z.shape[0] > 1:
                decoded_slices = [self._decode(z_slice, z_td, td_seq_len).sample for z_slice in z.split(1)]
                decoded = torch.cat(decoded_slices)
            else:
                decoded = self._decode(z, z_td, td_seq_len).sample
        else:
            if self.use_slicing and z.shape[0] > 1:
                decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
                decoded = torch.cat(decoded_slices)
            else:
                decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.FloatTensor,
        video_length: int, 
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        # posterior = self.encode(x).latent_dist
        output = self.encode(x, video_length)
        posterior = output.latent_dist
        posterior_td = output.latent_dist_td
        td_seq = output.td_seq
        
        if sample_posterior:
            z = posterior.sample(generator=generator)
            z_td = posterior_td.sample(generator=generator)
        else:
            z = posterior.mode()
            z_td = posterior_td.mode()
        dec = self.decode(z, video_length=video_length, z_td=z_td, td_seq_len=td_seq.shape[0]).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_name_or_path: str, subfolder: str=None, num_temp_down_blocks=1, num_temp_up_blocks=1):
        pretrained_model_path = pretrained_model_name_or_path
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["num_temp_down_blocks"] = num_temp_down_blocks
        config["num_temp_up_blocks"] = num_temp_up_blocks

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model