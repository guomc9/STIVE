"""
Code of attention storer AttentionStore, which is a base class for attention editor in attention_util.py

"""

import abc
import os
import copy
import torch
from stive.utils.pta_utils import get_time_string

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion

        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                # For inversion without null text file 
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # For classifier-free guidance scale!=1
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1

        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    def step_callback(self, x_t):


        x_t = super().step_callback(x_t)
        self.latents_store.append(x_t.cpu().detach())
        return x_t
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    @staticmethod
    def get_empty_cross_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[-2] <= 32 ** 2:  # avoid memory overhead
            # print(f"Store attention map {key} of shape {attn.shape}")
            if is_cross or self.save_self_attention:
                if attn.shape[-2] == 32**2:
                    append_tensor = attn.cpu().detach()
                else:
                    append_tensor = attn
                self.step_store[key].append(copy.deepcopy(append_tensor))
                # FIXME: Are these deepcopy all necessary?
                # self.step_store[key].append(append_tensor)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        
        if self.disk_store:
            path = self.store_dir + f'/{self.cur_step:03d}.pt'
            torch.save(copy.deepcopy(self.step_store), path)
            self.attention_store_all_step.append(path)
        else:
            self.attention_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()
        # print('clear step_store !!!!')

    def get_average_attention(self):
        "divide the attention map value in attention store by denoising steps"
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store_all_step = []
        self.attention_store = {}

    def __init__(self, save_self_attention:bool=True, disk_store=False):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []
        self.attention_store_all_step = []


class StepAttentionControl(abc.ABC):
    def __init__(self):
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1

    def reset(self):
        self.cur_att_layer = 0

import gc
class StepAttentionStore(StepAttentionControl):
    def __init__(self):
        super().__init__()
        self.attention_store = {}       # {'down-cross-1024': attn_prob}, attn_prob.shape: [B * F, M, Q, K]

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}-{'cross' if is_cross else 'self'}-{attn.shape[-2]}"
        if key not in self.attention_store:
            self.attention_store[key] = []
        self.attention_store[key].append(attn)

    def reset(self):
        super().reset()
        self.attention_store.clear()
        gc.collect()

    def get_mean_head_attns(self):
        attns = {}
        for key, attn_list in self.attention_store.items():      # [B * F, M, Q, K]
            if attn_list:
                attns[key]  = attn_list[0].mean(1).detach().cpu()
        return attns                                        # [B * F, Q, K]

import numpy as np
from einops import rearrange, repeat
from stive.prompt_attention.ptp_utils import pool_mask
class StepAttentionSupervisor(StepAttentionStore):
    def get_cross_attn_mask_loss(self, mask, target_indices, sub_sot=True, only_neg=False, loss_type='mae', reduction='mean'):
        """
        mask: [B, F, 1, H, W]
        target_indices: [B, T]
        attn: [B * F, M, Q, K]
        """
        losses = []
        b = len(target_indices)
        mask_check = {}
        for key, attns in self.attention_store.items():
            if 'cross' not in key:
                continue
            for i in range(b):
                mask_check[f'batch-{i}-{key}'] = {}
                
            for attn in attns:
                m, q = attn.shape[1:3]                                                                      # [B * F, M, Q, K]
                
                h = w = int(np.sqrt(q))
                f = mask.shape[1]
                
                attn = rearrange(attn, '(b f) m q k -> b (f m) q k', f=f)                                   # [B, F * M, Q, K]
                
                adapt_mask = pool_mask(mask, target_size=(h, w))                                            # [B, F, 1, H, W]
                adapt_mask = rearrange(adapt_mask, 'b f c h w -> b f (h w) c').squeeze(-1)                  # [B, F, Q]
                adapt_mask = repeat(adapt_mask, 'b f q -> b (f m) q', m=m).detach_()                        # [B, F * M, Q]
                if sub_sot and not only_neg:
                    adapt_mask = adapt_mask - adapt_mask * attn[..., 0]                                     # [B, F * M, Q]
                    # NO DETACH IS BETTER SINCE WEAKER SUPERVISE
                    # adapt_mask.detach_()
                    
                for i in range(b):
                    neg_mask = None
                    if only_neg:
                        neg_mask = (adapt_mask[i] < 1e-8).to(adapt_mask.dtype).detach()
                        mask_check[f'batch-{i}-{key}'] = rearrange(neg_mask, '(f m) (h w) -> f m h w', f=f, m=m, h=h, w=w).mean(dim=1).detach().cpu()  # [F, H, W]
                    else:
                        mask_check[f'batch-{i}-{key}'] = rearrange(adapt_mask[i], '(f m) (h w) -> f m h w', f=f, m=m, h=h, w=w).mean(dim=1).detach().cpu()  # [F, H, W]
                        
                    inds = torch.as_tensor(target_indices[i]).to(attn.device)
                    t = inds.shape[0]
                    if loss_type == 'mae':
                        if neg_mask is None:
                            losses.append((adapt_mask[i].unsqueeze(-1) - attn[i, ..., inds]).abs().mean())                  # [F * M, Q, T]
                        else:
                            losses.append((neg_mask.unsqueeze(-1) * (adapt_mask[i].unsqueeze(-1) - attn[i, ..., inds])).abs().sum() / neg_mask.sum())     # [F * M, Q, T]
                    elif loss_type == 'mse':
                        if neg_mask is None:
                            losses.append(((adapt_mask[i].unsqueeze(-1) - attn[i, ..., inds])**2).mean())                  # [F * M, Q, T]
                        else:
                            losses.append(((neg_mask.unsqueeze(-1) * (adapt_mask[i].unsqueeze(-1) - attn[i, ..., inds]))**2).sum() / neg_mask.sum())     # [F * M, Q, T]
                    elif loss_type == 'bce':
                        losses.append(torch.nn.functional.binary_cross_entropy(attn[i, ..., inds], adapt_mask[i].unsqueeze(-1).repeat(1, 1, t), reduction='mean'))
                    else:
                        raise ValueError(f"Unsupported loss type: {loss_type}")
        if reduction == 'mean':
            return torch.mean(torch.stack(losses)), mask_check
        else:
            return torch.sum(torch.stack(losses)), mask_check
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super().forward(attn, is_cross, place_in_unet)
