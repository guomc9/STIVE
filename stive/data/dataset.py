from datasets import Dataset
import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
import pickle

class VideoPromptDataset(Dataset):
    def __init__(self, data_dir, num_frames=12, sample_stride=1, height=512, width=512, rand_slice=False, concepts_prompt=False):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.resolution = (height, width)
        self.rand_slice = rand_slice
        self.concepts_prompt = concepts_prompt
        self.videos = self._load_data()

    def _preprocess(self, frames, resize=False, rescale=False, to_float=False):
        resized_frames = []
        for i, frame in enumerate(frames):
            if resize:
                resized_frames.append(cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4))
            else:
                resized_frames.append(frame)

        resized_frames = np.stack(resized_frames)
        if len(resized_frames.shape) == 3:
            resized_frames = resized_frames[..., np.newaxis]
        if rescale and to_float:
            resized_frames = (resized_frames / 255.0 * 2) - 1
        elif to_float:
            resized_frames = resized_frames / 255.0

        return resized_frames

    def _load_data(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'video_prompts.csv'))
        video_dir = os.path.join(self.data_dir, 'videos')
        videos = {}
        for _, row in df.iterrows():
            video_name = row['Video name']
            prompt = row['Our GT caption'] if not self.concepts_prompt else row['Concepts Prompt']
            video_path = os.path.join(video_dir, video_name)
            if os.path.exists(video_path):
                videos[video_name] = {}
                cap = cv2.VideoCapture(video_path)
                if len(cap) >= self.num_frames:
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if self.resolution[0] != int(frame.shape[0]) or self.resolution[1] != int(frame.shape[1]):
                            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    cap.release()
                    frames = torch.from_numpy(np.array(self._preprocess(frames, rescale=True, to_float=True)))
                    videos[video_name]['frames'] = frames
                    videos[video_name]['prompt'] = prompt

        return videos

    def _sample_frames(self, video: torch.FloatTensor):
        if self.rand_slice:
            sample_stride = self.sample_stride if len(video) // self.num_frames >= self.sample_stride else len(video) // self.num_frames
            F = video.shape[0]
            max_start_frame = F - (self.num_frames - 1) * sample_stride
            start_frame = random.randint(0, max_start_frame - 1)
            sample_indices = torch.arange(start_frame, start_frame + self.num_frames * sample_stride, sample_stride)[:self.num_frames]
            frames = video[sample_indices]
        else:
            sample_stride = self.sample_stride if len(video) // self.num_frames >= self.sample_stride else len(video) // self.num_frames
            sample_indices = torch.arange(0, len(video), sample_stride)[:self.num_frames]
            frames = video[sample_indices]
        
        return frames, sample_indices
    
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        prompts = []                            # [B]
        frames_list = []                        # [B, F, H, W, 3]
        if isinstance(idx, list):
            for i in idx:
                video_name = list(self.videos.keys())[i]
                video = self.videos[video_name]
                frames = video['frames']
                prompt = video['prompt']
                frames, sample_indices = self._sample_frames(frames)                    # [F, H, W, 3]
                prompts.append(prompt)
                frames_list.append(frames)
                
            return {'frames': torch.stack(frames_list), 'prompts': prompts}
        else:
            video_name = list(self.videos.keys())[idx]
            video = self.videos[video_name]
            frames = video['frames']
            prompt = video['prompt']
            frames, sample_indices = self._sample_frames(frames)                    # [F, H, W, 3]
            prompts.append(prompt)
            frames_list.append(frames)
                
            return {'frames': torch.stack(frames_list), 'prompts': prompts}
        
class VideoPromptTupleDataset(Dataset):
    def __init__(self, sources, prompts=None, num_frames=12, sample_stride=1, height=512, width=512, rand_slice=False):
        self.sources = sources
        if prompts is None:
            self.prompts = [None for _ in range(len(sources))]
        else:
            self.prompts = prompts
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.resolution = (height, width)
        self.video_captions = self._load_data()
        self.rand_slice = rand_slice

    def _preprocess(self, frames):
        frames = (frames / 255.0 * 2) - 1
        return frames

    def _load_data(self):
        video_captions = []
        for i, source in enumerate(self.sources):
            cap = cv2.VideoCapture(source)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frames.append(frame)
            cap.release()
            
            if len(frames) >= self.num_frames:
                frames = torch.from_numpy(self._preprocess(np.array(frames)))
                video_captions.append((frames, self.prompts[i]))
        
        return video_captions

    def _sample_frames(self, video: torch.FloatTensor):
        if self.rand_slice:
            sample_stride = self.sample_stride if len(video) // self.num_frames >= self.sample_stride else len(video) // self.num_frames
            F = video.shape[0]
            max_start_frame = F - (self.num_frames - 1) * sample_stride
            start_frame = random.randint(0, max_start_frame - 1)
            sample_indices = torch.arange(start_frame, start_frame + self.num_frames * sample_stride, sample_stride)[:self.num_frames]
            frames = video[sample_indices]
        else:
            sample_stride = self.sample_stride if len(video) // self.num_frames >= self.sample_stride else len(video) // self.num_frames
            sample_indices = torch.arange(0, len(video), sample_stride)[:self.num_frames]
            frames = video[sample_indices]
        
        return frames, sample_indices

    def __len__(self):
        return len(self.video_captions)

    def __getitem__(self, idx):
        prompts = []                            # [B]
        frames_list = []                        # [B, F, H, W, 3]
        if isinstance(idx, list):
            for i in idx:
                frames = self.video_captions[i][0]
                prompt = self.video_captions[i][1]
                frames, sample_indices = self._sample_frames(frames)                    # [F, H, W, 3]
                prompts.append(prompt)
                frames_list.append(frames)
                
            return {'frames': torch.stack(frames_list), 'prompts': prompts}
        else:
            frames = self.video_captions[idx][0]
            prompt = self.video_captions[idx][1]
            frames, sample_indices = self._sample_frames(frames)                        # [F, H, W, 3]
            prompts.append(prompt)
            frames_list.append(frames)
                
            return {'frames': torch.stack(frames_list), 'prompts': prompts}


class VideoEditPromptsDataset(Dataset):
    def __init__(self, source, source_prompt, edit_prompts=None, num_frames=12, sample_stride=1, height=512, width=512):
        self.source = source
        self.source_prompt = source_prompt
        if edit_prompts is None:
            self.prompts = []
        else:
            self.prompts = edit_prompts
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.resolution = (height, width)
        self.frames = self._load_data()

    def _preprocess(self, frames):
        frames = (frames / 255.0 * 2) - 1
        return frames

    def _load_data(self):
        cap = cv2.VideoCapture(self.source)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
        cap.release()
        
        if len(frames) >= self.num_frames:
            frames = torch.from_numpy(np.array(frames))
            frames = self._preprocess(frames)
        else:
            self.num_frames = len(frames)
            frames = torch.from_numpy(np.array(frames))
            frames = self._preprocess(frames)
        
        return frames

    def _sample_frames(self, video: torch.FloatTensor):
        sample_stride = self.sample_stride if len(video) // self.num_frames >= self.sample_stride else len(video) // self.num_frames
        sample_indices = torch.arange(0, len(video), sample_stride)[:self.num_frames]
        frames = video[sample_indices]
        
        return frames, sample_indices

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        source_prompts = []
        prompts = []                            # [B]
        frames_list = []                        # [B, F, H, W, 3]
        if isinstance(idx, list):
            for i in idx:
                source_prompts.append(self.source_prompt)
                prompt = self.prompts[i]
                frames, sample_indices = self._sample_frames(self.frames)                    # [F, H, W, 3]
                prompts.append(prompt)
                frames_list.append(frames)
                
            return {'frames': torch.stack(frames_list), 'prompts': prompts, 'source_prompts': source_prompts}
        else:
            source_prompts = []
            prompt = prompts[idx]
            source_prompts.append(self.source_prompt)
            frames, sample_indices = self._sample_frames(self.frames)                        # [F, H, W, 3]
            prompts.append(prompt)
            frames_list.append(frames)
                
            return {'frames': torch.stack(frames_list), 'prompts': prompts, 'source_prompts': source_prompts}
        
    def get_source(self):
        return {'frames': torch.stack([self._sample_frames(self.frames)[0]]), 'source_prompts': [self.source_prompt]}


class LatentPromptCacheDataset(Dataset):
    def __init__(self, data_dir, num_frames=12, sample_stride=1, height=512, width=512, rand_slice=False, concepts_prompt=False):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.height = height
        self.width = width
        self.rand_slice = rand_slice
        self.concepts_prompt = concepts_prompt
        self.latents, self.prompts = self._load_data()

    def _load_data(self):
        latents_dir = os.path.join(self.data_dir, 'latents')
        prompts_path = os.path.join(self.data_dir, 'video_prompts.csv')
        
        # Load prompts
        df = pd.read_csv(prompts_path)
        if self.concepts_prompt:
            prompts = df['Concepts Prompt'].tolist()
        else:
            prompts = df['Our GT caption'].tolist()

        latent_names = df['Video name'].tolist()
        # Load latents
        latents = []
        for i, latent_name in enumerate(latent_names):
            latent_path = os.path.join(latents_dir, f'{latent_name}.pt')
            latents.append(torch.load(latent_path))

        return latents, prompts

    def _sample_latents(self, latents: torch.FloatTensor):
        if self.rand_slice:
            sample_stride = self.sample_stride if len(latents) // self.num_frames >= self.sample_stride else len(latents) // self.num_frames
            F = latents.shape[0]
            max_start_frame = F - (self.num_frames - 1) * sample_stride
            start_frame = random.randint(0, max_start_frame - 1)
            sample_indices = torch.arange(start_frame, start_frame + self.num_frames * sample_stride, sample_stride)[:self.num_frames]
            latents = latents[sample_indices]
        else:
            sample_stride = self.sample_stride if len(latents) // self.num_frames >= self.sample_stride else len(latents) // self.num_frames
            sample_indices = torch.arange(0, len(latents), sample_stride)[:self.num_frames]
            latents = latents[sample_indices]
        
        return latents

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, torch.Tensor)):
            latents = torch.stack([self._sample_latents(self.latents[i]) for i in idx])
            prompts = [self.prompts[i] for i in idx]
        else:
            latents = torch.stack([self._sample_latents(self.latents[idx])])
            prompts = [self.prompts[idx]]
        
        return {'latents': latents, 'prompts': prompts}

    def __len__(self):
        return len(self.prompts)
    
class LatentPromptTupleCacheDataset(Dataset):
    def __init__(self, latents, prompts, num_frames=12, sample_stride=1, height=512, width=512, rand_slice=False):
        self.latents = latents
        self.prompts = prompts
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.height = height
        self.width = width
        self.rand_slice = rand_slice

    def _sample_latents(self, latents: torch.FloatTensor):
        if self.rand_slice:
            sample_stride = self.sample_stride if len(latents) // self.num_frames >= self.sample_stride else len(latents) // self.num_frames
            F = latents.shape[0]
            max_start_frame = F - (self.num_frames - 1) * sample_stride
            start_frame = random.randint(0, max_start_frame - 1)
            sample_indices = torch.arange(start_frame, start_frame + self.num_frames * sample_stride, sample_stride)[:self.num_frames]
            latents = latents[sample_indices]
        else:
            sample_stride = self.sample_stride if len(latents) // self.num_frames >= self.sample_stride else len(latents) // self.num_frames
            sample_indices = torch.arange(0, len(latents), sample_stride)[:self.num_frames]
            latents = latents[sample_indices]
        
        return latents

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, torch.Tensor)):
            latents = torch.stack([self._sample_latents(self.latents[i]) for i in idx])
            prompts = [self.prompts[i] for i in idx]
        else:
            latents = torch.stack([self._sample_latents(self.latents[idx])])
            prompts = [self.prompts[idx]]
        
        return {'latents': latents, 'prompts': prompts}

    def __len__(self):
        return len(self.prompts)

from einops import rearrange

class VideoSliceEditPromptDataset(Dataset):
    def __init__(self, latents, source_prompt, edit_prompt=None, num_slice=8, sample_stride=1):
        self.latents = latents
        self.source_prompt = source_prompt
        self.edit_prompt = edit_prompt
        self.num_slice = num_slice
        self.sample_stride = sample_stride
        self.latents = self._sample_latents(self.latents)

    def _sample_latents(self, latents: torch.FloatTensor):
        sample_indices = torch.arange(0, len(latents), self.sample_stride)
        latents = latents[sample_indices]
        clip_length = len(latents) - len(latents) % self.num_slice
        assert clip_length >= self.num_slice
        latents = latents[:clip_length]
        latents = rearrange(latents, '(b s) h w c -> b s h w c', s=self.num_slice)
        return latents

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        source_prompts = []
        prompts = []                            # [B]
        latents_list = []                       # [B, S, H, W, 3]
        if isinstance(idx, list):
            for i in idx:
                source_prompts.append(self.source_prompt)
                prompts.append(self.edit_prompt)
                latents_list.append(self.latents[i])
                
            return {'latents': torch.stack(latents_list), 'prompts': prompts, 'source_prompts': source_prompts}
        else:
            return {'latents': torch.stack(self.latents[idx]), 'prompts': [self.edit_prompt], 'source_prompts': [self.source_prompt]}


import copy
import random
from stive.utils.cache_latents_utils import encode_videos_latents

class LatentPromptDataset(Dataset):
    def __init__(
        self, 
        video_paths, 
        prompts, 
        mask_paths, 
        enable_temporal_modules=None, 
        repeat=None, 
        num_frames=12, 
        sample_stride=1, 
        height=512, 
        width=512, 
        enable_slice=True, 
        rand_slice=False, 
        rand_slice_length=False, 
        rand_slice_stride=False, 
        relax_mask=0, 
        concepts_prompt=False, 
        target_video_paths=None, 
        target_mask_paths=None, 
        target_prompts=None, 
        enable_target_slice=True, 
        target_rand_slice=False, 
        target_rand_slice_length=False, 
        target_rand_slice_stride=False, 
        relax_target_mask=0, 
    ):
        assert len(video_paths) == len(prompts) == len(mask_paths)
        self.video_paths = []
        self.prompts = []
        self.mask_paths = []
        self.enable_temporal_modules = enable_temporal_modules
        for video_path, prompt, mask_path in zip(video_paths, prompts, mask_paths):
            if os.path.exists(video_path) and os.path.exists(mask_path):
                self.video_paths.append(video_path)
                self.mask_paths.append(mask_path)
                self.prompts.append(prompt)
                
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.resolution = (height, width)
        self.concepts_prompt = concepts_prompt
        self.enable_slice = enable_slice
        self.repeat = repeat
        self.rand_slice = rand_slice
        self.rand_slice_length = rand_slice_length
        self.rand_slice_stride = rand_slice_stride

        self.relax_mask = relax_mask
        self.target_video_paths = []
        self.target_mask_paths = []
        self.target_prompts = []
        self.enable_target_slice = enable_target_slice
        self.target_rand_slice = target_rand_slice
        self.target_rand_slice_length = target_rand_slice_length
        self.target_rand_slice_stride = target_rand_slice_stride

        self.relax_target_mask = relax_target_mask
        if target_video_paths is not None and target_mask_paths is not None and target_prompts is not None:
            assert len(target_video_paths) == len(target_mask_paths) == len(target_prompts)
            for target_video_path, target_mask_path, target_prompt in zip(target_video_paths, target_mask_paths, target_prompts):
                if os.path.exists(target_video_path) and os.path.exists(target_mask_path):
                    self.target_mask_paths.append(target_mask_path)
                    self.target_video_paths.append(target_video_path)
                    self.target_prompts.append(target_prompt)
            print(f'target_prompts: {target_prompts}')
                    
        self.target_latents = None
        self.target_masks = None
        self.masks = self._load_masks(self.mask_paths, relax=self.relax_mask)                               # [B, F, 1, H, W]: List
        if len(self.target_video_paths) > 0:
            target_video_size = len(self.target_video_paths)
            video_paths = copy.deepcopy(self.video_paths)
            video_paths.extend(self.target_video_paths)
            latents = encode_videos_latents(video_paths, height=height, width=width)                        # [B, F, C, H, W]: List
            self.target_latents = latents[-target_video_size:]                                              # [T, F, C, H, W]
            self.latents = latents[:-target_video_size]
            self.target_masks = self._load_masks(self.target_mask_paths, relax=self.relax_target_mask)      # [T, F, 1, H, W]: List
        else:
            self.latents = encode_videos_latents(self.video_paths, height=height, width=width)              # [B, F, C, H, W]: List

    def _load_masks(self, mask_paths, relax=0):
        masks_list = []
        for mask_path in mask_paths:
            cap = cv2.VideoCapture(mask_path)
            masks = []
            while True:
                ret, mask = cap.read()
                if not ret:
                    break
                masks.append(mask)
            cap.release()
            if len(masks) >= self.num_frames:
                masks = torch.from_numpy(np.asarray(masks)[..., :1] / 255)              # [F, H, W, 1]
                masks = rearrange(masks, 'f h w c -> f c h w')                          # [F, 1, H, W]
                for _ in range(0, relax):
                    masks = self._relax_binary_tensor(masks)                            # [F, 1, H, W]
                masks_list.append(masks)
        return masks_list                                                               # [T, F, 1, H, W]

    @staticmethod
    def _relax_binary_tensor(tensor, kernel_size=3, stride=1, padding=1):
        assert tensor.dim() == 4
        
        relaxed = torch.nn.functional.max_pool2d(tensor, kernel_size=kernel_size, stride=stride, padding=padding)
        
        return relaxed

    def _sample_latents(self, latents: torch.FloatTensor, enable_slice: bool, rand_slice: bool, rand_slice_stride: bool=False, rand_slice_length: bool=False):
        if enable_slice:
            if rand_slice and not rand_slice_stride:
                sample_stride = self.sample_stride if len(latents) // self.num_frames >= self.sample_stride else len(latents) // self.num_frames
                F = latents.shape[0]
                max_start_frame = F - (self.num_frames - 1) * sample_stride
                start_frame = random.randint(0, max_start_frame - 1)
                sample_indices = torch.arange(start_frame, start_frame + self.num_frames * sample_stride, sample_stride)[:self.num_frames]
                latents = latents[sample_indices]
            elif rand_slice and rand_slice_stride:
                num_frames = self.num_frames if not rand_slice_length else random.randint(max(self.num_frames // 2, 1), max(self.num_frames, 1))
                F = latents.shape[0]
                sample_indices = torch.randint(0, F, (num_frames,))
                sample_indices, _ = torch.sort(sample_indices)
                print(f'sample_indices: {sample_indices}')
                latents = latents[sample_indices]
            else:
                sample_stride = self.sample_stride if len(latents) // self.num_frames >= self.sample_stride else len(latents) // self.num_frames
                sample_indices = torch.arange(0, len(latents), sample_stride)[:self.num_frames]
                latents = latents[sample_indices]
        else:
            sample_indices = torch.randperm(len(latents))[:self.num_frames]
            latents = latents[sample_indices]
        
        return latents, sample_indices
    
    def __len__(self):
        if self.repeat is not None:
            return self.repeat * len(self.prompts)
        
        return len(self.prompts)

    def __getitem__(self, idx):
        prompts = []                            # [B]
        latents_list = []                       # [B, F, C, H, W]
        masks_list = []                         # [B, F, 1, H, W]
        enable_temporal_modules = []            # [B]
        target_latents_list = []                # [B, F, C, H, W]
        target_masks_list = []                  # [B, F, 1, H, W]
        target_prompts = []                     # [B]
        if isinstance(idx, list):
            for i in idx:
                if self.repeat is not None:
                    i = i % len(self.prompts)
                latents = self.latents[i]
                prompt = self.prompts[i]
                masks = self.masks[i]
                latents, sample_indices = self._sample_latents(latents, enable_slice=self.enable_slice, rand_slice=self.rand_slice, \
                                                               rand_slice_length=self.rand_slice_length, rand_slice_stride=self.rand_slice_stride)   # [F, C, H, W]
                masks = masks[sample_indices]
                prompts.append(prompt)                                                              # [B]
                latents_list.append(latents)
                masks_list.append(masks)
                if self.enable_temporal_modules is not None:
                    enable_temporal_modules.append(self.enable_temporal_modules[i])
                else:
                    enable_temporal_modules.append(True)
                if self.target_latents is not None:
                    ti = random.randint(0, len(self.target_latents)-1)
                    target_latents = self.target_latents[ti]                                        # [F, C, H, W]
                    target_masks = self.target_masks[ti]                                            # [F, C, H, W]
                    target_latents, sample_indices = self._sample_latents(target_latents, enable_slice=self.enable_target_slice, rand_slice=self.target_rand_slice, \
                                                                          rand_slice_length=self.target_rand_slice_length, rand_slice_stride=self.target_rand_slice_stride)           # [F, C, H, W]
                    target_masks = target_masks[sample_indices]                                     # [F, 1, H, W]
                    target_latents_list.append(target_latents)                                      # [B, F, C, H, W]
                    target_masks_list.append(target_masks)                                          # [B, 1, C, H, W]
                    target_prompts.append(self.target_prompts[ti])                                  # [B]
        else:
            if self.repeat is not None:
                idx = idx % len(self.prompts)
            latents = self.latents[idx]
            prompt = self.prompts[idx]
            masks = self.masks[idx]
            if self.enable_temporal_modules is not None:
                enable_temporal_modules.append(self.enable_temporal_modules[i])
            else:
                enable_temporal_modules.append(True)
            latents, sample_indices = self._sample_latents(latents, enable_slice=self.enable_slice, rand_slice=self.rand_slice)        # [F, C, H, W]
            masks = masks[sample_indices]
            prompts.append(prompt)
            latents_list.append(latents)
            masks_list.append(masks)
            if self.target_latents is not None:
                ti = random.randint(0, len(self.target_latents)-1)
                target_latents = self.target_latents[ti]                                        # [F, C, H, W]
                target_masks = self.target_masks[ti]                                            # [F, C, H, W]
                target_latents, sample_indices = self._sample_latents(target_latents, enable_slice=self.enable_target_slice, rand_slice=self.target_rand_slice)           # [F, C, H, W]
                target_masks = target_masks[sample_indices]                                     # [F, C, H, W]
                target_latents_list.append(target_latents)                                      # [B, F, C, H, W]
                target_masks_list.append(target_masks)                                          # [B, 1, C, H, W]
                target_prompts.append(self.target_prompts[ti])                                  # [B]
                
        if self.target_latents is not None:
            return {
                'latents': torch.stack(latents_list, dim=0), 
                'masks': torch.stack(masks_list, dim=0), 
                'enable_temporal_modules': torch.as_tensor(enable_temporal_modules), 
                'prompts': prompts, 
                'target_latents': torch.stack(target_latents_list, dim=0), 
                'target_masks': torch.stack(target_masks_list, dim=0), 
                'target_prompts': target_prompts
                }
        else:
            return {
                'latents': torch.stack(latents_list, dim=0), 
                'masks': torch.stack(masks_list, dim=0), 
                'enable_temporal_modules': torch.as_tensor(enable_temporal_modules), 
                'prompts': prompts, 
                }
            