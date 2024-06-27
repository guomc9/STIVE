import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin

class TemporalSpatialDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.nonlinearity = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        skip = x
        x = self.downsample(x)
        return x, skip

class TemporalSpatialUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape != skip.shape:
            print(x.shape)
            print(skip.shape)
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat((skip, x), dim=1)
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        return x

class STDecoder(ModelMixin, ConfigMixin):
    def __init__(self, in_channels=4, out_channels=4, features=[64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(TemporalSpatialDownsampleBlock(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(TemporalSpatialUpsampleBlock(feature*2, feature))

        self.bottleneck = nn.Conv3d(features[-1], features[-1]*2, kernel_size=1)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for enc in self.encoder:
            x, skip = enc(x)
            print(f'{len(skips)} x.shape: {x.shape}')
            print(f'{len(skips)} skip.shape: {skip.shape}')
            skips.append(skip)

        x = self.bottleneck(x)
        print(f'{len(skips)} x.shape: {x.shape}')
        skips = skips[::-1]

        for i, dec in enumerate(self.decoder):
            x = dec(x, skips[i])

        return self.final_conv(x)
    
    def interpolate_frames(self, video_tensor):
        # [B, C, T, H, W] => [B, C, 2T-1, H, W]
        B, C, T, H, W = video_tensor.shape
        
        interpolated_tensor = self.forward(video_tensor)
        
        interpolated_frames = torch.zeros(B, C, 2*T - 1, H, W, device=video_tensor.device)
        
        for t in range(T - 1):
            interpolated_frames[:, :, t*2] = video_tensor[:, :, t]
            interpolated_frames[:, :, t*2 + 1] = interpolated_tensor[:, :, t]
        
        interpolated_frames[:, :, 2*T - 2] = video_tensor[:, :, T - 1]

        return interpolated_frames


model = STDecoder()
video_tensor = torch.randn(2, 4, 24, 64, 64)

# 2T = T+1 + T-1
# T predict T-1

# T
src_inds = torch.arange(0, 24, 2, dtype=torch.long)

# T - 1
gt_inds = torch.arange(1, 23, 2, dtype=torch.long)
print(f'src_inds: {src_inds}')
print(f'gt_inds: {gt_inds}')

interpolated_video = model(video_tensor)[:, :, :-1]
# interpolated_video = model.interpolate_frames(video_tensor)
print(video_tensor.shape)
print(interpolated_video.shape)