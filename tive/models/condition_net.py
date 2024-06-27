import torch
import torch.nn as nn

class ConditionNet3D(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 hidden_channels_list: list = [64, 128], 
                 out_channels: int = 320
                 ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=hidden_channels_list[0], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True), 
            
            nn.Conv3d(in_channels=hidden_channels_list[0], out_channels=hidden_channels_list[1], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True), 
            
            nn.Conv3d(in_channels=hidden_channels_list[1], out_channels=out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        )

    def forward(self, condition:torch.Tensor):
        assert condition.ndim == 5, f"Input 'condition.ndim' must be 5, not {condition.ndim}"
        return self.backbone(condition)
    
class ConditionDownBlock3D(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 down_sample: bool = True
                 ):
        super().__init__()
        self.down_sample = down_sample
        self.lin_in = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        if down_sample:
            self.down_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        else:
            self.down_conv = None
        self.lin_out = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        nn.init.zeros_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)

    def forward(self, condition:torch.Tensor):
        assert condition.ndim == 5, f"Input 'condition.ndim' must be 5, not {condition.ndim}"
        input_tensor = condition
        condition = torch.relu(self.lin_in(condition)) + input_tensor
        if self.down_sample:
            return self.lin_out(torch.relu(self.down_conv(condition))), condition
        else:
            return self.lin_out(condition), condition
            
    
class ConditionUpBlock3D(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 up_sample: bool = True
                 ):
        super().__init__()
        self.up_sample = up_sample
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.lin_out = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        nn.init.zeros_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)

    def forward(self, condition:torch.Tensor):
        assert condition.ndim == 5, f"Input 'condition.ndim' must be 5, not {condition.ndim}"
        if self.up_sample:
            condition = torch.nn.functional.interpolate(self.conv(condition), scale_factor=(1, 2, 2), mode='trilinear',align_corners=False)
        else:
            condition = self.conv(condition)
            
        return self.lin_out(torch.relu(condition))

class ConditionBlock3D(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int
                 ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.lin_out = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        nn.init.zeros_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)

    def forward(self, condition:torch.Tensor):
        assert condition.ndim == 5, f"Input 'condition.ndim' must be 5, not {condition.ndim}"
        return self.lin_out(torch.relu(self.conv(condition)))