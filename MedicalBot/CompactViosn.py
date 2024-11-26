import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class LightEfficientMedicalNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        
        # Reduced initial channels
        self.initial_channels = 16  # Reduced from 32
        self.reduction_ratio = 8    # Increased from 4 for fewer SE parameters
        
        # Initial convolution with reduced channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.initial_channels, kernel_size=5, stride=2, padding=2),  # Reduced kernel size
            nn.InstanceNorm2d(self.initial_channels, affine=True),
            nn.GELU()
        )
        
        # Reduced channel progression
        self.stage1 = self._make_stage(self.initial_channels, 32, stride=2)   # Reduced from 64
        self.stage2 = self._make_stage(32, 64, stride=2)                      # Reduced from 128
        self.stage3 = self._make_stage(64, 128, stride=2)                     # Reduced from 256
        self.stage4 = self._make_stage(128, 256, stride=2)                    # Reduced from 512
        
        # Global feature refinement
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        
        # Multi-scale feature fusion with reduced channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(480, 256, kernel_size=1),  # 480 = 32 + 64 + 128 + 256
            nn.InstanceNorm2d(256, affine=True),
            nn.GELU()
        )
        
        # Classifier with reduced dimensions
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),    # Reduced from 512, 256
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()

    def _make_stage(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Creates a lightweight stage"""
        return nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, stride),
            SEBlock(out_channels, self.reduction_ratio),
            LightResidualBlock(out_channels)  # Using lightweight residual block
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        
        f3_up = F.interpolate(f3, size=f2.shape[2:])
        f4_up = F.interpolate(f4, size=f2.shape[2:])
        f1_down = F.interpolate(f1, size=f2.shape[2:])
        
        fused = torch.cat([f1_down, f2, f3_up, f4_up], dim=1)
        fused = self.fusion_conv(fused)
        
        out = self.global_pool(fused)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.GELU(),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU()
        )
        
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)

class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class LightResidualBlock(nn.Module):
    """Lightweight residual block with reduced parameters"""
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
            # Using a single 3x3 conv instead of two
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)
