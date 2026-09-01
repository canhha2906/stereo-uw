"""Lightweight monocular depth student, distilled from a frozen teacher.

Reuses the same ImageNet-pretrained MobileNetV2 encoder as the stereo
branch (`models/backbone_pretrained.py`) so both directions of the paper
share one edge-friendly feature extractor; only the head differs (dense
depth regression here vs. cost-volume aggregation there).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone_pretrained import MobileNetV2Pretrained

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class MonoDepthStudent(nn.Module):
    def __init__(self, feat_channels: int = 32, output_stride: int = 8):
        super().__init__()
        self.encoder = MobileNetV2Pretrained(out_channels=feat_channels,
                                             output_stride=output_stride,
                                             pretrained=True)
        self.head = DepthHead(feat_channels)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, rgb_0_1: torch.Tensor) -> torch.Tensor:
        """rgb_0_1: (B,3,H,W) in [0,1] -- raw synthesized-underwater or
        real-underwater input. Returns depth at full input resolution."""
        H, W = rgb_0_1.shape[-2:]
        x = (rgb_0_1 - self.mean) / self.std
        feat = self.encoder(x)
        depth_low = self.head(feat)
        return F.interpolate(depth_low, size=(H, W), mode="bilinear", align_corners=False)
