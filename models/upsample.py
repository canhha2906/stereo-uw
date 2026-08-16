"""Upsample disparity from 1/s scale to full resolution.

Two implementations:

1. ConvexUpsample (RAFT-style): predict a 3x3 mask per output pixel and use it
   to weighted-sum a 3x3 neighborhood of the low-res disparity. Sharper edges
   than bilinear, but uses F.unfold + softmax which may need TRT plugins on
   older JetPacks.

2. BilinearUpsample: F.interpolate + value scaling. Falls out cleanly to ONNX.
   Use this as the GATE fallback if convex upsample blocks the TRT engine build.

The factor of `scale` ALSO multiplies the disparity values (disp lives in
pixel units of the OUTPUT resolution, not the input feature scale).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexUpsample(nn.Module):
    """RAFT-style convex upsampling. Predicts (B, scale^2 * 9, H, W) mask."""

    def __init__(self, feat_channels: int, scale: int):
        super().__init__()
        self.scale = scale
        self.mask_head = nn.Sequential(
            nn.Conv2d(feat_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale * scale * 9, 1, 1, 0),
        )

    def forward(self, disp_low: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # disp_low: (B, H, W), feat: (B, C, H, W) (left feature at same scale)
        B, H, W = disp_low.shape
        s = self.scale
        mask = self.mask_head(feat)                       # (B, s*s*9, H, W)
        mask = mask.view(B, 1, 9, s, s, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp_low.unsqueeze(1) * s, kernel_size=3, padding=1)
        up_disp = up_disp.view(B, 1, 9, 1, 1, H, W)

        up = torch.sum(mask * up_disp, dim=2)             # (B, 1, s, s, H, W)
        up = up.permute(0, 1, 4, 2, 5, 3).contiguous()    # (B, 1, H, s, W, s)
        return up.view(B, H * s, W * s)


class BilinearUpsample(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, disp_low: torch.Tensor, feat=None) -> torch.Tensor:
        # disp_low: (B, H, W)
        x = disp_low.unsqueeze(1) * self.scale            # values scale with resolution
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x.squeeze(1)


def build_upsample(kind: str, feat_channels: int, scale: int) -> nn.Module:
    if kind == "convex":
        return ConvexUpsample(feat_channels, scale)
    if kind == "bilinear":
        return BilinearUpsample(scale)
    raise ValueError(f"unknown upsample kind: {kind}")
