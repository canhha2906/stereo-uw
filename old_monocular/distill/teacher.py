"""Frozen monocular-depth teacher used only to generate pseudo ground
truth for distillation. Never trained; always eval() + no_grad.

Default is MiDaS-small (torch.hub, needs `timm`) because it loads with no
extra config/weight files and is cheap enough to run inference-only
during distillation. Swap in a stronger teacher (e.g. Depth Anything V2
via `transformers`) if teacher quality bottlenecks the student -- whatever
teacher is used, its only job is to seed a normalized relative depth map
for the physics simulator (see underwater_physics.py); it plays no role
at deployment time, so its cost doesn't matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MidasSmallTeacher(nn.Module):
    def __init__(self, infer_size: int = 256):
        super().__init__()
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.infer_size = infer_size
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, rgb_0_1: torch.Tensor) -> torch.Tensor:
        """rgb_0_1: (B,3,H,W) in [0,1]. Returns relative depth (B,1,H,W)
        resized back to the input resolution. Higher value = closer,
        per MiDaS convention; no metric meaning."""
        H, W = rgb_0_1.shape[-2:]
        x = F.interpolate(rgb_0_1, size=(self.infer_size, self.infer_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        depth = self.model(x).unsqueeze(1)  # (B,1,h,w)
        return F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)


def build_teacher(kind: str) -> nn.Module:
    if kind == "midas_small":
        return MidasSmallTeacher()
    raise ValueError(f"unknown teacher: {kind}")
