"""Soft-argmin disparity regression (parameter-free).

Input: aggregated cost (B, D, H, W) — lower cost = better match.
Output: disparity map (B, H, W) at the same scale, in DISPARITY UNITS of that scale.

For deployment, this is a softmax over D followed by a weighted sum with
indices 0..D-1. No learnable parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmin(nn.Module):
    def __init__(self, d_at_scale: int):
        super().__init__()
        self.D = d_at_scale
        # Register as buffer so it moves with .to(device) and exports cleanly.
        self.register_buffer("disp_range", torch.arange(d_at_scale, dtype=torch.float32))

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        # cost: (B, D, H, W). Lower cost = better match → negate before softmax.
        prob = F.softmax(-cost, dim=1)
        # weighted sum across D
        disp = (prob * self.disp_range.view(1, -1, 1, 1)).sum(dim=1)  # (B, H, W)
        return disp
