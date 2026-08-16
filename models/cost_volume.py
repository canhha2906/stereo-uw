"""Group-wise correlation cost volume (GwcNet-style).

Shape: features at 1/s scale are (B, C, H, W). Output cost volume is
(B, G, D/s, H, W) where G = number of groups, D = D_max.

This single cost volume feeds BOTH the 3D hourglass and the 2D-conv
aggregation paths — that's the non-negotiable from the spec.

TensorRT compatibility:
- Avoid torch.einsum and complex indexing (some opsets break).
- Use straight reshape + roll/slice + elementwise multiply + mean.
- For ONNX export, the disparity loop is unrolled at trace time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def groupwise_correlation(fL: torch.Tensor, fR: torch.Tensor, groups: int) -> torch.Tensor:
    """fL, fR: (B, C, H, W). Returns (B, G, H, W) inner product per group."""
    B, C, H, W = fL.shape
    assert C % groups == 0, f"channels {C} must be divisible by groups {groups}"
    cpg = C // groups
    cost = (fL * fR).view(B, groups, cpg, H, W).mean(dim=2)
    return cost  # (B, G, H, W)


class CostVolume(nn.Module):
    """Build the (B, G, D, H, W) group-wise correlation cost volume."""

    def __init__(self, max_disp_at_scale: int, groups: int):
        super().__init__()
        self.D = max_disp_at_scale  # e.g. D_max / 8 = 32 for D_max=256
        self.G = groups

    def forward(self, fL: torch.Tensor, fR: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fL.shape
        D, G = self.D, self.G
        # Pre-pad fR on the left by D-1 so for disparity d we can shift by d
        # without explicit per-d slicing inside the loop body.
        # Using F.pad on the W axis keeps it ONNX-friendly.
        cost = fL.new_zeros((B, G, D, H, W))
        for d in range(D):
            if d == 0:
                cost[:, :, 0, :, :] = groupwise_correlation(fL, fR, G)
            else:
                # right feature shifted by d to the right (matches left at col i to right at col i-d)
                cost[:, :, d, :, :-d] = groupwise_correlation(
                    fL[:, :, :, d:], fR[:, :, :, :-d], G
                )
                # leftmost d columns are out-of-bounds: leave at 0 (mask in loss)
        return cost
