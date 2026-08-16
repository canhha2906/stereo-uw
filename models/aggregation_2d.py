"""2D-conv aggregation alternative to the 3D hourglass.

Same input cost volume (B, G, D, H, W). Reshape to (B, G*D, H, W) and run
a 2D conv stack. Output is reshaped back to (B, D, H, W).

This makes the 2D-vs-3D comparison apples-to-apples: identical cost volume,
only the aggregation operator differs.
"""
import torch
import torch.nn as nn


def conv2d_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Aggregation2D(nn.Module):
    """2D-conv aggregation. If context_channels > 0, left-image context features
    are concatenated to the reshaped cost volume before the conv stack."""

    def __init__(self, groups: int, d_at_scale: int, hidden: int = 128,
                 context_channels: int = 0):
        super().__init__()
        self.G = groups
        self.D = d_at_scale
        self.context_channels = context_channels
        in_c = groups * d_at_scale + context_channels

        # Bottleneck so the parameter count doesn't blow up.
        self.stack = nn.Sequential(
            conv2d_bn_relu(in_c, hidden, k=3),
            conv2d_bn_relu(hidden, hidden, k=3),
            conv2d_bn_relu(hidden, hidden, k=3),
            conv2d_bn_relu(hidden, hidden, k=3),
            nn.Conv2d(hidden, d_at_scale, kernel_size=3, padding=1),
        )

    def forward(self, cost: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        B, G, D, H, W = cost.shape
        assert G == self.G and D == self.D, f"got G={G} D={D}, expected G={self.G} D={self.D}"
        x = cost.reshape(B, G * D, H, W)
        if self.context_channels > 0 and context is not None:
            x = torch.cat([x, context], dim=1)   # (B, G*D + Cc, H, W)
        out = self.stack(x)            # (B, D, H, W)
        return out
