"""3D hourglass aggregation over the (B, G, D, H, W) cost volume.

Trimmed encoder-decoder: 2 stride-2 downsamples then 2 transposed-conv
upsamples with skips. Output: (B, 1, D, H, W) — one cost value per disp/pixel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _match(x, ref):
    """Trilinearly interpolate x to ref's (D,H,W) shape. No-op if already equal."""
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


def conv3d_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


def deconv3d_bn_relu(in_c, out_c, k=4, s=2, p=1, op=0):
    return nn.Sequential(
        nn.ConvTranspose3d(in_c, out_c, k, s, p, output_padding=op, bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


class Aggregation3D(nn.Module):
    """Cost-volume aggregation via a small 3D hourglass.

    If context_channels > 0, left-image context features are projected to `base`
    channels and broadcast-added to the lifted cost volume across the disparity
    dimension (RAFT-style context injection, single-shot variant).
    """

    def __init__(self, groups: int, base: int = 16, context_channels: int = 0):
        super().__init__()
        # Lift G group-channels to `base`
        self.lift = conv3d_bn_relu(groups, base, k=3, s=1, p=1)

        self.context_channels = context_channels
        if context_channels > 0:
            self.ctx_proj = nn.Sequential(
                nn.Conv2d(context_channels, base, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base),
            )

        # Encoder
        self.down1 = conv3d_bn_relu(base, base * 2, k=3, s=2, p=1)  # /2
        self.enc1 = conv3d_bn_relu(base * 2, base * 2, k=3, s=1, p=1)
        self.down2 = conv3d_bn_relu(base * 2, base * 4, k=3, s=2, p=1)  # /4
        self.enc2 = conv3d_bn_relu(base * 4, base * 4, k=3, s=1, p=1)

        # Decoder
        self.up2 = deconv3d_bn_relu(base * 4, base * 2)
        self.dec2 = conv3d_bn_relu(base * 2, base * 2, k=3, s=1, p=1)
        self.up1 = deconv3d_bn_relu(base * 2, base)
        self.dec1 = conv3d_bn_relu(base, base, k=3, s=1, p=1)

        # Project to 1 channel per (D, H, W) cell
        self.head = nn.Conv3d(base, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, cost: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # cost: (B, G, D, H, W) ; context: (B, Cc, H, W) or None
        x0 = self.lift(cost)
        if self.context_channels > 0 and context is not None:
            ctx = self.ctx_proj(context)          # (B, base, H, W)
            x0 = x0 + ctx.unsqueeze(2)            # broadcast over D
        x1 = self.enc1(self.down1(x0))
        x2 = self.enc2(self.down2(x1))

        u2 = self.dec2(_match(self.up2(x2), x1) + x1)
        u1 = self.dec1(_match(self.up1(u2), x0) + x0)
        out = self.head(u1)  # (B, 1, D, H, W)
        return out.squeeze(1)  # (B, D, H, W)
