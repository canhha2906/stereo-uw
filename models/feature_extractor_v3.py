"""MobileNetV3-style Siamese feature extractor.

V3 adds two things over V2 (Howard et al. 2019):
  1. Squeeze-and-Excitation blocks (channel-wise attention).
  2. h-swish activation: x * relu6(x+3)/6  (smoother than ReLU, similar cost).

Both are TRT-supported on JetPack 6 / TRT 10. Older JetPack may need fallback.

This module exits at either 1/8 OR 1/16 stride, selected at construction time.
Channel counts trimmed (vs ImageNet V3-Small) to keep params low for stereo.

Output channels = 32 (must be divisible by groups=8 for the cost volume).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- activations --------------------------------------------------------------

class HSwish(nn.Module):
    """h-swish: x * ReLU6(x+3)/6.  Differentiable smooth alternative to swish."""

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class HSigmoid(nn.Module):
    """h-sigmoid: ReLU6(x+3)/6, used inside SE blocks."""

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


# ---- squeeze-and-excitation ---------------------------------------------------

class SEBlock(nn.Module):
    """Per-channel attention via global avg pool + 2 FC + h-sigmoid + scale.

    Reduction = 4 by default. TRT-compatible (all primitives are first-class).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gate = HSigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


# ---- V3 inverted residual bneck ----------------------------------------------

class BNeck(nn.Module):
    """MobileNetV3 inverted residual block.

    in -> 1x1 expand (+nl) -> kxk depthwise (+nl) -> [SE?] -> 1x1 project (linear)
    """

    def __init__(self, in_c, exp_c, out_c, kernel, stride, use_se: bool, nl: str):
        super().__init__()
        assert kernel in (3, 5)
        nl_layer = HSwish() if nl == "hs" else nn.ReLU(inplace=True)

        layers = []
        if exp_c != in_c:
            layers += [
                nn.Conv2d(in_c, exp_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp_c),
                nl_layer,
            ]
        # depthwise
        pad = kernel // 2
        layers += [
            nn.Conv2d(exp_c, exp_c, kernel, stride, pad, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            HSwish() if nl == "hs" else nn.ReLU(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock(exp_c))
        # project (linear)
        layers += [
            nn.Conv2d(exp_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.body = nn.Sequential(*layers)
        self.use_res = (stride == 1 and in_c == out_c)

    def forward(self, x):
        out = self.body(x)
        return x + out if self.use_res else out


# ---- the backbone -------------------------------------------------------------

class FeatureExtractorV3(nn.Module):
    """V3-Small-inspired backbone, trimmed for stereo. Outputs at stride 1/8 or 1/16.

    Args:
        out_channels: 32 (must be divisible by groups=8)
        output_stride: 8 or 16
    """

    def __init__(self, out_channels: int = 32, output_stride: int = 8):
        super().__init__()
        assert output_stride in (8, 16), output_stride

        # Stem: 3x3 stride 2, h-swish, 16 ch  → 1/2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HSwish(),
        )

        # 1/2 → 1/4: bneck k=3, exp=16, out=16, SE=Y, ReLU, stride=2
        # 1/4 → 1/8: bneck k=3, exp=72, out=24, SE=N, ReLU, stride=2
        #            bneck k=3, exp=88, out=24, SE=N, ReLU, stride=1
        self.blocks_to_8 = nn.Sequential(
            BNeck(16, 16, 16, kernel=3, stride=2, use_se=True,  nl="re"),  # 1/4
            BNeck(16, 72, 24, kernel=3, stride=2, use_se=False, nl="re"),  # 1/8
            BNeck(24, 88, 24, kernel=3, stride=1, use_se=False, nl="re"),
        )

        # 1/8 → 1/16: bneck k=5, exp=96, out=40, SE=Y, h-swish, stride=2
        #             bneck k=5, exp=240, out=40, SE=Y, h-swish, stride=1
        if output_stride == 16:
            self.blocks_to_16 = nn.Sequential(
                BNeck(24, 96,  40, kernel=5, stride=2, use_se=True, nl="hs"),  # 1/16
                BNeck(40, 240, 40, kernel=5, stride=1, use_se=True, nl="hs"),
            )
            project_in = 40
        else:
            self.blocks_to_16 = None
            project_in = 24

        # Project to out_channels at the chosen stride
        self.proj = nn.Sequential(
            nn.Conv2d(project_in, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_to_8(x)
        if self.blocks_to_16 is not None:
            x = self.blocks_to_16(x)
        return self.proj(x)
