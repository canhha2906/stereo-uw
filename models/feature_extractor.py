"""Siamese MobileNetV2-style feature extractor down to stride 1/8, 32 channels.

Why this shape: matches GwcNet-lite spec. 1/8 keeps cost-volume memory tractable
on the Orin Nano (8 GB). MobileNetV2 inverted residuals trade FLOPs for
parameters, which is what we want for edge deployment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_c, out_c, k=3, s=1, p=None, groups=1):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class InvertedResidual(nn.Module):
    """MobileNetV2 block: 1x1 expand → 3x3 depthwise → 1x1 project (linear)."""

    def __init__(self, in_c, out_c, stride, expand=4):
        super().__init__()
        hid = in_c * expand
        self.use_res = stride == 1 and in_c == out_c
        layers = []
        if expand != 1:
            layers += [
                nn.Conv2d(in_c, hid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out


class FeatureExtractor(nn.Module):
    """Outputs feature map at stride 1/8 with `out_channels` channels.

    Weights are SHARED between left and right (Siamese) by reusing this module.
    """

    def __init__(self, out_channels=32):
        super().__init__()
        # stem: 1/2
        self.stem = conv_bn_relu(3, 16, k=3, s=2)
        # stage 1: 1/2 -> 1/2
        self.s1 = InvertedResidual(16, 16, stride=1, expand=1)
        # stage 2: 1/2 -> 1/4
        self.s2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2),
            InvertedResidual(24, 24, stride=1),
        )
        # stage 3: 1/4 -> 1/8
        self.s3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2),
            InvertedResidual(32, 32, stride=1),
            InvertedResidual(32, 32, stride=1),
        )
        # project to out_channels (still at 1/8)
        self.proj = nn.Conv2d(32, out_channels, 1, 1, 0, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.proj_bn(self.proj(x))
        return x
