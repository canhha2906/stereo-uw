"""Context encoder (left image only).

Unlike the feature extractor (Siamese, shared between L/R, used to BUILD the cost
volume), the context encoder sees ONLY the left image and produces features that
are INJECTED into the aggregation stage. This is the RAFT idea adapted to a
single-shot (non-iterative) architecture: the network gets reference-image global
context to disambiguate low-texture regions (open water, sand) where pure
correlation has nothing to match.

Separate weights from the feature extractor — that's the point. The feature
extractor learns matching-friendly features; the context encoder is free to learn
scene-structure / semantic features instead.

Output: (B, ctx_channels, H/stride, W/stride), stride = output_stride (8 or 16).
"""
import torch
import torch.nn as nn

from .feature_extractor import InvertedResidual, conv_bn_relu


class ContextEncoder(nn.Module):
    def __init__(self, ctx_channels: int = 32, output_stride: int = 8,
                 pretrained: bool = False):
        super().__init__()
        assert output_stride in (8, 16), output_stride

        # ImageNet-pretrained variant: reuse the torchvision MobileNetV2 trunk
        # (separate weights from the matching feature extractor — that's the point
        # of a context branch).
        if pretrained:
            from .backbone_pretrained import MobileNetV2Pretrained
            self._pre = MobileNetV2Pretrained(out_channels=ctx_channels,
                                              output_stride=output_stride,
                                              pretrained=True)
            self.stem = self.s2 = self.s3 = self.s4 = self.proj = None
            return
        self._pre = None

        self.stem = conv_bn_relu(3, 16, k=3, s=2)          # 1/2
        self.s2 = nn.Sequential(                            # 1/4
            InvertedResidual(16, 24, stride=2),
            InvertedResidual(24, 24, stride=1),
        )
        self.s3 = nn.Sequential(                            # 1/8
            InvertedResidual(24, 32, stride=2),
            InvertedResidual(32, 32, stride=1),
        )
        if output_stride == 16:
            self.s4 = nn.Sequential(                        # 1/16
                InvertedResidual(32, 32, stride=2),
                InvertedResidual(32, 32, stride=1),
            )
        else:
            self.s4 = None
        self.proj = nn.Sequential(
            nn.Conv2d(32, ctx_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ctx_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, left: torch.Tensor) -> torch.Tensor:
        if self._pre is not None:
            return self._pre(left)
        x = self.stem(left)
        x = self.s2(x)
        x = self.s3(x)
        if self.s4 is not None:
            x = self.s4(x)
        return self.proj(x)
