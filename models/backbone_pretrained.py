"""ImageNet-pretrained MobileNetV2 backbone (torchvision), truncated for stereo.

We take torchvision's mobilenet_v2(weights=IMAGENET1K).features and cut it at
the desired output stride, then project to `out_channels` for the cost volume.

torchvision MobileNetV2 features layout (stride / out-ch after each block):
  [0]  conv s2          1/2   32
  [1]  IR               1/2   16
  [2]  IR s2            1/4   24
  [3]  IR               1/4   24
  [4]  IR s2            1/8   32
  [5]  IR               1/8   32
  [6]  IR               1/8   32
  [7]  IR s2            1/16  64
  ...  [11..13]         1/16  96
  [14] IR s2            1/32  160
  ...

So features[:7]  → stride 1/8, 32 ch
   features[:14] → stride 1/16, 96 ch

Input must be ImageNet-normalized (our data/transforms.py already uses ImageNet
mean/std), so the pretrained stats line up.
"""
import torch
import torch.nn as nn

# (slice end index, channels at that stride)
_TRUNC = {8: (7, 32), 16: (14, 96)}


class MobileNetV2Pretrained(nn.Module):
    def __init__(self, out_channels: int = 32, output_stride: int = 8,
                 pretrained: bool = True):
        super().__init__()
        assert output_stride in _TRUNC, f"output_stride must be 8 or 16, got {output_stride}"
        cut, in_ch = _TRUNC[output_stride]

        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        full = mobilenet_v2(weights=weights)
        self.features = full.features[:cut]

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.features(x)
        return self.proj(x)
