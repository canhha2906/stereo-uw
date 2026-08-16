"""GwcNet-lite end-to-end model.

Pipeline:
  left, right (B, 3, H, W)
    -> Siamese FeatureExtractor (shared weights) -> fL, fR at (B, C, H/s, W/s)
    -> group-wise correlation cost volume (B, G, D/s, H/s, W/s)
    -> aggregation (3D hourglass OR 2D conv stack) -> (B, D/s, H/s, W/s)
    -> soft-argmin -> disp_low (B, H/s, W/s)
    -> upsample (convex or bilinear) -> disp (B, H, W)

The forward returns BOTH the full-res disparity and the low-res disparity, so
the training loss can supervise both (the spec asks for an auxiliary loss on
the 1/8 disparity for early stability).
"""
import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .feature_extractor_v3 import FeatureExtractorV3
from .context_encoder import ContextEncoder
from .cost_volume import CostVolume
from .aggregation_3d import Aggregation3D
from .aggregation_2d import Aggregation2D
from .regression import SoftArgmin
from .upsample import build_upsample


def build_backbone(kind: str, out_channels: int, output_stride: int) -> nn.Module:
    if kind == "v2":
        if output_stride != 8:
            raise ValueError(f"v2 backbone only supports output_stride=8, got {output_stride}")
        return FeatureExtractor(out_channels=out_channels)
    if kind == "v3":
        return FeatureExtractorV3(out_channels=out_channels, output_stride=output_stride)
    if kind == "v2_imagenet":
        from .backbone_pretrained import MobileNetV2Pretrained
        return MobileNetV2Pretrained(out_channels=out_channels,
                                     output_stride=output_stride, pretrained=True)
    raise ValueError(f"unknown backbone: {kind}")


class GwcNetLite(nn.Module):
    def __init__(
        self,
        d_max: int = 256,
        res: int = 8,
        groups: int = 8,
        feat_channels: int = 32,
        agg: str = "3d",
        upsample: str = "bilinear",
        backbone: str = "v2",
        use_context: bool = False,
        context_channels: int = 32,
        context_pretrained: bool = False,
    ):
        super().__init__()
        assert d_max % res == 0, "d_max must be divisible by res"
        assert feat_channels % groups == 0, "feat_channels must be divisible by groups"

        self.d_max = d_max
        self.res = res
        self.d_at_scale = d_max // res
        self.use_context = use_context

        self.features = build_backbone(backbone, out_channels=feat_channels, output_stride=res)
        self.cost = CostVolume(max_disp_at_scale=self.d_at_scale, groups=groups)

        if use_context:
            self.context = ContextEncoder(ctx_channels=context_channels, output_stride=res,
                                          pretrained=context_pretrained)
            cc = context_channels
        else:
            self.context = None
            cc = 0

        if agg == "3d":
            self.agg = Aggregation3D(groups=groups, context_channels=cc)
        elif agg == "2d":
            self.agg = Aggregation2D(groups=groups, d_at_scale=self.d_at_scale,
                                     context_channels=cc)
        else:
            raise ValueError(f"unknown agg: {agg}")

        self.regress = SoftArgmin(d_at_scale=self.d_at_scale)
        self.upsample = build_upsample(upsample, feat_channels=feat_channels, scale=res)
        self._upsample_kind = upsample

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        fL = self.features(left)
        fR = self.features(right)
        cost = self.cost(fL, fR)              # (B, G, D/s, H/s, W/s)
        ctx = self.context(left) if self.use_context else None
        cost_agg = self.agg(cost, ctx)        # (B, D/s, H/s, W/s)
        disp_low = self.regress(cost_agg)     # (B, H/s, W/s), low-res scale
        if self._upsample_kind == "convex":
            disp = self.upsample(disp_low, fL)
        else:
            disp = self.upsample(disp_low)
        return disp, disp_low
