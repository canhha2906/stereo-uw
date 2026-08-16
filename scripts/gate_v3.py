"""GATE step for the 4 V3 configs: forward + ONNX export with random init.

If any of these fail, we don't proceed to training.
"""
import sys
from pathlib import Path
import yaml
import torch

# Make repo importable when run from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import GwcNetLite

CONFIGS = ["v3_r8_3d", "v3_r8_2d", "v3_r16_3d", "v3_r16_2d"]


def main():
    out_dir = ROOT / "onnx"
    out_dir.mkdir(exist_ok=True)
    for name in CONFIGS:
        with open(ROOT / "configs" / f"{name}.yaml") as f:
            cfg = yaml.safe_load(f)
        print(f"\n=== {name} ===")
        print(f"  backbone={cfg['backbone']} agg={cfg['agg']} res={cfg['res']} "
              f"d_max={cfg['d_max']}")
        m = GwcNetLite(
            d_max=cfg["d_max"], res=cfg["res"], groups=cfg["groups"],
            feat_channels=cfg["feat_channels"], agg=cfg["agg"],
            upsample="bilinear", backbone=cfg["backbone"],
        )
        m.eval()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  params = {n_params/1e6:.3f} M")

        # Forward on a dummy that respects the stride.
        # H,W must be multiple of res*4 so the 3D hourglass downsamples cleanly.
        mult = cfg["res"] * 4
        H, W = 480, 640
        if H % mult: H = ((H + mult - 1) // mult) * mult
        if W % mult: W = ((W + mult - 1) // mult) * mult
        L = torch.randn(1, 3, H, W)
        R = torch.randn(1, 3, H, W)
        with torch.no_grad():
            disp, disp_low = m(L, R)
        print(f"  forward ok: disp={tuple(disp.shape)} disp_low={tuple(disp_low.shape)}")

        # ONNX export
        out_path = out_dir / f"{name}.onnx"

        class W(torch.nn.Module):
            def __init__(self, mod): super().__init__(); self.mod = mod
            def forward(self, l, r): return self.mod(l, r)[0]

        torch.onnx.export(
            W(m), (L, R), str(out_path),
            input_names=["left", "right"], output_names=["disp"],
            opset_version=17, dynamic_axes=None,
            do_constant_folding=True, dynamo=False,
        )
        # Numerical match check
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"left": L.numpy(), "right": R.numpy()})[0]
            err = float(np.abs(ort_out - disp.numpy()).max())
            print(f"  onnx ok: ORT vs PyTorch max abs err = {err:.2e}")
        except Exception as e:
            print(f"  ORT check failed: {e}")


if __name__ == "__main__":
    main()
