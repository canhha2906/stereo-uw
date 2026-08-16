"""GATE one config by name: forward + ONNX export (random/pretrained init)."""
import sys
from pathlib import Path
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import GwcNetLite


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "agg2d_ctx"
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"=== {name} ===")
    m = GwcNetLite(
        d_max=cfg["d_max"], res=cfg["res"], groups=cfg["groups"],
        feat_channels=cfg["feat_channels"], agg=cfg["agg"], upsample="bilinear",
        backbone=cfg.get("backbone", "v2"),
        use_context=cfg.get("use_context", False),
        context_channels=cfg.get("context_channels", 32),
        context_pretrained=cfg.get("context_pretrained", False),
    )
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"  backbone={cfg.get('backbone','v2')} agg={cfg['agg']} "
          f"use_context={cfg.get('use_context',False)} "
          f"context_pretrained={cfg.get('context_pretrained',False)}")
    print(f"  params = {n/1e6:.3f} M")

    L = torch.randn(1, 3, 480, 640)
    R = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        disp, disp_low = m(L, R)
    print(f"  forward ok: disp={tuple(disp.shape)} disp_low={tuple(disp_low.shape)}")

    out_path = ROOT / "onnx" / f"{name}.onnx"
    out_path.parent.mkdir(exist_ok=True)

    class Wr(torch.nn.Module):
        def __init__(s, mod): super().__init__(); s.mod = mod
        def forward(s, l, r): return s.mod(l, r)[0]

    torch.onnx.export(
        Wr(m), (L, R), str(out_path),
        input_names=["left", "right"], output_names=["disp"],
        opset_version=17, dynamic_axes=None, do_constant_folding=True, dynamo=False,
    )
    try:
        import onnxruntime as ort, numpy as np
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"left": L.numpy(), "right": R.numpy()})[0]
        err = float(np.abs(o - disp.numpy()).max())
        print(f"  onnx ok: max abs err = {err:.2e}")
    except Exception as e:
        print(f"  ORT check failed: {e}")


if __name__ == "__main__":
    main()
