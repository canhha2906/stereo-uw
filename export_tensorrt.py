"""GATE-step entry point: export model to ONNX.

The actual TensorRT engine build runs on the Orin via `trtexec` — we cannot
build TRT engines from a Windows dev box. Workflow:

  Dev box (Windows):
    python export_tensorrt.py --agg 3d --out onnx/gwc_3d.onnx
    python export_tensorrt.py --agg 2d --out onnx/gwc_2d.onnx
    # copy onnx/ to the Orin

  Orin Nano (after `sudo nvpmodel -m 2 && sudo jetson_clocks`):
    trtexec --onnx=gwc_3d.onnx --saveEngine=gwc_3d.fp16.engine --fp16 \
            --memPoolSize=workspace:1024
    trtexec --onnx=gwc_2d.onnx --saveEngine=gwc_2d.fp16.engine --fp16 \
            --memPoolSize=workspace:1024

If a `.ckpt` is provided, the model is loaded with trained weights. For the
GATE step, leave --ckpt empty so the random-init model gets exported.
"""
import argparse
import os
from pathlib import Path

import torch
import yaml

from models import GwcNetLite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ref.yaml")
    ap.add_argument("--agg", choices=["3d", "2d"], required=True)
    ap.add_argument("--out", required=True, help="output ONNX path")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--upsample", choices=["bilinear", "convex"], default="bilinear",
                    help="bilinear is the TRT-safe choice; use convex only after GATE verifies it")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = GwcNetLite(
        d_max=cfg["d_max"],
        res=cfg["res"],
        groups=cfg["groups"],
        feat_channels=cfg["feat_channels"],
        agg=args.agg,
        upsample=args.upsample,
        backbone=cfg.get("backbone", "v2"),
        use_context=cfg.get("use_context", False),
        context_pretrained=cfg.get("context_pretrained", False),
    )
    model.eval()

    if args.ckpt and os.path.isfile(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("model", sd)
        model.load_state_dict(sd, strict=True)
        print(f"loaded weights from {args.ckpt}")
    else:
        print("exporting RANDOM-INIT model (GATE step: schema-only validation)")

    H, W = args.height, args.width
    assert H % cfg["res"] == 0 and W % cfg["res"] == 0, \
        f"H,W must be multiples of res={cfg['res']}"
    left = torch.randn(1, 3, H, W)
    right = torch.randn(1, 3, H, W)

    # Smoke-test forward on CPU before exporting.
    with torch.no_grad():
        disp, disp_low = model(left, right)
    print(f"forward ok: disp={tuple(disp.shape)} disp_low={tuple(disp_low.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class ExportWrapper(torch.nn.Module):
        """Wrap to export only the final disparity (drop the aux output)."""
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, left, right):
            disp, _ = self.m(left, right)
            return disp

    # Use the legacy TorchScript-based exporter (dynamo=False). The new dynamo
    # exporter (default in torch>=2.5) produces opsets that some TRT versions
    # don't yet handle, and emits unicode banners that crash cp1252 consoles.
    torch.onnx.export(
        ExportWrapper(model),
        (left, right),
        str(out_path),
        input_names=["left", "right"],
        output_names=["disp"],
        opset_version=args.opset,
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported ONNX: {out_path}")

    # Lightweight checker — won't catch all TRT issues but catches the obvious ones.
    try:
        import onnx
        m = onnx.load(str(out_path))
        onnx.checker.check_model(m)
        print("onnx.checker passed")
    except ImportError:
        print("(install `onnx` for schema check)")

    # Optional numerical match check against onnxruntime.
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"left": left.numpy(), "right": right.numpy()})[0]
        max_abs_err = float(np.abs(ort_out - disp.numpy()).max())
        print(f"onnxruntime vs PyTorch max abs err: {max_abs_err:.4e}")
    except ImportError:
        print("(install `onnxruntime` for numerical match check)")


if __name__ == "__main__":
    main()
