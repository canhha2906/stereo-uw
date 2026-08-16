"""Dump a small INT8 calibration set from UWStereo train.

TensorRT INT8 needs ~200-500 representative input samples to measure per-tensor
activation ranges. We dump preprocessed (left,right) pairs as .npy so the Orin's
calibrator can read them without needing the full dataset or the dataloader.

Run on the dev box (CPU is fine):
  python scripts/dump_calib.py \
      --uwstereo-root "...\\UWStereo" --out calib_uw --n 300 \
      --height 480 --width 640
Copy the resulting calib_uw/ folder to the Orin alongside the ONNX.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.uwstereo import build_index, split_indices
from data.transforms import normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uwstereo-root", required=True)
    ap.add_argument("--out", default="calib_uw")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    items = build_index(args.uwstereo_root)
    train_idx, _, _ = split_indices(len(items))   # calibrate on TRAIN only (never test)
    rng = np.random.default_rng(0)
    pick = rng.choice(train_idx, size=min(args.n, len(train_idx)), replace=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width

    for k, idx in enumerate(pick):
        lp, rp, _ = items[int(idx)]
        left = cv2.cvtColor(cv2.imread(lp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(rp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # center-crop/resize to the deploy resolution
        left = cv2.resize(left, (W, H))
        right = cv2.resize(right, (W, H))
        l = normalize(left).transpose(2, 0, 1)[None].astype(np.float32)   # (1,3,H,W)
        r = normalize(right).transpose(2, 0, 1)[None].astype(np.float32)
        np.save(out / f"{k:04d}_left.npy", l)
        np.save(out / f"{k:04d}_right.npy", r)

    print(f"dumped {len(pick)} calibration pairs to {out}/ at {H}x{W}")


if __name__ == "__main__":
    main()
