"""OpenCV SGBM baseline: accuracy/speed floor for the paper.

This runs on CPU and uses block-matching with semi-global aggregation.
Parameters tuned for 720x1280 underwater stereo. Tune if val EPE is poor.

Usage:
  python -m baselines.sgbm --uwstereo-root <path> --split test --d-max 256
"""
import argparse
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import UWStereoDataset


def make_sgbm(d_max: int):
    # numDisparities must be a multiple of 16
    num_disp = ((d_max + 15) // 16) * 16
    block_size = 5
    P1 = 8 * 3 * block_size * block_size
    P2 = 32 * 3 * block_size * block_size
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def unnormalize(t):
    """Undo ImageNet normalization back to uint8 for SGBM input."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = t.numpy().transpose(1, 2, 0) * std + mean
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uwstereo-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--d-max", type=int, default=256)
    args = ap.parse_args()

    ds = UWStereoDataset(args.uwstereo_root, split=args.split, d_max=args.d_max, augment=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    sgbm = make_sgbm(args.d_max)

    epe_sum = 0.0
    d1_sum = 0.0
    n = 0
    t0 = time.perf_counter()
    for batch in loader:
        left = unnormalize(batch["left"][0])
        right = unnormalize(batch["right"][0])
        disp_gt = batch["disp"][0].numpy()
        valid = batch["valid"][0].numpy() > 0.5

        l_g = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        r_g = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        disp_raw = sgbm.compute(l_g, r_g).astype(np.float32) / 16.0  # SGBM disparities are int16 fixed-point /16

        err = np.abs(disp_raw - disp_gt)
        valid_pred = (disp_raw > 0) & valid
        if valid_pred.sum() == 0:
            continue
        e = err[valid_pred]
        g = np.abs(disp_gt[valid_pred])
        epe = float(e.mean())
        d1 = float(((e > 3) & (e > 0.05 * g)).mean())
        epe_sum += epe
        d1_sum += d1
        n += 1
    elapsed = time.perf_counter() - t0
    print(f"SGBM | N={n} EPE={epe_sum/max(1,n):.4f} D1={d1_sum/max(1,n)*100:.2f}% "
          f"time/frame={elapsed/max(1,n)*1000:.1f} ms")


if __name__ == "__main__":
    main()
