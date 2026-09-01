"""Zero-shot evaluation: the student never trained on real or synthetic
underwater depth labels, so every eval here is out-of-distribution by
construction -- that's the point of the "zero shot learning" framing.

- UWStereo (quantitative): has real stereo disparity GT we can convert to
  relative depth (depth ~ 1/disparity, up to the unknown baseline*focal
  scale) -- good enough for the scale-shift-invariant metric below, even
  though the student was never shown this dataset during training.
- FLSea (qualitative only): real underwater photos, no depth GT at all --
  dumps side-by-side RGB/prediction images for the paper figures.

Run this once for the --ablation none checkpoint and once for --ablation
physics to get the "before vs after adding the physics model" comparison
the note describes.

Usage:
  python -m distill.evaluate_zeroshot --ckpt runs_distill/distill_uw-physics/last.ckpt ^
      --uwstereo-root "...\\UWStereo" --flsea-root "...\\FLSea-challenge"
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.pfm import read_pfm
from data.uwstereo import build_index
from distill.losses import _lstsq_scale_shift
from distill.student import MonoDepthStudent


def load_student(ckpt_path, device):
    m = MonoDepthStudent().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(sd.get("model", sd))
    m.eval()
    return m


def to_tensor_rgb(path: str, size: int = 384) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def eval_uwstereo(student, root: str, device, n_samples: int = 200) -> float:
    items = build_index(root)[:n_samples]
    abs_rels = []
    for left_path, _, disp_path in items:
        disp = read_pfm(disp_path)
        h, w = disp.shape
        rgb = to_tensor_rgb(left_path, size=384).to(device)
        pred = student(rgb)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)

        valid = disp > 1.0
        gt_depth = np.zeros_like(disp)
        gt_depth[valid] = 1.0 / disp[valid]
        gt_t = torch.from_numpy(gt_depth).float().to(device).view(1, 1, h, w)
        mask_t = torch.from_numpy(valid.astype(np.float32)).to(device).view(1, 1, h, w)

        s, t = _lstsq_scale_shift(pred, gt_t, mask_t)
        pred_aligned = pred * s.view(-1, 1, 1, 1) + t.view(-1, 1, 1, 1)

        err = (pred_aligned - gt_t).abs()
        rel = (err / gt_t.clamp(min=1e-3)) * mask_t
        abs_rel = (rel.sum() / mask_t.sum().clamp(min=1.0)).item()
        abs_rels.append(abs_rel)
    return float(np.mean(abs_rels))


@torch.no_grad()
def dump_flsea_qualitative(student, root: str, out_dir: str, device, n_samples: int = 20):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths = ([str(p) for p in Path(root).rglob("*.png")] +
            [str(p) for p in Path(root).rglob("*.jpg")])[:n_samples]
    for i, p in enumerate(paths):
        rgb = to_tensor_rgb(p, size=384).to(device)
        pred = student(rgb)[0, 0].cpu().numpy()
        pred_vis = (255 * (pred - pred.min()) / (np.ptp(pred) + 1e-6)).astype(np.uint8)
        rgb_vis = (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_path / f"{i:03d}_rgb.png"), cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_path / f"{i:03d}_depth.png"), pred_vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--uwstereo-root", default=None)
    ap.add_argument("--flsea-root", default=None)
    ap.add_argument("--out-dir", default="eval_distill_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = load_student(args.ckpt, device)

    if args.uwstereo_root:
        abs_rel = eval_uwstereo(student, args.uwstereo_root, device)
        print(f"UWStereo zero-shot AbsRel (scale-shift aligned) = {abs_rel:.4f}")

    if args.flsea_root:
        dump_flsea_qualitative(student, args.flsea_root, args.out_dir, device)
        print(f"FLSea qualitative dumps written to {args.out_dir}")


if __name__ == "__main__":
    main()
