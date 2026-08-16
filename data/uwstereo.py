"""UWStereo dataset loader.

Observed layout (verified 2026-06-05):
  <root>/UWStereo/
    default/default/{disparity/*.pfm, images/{left,right}/*.png}
    coral reef/coral/{disparity/*.pfm, images/{left,right}/*.png}
    industry/industry/{disparity/*.pfm, images/{left,right}/*.png}
    ship split/ship/{disparity/*.pfm, images/{left,right}/*.png}

Image shape: 720 × 1280 RGB. Disparity in PFM, float32.
Max disparity observed in sampled subset: ~448 (worst frame), ~240 globally.
We mask out d <= 0 and d >= d_max in the loss.

Total: 29,568 pairs across 4 scenes. Split (no official split file found):
  80% train / 10% val / 10% test, seeded for reproducibility, per scene.
"""
import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .pfm import read_pfm
from .transforms import normalize, to_chw_tensor, random_crop_stereo, asymmetric_color_jitter


SCENES = [
    ("default", "default"),
    ("coral reef", "coral"),
    ("industry", "industry"),
    ("ship split", "ship"),
]


def build_index(root: str):
    """Return list of (left_path, right_path, disp_path) for each frame."""
    items = []
    for outer, inner in SCENES:
        base = Path(root) / outer / inner
        disp_dir = base / "disparity"
        left_dir = base / "images" / "left"
        right_dir = base / "images" / "right"
        if not disp_dir.is_dir():
            continue
        for d in sorted(os.listdir(disp_dir)):
            if not d.endswith(".pfm"):
                continue
            stem = d[:-4]
            l = left_dir / f"{stem}.png"
            r = right_dir / f"{stem}.png"
            dp = disp_dir / d
            if l.is_file() and r.is_file():
                items.append((str(l), str(r), str(dp)))
    return items


def split_indices(n: int, seed: int = 0):
    """Deterministic 80/10/10 split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


class UWStereoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_h: int = 256,
        crop_w: int = 512,
        d_max: int = 256,
        augment: bool = True,
    ):
        self.items = build_index(root)
        if not self.items:
            raise RuntimeError(f"no UWStereo frames found under {root}")
        train_idx, val_idx, test_idx = split_indices(len(self.items))
        if split == "train":
            self.idx = train_idx
        elif split == "val":
            self.idx = val_idx
        elif split == "test":
            self.idx = test_idx
        else:
            raise ValueError(split)
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.d_max = d_max
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.idx)

    def scene_of(self, i):
        """Return the scene label (coral/industry/ship/default) for sample i."""
        lp, _, _ = self.items[int(self.idx[i])]
        norm = lp.replace("\\", "/")
        for _outer, inner in SCENES:
            if f"/{inner}/" in norm:
                return inner
        return "unknown"

    def _load(self, i):
        lp, rp, dp = self.items[int(self.idx[i])]
        limg = cv2.imread(lp, cv2.IMREAD_COLOR)
        rimg = cv2.imread(rp, cv2.IMREAD_COLOR)
        if limg is None or rimg is None:
            raise IOError(f"unreadable PNG: {lp if limg is None else rp}")
        left = cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        disp = read_pfm(dp)
        # Clean up infs / negatives (UWStereo has some out-of-range disp values)
        disp = np.where(np.isfinite(disp), disp, 0.0).astype(np.float32)

        if self.augment:
            # Crop FIRST (720x1280 -> 256x512), then jitter the small crop.
            left, right, disp = random_crop_stereo(left, right, disp, self.crop_h, self.crop_w)
            left = asymmetric_color_jitter(left)
            right = asymmetric_color_jitter(right)

        left_t = to_chw_tensor(normalize(left))
        right_t = to_chw_tensor(normalize(right))
        disp_t = torch.from_numpy(disp.copy())
        valid = ((disp_t > 0) & (disp_t < self.d_max)).float()
        return {"left": left_t, "right": right_t, "disp": disp_t, "valid": valid}

    def __getitem__(self, i):
        # Robust to corrupt PNG/PFM: skip bad sample, substitute a random valid one.
        import random
        for attempt in range(20):
            idx = i if attempt == 0 else random.randint(0, len(self.idx) - 1)
            try:
                return self._load(idx)
            except Exception as e:
                bad = self.items[int(self.idx[idx])]
                print(f"[uwstereo] skipping bad sample {bad}: {e}", flush=True)
        raise RuntimeError("too many corrupt UWStereo samples in a row")
