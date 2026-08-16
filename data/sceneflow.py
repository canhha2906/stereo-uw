"""SceneFlow dataset loader.

Verified layouts on disk 2026-06-06:

  D:\\SCENEFLOW\\flying things\\
    frames_cleanpass\\{TRAIN,TEST}\\{A,B,C}\\<sceneid>\\{left,right}\\*.png
    disparity\\{TRAIN,TEST}\\{A,B,C}\\<sceneid>\\{left,right}\\*.pfm

  D:\\SCENEFLOW\\driving\\
    frames_cleanpass\\{15mm,35mm}_focallength\\scene_*\\{fast,slow}\\{left,right}\\*.png
    disparity\\{15mm,35mm}_focallength\\scene_*\\{fast,slow}\\{left,right}\\*.pfm

  D:\\SCENEFLOW\\monka\\
    frames_cleanpass\\<scene>\\{left,right}\\*.png       (only frames so far)
    disparity\\<scene>\\{left,right}\\*.pfm              (download in progress)

Key invariant: for every `<x>/frames_cleanpass/<rel>/left/<stem>.png` there is a
matching `<x>/disparity/<rel>/left/<stem>.pfm`. We exploit that with a single
string substitution instead of separate frames_root / disp_root paths.

Splits:
- FT3D ships its own TRAIN/TEST subtrees → respect them.
- Driving and Monkaa have no official split → put them all in TRAIN.
"""
import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .pfm import read_pfm
from .transforms import normalize, to_chw_tensor, random_crop_stereo, asymmetric_color_jitter


# Subset specs: (frames_dir_relative_to_root, split_root_inside_frames_dir, label)
# `split_root_inside_frames_dir` is "" if the subset has no TRAIN/TEST subdivision.
# Monkaa dropped 2026-06-09 (corrupt disparity download). Re-add the line below
# if a clean Monkaa is ever extracted: ("monka/frames_cleanpass", "monkaa").
SUBSETS_TRAIN = [
    ("flying things/frames_cleanpass/TRAIN", "ft3d_train"),
    ("driving/frames_cleanpass",             "driving"),
]
SUBSETS_TEST = [
    ("flying things/frames_cleanpass/TEST",  "ft3d_test"),
]


def _frames_to_disp(frames_path: Path) -> Path:
    """Map .../frames_cleanpass/... → .../disparity/..."""
    parts = list(frames_path.parts)
    swapped = False
    for i, p in enumerate(parts):
        if p == "frames_cleanpass":
            parts[i] = "disparity"
            swapped = True
            break
    if not swapped:
        raise ValueError(f"path missing 'frames_cleanpass': {frames_path}")
    return Path(*parts)


def _index_subset(scene_flow_root: Path, frames_rel: str):
    """Walk frames root; pair every left PNG with same-stem PFM under disparity tree."""
    frames_root = scene_flow_root / frames_rel
    if not frames_root.is_dir():
        return []
    items = []
    for left_dir in frames_root.rglob("left"):
        if not left_dir.is_dir():
            continue
        right_dir = left_dir.parent / "right"
        if not right_dir.is_dir():
            continue
        disp_dir = _frames_to_disp(left_dir)
        if not disp_dir.is_dir():
            continue
        for png in os.listdir(left_dir):
            if not png.lower().endswith(".png"):
                continue
            stem = os.path.splitext(png)[0]
            lp = left_dir / png
            rp = right_dir / png
            dp = disp_dir / f"{stem}.pfm"
            if rp.is_file() and dp.is_file():
                items.append((str(lp), str(rp), str(dp)))
    return items


def build_index(scene_flow_root: str, split: str):
    """Return list of (left_png, right_png, disp_pfm) for the requested split.

    `split='train'` includes FT3D TRAIN + Driving + Monkaa (if extracted).
    `split='test'`  uses FT3D TEST only — that's the conventional held-out.
    """
    root = Path(scene_flow_root)
    subsets = SUBSETS_TRAIN if split == "train" else SUBSETS_TEST
    all_items = []
    counts = {}
    for rel, label in subsets:
        # Retry a few times for TRANSIENT drive blips, but ABORT (don't silently
        # train on partial data) if a subset persistently fails to index.
        sub = None
        for attempt in range(3):
            try:
                sub = _index_subset(root, rel)
                break
            except OSError as e:
                print(f"[sceneflow] I/O error indexing {rel} (attempt {attempt+1}/3): {e}", flush=True)
                import time as _t; _t.sleep(2)
        if sub is None:
            raise RuntimeError(
                f"FATAL: could not index subset '{rel}' after retries. The drive/filesystem "
                f"is likely corrupted (run chkdsk). REFUSING to train on partial data."
            )
        counts[label] = len(sub)
        all_items.extend(sub)
    return all_items, counts


class SceneFlowDataset(Dataset):
    """Combined FT3D + Driving (+ Monkaa) loader."""

    def __init__(
        self,
        scene_flow_root: str,
        split: str = "train",
        crop_h: int = 256,
        crop_w: int = 512,
        d_max: int = 256,
        augment: bool = True,
        verbose: bool = True,
        holdout_frac: float = 0.03,
    ):
        # If FT3D TEST is present, use it as the val/test split (original behavior).
        # If it's absent (e.g. we only copied a TRAIN subset to the fast drive),
        # deterministically carve `holdout_frac` of TRAIN as val so we don't need
        # to copy the 54 GB FT3D TEST tree.
        test_items, _ = build_index(scene_flow_root, "test")
        if test_items:
            self.items, counts = build_index(scene_flow_root, split)
        else:
            train_all, counts = build_index(scene_flow_root, "train")
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(train_all))
            n_val = max(1, int(holdout_frac * len(train_all)))
            val_set = set(perm[:n_val].tolist())
            if split == "train":
                self.items = [train_all[i] for i in range(len(train_all)) if i not in val_set]
                counts = {"train_holdout": len(self.items)}
            else:
                self.items = [train_all[i] for i in perm[:n_val]]
                counts = {"val_from_train": len(self.items)}

        if not self.items:
            raise RuntimeError(
                f"no SceneFlow pairs found under {scene_flow_root} for split={split}.\n"
                f"  Per-subset counts: {counts}\n"
                f"  Expected layout: <root>/{{flying things,driving}}/{{frames_cleanpass,disparity}}/..."
            )
        if verbose:
            print(f"SceneFlow {split}: {len(self.items)} pairs " + str(counts))
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.d_max = d_max
        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        lp, rp, dp = self.items[i]
        limg = cv2.imread(lp, cv2.IMREAD_COLOR)
        rimg = cv2.imread(rp, cv2.IMREAD_COLOR)
        if limg is None or rimg is None:
            raise IOError(f"unreadable PNG: {lp if limg is None else rp}")
        left = cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        disp = read_pfm(dp)
        disp = np.where(np.isfinite(disp), disp, 0.0).astype(np.float32)

        if self.augment:
            # Crop FIRST so the photometric jitter (and its float32 copies) runs
            # on the small crop, not the full image — big RAM + speed saving.
            left, right, disp = random_crop_stereo(left, right, disp, self.crop_h, self.crop_w)
            left = asymmetric_color_jitter(left)
            right = asymmetric_color_jitter(right)

        left_t = to_chw_tensor(normalize(left))
        right_t = to_chw_tensor(normalize(right))
        disp_t = torch.from_numpy(disp.copy())
        valid = ((disp_t > 0) & (disp_t < self.d_max)).float()
        return {"left": left_t, "right": right_t, "disp": disp_t, "valid": valid}

    def __getitem__(self, i):
        # Robust to the occasional corrupt PNG/PFM: skip a bad sample and
        # substitute a random valid one rather than crashing a multi-day run.
        import random
        for attempt in range(20):
            idx = i if attempt == 0 else random.randint(0, len(self.items) - 1)
            try:
                return self._load(idx)
            except Exception as e:
                print(f"[sceneflow] skipping bad sample {self.items[idx]}: {e}", flush=True)
        raise RuntimeError("too many corrupt SceneFlow samples in a row")
