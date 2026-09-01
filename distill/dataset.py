"""Generic 'land' RGB image folder -- the distillation source domain.

No labels needed: the frozen teacher (see teacher.py) supplies pseudo
ground-truth depth on these clean images. Point --clean-images-root at
any sizeable, diverse photo collection (COCO/ImageNet unlabeled subset,
stock photos, even SceneFlow's left frames) -- diversity of scene depth
structure matters more than the images being underwater-relevant, since
the underwater look is added synthetically (see underwater_physics.py).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class CleanImageFolder(Dataset):
    def __init__(self, root: str, crop_h: int = 384, crop_w: int = 384):
        self.paths = [str(p) for p in Path(root).rglob("*")
                      if p.suffix.lower() in _IMG_EXTS]
        if not self.paths:
            raise RuntimeError(f"no images found under {root}")
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.crop_w, self.crop_h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)
