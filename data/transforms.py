"""Stereo-aware augmentation. Watch out for the gotchas from the spec:

- No naive horizontal flip (would swap L/R and require disparity convention swap).
- Any horizontal resize must scale disparity values by the same factor.
- Photometric jitter is ASYMMETRIC (different params per eye).
"""
import numpy as np
import torch


# ImageNet means/stds for ImageNet-pretrained-compatible normalization.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_u8: np.ndarray) -> np.ndarray:
    """uint8 H,W,3 RGB -> float32 H,W,3 normalized."""
    img = img_u8.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img


def to_chw_tensor(img_norm: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_norm.transpose(2, 0, 1)))


def random_crop_stereo(left, right, disp, crop_h, crop_w, rng=None):
    """Crop all three with the same (y, x). Disparity values are pixel-shifts
    so cropping doesn't change them (no scaling needed)."""
    H, W = disp.shape
    if rng is None:
        rng = np.random
    y = rng.randint(0, max(1, H - crop_h + 1))
    x = rng.randint(0, max(1, W - crop_w + 1))
    return (
        left[y:y + crop_h, x:x + crop_w],
        right[y:y + crop_h, x:x + crop_w],
        disp[y:y + crop_h, x:x + crop_w],
    )


def asymmetric_color_jitter(img_u8: np.ndarray, rng=None) -> np.ndarray:
    """Per-eye independent brightness/contrast/gamma jitter. Mild, since
    underwater color casts are part of what the model must learn to handle."""
    if rng is None:
        rng = np.random
    img = img_u8.astype(np.float32) / 255.0
    # brightness
    img = img + rng.uniform(-0.05, 0.05)
    # contrast
    img = (img - 0.5) * rng.uniform(0.9, 1.1) + 0.5
    # gamma
    img = np.clip(img, 1e-3, 1.0) ** rng.uniform(0.9, 1.1)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
