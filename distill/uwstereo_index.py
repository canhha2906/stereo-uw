"""Minimal UWStereo file-index builder, for zero-shot eval only -- this
package never trains on UWStereo, it only evaluates against its disparity
GT as an out-of-distribution check. Self-contained copy so `distill/`
doesn't depend on the deleted `data/` package.

Observed layout:
  <root>/UWStereo/
    default/default/{disparity/*.pfm, images/{left,right}/*.png}
    coral reef/coral/{disparity/*.pfm, images/{left,right}/*.png}
    industry/industry/{disparity/*.pfm, images/{left,right}/*.png}
    ship split/ship/{disparity/*.pfm, images/{left,right}/*.png}
"""
import os
from pathlib import Path

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
