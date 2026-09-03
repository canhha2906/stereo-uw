"""Minimal PFM reader (SceneFlow/UWStereo disparity format). Self-contained
copy so `distill/` doesn't depend on the deleted `data/` package."""
import re

import numpy as np


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        color = header == "PF"
        if header not in ("PF", "Pf"):
            raise ValueError(f"not a PFM file: {path}")
        dims = f.readline().decode("ascii").rstrip()
        while dims.startswith("#"):
            dims = f.readline().decode("ascii").rstrip()
        w, h = map(int, re.split(r"\s+", dims))
        scale = float(f.readline().decode("ascii").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), endian + "f")
        if color:
            data = data.reshape(h, w, 3)
        else:
            data = data.reshape(h, w)
        return np.flipud(data).astype(np.float32)
