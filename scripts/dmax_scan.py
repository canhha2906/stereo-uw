"""Full D_max scan over the UWStereo training split.

Reads every PFM in the train subset and prints the global max disparity.
Use the result to set d_max in the configs.
"""
import argparse
import numpy as np
from tqdm import tqdm

from data.pfm import read_pfm
from data.uwstereo import build_index, split_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uwstereo-root", required=True)
    args = ap.parse_args()

    items = build_index(args.uwstereo_root)
    train_idx, _, _ = split_indices(len(items))

    gmax = 0.0
    gmin = float("inf")
    n_truncated_at_256 = 0
    n_truncated_at_192 = 0
    n_total = 0
    n_frames = len(train_idx)

    for i in tqdm(train_idx, total=n_frames):
        _, _, dp = items[int(i)]
        d = read_pfm(dp)
        v = d[np.isfinite(d)]
        v = v[v > 0]
        if v.size == 0:
            continue
        m = float(v.max())
        gmax = max(gmax, m)
        gmin = min(gmin, float(v.min()))
        n_total += v.size
        n_truncated_at_192 += int((v > 192).sum())
        n_truncated_at_256 += int((v > 256).sum())

    print(f"frames scanned: {n_frames}")
    print(f"global max disparity: {gmax:.2f}")
    print(f"global min disparity: {gmin:.2f}")
    print(f"fraction of valid pixels exceeding D_max=192: {n_truncated_at_192 / max(1, n_total):.4f}")
    print(f"fraction of valid pixels exceeding D_max=256: {n_truncated_at_256 / max(1, n_total):.4f}")

    # Recommend D_max as next multiple of 8 above the global max
    rec = int(np.ceil(gmax / 8.0) * 8)
    print(f"recommended D_max = {rec} (next multiple of 8 above observed max)")


if __name__ == "__main__":
    main()
