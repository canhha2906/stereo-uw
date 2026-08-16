"""Build the Pareto-front figure (EPE vs latency, EPE vs energy/frame).

Reads results from `figures/results.csv` with columns:
  name, agg, res, precision, epe, d1_pct, latency_ms, energy_mj, mem_mb

Plots two scatter panels with Pareto-front lines.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def pareto_front(points):
    """Return indices on the lower-left Pareto front (lower is better on both axes)."""
    pts = sorted(enumerate(points), key=lambda x: x[1])
    front = []
    best_y = float("inf")
    for i, (x, y) in pts:
        if y < best_y:
            front.append((x, y, i))
            best_y = y
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="figures/results.csv")
    ap.add_argument("--out", default="figures/pareto.png")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["name"],
                "epe": float(r["epe"]),
                "lat": float(r["latency_ms"]),
                "energy": float(r["energy_mj"]),
            })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, key, xlabel in [(axes[0], "lat", "Latency (ms)"),
                            (axes[1], "energy", "Energy / frame (mJ)")]:
        xs = [r[key] for r in rows]
        ys = [r["epe"] for r in rows]
        ax.scatter(xs, ys, s=40)
        for r in rows:
            ax.annotate(r["name"], (r[key], r["epe"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        pts = list(zip(xs, ys))
        front = pareto_front(pts)
        if len(front) >= 2:
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.plot(fx, fy, "k--", linewidth=1, alpha=0.5, label="Pareto front")
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("EPE (px)")
        ax.set_title(f"EPE vs {xlabel}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
