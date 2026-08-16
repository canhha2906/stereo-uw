"""Evaluate a checkpoint on the UWStereo held-out test split.

Reports overall EPE + 3px/D1, AND a per-scene breakdown (coral / industry /
ship / default) — the "does the lightweight net degrade gracefully on
low-texture scenes, or collapse?" finding.

Usage:
  python evaluate.py --config configs/ref.yaml --ckpt runs/ref-finetune/best.ckpt \
      --uwstereo-root "...\\UWStereo" [--split test] [--csv results.csv]
"""
import argparse
import csv
import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from engine import TrainCfg, build_model, run_eval
from data import UWStereoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--uwstereo-root", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--csv", default=None, help="append results row to this CSV")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainCfg(
        name=cfg_dict["name"], agg=cfg_dict["agg"], res=cfg_dict["res"],
        d_max=cfg_dict["d_max"], groups=cfg_dict["groups"],
        feat_channels=cfg_dict["feat_channels"],
        crop_h=cfg_dict["crop_h"], crop_w=cfg_dict["crop_w"],
        batch_size=1, num_workers=0, amp=False,
        lr=0, epochs=0, weight_decay=0,
        backbone=cfg_dict.get("backbone", "v2"),
        use_context=cfg_dict.get("use_context", False),
        context_pretrained=cfg_dict.get("context_pretrained", False),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    sd = sd.get("model", sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    test_set = UWStereoDataset(args.uwstereo_root, split=args.split,
                               d_max=cfg.d_max, augment=False)

    # ---- overall ----
    loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    epe, d1 = run_eval(model, loader, device, return_d1=True)
    print(f"\n=== {cfg.name} | split={args.split} | N={len(test_set)} ===")
    print(f"OVERALL  EPE={epe:.4f}  D1={d1*100:.2f}%")

    # ---- per-scene breakdown ----
    scene_idx = {}
    for i in range(len(test_set)):
        scene_idx.setdefault(test_set.scene_of(i), []).append(i)
    per_scene = {}
    print("per-scene:")
    for scene in sorted(scene_idx):
        sub = Subset(test_set, scene_idx[scene])
        sl = DataLoader(sub, batch_size=1, num_workers=0, shuffle=False)
        s_epe, s_d1 = run_eval(model, sl, device, return_d1=True)
        per_scene[scene] = (s_epe, s_d1)
        print(f"  {scene:10s} N={len(sub):5d}  EPE={s_epe:.4f}  D1={s_d1*100:.2f}%")

    # ---- optional CSV row (for the master results table) ----
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["config", "backbone", "agg", "context", "split",
                            "N", "epe", "d1_pct",
                            "epe_coral", "epe_industry", "epe_ship", "epe_default"])
            w.writerow([cfg.name, cfg.backbone, cfg.agg, cfg.use_context, args.split,
                        len(test_set), f"{epe:.4f}", f"{d1*100:.2f}",
                        f"{per_scene.get('coral',(float('nan'),))[0]:.4f}",
                        f"{per_scene.get('industry',(float('nan'),))[0]:.4f}",
                        f"{per_scene.get('ship',(float('nan'),))[0]:.4f}",
                        f"{per_scene.get('default',(float('nan'),))[0]:.4f}"])
        print(f"appended row to {args.csv}")


if __name__ == "__main__":
    main()
