"""UWStereo-from-scratch training (skips SceneFlow pretrain).

SceneFlow disparity GT was not on disk as of 2026-06-05, so we cannot
pretrain. UWStereo alone has 29,568 synthetic underwater pairs which is
plenty for the GwcNet-lite scale. Frame this honestly in the paper.

Usage:
  python train_uw.py --config configs/ref.yaml \
      --uwstereo-root "C:\\Users\\canhh\\Workspace\\conference paper, computer vision\\data set\\UWStereo"

This script runs `pretrain` LR schedule (warmup + cosine) directly on
UWStereo. There is no separate finetune stage.
"""
import argparse
import yaml
import torch

from engine import TrainCfg, train_one_run
from data import UWStereoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--uwstereo-root", required=True)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config epochs (use a small number for a smoke run)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    pre = cfg_dict["pretrain"]

    cfg = TrainCfg(
        name=cfg_dict["name"],
        agg=cfg_dict["agg"],
        res=cfg_dict["res"],
        d_max=cfg_dict["d_max"],
        groups=cfg_dict["groups"],
        feat_channels=cfg_dict["feat_channels"],
        crop_h=cfg_dict["crop_h"],
        crop_w=cfg_dict["crop_w"],
        batch_size=cfg_dict["batch_size"],
        num_workers=cfg_dict.get("num_workers", 4),
        amp=cfg_dict.get("amp", True),
        lr=pre["lr"],
        epochs=args.epochs if args.epochs is not None else pre["epochs"],
        weight_decay=pre["weight_decay"],
        warmup_iters=pre.get("warmup_iters", 0),
        out_dir=args.out_dir,
        precision=args.precision,
    )

    train_set = UWStereoDataset(args.uwstereo_root, split="train",
                                crop_h=cfg.crop_h, crop_w=cfg.crop_w,
                                d_max=cfg.d_max, augment=True)
    val_set = UWStereoDataset(args.uwstereo_root, split="val",
                              crop_h=cfg.crop_h, crop_w=cfg.crop_w,
                              d_max=cfg.d_max, augment=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | agg={cfg.agg} | bs={cfg.batch_size} | "
          f"train={len(train_set)} | val={len(val_set)}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | "
              f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    ckpt = train_one_run(cfg, train_set, val_set, device=device, tag="uw")
    print(f"best ckpt: {ckpt}")


if __name__ == "__main__":
    main()
