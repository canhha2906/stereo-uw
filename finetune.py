"""UWStereo finetune. Loads SceneFlow-pretrained weights and continues training.

Usage:
  python finetune.py --config configs/ref.yaml \
      --uwstereo-root "C:\\Users\\canhh\\Workspace\\conference paper, computer vision\\data set\\UWStereo" \
      --pretrained runs/ref-pretrain/best.ckpt
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
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--resume", default=None,
                    help="path to last.ckpt or best.ckpt to resume from")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="epoch index to resume at (skips earlier epochs in LR schedule)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    ft = cfg_dict["finetune"]

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
        lr=ft["lr"],
        epochs=args.epochs if args.epochs is not None else ft["epochs"],
        weight_decay=ft["weight_decay"],
        warmup_iters=0,
        out_dir=args.out_dir,
        precision=args.precision,
        backbone=cfg_dict.get("backbone", "v2"),
        use_context=cfg_dict.get("use_context", False),
        context_pretrained=cfg_dict.get("context_pretrained", False),
    )

    train_set = UWStereoDataset(args.uwstereo_root, split="train",
                                crop_h=cfg.crop_h, crop_w=cfg.crop_w,
                                d_max=cfg.d_max, augment=True)
    val_set = UWStereoDataset(args.uwstereo_root, split="val",
                              crop_h=cfg.crop_h, crop_w=cfg.crop_w,
                              d_max=cfg.d_max, augment=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | train={len(train_set)} | val={len(val_set)}")
    init_ckpt = args.resume if args.resume else args.pretrained
    if init_ckpt is None:
        ap.error("must supply --resume or --pretrained")
    ckpt = train_one_run(cfg, train_set, val_set, device=device,
                         pretrained_ckpt=init_ckpt, tag="finetune",
                         start_epoch=args.start_epoch)
    print(f"best finetune ckpt: {ckpt}")


if __name__ == "__main__":
    main()
