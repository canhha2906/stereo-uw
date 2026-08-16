"""SceneFlow pretraining entry point.

Usage:
  python train.py --config configs/ref.yaml --sceneflow-root D:\\SCENEFLOW

Dataset auto-discovery: the loader walks <root>/{flying things,driving,monka}
and picks up whatever has both frames + disparity extracted.
"""
import argparse
import yaml
import torch

from engine import TrainCfg, train_one_run
from data import SceneFlowDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sceneflow-root", required=True,
                    help="root containing 'flying things', 'driving', 'monka' subdirs")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config epochs (use 1 for a smoke run)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto picks BF16 on Ada/Hopper/Blackwell; fp16 forces FP16+GradScaler")
    ap.add_argument("--resume", default=None,
                    help="path to last.ckpt to resume an interrupted pretrain")
    ap.add_argument("--start-epoch", type=int, default=0)
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
        backbone=cfg_dict.get("backbone", "v2"),
        use_context=cfg_dict.get("use_context", False),
        context_pretrained=cfg_dict.get("context_pretrained", False),
    )

    train_set = SceneFlowDataset(
        scene_flow_root=args.sceneflow_root,
        split="train",
        crop_h=cfg.crop_h, crop_w=cfg.crop_w,
        d_max=cfg.d_max, augment=True,
    )
    val_set = SceneFlowDataset(
        scene_flow_root=args.sceneflow_root,
        split="test",
        crop_h=cfg.crop_h, crop_w=cfg.crop_w,
        d_max=cfg.d_max, augment=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | train={len(train_set)} | val={len(val_set)} | agg={cfg.agg}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | BF16: {torch.cuda.is_bf16_supported()}")
    ckpt = train_one_run(cfg, train_set, val_set, device=device, tag="pretrain",
                         pretrained_ckpt=args.resume, start_epoch=args.start_epoch)
    print(f"best pretrain ckpt: {ckpt}")


if __name__ == "__main__":
    main()
