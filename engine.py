"""Shared training/eval engine. Used by train.py, finetune.py, evaluate.py."""
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import GwcNetLite


def masked_smooth_l1(pred, target, valid):
    """Smooth-L1 over valid pixels only. `valid` is a 0/1 float mask."""
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    denom = valid.sum().clamp(min=1.0)
    return (loss * valid).sum() / denom


def metric_epe(pred, target, valid):
    err = (pred - target).abs()
    denom = valid.sum().clamp(min=1.0)
    return (err * valid).sum() / denom


def metric_d1(pred, target, valid, thresh_px=3.0, thresh_rel=0.05):
    """KITTI-style D1: pixel is bad if |err| > 3px AND |err|/|gt| > 5%."""
    err = (pred - target).abs()
    bad = ((err > thresh_px) & (err > thresh_rel * target.abs())).float()
    denom = valid.sum().clamp(min=1.0)
    return (bad * valid).sum() / denom


@dataclass
class TrainCfg:
    name: str
    agg: str
    res: int
    d_max: int
    groups: int
    feat_channels: int
    crop_h: int
    crop_w: int
    batch_size: int
    num_workers: int
    amp: bool
    lr: float
    epochs: int
    weight_decay: float
    warmup_iters: int = 0
    aux_weight: float = 0.3
    log_every: int = 50
    val_every: int = 1
    out_dir: str = "runs"
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    backbone: str = "v2"     # "v2" or "v3"
    use_context: bool = False  # left-image context branch into aggregation
    context_pretrained: bool = False  # ImageNet-pretrained context encoder


def build_model(cfg: TrainCfg, upsample="bilinear") -> GwcNetLite:
    return GwcNetLite(
        d_max=cfg.d_max,
        res=cfg.res,
        groups=cfg.groups,
        feat_channels=cfg.feat_channels,
        agg=cfg.agg,
        upsample=upsample,
        backbone=cfg.backbone,
        use_context=cfg.use_context,
        context_pretrained=cfg.context_pretrained,
    )


def warmup_lr(opt, base_lr, step, warmup_iters):
    lr = base_lr * min(1.0, (step + 1) / max(1, warmup_iters))
    for g in opt.param_groups:
        g["lr"] = lr


def train_one_run(
    cfg: TrainCfg,
    train_set,
    val_set,
    device: str = "cuda",
    pretrained_ckpt: str | None = None,
    tag: str = "pretrain",
    start_epoch: int = 0,
):
    out_dir = Path(cfg.out_dir) / f"{cfg.name}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)

    model = build_model(cfg).to(device)
    if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
        sd = torch.load(pretrained_ckpt, map_location=device)
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    # Fast-forward LR schedule if resuming from a partial run.
    for _ in range(start_epoch):
        sched.step()
    if start_epoch > 0:
        print(f"  RESUME: starting at epoch {start_epoch}, LR = {opt.param_groups[0]['lr']:.2e}")
    # Precision selection.
    #  auto: BF16 if GPU supports it (Ada/Hopper/Blackwell), else FP16.
    #  bf16: force BF16 (no GradScaler needed).
    #  fp16: force FP16 (uses GradScaler for stability).
    #  fp32: no autocast.
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    prec = cfg.precision
    if prec == "auto":
        prec = "bf16" if bf16_ok else "fp16"
    if prec == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = cfg.amp and device == "cuda"
        use_scaler = False
    elif prec == "fp16":
        amp_dtype = torch.float16
        use_amp = cfg.amp and device == "cuda"
        use_scaler = use_amp
    elif prec == "fp32":
        amp_dtype = torch.float32
        use_amp = False
        use_scaler = False
    else:
        raise ValueError(f"unknown precision: {cfg.precision}")
    print(f"  precision={prec} | autocast={use_amp} | grad_scaler={use_scaler}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # pin_memory locks host RAM (can't be paged) — a liability when RAM is tight,
    # so we disable it. num_workers comes from config (set low/0 on low-RAM boxes).
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=False,
        persistent_workers=False,
    )

    global_step = 0
    best_epe = float("inf")
    # If resuming, evaluate the current checkpoint to seed best_epe so we don't
    # accidentally overwrite the saved best with a worse epoch.
    if start_epoch > 0:
        seed_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True)
        print(f"  RESUME: evaluating loaded checkpoint on val set ({len(val_set)} items)...")
        best_epe = run_eval(model, seed_loader, device)
        print(f"  RESUME: existing model val EPE = {best_epe:.4f} (kept as floor for best.ckpt)")
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"ep{epoch}"):
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            disp_gt = batch["disp"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            if global_step < cfg.warmup_iters:
                warmup_lr(opt, cfg.lr, global_step, cfg.warmup_iters)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                disp, disp_low = model(left, right)
                loss_full = masked_smooth_l1(disp, disp_gt, valid)
                # auxiliary loss on the low-res disparity for stability
                disp_gt_low = F.avg_pool2d(disp_gt.unsqueeze(1), cfg.res).squeeze(1) / cfg.res
                valid_low = F.avg_pool2d(valid.unsqueeze(1), cfg.res).squeeze(1)
                valid_low = (valid_low > 0.5).float()
                loss_aux = masked_smooth_l1(disp_low, disp_gt_low, valid_low)
                loss = loss_full + cfg.aux_weight * loss_aux

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                # BF16 autocast or full FP32 — no scaler needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if global_step % cfg.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_full", loss_full.item(), global_step)
                writer.add_scalar("train/loss_aux", loss_aux.item(), global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
            global_step += 1

        sched.step()

        if (epoch + 1) % cfg.val_every == 0:
            val_epe = run_eval(model, val_loader, device)
            writer.add_scalar("val/epe", val_epe, epoch)
            print(f"epoch {epoch} val EPE = {val_epe:.4f}  (best so far {best_epe:.4f})")
            if val_epe < best_epe:
                best_epe = val_epe
                torch.save({"model": model.state_dict(), "epoch": epoch, "epe": val_epe},
                           out_dir / "best.ckpt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last.ckpt")
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s")

    writer.close()
    return str(out_dir / "best.ckpt")


@torch.no_grad()
def run_eval(model, loader, device, return_d1=False):
    model.eval()
    total_epe = 0.0
    total_d1 = 0.0
    n = 0
    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        disp_gt = batch["disp"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        # Pad to multiple of 8 (1/8 stride)
        H, W = disp_gt.shape[-2:]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        left = F.pad(left, (0, pad_w, 0, pad_h))
        right = F.pad(right, (0, pad_w, 0, pad_h))
        disp, _ = model(left, right)
        if pad_h or pad_w:
            disp = disp[..., :H, :W]
        total_epe += metric_epe(disp, disp_gt, valid).item()
        total_d1 += metric_d1(disp, disp_gt, valid).item()
        n += 1
    if return_d1:
        return total_epe / max(1, n), total_d1 / max(1, n)
    return total_epe / max(1, n)
