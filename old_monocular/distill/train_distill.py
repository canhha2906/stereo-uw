"""Distillation entry point: land-domain teacher -> physics-synthesized
underwater image -> student. The student never sees a real underwater
depth label; it only ever matches the teacher's pseudo-depth on the
*clean* image while looking at the *synthesized-underwater* image. That
mismatch (clean supervision signal, underwater-looking input) is what
teaches the student to be attenuation/backscatter-invariant.

Usage:
  # ablation "none": distill directly on clean images (no physics) -- this
  # is the "land model dropped underwater, works badly" baseline.
  python -m distill.train_distill --config configs/distill_uw.yaml ^
      --clean-images-root D:\\clean_images --ablation none

  # ablation "physics": the actual proposal.
  python -m distill.train_distill --config configs/distill_uw.yaml ^
      --clean-images-root D:\\clean_images --ablation physics
"""
import argparse
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from distill.dataset import CleanImageFolder
from distill.losses import scale_shift_invariant_l1
from distill.student import MonoDepthStudent
from distill.teacher import build_teacher
from distill.underwater_physics import PhysicsParams, synthesize_underwater


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clean-images-root", required=True)
    ap.add_argument("--out-dir", default="runs_distill")
    ap.add_argument("--ablation", choices=["none", "physics"], default="physics",
                    help="'none' skips the underwater simulator (baseline); "
                         "'physics' is the proposed method")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | ablation={args.ablation}")

    teacher = build_teacher(cfg["teacher"]).to(device).eval()
    student = MonoDepthStudent(feat_channels=32,
                               output_stride=cfg["student_out_stride"]).to(device)

    ds = CleanImageFolder(args.clean_images_root,
                          crop_h=cfg["input_h"], crop_w=cfg["input_w"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"],
                        num_workers=cfg.get("num_workers", 0), shuffle=True)
    print(f"clean images: {len(ds)}")

    opt = optim.AdamW(student.parameters(), lr=cfg["lr"],
                      weight_decay=cfg["weight_decay"])
    physics = PhysicsParams(**cfg["physics"]) if args.ablation == "physics" else None

    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    out_dir = Path(args.out_dir) / f"{cfg['name']}-{args.ablation}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        student.train()
        running = 0.0
        for i, clean in enumerate(loader):
            clean = clean.to(device)
            with torch.no_grad():
                depth_teacher = teacher(clean)

            student_input = (synthesize_underwater(clean, depth_teacher, physics)
                             if physics is not None else clean)

            depth_student = student(student_input)
            loss = scale_shift_invariant_l1(depth_student, depth_teacher)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

            if i % 50 == 0:
                print(f"ep{epoch} it{i}/{len(loader)} loss={loss.item():.4f}")

        print(f"epoch {epoch} mean loss = {running / len(loader):.4f}")
        torch.save({"model": student.state_dict()}, out_dir / "last.ckpt")

    print(f"done: {out_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
