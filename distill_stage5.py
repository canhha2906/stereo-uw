"""Stage 5: distil the physics-retrained Fast-ACVNet+ teacher into GwcNet-lite.

Teacher : Fast-ACVNet+ finetuned on physics-rendered KITTI (Stage 3), 3.203 M params.
Student : GwcNet-lite from Paper 1, 0.467 M params - 6.9x smaller.
Data    : the same physics-rendered KITTI used in Stage 3. No underwater data is
          used for training at any point.

Offline distillation: the teacher's disparity was dumped to .npy by
Fast-ACVNet/dump_teacher_disp.py. That avoids importing Fast-ACVNet here, since both
repos define a package called `models` and they collide on sys.path.

Why distillation earns its place, rather than just training on ground truth: KITTI's
disparity labels are only ~19% dense, while the teacher supplies a prediction for every
pixel (measured EPE 0.31 px against GT on the valid 19%). The student therefore gets
dense supervision it could not get from the labels alone.

Loss = w_gt * L1(student, GT) on valid pixels  +  w_kd * L1(student, teacher) everywhere
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models import GwcNetLite

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class RenderedKitti(Dataset):
    """Rendered KITTI + sparse GT + dense teacher disparity, random crops."""

    def __init__(self, root, teacher_dir, crop_h=256, crop_w=512, train=True):
        self.src = Path(root) / "training"
        self.tdir = Path(teacher_dir)
        self.names = sorted(p.name for p in (self.src / "image_2").glob("*_10.png"))
        self.ch, self.cw, self.train = crop_h, crop_w, train

    def __len__(self):
        return len(self.names)

    def _img(self, p):
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        return ((a - MEAN) / STD).transpose(2, 0, 1)

    def __getitem__(self, i):
        n = self.names[i]
        L = self._img(self.src / "image_2" / n)
        R = self._img(self.src / "image_3" / n)
        gt = np.array(Image.open(self.src / "disp_occ_0" / n), np.float32) / 256.0
        td = np.load(self.tdir / n.replace(".png", ".npy"))

        h = min(L.shape[1], gt.shape[0], td.shape[0])
        w = min(L.shape[2], gt.shape[1], td.shape[1])
        L, R, gt, td = L[:, :h, :w], R[:, :h, :w], gt[:h, :w], td[:h, :w]

        if self.train:
            y = np.random.randint(0, max(h - self.ch, 1))
            x = np.random.randint(0, max(w - self.cw, 1))
            sl = (slice(y, y + self.ch), slice(x, x + self.cw))
            L, R = L[:, sl[0], sl[1]], R[:, sl[0], sl[1]]
            gt, td = gt[sl], td[sl]
        return (torch.from_numpy(np.ascontiguousarray(L)),
                torch.from_numpy(np.ascontiguousarray(R)),
                torch.from_numpy(np.ascontiguousarray(gt)),
                torch.from_numpy(np.ascontiguousarray(td)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\canhh\Workspace\code\Fast-ACVNet\data\kitti15_uw_III")
    ap.add_argument("--teacher-disp", default=r"C:\Users\canhh\Workspace\code\Fast-ACVNet\data\kitti15_uw_III\teacher_disp")
    ap.add_argument("--out", default="runs/stage5-distill")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-max", type=int, default=192)
    ap.add_argument("--w-gt", type=float, default=1.0)
    ap.add_argument("--w-kd", type=float, default=1.0)
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = RenderedKitti(a.data, a.teacher_disp)
    dl = DataLoader(ds, a.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = GwcNetLite(d_max=a.d_max, res=8, groups=8, feat_channels=32,
                       agg="3d", upsample="bilinear",
                       backbone="v2_imagenet", use_context=False).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student GwcNet-lite: {n_par:.3f} M params, {len(ds)} training pairs", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(a.epochs * 0.7)], gamma=0.1)

    for ep in range(a.epochs):
        model.train()
        tot = tot_gt = tot_kd = 0.0
        for L, R, gt, td in dl:
            L, R, gt, td = L.cuda(), R.cuda(), gt.cuda(), td.cuda()
            disp, _ = model(L, R)
            m = (gt > 0) & (gt < a.d_max)
            l_gt = F.smooth_l1_loss(disp[m], gt[m]) if m.any() else disp.sum() * 0.0
            mt = (td > 0) & (td < a.d_max)
            l_kd = F.smooth_l1_loss(disp[mt], td[mt]) if mt.any() else disp.sum() * 0.0
            loss = a.w_gt * l_gt + a.w_kd * l_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); tot_gt += l_gt.item(); tot_kd += l_kd.item()
        sched.step()
        nb = max(len(dl), 1)
        print(f"epoch {ep+1}/{a.epochs}  loss {tot/nb:.4f}  (gt {tot_gt/nb:.4f}  kd {tot_kd/nb:.4f})"
              f"  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        torch.save({"epoch": ep, "model": model.state_dict()}, Path(a.out) / "last.ckpt")

    print("done ->", Path(a.out) / "last.ckpt")


if __name__ == "__main__":
    main()
