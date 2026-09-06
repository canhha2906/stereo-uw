"""Few-shot adaptation: how much do a handful of real underwater labels buy?

The main result stays zero-underwater-label (turbid4, EPE 4.2777). This adds a curve
on top of it: starting from that checkpoint, fine-tune on N labelled UWStereo pairs and
re-evaluate.

METHODOLOGY - the thing that must not be got wrong:
the N training pairs are drawn from UWStereo's TRAIN split only. The 2,958-pair TEST
split is the same deterministic seed-0 split used by every other number in this repo and
is never touched. N = 0 is the unmodified turbid4 model, so the curve starts exactly at
the zero-label result.
"""
import argparse, os, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.Fast_ACV_plus import Fast_ACVNet_plus
from models.loss import model_loss_train

ROOT = r"C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo"
SCENES = [("default", "default"), ("coral reef", "coral"),
          ("industry", "industry"), ("ship split", "ship")]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_pfm(path):
    with open(path, "rb") as f:
        if f.readline().rstrip() not in (b"PF", b"Pf"):
            raise ValueError("not PFM")
        while True:
            line = f.readline().decode("latin-1")
            if not line.startswith("#"):
                break
        w, h = map(int, re.match(r"^(\d+)\s+(\d+)\s*$", line).groups())
        scale = float(f.readline().rstrip())
        return np.flipud(np.fromfile(f, "<f" if scale < 0 else ">f").reshape(h, w))


def build_index(root):
    items = []
    for outer, inner in SCENES:
        base = Path(root) / outer / inner
        dd, ld, rd = base / "disparity", base / "images" / "left", base / "images" / "right"
        if not dd.is_dir():
            continue
        for d in sorted(os.listdir(dd)):
            if d.endswith(".pfm"):
                s = d[:-4]
                if (ld / f"{s}.png").is_file():
                    items.append((str(ld / f"{s}.png"), str(rd / f"{s}.png"), str(dd / d)))
    return items


def split_indices(n, seed=0):
    """The one split used everywhere in this repo. Do not change."""
    perm = np.random.default_rng(seed).permutation(n)
    ntr, nva = int(0.8 * n), int(0.1 * n)
    return perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


class FewShotSet(Dataset):
    def __init__(self, items, crop_h=256, crop_w=512, maxdisp=192):
        self.items, self.ch, self.cw, self.maxdisp = items, crop_h, crop_w, maxdisp

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lp, rp, dp = self.items[i]
        L, R = load_img(lp), load_img(rp)
        g = np.ascontiguousarray(read_pfm(dp)).astype(np.float32)
        h, w = g.shape
        y = np.random.randint(0, max(h - self.ch, 1))
        x = np.random.randint(0, max(w - self.cw, 1))
        L = L[:, y:y + self.ch, x:x + self.cw]
        R = R[:, y:y + self.ch, x:x + self.cw]
        g = g[y:y + self.ch, x:x + self.cw]
        import cv2
        g_low = cv2.resize(g, (self.cw // 4, self.ch // 4), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(np.ascontiguousarray(L)),
                torch.from_numpy(np.ascontiguousarray(R)),
                torch.from_numpy(np.ascontiguousarray(g)),
                torch.from_numpy(np.ascontiguousarray(g_low)))


@torch.no_grad()
def evaluate(model, test, maxdisp=192, limit=0):
    if limit:
        test = test[:limit]
    epes, d1s, p3 = [], [], []
    for lp, rp, dp in test:
        L = torch.from_numpy(load_img(lp)).unsqueeze(0).cuda()
        R = torch.from_numpy(load_img(rp)).unsqueeze(0).cuda()
        g = torch.from_numpy(np.ascontiguousarray(read_pfm(dp))).float().cuda()
        ph = (32 - L.shape[2] % 32) % 32
        if ph:
            L = F.pad(L, (0, 0, ph, 0)); R = F.pad(R, (0, 0, ph, 0))
        o = model(L, R)
        p = (o[0] if isinstance(o, (list, tuple)) else o).squeeze(0)
        if ph:
            p = p[ph:, :]
        m = (g > 0) & (g < maxdisp) & torch.isfinite(g)
        if m.sum() == 0:
            continue
        e = (p[m] - g[m]).abs()
        epes.append(e.mean().item())
        p3.append((e > 3).float().mean().item() * 100)
        d1s.append((((e > 3) & (e > 0.05 * g[m])).float().mean() * 100).item())
    return float(np.mean(epes)), float(np.mean(p3)), float(np.mean(d1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./checkpoints/exp_turbid4/checkpoint_000059.ckpt")
    ap.add_argument("--n", type=int, required=True, help="labelled UWStereo pairs to use")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = build_index(ROOT)
    tr_idx, _, te_idx = split_indices(len(items))
    test = [items[i] for i in te_idx]
    train_pool = [items[i] for i in tr_idx]
    print(f"UWStereo: {len(items)} pairs | train pool {len(train_pool)} | test {len(test)}")

    model = Fast_ACVNet_plus(192, False).cuda()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})

    if a.n == 0:
        model.eval()
        epe, p3, d1 = evaluate(model, test, limit=a.eval_limit)
        print(f"\nN=0 (zero underwater labels)  EPE {epe:.4f}  >3px {p3:.4f}%  D1 {d1:.4f}%")
        return

    rng = np.random.default_rng(a.seed)
    shots = [train_pool[i] for i in rng.choice(len(train_pool), a.n, replace=False)]
    print(f"few-shot: {a.n} pairs drawn from the TRAIN split (test split untouched)")

    dl = DataLoader(FewShotSet(shots), batch_size=min(a.batch_size, a.n),
                    shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999))

    model.train()
    for ep in range(a.epochs):
        tot = 0.0
        for L, R, g, g_low in dl:
            L, R, g, g_low = L.cuda(), R.cuda(), g.cuda(), g_low.cuda()
            disp_ests = model(L, R)
            # the model returns four outputs at two scales; mirror main_kitti.py exactly
            mask = (g > 0) & (g < 192)
            mask_low = (g_low > 0) & (g_low < 192)
            if mask.sum() == 0:
                continue
            loss = model_loss_train(disp_ests,
                                    [g, g_low, g, g_low],
                                    [mask, mask_low, mask, mask_low])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{a.epochs}  loss {tot/max(len(dl),1):.4f}", flush=True)

    model.eval()
    epe, p3, d1 = evaluate(model, test, limit=a.eval_limit)
    print(f"\nN={a.n}  EPE {epe:.4f}  >3px {p3:.4f}%  D1 {d1:.4f}%")


if __name__ == "__main__":
    main()
