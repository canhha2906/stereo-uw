"""PSMNet (Chang & Chen 2018, CVPR) wrapped for UWStereo evaluation.

PSMNet is the heavy ceiling baseline per spec §5. We use the authors'
SceneFlow-pretrained checkpoint, run inference on the UWStereo test split,
report EPE/D1, and (later) export to TRT for Orin benchmarking.

The PSMNet repo lives outside this repo:
  Clone: git clone https://github.com/JiaRenChang/PSMNet "D:\\PSMNet\\repo"
  Weight: download `pretrained_sceneflow_new.tar` to D:\\PSMNet\\weights\\
          (link in PSMNet README, ~150 MB)

We add D:\\PSMNet\\repo to sys.path at runtime so we can import their model
without copying files into this repo.

Usage:
  python -m baselines.psmnet_wrap \
      --psmnet-root  "D:\\PSMNet\\repo" \
      --psmnet-ckpt  "D:\\PSMNet\\weights\\pretrained_sceneflow_new.tar" \
      --uwstereo-root "C:\\Users\\canhh\\Workspace\\conference paper, computer vision\\data set\\UWStereo" \
      --split test \
      --max-disp 192
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_psmnet(psmnet_root: str, ckpt_path: str, max_disp: int, device: str):
    """Import PSMNet model code from the cloned repo and load weights."""
    sys.path.insert(0, str(Path(psmnet_root).resolve()))
    # PSMNet repo provides models.stackhourglass.PSMNet(maxdisp)
    from models import stackhourglass  # type: ignore

    model = stackhourglass.PSMNet(max_disp)
    model = torch.nn.DataParallel(model).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # PSMNet ckpts are dicts with 'state_dict' key
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def pad_to_multiple(t: torch.Tensor, mult: int = 32):
    """Pad H,W up to multiples of `mult`. Returns padded tensor and (top, left)."""
    *_, H, W = t.shape
    ph = (mult - H % mult) % mult
    pw = (mult - W % mult) % mult
    # PSMNet pads top-left so right/bottom seam doesn't break disparity
    return F.pad(t, (pw, 0, ph, 0)), ph, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psmnet-root", required=True,
                    help="path to cloned PSMNet repo (must contain models/stackhourglass.py)")
    ap.add_argument("--psmnet-ckpt", required=True,
                    help="path to pretrained_sceneflow_new.tar or pretrained_model_KITTI*.tar")
    ap.add_argument("--uwstereo-root", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-disp", type=int, default=192,
                    help="PSMNet's max disp (192 is the SceneFlow default)")
    ap.add_argument("--d-max-eval", type=int, default=256,
                    help="UWStereo valid-range cap (matches our model configs)")
    args = ap.parse_args()

    # Defer this import until after we've parsed args, so the script is importable
    # even when the local repo's own dependencies aren't on path yet.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data import UWStereoDataset
    from engine import metric_epe, metric_d1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_psmnet(args.psmnet_root, args.psmnet_ckpt, args.max_disp, device)
    print(f"PSMNet loaded on {device} | max_disp={args.max_disp}")

    ds = UWStereoDataset(args.uwstereo_root, split=args.split,
                         d_max=args.d_max_eval, augment=False)
    loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)

    epe_sum, d1_sum, n = 0.0, 0.0, 0
    t_sum = 0.0
    for batch in tqdm(loader):
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        disp_gt = batch["disp"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        l_pad, ph, pw = pad_to_multiple(left, mult=32)
        r_pad, _, _ = pad_to_multiple(right, mult=32)

        with torch.no_grad():
            t0 = time.perf_counter()
            disp = model(l_pad, r_pad)
            torch.cuda.synchronize() if device == "cuda" else None
            t_sum += time.perf_counter() - t0
        # PSMNet stackhourglass returns 1 tensor in eval mode (3 in train).
        # crop back to original H,W
        if disp.dim() == 4:
            disp = disp.squeeze(1)
        disp = disp[..., ph:, pw:]

        # PSMNet caps disparities at max_disp; UWStereo has disp > 192 in
        # some frames. We mask GT > max_disp out of the metric so PSMNet
        # isn't unfairly penalised for out-of-range pixels.
        m_valid = valid * (disp_gt < args.max_disp).float()
        epe = metric_epe(disp, disp_gt, m_valid).item()
        d1 = metric_d1(disp, disp_gt, m_valid).item()
        epe_sum += epe
        d1_sum += d1
        n += 1

    print(f"PSMNet on UWStereo[{args.split}] | N={n}")
    print(f"  EPE = {epe_sum/max(1,n):.4f} px  (D_max eval={args.max_disp})")
    print(f"  D1  = {d1_sum/max(1,n)*100:.2f}%")
    print(f"  latency = {t_sum/max(1,n)*1000:.1f} ms/frame on {device}")


if __name__ == "__main__":
    main()
