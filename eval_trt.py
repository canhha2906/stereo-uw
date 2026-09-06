"""Stage 6c: does quantisation break disparity regression?

Runs a TensorRT engine on the UWStereo test split and reports EPE / D1, so FP32, FP16
and INT8 can be compared under one protocol.

Resolution note: the engines are built at a fixed 480x640, while UWStereo is 720x1280.
Images are resized to 640 wide, so **disparity values are scaled by 640/1280 = 0.5**.
Absolute EPE here is therefore in 640-wide pixels and is NOT comparable to the
native-resolution numbers elsewhere in this repo. What is comparable is FP32 vs FP16 vs
INT8 within this script, which is the question being asked.
"""
import argparse, os, re, time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

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


def split_test(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[int(0.8 * n) + int(0.1 * n):]


ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--uwstereo-root", required=True)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--maxdisp", type=float, default=192.0)
a = ap.parse_args()

logger = trt.Logger(trt.Logger.ERROR)
engine = trt.Runtime(logger).deserialize_cuda_engine(Path(a.engine).read_bytes())
ctx = engine.create_execution_context()
names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
ins = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

bufs = {}
for n in names:
    shape = tuple(ctx.get_tensor_shape(n))
    bufs[n] = torch.empty(int(np.prod(shape)), dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
out_shape = tuple(ctx.get_tensor_shape(outs[0]))

items = build_index(a.uwstereo_root)
test = [items[i] for i in split_test(len(items))][: a.limit]
scale = a.width / 1280.0
stream = torch.cuda.Stream()

epes, d1s, times = [], [], []
for k, (lp, rp, dp) in enumerate(test, 1):
    l = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
    r = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
    l = ((cv2.resize(l, (a.width, a.height)).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    r = ((cv2.resize(r, (a.width, a.height)).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    bufs[ins[0]].copy_(torch.from_numpy(np.ascontiguousarray(l)).ravel().cuda())
    bufs[ins[1]].copy_(torch.from_numpy(np.ascontiguousarray(r)).ravel().cuda())

    torch.cuda.synchronize(); t0 = time.perf_counter()
    ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

    pred = bufs[outs[0]].view(out_shape).squeeze().float()
    gt = read_pfm(dp).astype(np.float32)
    gt = cv2.resize(gt, (a.width, a.height), interpolation=cv2.INTER_NEAREST) * scale
    gt = torch.from_numpy(np.ascontiguousarray(gt)).cuda()

    m = (gt > 0) & (gt < a.maxdisp * scale) & torch.isfinite(gt)
    if m.sum() == 0:
        continue
    e = (pred[m] - gt[m]).abs()
    epes.append(e.mean().item())
    d1s.append((((e > 3 * scale) & (e > 0.05 * gt[m])).float().mean() * 100).item())
    if k % 100 == 0:
        print(f"  {k}/{len(test)} running EPE {np.mean(epes):.4f}", flush=True)

print(f"\nengine  : {a.engine}")
print(f"N       : {len(epes)}  @ {a.height}x{a.width} (disparity scaled x{scale:g})")
print(f"EPE     : {np.mean(epes):.4f} px")
print(f"D1-all  : {np.mean(d1s):.4f} %")
print(f"latency : {np.median(times):.2f} ms median")
