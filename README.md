# stereo-uw

Lightweight learned stereo for underwater depth on the Jetson Orin Nano.
SceneFlow pretrain → UWStereo finetune → TensorRT deployment → accuracy / latency / energy characterization.

This is the codebase for a 6-page conference paper. See the original brief
in `instruction/CLAUDE.md` (one directory up from the data set folder).

## Scope (unranked-venue trim, 2026-06-05)
- 2 checkpoints: `--agg 3d` and `--agg 2d`, both at feature stride 1/8.
- Precision sweep: FP32 + FP16 (INT8 deferred to future work).
- Baseline: OpenCV SGBM only.
- Hardware: Jetson Orin Nano (8 GB), 25 W "Super" mode, `jetson_clocks` pinned.

## What gates everything
**Do not train anything until `export_tensorrt.py` produces working `.engine` files
for BOTH `--agg 3d` and `--agg 2d` on the target Orin.** Group-wise correlation and
`grid_sample` (convex upsample) are the known TensorRT failure points.

## Quickstart

```powershell
# Windows / dev box
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# GATE step (run on Orin after copying gwc_3d.onnx and gwc_2d.onnx)
python export_tensorrt.py --agg 3d --out onnx/gwc_3d.onnx
python export_tensorrt.py --agg 2d --out onnx/gwc_2d.onnx
# then on the Orin:
# trtexec --onnx=gwc_3d.onnx --saveEngine=gwc_3d.engine --fp16

# Train (after GATE passes)
python train.py     --config configs/ref.yaml
python finetune.py  --config configs/ref.yaml --pretrained runs/ref/last.ckpt
python evaluate.py  --config configs/ref.yaml --ckpt runs/ref-ft/best.ckpt

# Benchmark on Orin
python benchmark.py --engine gwc_3d.engine --precision fp16
```

## Dataset locations (Windows dev box)
- SceneFlow: `C:\Users\canhh\Workspace\conference paper, computer vision\data set\` (flyingthings + monka + driving — partial; disparity GT pending download)
- UWStereo:  `C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo\` (29,568 pairs, 720×1280, D_max≈240–448 observed)
- FLSea:     `...\data set\FLSea-challenge\` (real underwater — qualitative-only, no quantitative training/eval)

## Honest disclaimers (for the paper)
- Synthetic-domain study. No real-water ground truth. Sim-to-real adaptation is future work.
- The contribution is **characterization** — not the network. GwcNet-lite is the measurement vehicle.
