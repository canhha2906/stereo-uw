# stereo-uw

Underwater stereo depth. Two papers share this repo.

**Current direction (Paper 2):** take a stereo network pretrained on land
(Fast-ACVNet, KITTI), measure how badly it fails underwater, then retrain it on
KITTI **rendered through a physical underwater light-attenuation model** and measure
how much that recovers — without ever training on underwater data. Then distill, then
quantize.

Read **`CLAUDE.md`** for the plan and **`TRAINING_AND_RESULTS.md`** for the commands
and results tables.

> Status: nothing has been run yet. Fast-ACVNet is not cloned, KITTI is not
> downloaded, and every results table is empty on purpose.

---

## Layout

| Path | What it is |
|---|---|
| `CLAUDE.md` | Paper 2 plan. Sections 0–5 are the senior's plan; section 6 is everything added on top, kept separate on purpose. |
| `TRAINING_AND_RESULTS.md` | Per-stage commands, prerequisites, empty results tables. |
| `models/` | **Paper 1.** GwcNet-lite: feature extractor, group-wise correlation cost volume, 2D and 3D aggregation, context encoder, soft-argmin, upsample. |
| `data/` | SceneFlow + UWStereo loaders, PFM reader, stereo-safe augmentation. |
| `engine.py`, `train.py`, `finetune.py`, `evaluate.py` | Paper 1 training and evaluation. |
| `configs/` | Paper 1's 2×2 ablation: 2D/3D aggregation × context on/off. |
| `runs/` | 16 trained checkpoints + TensorBoard logs for those 4 cells. |
| `baselines/` | SGBM (floor) and PSMNet (ceiling). |
| `export_tensorrt.py`, `build_int8.py`, `benchmark.py` | Jetson Orin Nano deployment: ONNX export, INT8 engine, latency/power/energy. Reused by Paper 2 at Stage 6. |
| `scripts/` | GATE checks, D_max scan, INT8 calibration dump, smoke tests. |
| `old_monocular/` | **Superseded.** A monocular distillation direction explored on 2026-09-01. Kept for reference, not deleted. |

## Environments

- Paper 1 code runs in the conda env `stereo` (recent PyTorch, Blackwell GPU).
- Fast-ACVNet pins PyTorch 1.10 / CUDA 11.3 and needs its own environment. Whether
  that runs on the RTX 5060 is untested — check before relying on it.

## Paper 1 (previous direction, complete)

Lightweight learned stereo characterized on the Jetson Orin Nano: accuracy, latency
and energy across 2D vs 3D aggregation and a precision sweep. Its spec lives at
`conference paper, computer vision/instruction/CLAUDE.md`. Its code and checkpoints
here stay untouched.

## Branches

- `main` — both papers, nothing deleted
- `monocular-distill` — the 2026-09-01 monocular lineage, preserved as it was
