# CLAUDE.md

This file gives Claude Code the context it needs to work in this repo.

## Active direction (2026-09-01): distillation + physics-based underwater adaptation

The paper pivoted. The primary experiment is now **not** the stereo
GwcNet-lite pipeline below — it's a monocular-depth distillation approach:
a frozen pretrained monocular depth model (teacher) supplies pseudo-depth
on ordinary "land" photos, a physics-based underwater light-transmission
simulator turns those photos underwater-looking, and a small student is
trained to recover the teacher's clean-image depth from the
synthesized-underwater input — so it works zero-shot on real/synthetic
underwater images without ever training on underwater depth labels.

See **`PLAN_DISTILLATION.md`** for the full design, code map, setup steps,
and results-table template. New code lives under `distill/` and
`configs/distill_uw.yaml`.

The stereo/GwcNet-lite code described in the rest of this file (`models/`,
`data/`, `train.py`, `finetune.py`, `evaluate.py`, `export_tensorrt.py`,
`benchmark.py`, etc.) is kept as-is for reference and as a possible
baseline/comparison point — it has not been deleted, but it is no longer
the paper's main line of work. Read `PLAN_DISTILLATION.md` first; treat
everything below as legacy context for the older direction.

## What this is (legacy stereo direction)

A research codebase for a 6-page conference paper: lightweight learned
stereo depth estimation for **underwater imagery**, targeting real-time
inference on a **Jetson Orin Nano**. The paper's contribution is
**characterization** (accuracy/latency/energy tradeoffs), not a novel
network — GwcNet-lite is the measurement vehicle, not the point.

Pipeline: SceneFlow pretrain → UWStereo finetune → ONNX export → TensorRT
engine build on-device → benchmark (latency/power/energy) on the Orin.

Paper scope was trimmed 2026-06-05 to: 2 checkpoints (`--agg 3d` and
`--agg 2d`), FP32+FP16 only (INT8 deferred), OpenCV SGBM as the only
baseline, single target device (Orin Nano, 25 W mode, clocks pinned).

## The one rule that gates everything

**Do not invest in training runs until `export_tensorrt.py` produces a
working ONNX graph that TensorRT can actually build into an `.engine` on
the target Orin, for both `--agg 3d` and `--agg 2d`.** Group-wise
correlation (the cost volume) and `grid_sample`-based convex upsampling are
the known TensorRT failure points on this stack. If asked to change the
model architecture, cost volume, or upsampling path, flag that it needs to
be re-validated through the ONNX/TensorRT export gate before it's trusted
for training.

Practical implication: `export_tensorrt.py --upsample bilinear` is the
TRT-safe default. Only use `--upsample convex` after confirming it survives
the gate on the real Orin.

## Architecture

`models/gwcnet_lite.py` (`GwcNetLite`) wires together:

```
left, right (B,3,H,W)
  -> Siamese FeatureExtractor (shared weights)     -> fL, fR  (B,C,H/s,W/s)
  -> group-wise correlation cost volume            -> (B,G,D/s,H/s,W/s)
  -> aggregation: Aggregation3D (hourglass) OR Aggregation2D (conv stack)
  -> soft-argmin regression                        -> disp_low (B,H/s,W/s)
  -> upsample (bilinear, TRT-safe | convex, untested on TRT) -> disp (B,H,W)
```

Forward returns `(disp, disp_low)` — full-res and 1/8-res disparity — so
training can supervise both (auxiliary loss on the low-res head for early
stability).

Key modules, all under `models/`:
- `feature_extractor.py` — from-scratch backbone (`backbone: v2`)
- `backbone_pretrained.py` — ImageNet-pretrained MobileNetV2 backbone
  (`backbone: v2_imagenet`, output_stride must be 8)
- `context_encoder.py` — optional guidance branch (`use_context: true`),
  feeds extra channels into aggregation
- `cost_volume.py` — group-wise correlation
- `aggregation_3d.py` / `aggregation_2d.py` — the `--agg 3d` / `--agg 2d` paths
- `regression.py` — soft-argmin disparity regression
- `upsample.py` — bilinear vs. convex (`grid_sample`) upsampling

Model variants are driven entirely by `configs/*.yaml`:
| config | agg | context |
|---|---|---|
| `ref.yaml` | 3d | off |
| `ref_ctx.yaml` | 3d | on |
| `agg2d.yaml` | 2d | off |
| `agg2d_ctx.yaml` | 2d | on |

## Data

- **SceneFlow** (`data/sceneflow.py`): synthetic pretrain set (FlyingThings3D
  + Driving + Monkaa). As of 2026-06-05, disparity GT for SceneFlow was not
  fully on disk, so `train_uw.py` exists as a fallback that trains directly
  on UWStereo (no pretrain stage) — check which path is actually in use
  before assuming a pretrain checkpoint exists.
- **UWStereo** (`data/uwstereo.py`): 29,568 synthetic underwater stereo
  pairs, 720×1280, across 4 scenes (`default`, `coral reef`, `industry`,
  `ship split`). Disparity in PFM (`data/pfm.py`). Deterministic 80/10/10
  train/val/test split, seeded per scene (see `split_indices` in
  `data/uwstereo.py`). Observed max disparity ~240–448px.
- **FLSea**: real underwater images, qualitative-only — no ground truth, not
  used for quantitative training or eval.

Dataset roots are **local Windows paths** passed via CLI flags
(`--sceneflow-root`, `--uwstereo-root`), not checked into the repo or configs.
Don't hardcode a path found in one script (e.g. `scripts/smoke_e2e.py`) as
the canonical location — treat those as whatever-was-true-that-day values.

## Training entry points

- `train.py` — SceneFlow pretrain
- `train_uw.py` — UWStereo-from-scratch (skips SceneFlow; runs the
  `pretrain` LR schedule directly on UWStereo)
- `finetune.py` — loads a pretrained checkpoint, continues on UWStereo
- `evaluate.py` — EPE + D1 (KITTI-style, 3px & 5%-relative) overall and
  per-scene, optional `--csv` row append for a master results table
- `engine.py` — shared `TrainCfg`, `build_model`, `train_one_run`,
  `run_eval`, loss/metric functions. Read this before touching training
  loop behavior — all four scripts above depend on it.

`--precision auto` picks BF16 on Ada/Hopper/Blackwell GPUs, otherwise use
`fp16` (forces FP16 + `GradScaler`) or `fp32`. `--config` selects the model
variant; `pretrain`/`finetune` LR/epoch/weight-decay come from the matching
YAML section, overridable via `--epochs`.

`run_matrix_final.bat` is the actual paper-run driver: runs all 4 configs
(2D first, 3D last) with `--precision bf16`, 10-epoch pretrain + 10-epoch
finetune, logging each stage to `log_*.log`. `status.ps1`/`watch.ps1`/`bar.ps1`
are local monitoring dashboards that tail those logs — they hardcode a
`C:\Users\canhh\...` path fallback and aren't meant to be portable.

## Deployment path

1. `python export_tensorrt.py --agg {3d,2d} --out onnx/gwc_X.onnx` (Windows
   dev box; produces + sanity-checks ONNX via `onnx.checker` and an
   onnxruntime numerical diff against the PyTorch output)
2. Copy the `.onnx` files to the Orin
3. On the Orin: `sudo nvpmodel -m 2 && sudo jetson_clocks`, then
   `trtexec --onnx=... --saveEngine=... --fp16` (TensorRT engines cannot be
   built on the Windows dev box)
4. `benchmark.py` (runs **on the Orin only** — needs `tensorrt`, `pycuda`,
   `tegrastats`) — reports latency/FPS, idle-subtracted net energy/frame,
   and frames-per-Wh, using `VDD_IN` as the power rail.

`build_int8.py` exists but INT8 is explicitly out of scope per the current
paper trim — don't wire it into the main flow without checking with the
user first.

## Baselines (`baselines/`)

- `sgbm.py` — OpenCV SGBM, CPU floor, no setup required.
- `psmnet_wrap.py` — wraps the original PSMNet (Chang & Chen 2018) as the
  "heavy ceiling" baseline. Requires a separate clone + pretrained checkpoint
  outside this repo (see `baselines/README.md`). Masks out pixels with
  disparity > 192 (PSMNet's trained `max_disp`) for a fair EPE/D1 comparison
  — this is a paper-methodology detail, don't silently change the mask.

## Conventions / gotchas

- Windows-first repo: dev/training happens on a Windows box with a conda
  env, PowerShell helper scripts, `.bat` matrix runner. Don't assume a Unix
  shell or `/tmp`-style paths belong here.
- `runs/<name>-{pretrain,finetune}/{best,last}.ckpt` plus TensorBoard event
  files are checked into the repo under `runs/` — actual experiment
  artifacts, not just a `.gitignore`'d output dir. Be careful not to
  overwrite or "clean up" these on the assumption they're disposable.
- `export_tensorrt.py` deliberately uses the legacy TorchScript-based ONNX
  exporter (`dynamo=False`) — the newer dynamo exporter produces opsets
  some TensorRT versions can't handle yet, and its unicode banners crash
  cp1252 (Windows default) consoles. Don't "modernize" this to
  `dynamo=True` without re-validating the export gate.
- Checkpoints store the model under a `"model"` key (`sd.get("model", sd)`
  pattern appears in `evaluate.py`/`export_tensorrt.py`) — handle both
  wrapped and raw state-dict checkpoints.
- The codebase is explicit about being an honest, synthetic-domain study:
  no real-water ground truth exists, sim-to-real is future work. Don't
  introduce claims or code that imply real-world validation beyond FLSea
  qualitative checks.
