# CLAUDE.md

This file gives Claude Code the context it needs to work in this repo.

## What this is

A research codebase for a conference paper: zero-shot monocular depth
estimation for **underwater robotics**, via **knowledge distillation
through a physics-based domain simulator**, targeting real-time inference
on resource-constrained edge hardware.

The core idea: a frozen pretrained monocular depth model (teacher)
produces pseudo-depth on ordinary "land" photos. A physics-based
underwater light-transmission simulator (Jaffe-McGlamery model) turns
those photos underwater-looking. A small student network is trained to
recover the teacher's *clean-image* depth while only ever looking at the
*synthesized-underwater* image. Because the student never trains on real
underwater depth labels, every evaluation on underwater imagery is
inherently zero-shot.

This repo previously contained a stereo-matching (GwcNet-lite) pipeline
for the same underwater-depth problem. That direction was deleted on
2026-09-01 in favor of this distillation approach (still recoverable from
git history if ever needed, e.g. `git log --diff-filter=D -- models/`).
Do not try to resurrect or reference the old stereo code as if it's
still present — it isn't.

See **`PLAN_DISTILLATION.md`** for the full design rationale, code map,
setup steps, and results-table template. This file covers conventions and
gotchas; that file covers the "what and why."

## Code map

| Path | Role |
|---|---|
| `distill/underwater_physics.py` | `synthesize_underwater()` — the Jaffe-McGlamery simulator, `I = J*e^(-βd) + A*(1-e^(-βd))`, with randomized per-channel attenuation (β) and backscatter (A) |
| `distill/teacher.py` | Frozen MiDaS-small wrapper (`torch.hub`) — the only source of pseudo-depth labels; never trained |
| `distill/encoder.py` | Self-contained ImageNet-pretrained MobileNetV2 encoder (no cross-package deps) |
| `distill/student.py` | `MonoDepthStudent` — `distill/encoder.py` + a small dense-depth head |
| `distill/losses.py` | Scale-and-shift-invariant L1 (teacher depth has no metric scale) |
| `distill/dataset.py` | `CleanImageFolder` — any generic land-photo folder, no labels needed |
| `distill/pfm.py` | Minimal PFM reader, for UWStereo disparity GT during eval only |
| `distill/uwstereo_index.py` | Minimal UWStereo file-index builder, for eval only |
| `distill/train_distill.py` | Training loop; `--ablation none` (baseline) vs `--ablation physics` (proposed) |
| `distill/evaluate_zeroshot.py` | Quantitative zero-shot AbsRel on UWStereo (disparity→depth), qualitative dumps on FLSea |
| `configs/distill_uw.yaml` | Model / training / physics hyperparameters |

`distill/` is fully self-contained — it has its own PFM reader and
UWStereo index builder rather than importing anything from the deleted
`data/` or `models/` packages. Keep it that way: don't reintroduce a
`from models...` or `from data...` import here.

## Conventions / gotchas

- **The physics simulator's `d(x)` is not a calibrated distance.** It's
  the teacher's *relative* depth, min-max normalized per-sample, times a
  randomized scale factor (`depth_scale_range` in the config). It's a
  domain-randomization tool, not a metrically accurate underwater
  renderer — don't let code or docs imply otherwise.
- **The teacher is frozen, always.** `distill/teacher.py` wraps it with
  `requires_grad_(False)` and calls it under `@torch.no_grad()`. If a
  change accidentally lets gradients flow into the teacher, that's a bug,
  not a feature — the whole point is the student learns to be
  underwater-invariant, not that the teacher adapts.
- **`--ablation none` vs `--ablation physics` is the paper's core
  comparison.** `none` is the "land model looks bad underwater" baseline
  (distills on clean images only); `physics` is the proposed method. Any
  change to the physics simulator's parameter ranges should be evaluated
  against both, not just `physics` in isolation.
- **UWStereo and FLSea are eval-only here**, not training data — see
  `distill/evaluate_zeroshot.py`. UWStereo's disparity GT is converted to
  relative depth (`depth ~ 1/disparity`) for a quantitative check; FLSea
  has no GT at all and is qualitative-only. Neither dataset is touched
  during `train_distill.py`, which only needs `--clean-images-root`
  (ordinary land photos, no labels).
- **No metric scale anywhere in this pipeline.** The student outputs
  relative depth. Any quantitative comparison must scale-shift-align
  first (see `distill/losses._lstsq_scale_shift`) — don't compute raw
  L1/RMSE between predicted and ground-truth depth without aligning them.
- Windows-first repo: dev happens on a Windows box; commands in docs use
  PowerShell (`` ` `` line continuation, `^` in `.bat`-style examples).
  Don't assume a Unix shell.
- Edge deployment (`export_tensorrt.py`-style ONNX/TensorRT export) is
  **not wired up yet** for the student model — see the "optional, later"
  note in `PLAN_DISTILLATION.md`. Don't assume a deployment gate exists
  until it's actually built and validated on target hardware.
