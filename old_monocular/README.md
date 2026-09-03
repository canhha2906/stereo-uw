# old_monocular — superseded direction

A **monocular** depth approach explored on 2026-09-01, before the senior's stereo plan
became the direction. Kept here for reference. **Not deleted, not being worked on.**

## What it was

A frozen pretrained monocular depth model (teacher) produces pseudo-depth on ordinary
land photos. A physics-based underwater light-transmission simulator (Jaffe-McGlamery)
makes those photos look underwater. A small student learns to recover the teacher's
*clean-image* depth while only ever seeing the *synthesized-underwater* image.

## Why it was superseded

The senior's plan is **stereo**, not monocular: Fast-ACVNet with its KITTI-pretrained
weights, retrained on physics-rendered KITTI. See `../CLAUDE.md`.

## What is still useful

`distill/underwater_physics.py` implements the underwater image formation model:

```
I_c(x) = J_c(x) · t_c(x) + A_c · (1 − t_c(x))
t_c(x) = exp(−β_c · d(x))
```

This is exactly the equation Paper 2 needs at Stage 2, so it is reused rather than
rewritten. Two changes are required when reusing it:

1. It expects a monocular teacher's **relative** depth, min-max normalized per sample
   with a randomized depth-scale factor. KITTI gives **true metric depth**
   (`Z = f·B/disparity`), so that normalization and random scale must be bypassed.
2. Its β values are sampled from random ranges. Paper 2 uses **fixed Jerlov water
   types**, so β becomes a constant per water type from UWCNN's tables.

## Contents

| Path | What |
|---|---|
| `distill/underwater_physics.py` | The image formation model — the reusable part |
| `distill/teacher.py` | Frozen MiDaS-small wrapper, source of pseudo-depth |
| `distill/student.py` | Small MobileNetV2-encoder depth student |
| `distill/losses.py` | Scale-and-shift-invariant distillation loss |
| `distill/dataset.py` | Land-photo folder loader (no labels needed) |
| `distill/train_distill.py` | Training entry point |
| `distill/evaluate_zeroshot.py` | Zero-shot evaluation on underwater imagery |
| `distill/uwstereo_index.py`, `distill/pfm.py`, `distill/encoder.py` | Support code |
| `configs/distill_uw.yaml` | Its training config |
| `PLAN_DISTILLATION.md` | Its full design document |
| `CLAUDE_monocular_distill.md` | Its working spec |

The full original lineage is also preserved on branch `monocular-distill`.

> Paths inside these files assume the package was at the repo root (`distill/`, not
> `old_monocular/distill/`). Fix imports before running any of it.
