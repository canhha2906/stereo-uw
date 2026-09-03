# stereo-uw

Zero-shot monocular depth for underwater robotics via **distillation +
physics-based domain adaptation**, targeting real-time inference on
resource-constrained hardware.

A frozen pretrained monocular depth model (teacher) supplies pseudo-depth
on ordinary "land" photos. A physics-based underwater light-transmission
simulator turns those photos underwater-looking. A small student network
is trained to recover the teacher's clean-image depth while only ever
looking at the synthesized-underwater image — so it generalizes to real
underwater scenes without ever training on underwater depth labels.

See **`PLAN_DISTILLATION.md`** for the full design, code map, and results
template.

## Why this approach

Underwater depth training data is scarce. A monocular depth model trained
purely on land imagery degrades badly underwater (color cast + light
attenuation are out-of-distribution for it). Rather than collecting real
underwater depth ground truth, this project synthesizes the underwater
domain shift with a physics model and distills a small student through
that synthetic shift — turning the scarce-data problem into a
domain-randomization problem.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# first run downloads MiDaS_small weights via torch.hub (needs internet once)

# baseline ablation: distill directly on clean images (no physics) --
# the "land model dropped underwater, works badly" case
python -m distill.train_distill --config configs\distill_uw.yaml `
    --clean-images-root D:\clean_images --ablation none --epochs 20

# proposed method: distillation through the physics simulator
python -m distill.train_distill --config configs\distill_uw.yaml `
    --clean-images-root D:\clean_images --ablation physics --epochs 20

# zero-shot evaluation of either checkpoint
python -m distill.evaluate_zeroshot --ckpt runs_distill\distill_uw-physics\last.ckpt `
    --uwstereo-root "...\UWStereo" --flsea-root "...\FLSea-challenge"
```

## Code map

| Path | Role |
|---|---|
| `distill/underwater_physics.py` | Jaffe-McGlamery underwater image formation simulator |
| `distill/teacher.py` | Frozen MiDaS-small wrapper, source of pseudo-depth labels |
| `distill/student.py` | Small MobileNetV2-encoder depth student |
| `distill/losses.py` | Scale-and-shift-invariant distillation loss |
| `distill/dataset.py` | Generic land-photo folder loader (no labels needed) |
| `distill/train_distill.py` | Training entry point (`--ablation none` vs `physics`) |
| `distill/evaluate_zeroshot.py` | Zero-shot eval: UWStereo (quantitative), FLSea (qualitative) |
| `configs/distill_uw.yaml` | Model / training / physics hyperparameters |

## Honest disclaimers

- The physics simulator's depth input is a monocular teacher's *relative*
  depth with a randomized scale factor, not a calibrated water distance —
  it's a domain-randomization tool, not a metrically accurate underwater
  renderer.
- FLSea has no depth ground truth; all FLSea results are qualitative only.
- The student has no metric scale, so quantitative comparisons use
  scale-shift-aligned metrics (see `PLAN_DISTILLATION.md`).
