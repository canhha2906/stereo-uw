# Distillation + Physics-Based Underwater Adaptation

Active direction as of 2026-09-01, replacing the earlier stereo-only
approach entirely — the old GwcNet-lite stereo pipeline (`models/`,
`data/`, `train.py`, `finetune.py`, `evaluate.py`, `engine.py`,
`export_tensorrt.py`, `benchmark.py`, `baselines/`, `scripts/`, old
`configs/*.yaml`, and `runs/` checkpoints) was deleted from this repo in
the same commit that added the files below. It remains recoverable from
git history (`git log --diff-filter=D`) if ever needed for comparison.

## The idea

1. A pretrained monocular depth model (**teacher**, frozen) runs on
   ordinary "land" photos and produces a relative depth map.
2. That depth map drives a **physics-based underwater light-transmission
   simulator** (Jaffe-McGlamery model) which synthesizes an
   underwater-looking version of the same photo.
3. A small **student** network is trained to predict the teacher's
   *clean-image* depth while only ever looking at the
   *synthesized-underwater* image.
4. Because the student is never shown real underwater depth labels
   (there aren't enough of them), evaluation on real/synthetic underwater
   data is inherently **zero-shot**.

The expected story (matching the original note): a land-domain model
applied directly underwater degrades badly (color cast + attenuation is
out-of-distribution for it); once the physics-based simulator is added to
the training loop, the resulting student handles underwater images well
despite never training on real underwater depth.

## Code map

| File | Role |
|---|---|
| `distill/underwater_physics.py` | `synthesize_underwater()` — the Jaffe-McGlamery simulator, `I = J*e^(-βd) + A*(1-e^(-βd))`, with randomized per-channel attenuation (β) and backscatter (A) for domain randomization |
| `distill/teacher.py` | Frozen MiDaS-small wrapper — the only source of pseudo depth labels |
| `distill/encoder.py` | Self-contained ImageNet-pretrained MobileNetV2 encoder |
| `distill/student.py` | `MonoDepthStudent` — `distill/encoder.py` + a small dense-depth head |
| `distill/losses.py` | Scale-and-shift-invariant L1 (teacher depth has no metric scale) |
| `distill/dataset.py` | `CleanImageFolder` — any generic land-photo folder, no labels needed |
| `distill/pfm.py` | Minimal PFM reader, for UWStereo disparity GT during eval only |
| `distill/uwstereo_index.py` | Minimal UWStereo file-index builder, for eval only |
| `distill/train_distill.py` | Training loop, `--ablation none` (baseline) vs `--ablation physics` (proposed) |
| `distill/evaluate_zeroshot.py` | Quantitative zero-shot AbsRel on UWStereo (disparity→depth), qualitative dumps on FLSea |
| `configs/distill_uw.yaml` | Model/training/physics hyperparameters |

`distill/` has no dependency on any deleted stereo-pipeline package — it's
fully self-contained (its own PFM reader and UWStereo index builder,
rather than importing the old `data/` package).

## Setup steps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # now includes timm, needed by torch.hub MiDaS

# first run downloads MiDaS_small weights via torch.hub (needs internet once)
```

**1. Baseline ablation — "land model dropped underwater" (expected: bad)**
```powershell
python -m distill.train_distill --config configs\distill_uw.yaml `
    --clean-images-root D:\clean_images --ablation none --epochs 20
```

**2. Proposed method — distillation + physics simulator (expected: good)**
```powershell
python -m distill.train_distill --config configs\distill_uw.yaml `
    --clean-images-root D:\clean_images --ablation physics --epochs 20
```

`--clean-images-root` should point at any sizeable, diverse RGB photo
collection — depth-structure diversity matters, underwater relevance
does not, since the underwater look is added synthetically.

**3. Zero-shot evaluation, both checkpoints**
```powershell
python -m distill.evaluate_zeroshot --ckpt runs_distill\distill_uw-none\last.ckpt `
    --uwstereo-root "...\UWStereo" --flsea-root "...\FLSea-challenge" --out-dir eval_none

python -m distill.evaluate_zeroshot --ckpt runs_distill\distill_uw-physics\last.ckpt `
    --uwstereo-root "...\UWStereo" --flsea-root "...\FLSea-challenge" --out-dir eval_physics
```

**4. (Optional, later) Edge deployment** — once the physics-augmented
model's accuracy is validated, `MonoDepthStudent` should go through an
ONNX export + TensorRT build gate (single-image input, so simpler than
the old stereo pair export) before any deployment claims are made. No
export script exists yet in this repo — write and validate one only
after the accuracy story above holds up.

## Results table (fill in after running the two ablations)

**Quantitative — zero-shot on UWStereo (disparity→depth, scale-shift aligned)**

| Ablation | Physics simulator | AbsRel (UWStereo, zero-shot) |
|---|---|---|
| `none` | off | TBD |
| `physics` | on | TBD |

This is the core number for the paper's claim: `physics` AbsRel should be
meaningfully lower than `none`.

**Qualitative — FLSea (real underwater, no GT)**

| Ablation | Notes on `eval_*/00X_depth.png` vs `00X_rgb.png` |
|---|---|
| `none` | TBD — expect flat/collapsed depth or artifacts driven by color cast |
| `physics` | TBD — expect depth structure that tracks scene layout despite haze/color cast |

**Per-scene breakdown (optional, mirrors `evaluate.py`'s per-scene split on UWStereo)**

| Ablation | coral | industry | ship | default |
|---|---|---|---|---|
| `none` | TBD | TBD | TBD | TBD |
| `physics` | TBD | TBD | TBD | TBD |

## Honest disclaimers (carry into the paper, consistent with the stereo study)

- `d(x)` used by the physics simulator is the teacher's *relative* depth
  with a randomized scale factor, not a calibrated water depth — the
  simulator is a domain-randomization tool, not a metrically accurate
  underwater renderer.
- FLSea has no depth ground truth; all FLSea results are qualitative only,
  same caveat as in the stereo study.
- UWStereo zero-shot numbers use scale-shift-aligned AbsRel because the
  student never learns a metric scale — this is standard practice for
  relative-depth models but should be stated explicitly in the paper.
