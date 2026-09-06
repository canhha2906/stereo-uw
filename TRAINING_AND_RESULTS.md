# Training procedure and results tables — Paper 2

Companion to `CLAUDE.md`. That file says *what* to do; this one says *how to run it*
and *where the numbers go*.

> Cells are filled ONLY from real logs. Anything still blank has not been run.
> Stages 0-5 are done and measured. Stage 6 (quantization) not started.

---

## 0. Prerequisites — none of this exists on the machine yet

| Item | Status | Where it goes |
|---|---|---|
| Fast-ACVNet repo | not cloned | `Workspace/code/Fast-ACVNet/` |
| Fast-ACVNet+ pretrained weights (SceneFlow → KITTI) | not downloaded | authors' Google Drive |
| KITTI 2015 | not downloaded | `data set/KITTI2015/` |
| KITTI 2012 | not downloaded | `data set/KITTI2012/` |
| SQUID | not downloaded | `data set/SQUID/` |
| UWStereo | **already on disk** | `data set/UWStereo/` |
| Python 3.8 + PyTorch 1.10/cu113 + `timm==0.5.4` | separate env needed | conda env, e.g. `facv` |

The existing `stereo` conda env is on a much newer PyTorch for the Blackwell GPU.
Fast-ACVNet pins PyTorch 1.10, so it needs its own environment. Check whether
1.10/cu113 actually runs on the RTX 5060 before assuming this works — if it does
not, the fallback is a newer torch and fixing whatever breaks in their code.

### Data sizes to plan around

KITTI 2015 ships **200 training pairs** with sparse LiDAR disparity, and KITTI 2012
ships **194**. That is the entire supervised pool for Stage 3 unless KITTI raw is
added. Small, but Stage 3 is a *finetune* of an already-trained checkpoint, not
training from scratch.

---

## 1. Stage 0 — Setup and sanity check

```bash
git clone https://github.com/gangweiX/Fast-ACVNet
cd Fast-ACVNet
# download Fast-ACVNet+ weights (SceneFlow + KITTI finetuned) into ./pretrained/
```

Before changing anything, reproduce the authors' published numbers to confirm the
checkout and weights are intact:

| Metric | Authors report | We measure | Note |
|---|---|---|---|
| KITTI 2015 D1-all | 2.01% (test server) | **0.846%** | val split, 20 imgs - not the same images, so not a like-for-like reproduction |
| KITTI 2015 EPE | not reported | **0.4049 px** | val split, 20 imgs |

Checkpoint loads with 0 missing / 0 unexpected keys, 3.203M params. It is genuine and
the pipeline runs; that is all this check was for.

If these do not reproduce, stop and fix that first. Every later comparison depends
on this checkpoint being what it claims to be.

---

## 2. Stage 1 — Baseline: land model straight into water

No training. Run the unmodified Fast-ACVNet+ KITTI checkpoint on underwater data.

```bash
python test_uw.py \
    --loadckpt ./pretrained/fast_acvnet_plus_kitti.ckpt \
    --dataset uwstereo \
    --datapath "<...>/data set/UWStereo" \
    --outdir ./results/stage1_uwstereo
```

`test_uw.py` does not exist in their repo — it has to be written, modelled on their
`save_disp.py`, adding UWStereo/SQUID loading and EPE/D1 computation. `data/uwstereo.py`
in this repo already parses the UWStereo layout and PFM disparity, so reuse it.

**Results — Stage 1**

| Eval set | N pairs | EPE (px) | >3px (%) | D1-all (%) |
|---|---|---|---|---|
| **UWStereo test** | **2,958** | **5.8494** | **22.8836** | **21.5305** |
| SQUID | | | | | *(not downloaded yet)* |

For scale: the same checkpoint scores EPE 0.4049 on KITTI val, so underwater is
**14.4x worse**. It is also worse than the classical SGBM floor from Paper 1
(EPE 4.5620, D1 18.73%) - a land-trained network underwater loses to block matching.

Also save qualitative disparity maps — the failure mode matters as much as the number.

---

## 3. Stage 2 — Render KITTI through the physics model

### Reuse what already exists

`old_monocular/distill/underwater_physics.py` already implements the exact equation in `CLAUDE.md`:

```
I_c(x) = J_c(x) · t_c(x) + A_c · (1 − t_c(x))
t_c(x) = exp(−β_c · d(x))
```

**But it was written for the monocular pipeline and needs two changes for KITTI:**

1. It takes a monocular teacher's *relative* depth, min-max normalized per sample,
   with a randomized `depth_scale_range` standing in for real distance. KITTI gives
   **true metric depth**, so that normalization and the random scale must be
   bypassed — feed metres directly.
2. Its β ranges are randomized (`beta_r_range=(0.3,1.2)` etc.). Thái's plan uses
   **fixed Jerlov water types**, so β becomes a chosen constant per water type, taken
   from UWCNN's tables, not a sampled range.

### Getting metric depth from KITTI

```
Z(x) = f · B / disparity(x)
```

- `Z(x)` — distance from camera to the pixel, in **metres**
- `f` — focal length in pixels, from KITTI `calib_cam_to_cam.txt`
- `B` — stereo baseline in metres, same file (≈0.54 m for KITTI)
- `disparity(x)` — KITTI's ground-truth disparity in pixels

### Rules that must hold

- **The same water parameters for left and right image of a pair.** Different
  parameters per eye destroys stereo correspondence.
- **Disparity labels are unchanged.** Rendering recolours pixels; it does not move
  them. Reuse KITTI's `.png` disparity files as-is.
- KITTI GT is sparse. Pixels with no GT still get rendered, they are just not
  supervised — normal KITTI training behaviour.

**Sanity check before training on it:** render a handful of pairs, view them, and
confirm they look like the corresponding Jerlov type. A rendering bug here silently
poisons Stage 3.

---

## 4. Stage 3 — Retrain on rendered KITTI

```bash
python main_kitti.py \
    --loadckpt ./pretrained/fast_acvnet_plus_kitti.ckpt \
    --datapath ./data/kitti_underwater_<watertype>/ \
    --logdir ./checkpoints/kitti_uw_<watertype> \
    --epochs <as authors' default>
```

Optimizer, LR schedule, augmentation and loss stay exactly as the authors shipped
them. The only change in the whole pipeline is that the input images have been
rendered.

**Training log — one row per run**

| Water type | β (R, G, B) | Epochs | Final train loss | KITTI val D1 | Wall-clock |
|---|---|---|---|---|---|
| | | | | | |

Keeping KITTI val D1 here is worth it: if it collapses, the model has forgotten how
to match rather than learned to see through water, and Stage 4 would be misleading.

---

## 5. Stage 4 — The headline comparison

Same eval code path as Stage 1, same sets, same metrics. Only the checkpoint differs.

**Results — the two rows the paper rests on**

| Model | UWStereo EPE | >3px (%) | D1-all (%) |
|---|---|---|---|
| Fast-ACVNet+ KITTI, direct transfer — *Stage 1* | 5.8494 | 22.8836 | 21.5305 |
| **Fast-ACVNet+ retrained on rendered KITTI (type III)** | **5.4623** | **20.8844** | **19.6414** |
| Δ | **−0.3871 (−6.6%)** | **−2.00 pts (−8.7%)** | **−1.89 pts (−8.8%)** |

N = 2,958 pairs, identical eval code path and disparity mask (0 < d < 192) in both rows.
The only difference between the two rows is the checkpoint.

**The direction holds: rendering KITTI through the physics model and finetuning on it
improves underwater accuracy, consistently across all three metrics, without ever
training on underwater data.**

Two things to state honestly alongside that:

1. **The gain is modest** — 6.6% EPE. The model is better underwater, not fixed.
2. **It is still worse than the classical floor.** Paper 1's SGBM scores EPE 4.5620 /
   D1 18.73% on this same split. A land-trained network retrained through simulated
   water still loses to block matching underwater.

Run conditions: water type III only; training stopped at epoch 54/60 when a harness
timeout killed it mid-epoch (no crash, no error in the log), checkpoint_000053 used.
LR had already dropped 10x at epoch 40, so most of the refinement phase had run.

If the second row is better, the core claim holds. If it is not, that is the finding
and it gets reported as such.

**Metric definitions** (state these in the paper so numbers are comparable):
- **EPE** — end-point error, mean absolute disparity error in pixels
- **>3px** — percentage of pixels whose disparity error exceeds 3 px
- **D1-all** — percentage of pixels with error > max(3 px, 5% of ground truth)

---

## 6. Stage 5 — Distillation

Only after Stage 4 produces a result. Teacher = the Stage 3 model.

Teacher = the Stage 3 model. Student = GwcNet-lite from Paper 1. Trained on the same
200 rendered KITTI pairs, with dense teacher disparity plus KITTI's sparse GT.
Fast-ACVNet's own base variant was rejected as a student: 3.075 M vs 3.203 M is only
4% smaller.

| Model | Params (M) | UWStereo EPE | D1-all (%) |
|---|---|---|---|
| Teacher — Fast-ACVNet+, physics-retrained | 3.203 | 5.4623 | 19.64 |
| Student — GwcNet-lite, from scratch | 0.467 | 6.6738 | 27.84 |
| Student — GwcNet-lite, SceneFlow-initialised | 0.467 | 6.8941 | 27.58 |

**Distillation costs accuracy here: about 1.4 px EPE and 8 points of D1 for a 6.9x
smaller model.** The student also lands worse than the Stage 1 direct-transfer baseline
(5.8494), so at this size and data scale the compression is not worth it as it stands.

**Initialisation was not the bottleneck.** The first student started essentially random
apart from its ImageNet backbone, so it was re-run initialised from Paper 1's
SceneFlow-pretrained checkpoint (26,790 pairs, loads 0 missing / 0 unexpected). Starting
loss dropped from 14.77 to 2.69 — but final accuracy did not move (6.67 -> 6.89 EPE).
Easier optimisation, same end point.

The remaining suspect is data scale: 200 KITTI pairs is very little to adapt a network
that never sees underwater imagery. For contrast, Paper 1's same architecture reaches
EPE 1.62 on this split — but that model was finetuned on 29k UWStereo pairs, i.e. on the
target domain, which this experiment forbids.

---

## 7. Stage 6 — Quantization

Both models were taken through this stage, per the user's decision.

### Done on the dev box: ONNX export + PyTorch reference timing

| Model | Params (M) | ONNX | ORT vs PyTorch | Latency (ms) | Peak mem (MB) | UWStereo EPE |
|---|---|---|---|---|---|---|
| Teacher — Fast-ACVNet+ | 3.203 | 16.3 MB | 9.92e-05 | 34.2 | 371 | 5.4623 |
| Student — GwcNet-lite | 0.467 | 5.6 MB | 1.53e-04 | 6.5 | 70 | 6.8941 |
| ratio | 6.9x smaller | | | **5.3x faster** | 5.3x less | +1.43 px |

Timing is 50 iterations at 480x640 after 10 warm-up passes, RTX 5060 Laptop, FP32
PyTorch. **This is the dev GPU, not the Orin Nano** - the absolute numbers do not
transfer, but the ratio between the two models is the point.

This reframes the Stage 5 result. Distillation costs 1.43 px EPE and buys a 5.3x
speed-up and 5.3x memory cut. Whether that trade is worth taking is a deployment-budget
question, not a modelling one, and the paper should present it as such rather than
calling distillation a failure.

### Deployment blocker found: Fast-ACVNet+ does not export to ONNX as shipped

Three places select a top-k by `sort()` followed by a slice. PyTorch exports that as a
`TopK` node whose K operand both onnxruntime and TensorRT reject:

    Node (/m/TopK) Op (TopK) [ShapeInferenceError]
    K input must be a one-dimensional tensor of size 1

Fixed by using `torch.topk` explicitly, which emits K as a constant:

| File | Line | Was | Now |
|---|---|---|---|
| `models/Fast_ACV_plus.py` | 278 | `att_weights_prob.sort(2, True)` + `[:, :, :24]` | `torch.topk(..., 24, dim=2, largest=True)` |
| `models/Fast_ACV_plus.py` | 284 | `ind_k.sort(2, False)[0]` | `torch.topk(..., 24, dim=2, largest=False)` |
| `models/submodule.py` | 253 | `cost.sort(1, True)` + `[:, :2]` | `torch.topk(cost, 2, dim=1, largest=True)` |

Verified **bit-identical** output before and after (max abs diff 0.000e+00). Originals
kept as `.orig` beside each file. This belongs in the paper: it is exactly the class of
edge-deployment obstacle the work is about, and it is not documented by the authors.

### Still to do, and it needs the hardware

TensorRT engine build, INT8 calibration, FPS, and energy per frame **must run on the
Jetson Orin Nano** - there is no TensorRT or `trtexec` on this Windows box, so no engine
can be built here. Handoff: copy `onnx/stage6_student.onnx` and
`Fast-ACVNet/onnx/stage6_teacher.onnx` to the Orin, then use Paper 1's
`build_int8.py`, `scripts/dump_calib.py` and `benchmark.py` at 25 W with `jetson_clocks`
pinned, logging the JetPack and TensorRT versions.

| Precision | Model | Latency (ms) | FPS | Peak mem (MB) | Energy (mJ/frame) |
|---|---|---|---|---|---|
| FP16 | teacher | | | | |
| FP16 | student | | | | |
| INT8 | teacher | | | | |
| INT8 | student | | | | |

---

## 8. Order of execution

```
Stage 0 sanity  →  Stage 1 baseline  →  Stage 2 render  →  Stage 3 retrain
                                                                  ↓
              Stage 6 quantize  ←  Stage 5 distill  ←  Stage 4 compare
```

Stage 1 must be measured **before** Stage 3 exists. Running them out of order and
comparing against a remembered number is how a result gets quietly invalidated.
