# Training procedure and results tables — Paper 2

Companion to `CLAUDE.md`. That file says *what* to do; this one says *how to run it*
and *where the numbers go*.

> **No numbers in this file are filled in, and none should be invented.**
> Every table below is a skeleton. A cell gets a value only after that exact run
> has finished and its log exists. Nothing here has been executed yet.

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

| Metric | Authors report | We measure | Match? |
|---|---|---|---|
| KITTI 2015 D1-all | 2.01% | | |
| KITTI 2012 3-all | 1.85% | | |
| Runtime | 45 ms | | |

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
| UWStereo test | 2,958 | | | |
| SQUID | | | | |

Also save qualitative disparity maps — the failure mode matters as much as the number.

---

## 3. Stage 2 — Render KITTI through the physics model

### Reuse what already exists

`distill/underwater_physics.py` already implements the exact equation in `CLAUDE.md`:

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

| Model | UWStereo EPE | UWStereo >3px | SQUID EPE | SQUID >3px |
|---|---|---|---|---|
| Fast-ACVNet+ KITTI (direct transfer) — *from Stage 1* | | | | |
| Fast-ACVNet+ retrained on rendered KITTI | | | | |
| Δ (improvement) | | | | |

If the second row is better, the core claim holds. If it is not, that is the finding
and it gets reported as such.

**Metric definitions** (state these in the paper so numbers are comparable):
- **EPE** — end-point error, mean absolute disparity error in pixels
- **>3px** — percentage of pixels whose disparity error exceeds 3 px
- **D1-all** — percentage of pixels with error > max(3 px, 5% of ground truth)

---

## 6. Stage 5 — Distillation

Only after Stage 4 produces a result. Teacher = the Stage 3 model.

| Model | Params (M) | UWStereo EPE | >3px | Latency (ms) |
|---|---|---|---|---|
| Teacher (Stage 3) | | | | |
| Student (distilled) | | | | |

---

## 7. Stage 6 — Quantization

Reuse Paper 1's tooling: `export_tensorrt.py`, `build_int8.py`, `scripts/dump_calib.py`,
`benchmark.py`. Benchmark on the Orin Nano at 25 W with `jetson_clocks` pinned, and log
the JetPack and TensorRT versions.

| Precision | UWStereo EPE | Latency (ms) | FPS | Peak mem (MB) | Energy (mJ/frame) |
|---|---|---|---|---|---|
| FP32 | | | | | |
| FP16 | | | | | |
| INT8 | | | | | |

---

## 8. Order of execution

```
Stage 0 sanity  →  Stage 1 baseline  →  Stage 2 render  →  Stage 3 retrain
                                                                  ↓
              Stage 6 quantize  ←  Stage 5 distill  ←  Stage 4 compare
```

Stage 1 must be measured **before** Stage 3 exists. Running them out of order and
comparing against a remembered number is how a result gets quietly invalidated.
