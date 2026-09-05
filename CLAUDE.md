# CLAUDE.md — Physics-Guided Underwater Stereo (Paper 2)

> Working context for an agent executing this project end-to-end.
> This plan came from the senior (Thái). Execute it as written.
> Do not redesign it, do not substitute components, do not reorder the stages.

---

## 0. One-line goal

Take an off-the-shelf stereo network pretrained on land (Fast-ACVNet, KITTI 2015),
measure how badly it fails underwater, then retrain it on KITTI **rendered through
an underwater physical light-attenuation model** and measure how much that
recovers. Then distill, then quantize.

---

## 1. The claim

*A land-trained stereo network degrades underwater. Retraining it on terrestrial
data pushed through a physical underwater imaging model recovers accuracy on real
underwater data — without ever training on underwater data.*

If Stage 4 beats Stage 1 on the underwater evaluation set, the result is publishable.

---

## 2. NON-NEGOTIABLES

1. **Underwater data is EVALUATION ONLY.** Never train on it, never finetune on
   it, never use it to pick hyperparameters. It is the held-out test domain. This
   is the whole point of the experiment — violating it destroys the claim.
2. **Base network is Fast-ACVNet.** Do not substitute GwcNet-lite, IGEV,
   RAFT-Stereo, or anything else. Use the authors' repo and their released
   pretrained weights.
3. **Training data is KITTI.** The physics model is applied to KITTI images.
4. **Stage order is fixed:** baseline → physics retrain → evaluate → distill →
   quantize. Do not start distillation before the physics result exists. Do not
   quantize before distillation.
5. **The only thing added to the training pipeline is the physics rendering.**
   Everything else about Fast-ACVNet's training stays as the authors shipped it.
6. Report the Stage 1 vs Stage 4 comparison honestly. If physics rendering does
   not help, that is the result.

---

## 3. Pipeline

### Stage 0 — Setup

- Clone https://github.com/gangweiX/Fast-ACVNet (MIT license).
- Download the authors' pretrained weights. Two sets are published:
  - **Fast-ACVNet** — SceneFlow only
  - **Fast-ACVNet+** — SceneFlow then finetuned on KITTI  ← **this is the one
    Thái means by "pretrained KITTI 2015"**
- Environment per their README: Python 3.8, PyTorch 1.10 + CUDA 11.3,
  `timm==0.5.4`, opencv, scikit-image, tensorboard, matplotlib, tqdm.
- Reproduce their reported KITTI numbers first (Fast-ACVNet+: KITTI 2015 D1-all
  2.01%, KITTI 2012 3-all 1.85%, 45 ms) to confirm the checkout is sane before
  changing anything.

### Stage 1 — Baseline: land model straight into water

Run the **unmodified** KITTI-pretrained checkpoint on the underwater evaluation
set. No finetuning, no adaptation. This is the "direct transfer" number the paper
argues against.

Evaluation sets:
- UWStereo held-out test split (2,958 pairs, `data set/UWStereo/`)
- SQUID real underwater pairs

Report: EPE, D1 / >3px. Save qualitative disparity maps.

### Stage 2 — Physics rendering of KITTI

Apply the underwater image formation model to KITTI's left and right images.

    U_c(x) = I_c(x) · t_c(x) + A_c · (1 − t_c(x))
    t_c(x) = exp(−β_c · d(x))

where, per colour channel c ∈ {R, G, B}:
- `I_c(x)` = original clear KITTI pixel value
- `U_c(x)` = rendered underwater pixel value
- `d(x)` = distance from camera to that pixel, in metres
- `β_c` = attenuation coefficient for that channel, per metre — depends on water type
- `A_c` = global ambient (background) light for that channel
- `t_c(x)` = transmission, fraction of light surviving the path

Implementation notes:
- `d(x)` comes from KITTI disparity via `Z = f · B / disparity`, with focal length
  `f` and baseline `B` read from KITTI's `calib_cam_to_cam` files. KITTI depth is
  already in real metres, so β values apply directly.
- **Disparity ground truth does not change.** Rendering recolours pixels; it does
  not move them. Reuse KITTI's existing labels unchanged.
- Apply the **same** water parameters to the left and right image of a pair.
  Different parameters per eye would break stereo matching.
- β per Jerlov water type: take from UWCNN (Li et al., *Underwater scene prior
  inspired deep underwater image and video enhancement*), code and tables at
  https://github.com/saeed-anwar/UWCNN — the standard source for these numbers.
- KITTI disparity GT is sparse (LiDAR). Pixels without GT still get rendered; they
  are simply not supervised, exactly as in normal KITTI training.

### Stage 3 — Retrain on physics-rendered KITTI

Load the KITTI-pretrained checkpoint from Stage 0 and continue training on the
rendered KITTI data from Stage 2, using Fast-ACVNet's own `main_kitti.py` training
procedure.

    python main_kitti.py --loadckpt <kitti_pretrained.ckpt> --logdir ./checkpoints/kitti_uw

Keep their optimizer, schedule, augmentation and loss unchanged.

### Stage 4 — Evaluate and compare

Run the Stage 3 model on the **same** underwater evaluation sets as Stage 1, with
the same metrics and the same code path.

The headline table is exactly two rows:

| Model | UWStereo EPE | UWStereo >3px | SQUID |
|---|---|---|---|
| Fast-ACVNet+ KITTI-pretrained (direct transfer) | | | |
| Fast-ACVNet+ retrained on physics-rendered KITTI | | | |

If row 2 beats row 1, the core result exists.

### Stage 5 — Distillation

Distill the Stage 3 model into a smaller student.

### Stage 6 — Quantization

Quantize the distilled student. The existing INT8 / TensorRT / Orin Nano tooling
from Paper 1 is reusable: `build_int8.py`, `export_tensorrt.py`, `benchmark.py`,
`scripts/dump_calib.py`.

---

## 4. Data locations (Windows dev box)

- UWStereo: `C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo\`
- FLSea:    `C:\Users\canhh\Workspace\conference paper, computer vision\data set\FLSea-challenge\`
- KITTI 2012 / 2015: to download
- SQUID: to download

---

## 5. Relationship to Paper 1

Paper 1 (GwcNet-lite accuracy/latency/energy characterization) is a **separate**
project living in the same repo. Its spec is at
`conference paper, computer vision/instruction/CLAUDE.md` and its code
(`models/`, `train.py`, `finetune.py`, `evaluate.py`, `configs/`, `runs/`) stays
untouched. Reuse from it is limited to the deployment tooling named in Stage 6.

---

## 6. Deviations and extensions beyond Thái's plan

**Rule:** Sections 0–5 above are Thái's plan. Anything *not* literally in his
instructions is recorded here, labelled, so it is always clear what came from him and
what was added while implementing. If anything in this section ever conflicts with
Sections 0–5, **his plan wins.**

**Venue target:** unranked or low-ranked conference. The bar is a sound experiment
honestly reported, not a SOTA result. Do not inflate scope to chase a stronger venue.

### 6.1 Forced by implementation — not design choices

These are not alternatives; the experiment is wrong without them.

- **Same water parameters for the left and right image of a pair.** Different β or A
  per eye breaks stereo correspondence and the network would be learning from an
  impossible image pair.
- **Disparity ground truth is reused unchanged.** Rendering recolours pixels, it does
  not move them, so KITTI's labels stay valid.
- **An evaluation script has to be written.** Fast-ACVNet ships `save_disp.py` for
  KITTI benchmark submission, not an EPE/D1 evaluator for UWStereo or SQUID.

### 6.2 Added while writing the spec

Each of these fills a gap Thái's message left open. None of them change the pipeline.

| Addition | Why |
|---|---|
| SQUID as a second evaluation set | Thái said "tập dữ liệu dưới nước" without naming one. UWStereo is synthetic; SQUID is real water, so the claim is not synthetic-only. |
| Metrics fixed as EPE, >3px, D1-all | Needed so Stage 1 and Stage 4 are comparable. |
| β taken from UWCNN's Jerlov tables | Thái said "mô hình vật lý" without specifying coefficients. UWCNN is the standard published source. |
| Stage 0 reproduces the authors' KITTI numbers first | If the downloaded checkpoint is not what it claims, every later comparison is meaningless. |
| KITTI val D1 tracked during Stage 3 | Detects the failure where the model forgets how to match instead of learning to see through water. |
| Reuse `old_monocular/distill/underwater_physics.py` | It already implements the exact equation; rewriting it would risk a different bug. |
| Reuse Paper 1's Orin tooling at Stage 6 | Thái said "quantize"; `build_int8.py`, `export_tensorrt.py`, `benchmark.py` already do it. |

### 6.2b Found while actually running it (2026-09-03/05)

Discovered during execution, not planned. All are recorded because they change results
or were required to get anything to run at all.

**Wavelength-dependent veiling light — a real deviation from UWCNN.**
UWCNN draws the background light `B_c` uniformly in (0.8, 1): bright and neutral. On
KITTI that inverts the colour cast, because

    U_R - U_B = (I - B) * (T_R - T_B),  and  T_R < T_B

so the sign is set by `(I - B)`. KITTI scenes are dark (mean 0.38) against a bright
neutral B (~0.9), giving a *reddish* haze: measured R-B = +0.062. The evaluation domain
is the opposite - UWStereo measures R-B between -0.10 and -0.27 with red near zero.
Training reddish to test blue-green would fight the experiment. Fix: the veiling light
is itself sunlight already filtered by water, so

    B_c = B_surface * N_c ** d_ambient,   d_ambient = 10 m

Rendered KITTI then measures R-B = -0.146, mean RGB [0.223, 0.368, 0.369] - inside the
target range. **This is a deviation from the cited recipe and must be stated in the paper.**

**Depth band.** UWCNN maps scene depth to 0.5-15 m (NYU indoor). KITTI is outdoor, true
depth 6-83 m, so the whole range cannot be used or everything saturates to veiling
light. Scene depth is rescaled into 0.5-8 m, 1st/99th percentile clipped.

**Densification.** KITTI disparity GT is only ~19% dense, but rendering needs a distance
for every pixel. Holes are filled by nearest valid neighbour, used ONLY to drive the
physics. Supervision still uses the original sparse ground truth.

**Transmission is base-N, not base-e.** UWCNN Eq. 2 is `T = N ** d`, i.e. 10^(-beta*d),
not `exp(-beta*d)`. Using the wrong base silently changes the water by a factor of ln(10).

**N_lambda values** are UWCNN Table 1, transcribed in `render_kitti_underwater.py`.

**Environment.** Fast-ACVNet pins PyTorch 1.10/cu113, which does NOT support the Blackwell
5060. It runs fine on torch 2.11/cu128 instead. Required fixes: `timm==0.5.4` (timm 1.x
removed `EfficientNetFeatures.act1`), `np.lib.pad` -> `np.pad` in 5 files under `datasets/`
(NumPy 2 removed the alias), plus matplotlib / scikit-image / opencv-python / scipy /
tensorboardX.

**main_kitti.py fork bomb on Windows.** All setup (SummaryWriter, datasets, model,
optimizer) sat at module level with only `train()` behind the `__main__` guard. When the
process is interrupted, tensorboardX's atexit cleanup touches a multiprocessing.Queue,
which spawns a child, which re-imports the module, which builds another SummaryWriter,
which spawns again - an unbounded cascade (25 processes observed, one every 3 s, GPU
pinned at 100%). Fixed by moving the setup block into `_setup()` called from the
`__main__` guard. `num_workers` also set 24 -> 0. Original kept as `main_kitti.py.orig`.

**Training length.** Authors default to 500 epochs (`lrepochs 300:10`). Run at 60 epochs
(`lrepochs 40:10`), batch 4, on 180 KITTI-2015 train images.

### 6.3 Deliberately NOT being done

Discussed in earlier sessions, **rejected as out of scope.** Recorded here so no future
session re-proposes them as if they were new ideas.

- Turbidity sweep with β as a plotted axis (EPE vs water type curve)
- Energy-per-frame / mJ characterization as a headline result
- Head-to-head comparison against AquaStereo's diffusion rendering
- Adding ULAP as an input prior or context-branch feature

These are not part of Thái's plan. Do not add them, and do not pitch them again.
