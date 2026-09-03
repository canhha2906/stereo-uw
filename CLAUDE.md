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
