# Ablation results — rendering variants

All rows: UWStereo test split, 2,958 pairs, identical eval script and disparity
mask (0 < d < 192). Every run starts from the same `kitti_2015.ckpt`, 60 epochs,
lrepochs 40:10, batch 4. Only the named variable differs.

| Run | Training data | Scale aug | EPE | >3px | D1-all |
|---|---|---|---|---|---|
| direct transfer | none (no finetune) | — | 5.8494 | 22.8836 | 21.5305 |
| typeIII | KITTI rendered, Jerlov III | no | 5.4623 | 20.8844 | 19.6414 |
| scaleaug | KITTI rendered, Jerlov III | s in [1,3] | 5.4195 | 23.0412 | 21.5187 |
| mixed10 | KITTI rendered, all 10 Jerlov types | no | 4.4677 | 18.8570 | 17.6471 |
| turbid4 | KITTI rendered, turbid coastal 3/5/7/9 | no | 4.2777 | 18.1071 | 16.9461 |

Reference: Paper 1's classical SGBM floor on this split is EPE 4.5620 / D1 18.73%.

_Generated automatically by finalize_results.sh when the ablation driver finished._

---

## Few-shot adaptation — what a handful of real underwater labels buys

Thái asked for this. Starting from the **turbid4** checkpoint (the zero-label winner), the
model is fine-tuned on N labelled UWStereo pairs and re-evaluated.

**Methodology:** the N pairs come from UWStereo's **train** split (23,654 pairs). The
2,958-pair test split is the same deterministic seed-0 split used everywhere else in this
repo and is never touched. Evaluation here uses a fixed 600-image subset of that test
split, so these numbers are comparable to each other but **not** to the 2,958-pair table
above — N=0 scores 3.9314 here against 4.2777 on the full split.

| N labelled pairs | EPE | >3px (%) | D1-all (%) |
|---|---|---|---|
| 0 (zero underwater labels) | 3.9314 | 17.61 | 16.40 |
| 1 | **5.2915** | 18.82 | 17.58 |
| 5 | 3.3388 | 15.03 | 13.85 |
| 10 | 3.3061 | 14.30 | 13.07 |
| 25 | 2.7593 | 13.04 | 11.83 |
| 50 | **2.5448** | 10.35 | 9.24 |

Two things worth stating in the paper:

**One labelled image is worse than none.** N=1 degrades EPE from 3.93 to 5.29 — worse
than the zero-label model it started from. 200 epochs on a single scene overfits it away.
The curve only turns useful from N=5.

**After that it keeps paying.** 50 pairs reach EPE 2.5448 and D1 9.24%, a 35% EPE
reduction against zero labels. There is no plateau in this range, so the honest statement
is that more labels keep helping and 50 is not the ceiling.

Setup: lr 1e-5, 200 epochs, batch ≤2, the authors' own `model_loss_train` with their
`[1.0, 0.3, 0.5, 0.3]` weighting over the four multi-scale outputs — identical to
`main_kitti.py`, so these runs share the training recipe with everything else here.
