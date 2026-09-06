# Why the gain is only 6.6%, and what to do about it

> **SUPERSEDED 2026-09-06 — the main diagnosis below is wrong.** It argued that a
> disparity-range mismatch was the dominant failure. That was tested directly with scale
> augmentation and it did essentially nothing (EPE 5.4623 → 5.4195, D1 worse). What
> actually moved the result was **water-type randomisation**: EPE 4.2777, which beats
> Paper 1's SGBM floor of 4.5620. The gap is optical, not geometric. See
> `RESULTS_ablation.md` and CLAUDE.md section 5b. The distribution measurements below are
> still correct as measurements; the conclusion drawn from them is not.

Thái's read after checking the repo: *"kết quả chưa khả quan lắm"*. Agreed. This is the
diagnosis, run on the Stage-4 model against the UWStereo test split, plus what the
evidence says to do next.

---

## The dominant failure is a disparity-range mismatch, not a data-volume problem

Measured distributions, same disparity units (native resolution):

| Dataset | p50 | p90 | p99 | >64 px | >128 px |
|---|---|---|---|---|---|
| **KITTI 2015** — what we finetune on | 32.7 | 59.3 | 81.9 | **5.4%** | **0.0%** |
| **UWStereo** — the target | 72.5 | 145.1 | 188.4 | **56.3%** | **17.2%** |
| SceneFlow FlyingThings3D | 31.4 | 81.6 | 128.8 | 18.2% | 1.0% |

**Over half of UWStereo's pixels sit above 64 px of disparity. KITTI has essentially
none there, and literally zero above 128 px.** Finetuning on KITTI therefore teaches the
network a disparity range that covers well under half the pixels it is tested on.

The per-range error confirms it exactly:

| GT disparity | EPE |
|---|---|
| 0–32 px | 13.19 |
| 32–64 px | **4.76** |
| 64–128 px | 6.88 |
| 128–192 px | **20.53** |

The model is competent precisely where KITTI has mass (32–64 px) and collapses at both
ends. The 128–192 band — 17% of target pixels — is where KITTI supplies **no training
signal at all**, and it is 4x worse than the band KITTI covers.

This reframes the whole result. The physics rendering is doing its job on appearance;
the geometry the model was finetuned on is simply the wrong shape.

## Secondary: scene content

| Scene | n | EPE |
|---|---|---|
| ship | 136 | **7.49** |
| default | 73 | 4.23 |
| coral | 52 | 3.69 |
| industry | 139 | 3.52 |

`ship` is twice as bad as anything else and is the largest group, so it drags the
headline number. Ship hulls are large, smooth, low-texture surfaces at close range —
KITTI contains nothing remotely like that, and close range is also exactly the
high-disparity band the training data lacks. The two problems compound.

Also worth noting: only 0.23% of GT pixels exceed the model's 192 limit, so the max-disp
setting is not the bottleneck. The bottleneck is the *training distribution* inside that
range.

---

## What to do, ranked by evidence

### 1. Fix the disparity range — highest impact, cheap

Two ways, and they combine:

- **Scale augmentation during finetuning.** Upscaling an image by a factor s multiplies
  its disparity by s. Random s in roughly [1, 3] on KITTI turns a 5%-above-64px training
  set into one that actually covers 64–192. No new data required.
- **Add SceneFlow.** It is already on disk at `C:\SCENEFLOW` — FlyingThings3D and
  Driving, **with dense PFM disparity**, roughly 26,790 pairs against KITTI's 200. It
  reaches 18.2% above 64 px on its own, and with scale augmentation it covers the target
  range properly. Dense labels also remove the 19%-sparse problem that made distillation
  necessary in the first place.

Rendering SceneFlow through the same physics model is the same code path as KITTI; only
the depth source changes, and SceneFlow ships dense depth.

### 2. Train across water types, not just type III

Every rendered image so far used Jerlov III. Sampling a water type per image from the ten
in UWCNN Table 1 is domain randomisation at essentially zero cost, and it is the standard
answer to "the model saw one appearance and the test set has four scenes".

### 3. Re-run distillation once the teacher is actually good

The student is currently distilled from a teacher that is itself limited by the above.
Fixing the teacher first is the right order — distilling a better teacher on dense
SceneFlow labels is a different experiment from distilling the current one on 200 sparse
KITTI pairs.

### 4. INT8 with the regression tail kept in FP16

Already scripted as `--keep-fp16-output` in Paper 1's `build_int8.py`. INT8 currently
costs +16.5% EPE on the student and +21% on the teacher; this is the standard fix and it
is cheap to test.

---

## What this does not fix

Even with all of the above, beating Paper 1's SGBM floor (EPE 4.5620) is not guaranteed.
That comparison should stay in the paper either way — a land-trained network losing to
block matching underwater is a legitimate and interesting result, and hiding it would be
the wrong call.

The honest framing of the contribution does not depend on winning: *physics-guided
rendering of terrestrial data measurably improves underwater stereo without any
underwater training data, and here is exactly where it still falls short and why.*
