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
