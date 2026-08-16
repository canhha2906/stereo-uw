# Baselines

## SGBM (OpenCV, CPU floor)

Built-in, no setup. Run:
```powershell
python -m baselines.sgbm --uwstereo-root "<path>"
```

## PSMNet (heavy ceiling, per spec §5)

PSMNet is cited as the ancestor of our 3D-aggregation path. We use the authors'
SceneFlow-pretrained checkpoint as the heavy ceiling baseline — it will not run
real-time on the Orin, which is the point.

**Reference:** [JiaRenChang/PSMNet](https://github.com/JiaRenChang/PSMNet) — Chang & Chen 2018, CVPR.

### One-time setup (after D drive is mounted)

```powershell
# clone repo + create folders
New-Item -ItemType Directory -Force -Path "D:\PSMNet\weights" | Out-Null
git clone https://github.com/JiaRenChang/PSMNet "D:\PSMNet\repo"

# download pretrained_sceneflow_new.tar (~150 MB)
# (link on PSMNet README under "Pretrained Model" -> SceneFlow row)
# place at: D:\PSMNet\weights\pretrained_sceneflow_new.tar
```

### Run on UWStereo test split

```powershell
python -m baselines.psmnet_wrap `
    --psmnet-root  "D:\PSMNet\repo" `
    --psmnet-ckpt  "D:\PSMNet\weights\pretrained_sceneflow_new.tar" `
    --uwstereo-root "C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo" `
    --split test
```

Reports EPE, 3px/D1, latency on whatever device PyTorch finds (5060 if available).

### Notes for the paper
- **Max disparity:** PSMNet was trained at `max_disp=192`. UWStereo has frames with
  disparity > 192. We mask those pixels out of PSMNet's EPE/D1 to keep the
  comparison fair. Document this in the paper.
- **PSMNet on Orin:** likely won't run real-time. Try `trtexec` on a PSMNet ONNX
  export; if it OOMs, report "could not deploy" — that's the "this is too heavy
  for edge" story you want to anchor the Pareto figure.
