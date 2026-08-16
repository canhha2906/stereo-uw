@echo off
REM ===========================================================================
REM FINAL PAPER MATRIX  -  all consistent:
REM   backbone = v2_imagenet (ImageNet MobileNetV2)
REM   pretrain = SceneFlow FT3D + Driving (26,790 pairs)  [Monkaa dropped]
REM   D_max=192, 10-epoch pretrain + 10-epoch finetune, BF16, num_workers=0
REM   2x2 ablation: {3d,2d} x {context off/on}
REM Order: 2D configs first (faster), 3D last.
REM ~2.5 days total. Survives nothing if window closes -> logs to .log files.
REM ===========================================================================
set PY="C:\Users\canhh\miniconda3\envs\stereo\python.exe"
set UW="C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo"
REM SceneFlow now on the reliable INTERNAL drive (FT3D TRAIN/A + Driving subset).
REM Was D:\SCENEFLOW (external USB enclosure kept dropping under sustained load).
set SF="C:\SceneFlow"
cd /d "C:\Users\canhh\Workspace\code\stereo-uw"

echo ===== agg2d (2D, no context) =====
%PY% train.py    --config configs\agg2d.yaml     --sceneflow-root %SF% --epochs 10 --precision bf16 > log_agg2d_pre.log 2>&1
%PY% finetune.py --config configs\agg2d.yaml     --uwstereo-root %UW% --pretrained runs\agg2d-pretrain\best.ckpt --epochs 10 --precision bf16 > log_agg2d_ft.log 2>&1

echo ===== agg2d_ctx (2D, context) =====
%PY% train.py    --config configs\agg2d_ctx.yaml --sceneflow-root %SF% --epochs 10 --precision bf16 > log_agg2dctx_pre.log 2>&1
%PY% finetune.py --config configs\agg2d_ctx.yaml --uwstereo-root %UW% --pretrained runs\agg2d_ctx-pretrain\best.ckpt --epochs 10 --precision bf16 > log_agg2dctx_ft.log 2>&1

echo ===== ref (3D, no context) =====
%PY% train.py    --config configs\ref.yaml       --sceneflow-root %SF% --epochs 10 --precision bf16 > log_ref_pre.log 2>&1
%PY% finetune.py --config configs\ref.yaml       --uwstereo-root %UW% --pretrained runs\ref-pretrain\best.ckpt --epochs 10 --precision bf16 > log_ref_ft.log 2>&1

echo ===== ref_ctx (3D, context) =====
%PY% train.py    --config configs\ref_ctx.yaml   --sceneflow-root %SF% --epochs 10 --precision bf16 > log_refctx_pre.log 2>&1
%PY% finetune.py --config configs\ref_ctx.yaml   --uwstereo-root %UW% --pretrained runs\ref_ctx-pretrain\best.ckpt --epochs 10 --precision bf16 > log_refctx_ft.log 2>&1

echo ===== MATRIX DONE =====
pause
