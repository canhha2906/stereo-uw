@echo off
REM Full 2x2 matrix: {3d,2d} x {context on/off}, D_max=192, V2 backbone.
REM Pretrain on SceneFlow then finetune on UWStereo, each 10 epochs.
REM Order: cheapest/most-important first (ref = headline), context variants last.

set PY="C:\Users\canhh\miniconda3\envs\stereo\python.exe"
set UW="C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo"
set SF="D:\SCENEFLOW"
cd /d "C:\Users\canhh\Workspace\code\stereo-uw"

echo ===== ref (3D, no context) =====
%PY% train.py    --config configs\ref.yaml       --sceneflow-root %SF% --epochs 10
%PY% finetune.py --config configs\ref.yaml       --uwstereo-root %UW% --pretrained runs\ref-pretrain\best.ckpt --epochs 10

echo ===== agg2d (2D, no context) =====
%PY% train.py    --config configs\agg2d.yaml     --sceneflow-root %SF% --epochs 10
%PY% finetune.py --config configs\agg2d.yaml     --uwstereo-root %UW% --pretrained runs\agg2d-pretrain\best.ckpt --epochs 10

echo ===== ref_ctx (3D, context) =====
%PY% train.py    --config configs\ref_ctx.yaml   --sceneflow-root %SF% --epochs 10
%PY% finetune.py --config configs\ref_ctx.yaml   --uwstereo-root %UW% --pretrained runs\ref_ctx-pretrain\best.ckpt --epochs 10

echo ===== agg2d_ctx (2D, context) =====
%PY% train.py    --config configs\agg2d_ctx.yaml --sceneflow-root %SF% --epochs 10
%PY% finetune.py --config configs\agg2d_ctx.yaml --uwstereo-root %UW% --pretrained runs\agg2d_ctx-pretrain\best.ckpt --epochs 10

echo ===== ALL DONE =====
pause
