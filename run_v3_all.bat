@echo off
REM Sequential V3 training pipeline: 1/16 first, then 1/8.
REM ~40-50 hours total. Each command waits for the previous.
REM If any step fails, the rest still run. Comment lines out if you want to skip.

set PY="C:\Users\canhh\miniconda3\envs\stereo\python.exe"
set UW="C:\Users\canhh\Workspace\conference paper, computer vision\data set\UWStereo"
set SF="D:\SCENEFLOW"

cd /d "C:\Users\canhh\Workspace\code\stereo-uw"

echo === V3 1/16 3D pretrain ===
%PY% train.py --config configs\v3_r16_3d.yaml --sceneflow-root %SF% --epochs 10

echo === V3 1/16 3D finetune ===
%PY% finetune.py --config configs\v3_r16_3d.yaml --uwstereo-root %UW% --pretrained runs\v3_r16_3d-pretrain\best.ckpt --epochs 10

echo === V3 1/16 2D pretrain ===
%PY% train.py --config configs\v3_r16_2d.yaml --sceneflow-root %SF% --epochs 10

echo === V3 1/16 2D finetune ===
%PY% finetune.py --config configs\v3_r16_2d.yaml --uwstereo-root %UW% --pretrained runs\v3_r16_2d-pretrain\best.ckpt --epochs 10

echo === V3 1/8 3D pretrain ===
%PY% train.py --config configs\v3_r8_3d.yaml --sceneflow-root %SF% --epochs 10

echo === V3 1/8 3D finetune ===
%PY% finetune.py --config configs\v3_r8_3d.yaml --uwstereo-root %UW% --pretrained runs\v3_r8_3d-pretrain\best.ckpt --epochs 10

echo === V3 1/8 2D pretrain ===
%PY% train.py --config configs\v3_r8_2d.yaml --sceneflow-root %SF% --epochs 10

echo === V3 1/8 2D finetune ===
%PY% finetune.py --config configs\v3_r8_2d.yaml --uwstereo-root %UW% --pretrained runs\v3_r8_2d-pretrain\best.ckpt --epochs 10

echo === ALL DONE ===
pause
