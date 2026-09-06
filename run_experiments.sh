#!/bin/bash
# Runs the remaining ablations sequentially and evaluates each on the SAME
# UWStereo test split, so every number is comparable. Sequential on purpose:
# two trainings at once contend for the GPU and make the timings meaningless.
set -u
cd "$(dirname "$0")"
PY="/c/Users/canhh/miniconda3/envs/stereo/python.exe"
UW="C:/Users/canhh/Workspace/conference paper, computer vision/data set/UWStereo"
BASE="./pretrained/Fast-ACVNet+/kitti_2015.ckpt"

wait_for_gpu_free() {
  while true; do
    n=$(powershell.exe -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
    [ -z "$n" ] && n=0
    [ "$n" = "0" ] && break
    sleep 20
  done
}

train_and_eval() {           # $1 name  $2 data dir  $3 scale_max
  local name=$1 data=$2 smax=$3
  local logdir="./checkpoints/exp_$name"
  echo "=============== $name  (data=$data  KITTI_SCALE_MAX=$smax) ==============="
  rm -rf "$logdir"
  KITTI_SCALE_MAX="$smax" PYTHONUNBUFFERED=1 "$PY" main_kitti.py \
      --kitti15_datapath "$data" --kitti12_datapath "$data" \
      --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
      --loadckpt "$BASE" --logdir "$logdir" \
      --epochs 60 --lrepochs "40:10" --batch_size 4 --test_batch_size 2 \
      > "train_$name.log" 2>&1
  local ck
  ck=$(ls "$logdir"/*.ckpt 2>/dev/null | sort | tail -1)
  if [ -z "$ck" ]; then echo "$name: NO CHECKPOINT, training failed"; tail -3 "train_$name.log"; return; fi
  echo "$name: trained $(grep -c avg_test_scalars "train_$name.log") epochs, evaluating $ck"
  PYTHONUNBUFFERED=1 "$PY" eval_uwstereo.py --ckpt "$ck" > "eval_$name.log" 2>&1
  grep -E "^N |^EPE|^>3px|^D1-all" "eval_$name.log"
}

echo "waiting for the in-flight scale-aug run to finish ..."
wait_for_gpu_free

# The scale-aug run already trained; just evaluate it.
SA=$(ls ./checkpoints/kitti_uw_III_scaleaug/*.ckpt 2>/dev/null | sort | tail -1)
if [ -n "$SA" ]; then
  echo "=============== scaleaug (already trained) ==============="
  echo "scaleaug: trained $(grep -c avg_test_scalars stage3_scaleaug.log) epochs, evaluating $SA"
  PYTHONUNBUFFERED=1 "$PY" eval_uwstereo.py --ckpt "$SA" > eval_scaleaug.log 2>&1
  grep -E "^N |^EPE|^>3px|^D1-all" eval_scaleaug.log
fi

train_and_eval "mixed10"  "C:/Users/canhh/Workspace/code/Fast-ACVNet/data/kitti15_uw_mixed"  "1.0"
train_and_eval "turbid4"  "C:/Users/canhh/Workspace/code/Fast-ACVNet/data/kitti15_uw_turbid" "1.0"

echo "=============== ALL EXPERIMENTS DONE ==============="
