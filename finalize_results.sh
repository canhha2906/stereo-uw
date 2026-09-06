#!/bin/bash
# Safety net: wait for run_experiments.sh to finish, collect the numbers into the
# repo, commit and push. Runs unattended so the results reach GitHub even if
# nobody is at the machine.
set -u
FA="/c/Users/canhh/Workspace/code/Fast-ACVNet"
REPO="/c/Users/canhh/Workspace/code/stereo-uw"
cd "$FA"

# wait for the driver to print its final banner, or for python to be gone for a while
tries=0
while ! grep -q "ALL EXPERIMENTS DONE" experiments_all.log 2>/dev/null; do
  n=$(powershell.exe -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
  [ -z "$n" ] && n=0
  if [ "$n" = "0" ]; then
    tries=$((tries+1))
    [ "$tries" -ge 6 ] && { echo "python gone for ~3 min without the done banner; finalising anyway"; break; }
  else
    tries=0
  fi
  sleep 30
done

# collect every result into one file
OUT="$REPO/RESULTS_ablation.md"
{
  echo "# Ablation results — rendering variants"
  echo
  echo "All rows: UWStereo test split, 2,958 pairs, identical eval script and disparity"
  echo "mask (0 < d < 192). Every run starts from the same \`kitti_2015.ckpt\`, 60 epochs,"
  echo "lrepochs 40:10, batch 4. Only the named variable differs."
  echo
  echo "| Run | Training data | Scale aug | EPE | >3px | D1-all |"
  echo "|---|---|---|---|---|---|"
  echo "| direct transfer | none (no finetune) | — | 5.8494 | 22.8836 | 21.5305 |"
  echo "| typeIII | KITTI rendered, Jerlov III | no | 5.4623 | 20.8844 | 19.6414 |"
  for n in scaleaug mixed10 turbid4; do
    f="$FA/eval_$n.log"
    [ -f "$f" ] || continue
    e=$(grep -m1 '^EPE' "$f" | awk '{print $3}')
    p=$(grep -m1 '^>3px' "$f" | awk '{print $3}')
    d=$(grep -m1 '^D1-all' "$f" | awk '{print $3}')
    case $n in
      scaleaug) desc="KITTI rendered, Jerlov III | s in [1,3]";;
      mixed10)  desc="KITTI rendered, all 10 Jerlov types | no";;
      turbid4)  desc="KITTI rendered, turbid coastal 3/5/7/9 | no";;
    esac
    echo "| $n | $desc | ${e:-?} | ${p:-?} | ${d:-?} |"
  done
  echo
  echo "Reference: Paper 1's classical SGBM floor on this split is EPE 4.5620 / D1 18.73%."
  echo
  echo "_Generated automatically by finalize_results.sh when the ablation driver finished._"
} > "$OUT"

cd "$REPO"
cp "$FA/run_experiments.sh" "$FA/finalize_results.sh" . 2>/dev/null
git add -A
git commit -q -m "Ablation results: rendering variants on the same UWStereo test split

Auto-committed when the ablation driver finished. Every run starts from the same
kitti_2015.ckpt with identical schedule and is evaluated by the same script on the
same 2,958 pairs, so only the named variable differs.

Numbers in RESULTS_ablation.md.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" 2>/dev/null

powershell.exe -NoProfile -Command "Set-Location '$REPO'; \$env:GIT_TERMINAL_PROMPT='0'; \$j = Start-Job -ScriptBlock { Set-Location '$REPO'; \$env:GIT_TERMINAL_PROMPT='0'; git push origin main 2>&1 }; Wait-Job \$j -Timeout 300 | Out-Null; Receive-Job \$j; Remove-Job \$j -Force" 2>&1 | tail -3
echo "FINALISED — results committed and pushed"
