#!/bin/bash
# Keep stderr visible. A previous version piped through grep and swallowed the
# traceback, so every run "finished" with no result and nothing said why.
set -u
cd "$(dirname "$0")"
PY="/c/Users/canhh/miniconda3/envs/stereo/python.exe"
for n in 0 1 5 10 25 50; do
  echo "=============== N=$n ==============="
  PYTHONUNBUFFERED=1 "$PY" fewshot_uw.py --n "$n" --eval-limit 600 2>&1 \
    | grep -viE "UserWarning|warnings.warn|meshgrid|return _VF|_Reduction|size_average"
done
echo "=============== FEWSHOT DONE ==============="
