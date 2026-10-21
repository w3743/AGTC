#!/usr/bin/env bash
set -uo pipefail
cd '/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境' || exit 2
OUT='results/project_history/training_method_solutions_20261020'
export CUDA_VISIBLE_DEVICES=''
bash scripts/isaaclab_python.sh -m pytest -q tests/test_task.py tests/test_isaac_field_learning.py tests/test_training_observability.py > "$OUT/targeted_attempt1.log" 2>&1
rc=$?
printf '{"actual_exit_code":%d,"cpu_only":true}\n' "$rc" > "$OUT/targeted_attempt1.exit.json"
cat "$OUT/targeted_attempt1.log"
exit "$rc"
