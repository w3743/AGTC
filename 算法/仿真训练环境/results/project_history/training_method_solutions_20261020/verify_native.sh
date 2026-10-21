#!/usr/bin/env bash
set -uo pipefail
cd '/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境' || exit 2
OUT='results/project_history/training_method_solutions_20261020'
export RUN_ISAAC_PHYSICS=1
bash scripts/isaaclab_python.sh -m pytest -q tests/test_training_diagnostics_native.py > "$OUT/native_attempt1.log" 2>&1
rc=$?
printf '{"actual_exit_code":%d,"native_physics":true,"formal_training_started":false}\n' "$rc" > "$OUT/native_attempt1.exit.json"
tail -n 45 "$OUT/native_attempt1.log"
exit "$rc"
