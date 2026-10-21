#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/fast_stop_objective_20261020"
LABEL="${1:-attempt1}"
case "$LABEL" in attempt[0-9]*) ;; *) exit 79 ;; esac
cd "$ROOT" || exit 80
if [[ -e "$OUT/native_${LABEL}.log" ]]; then exit 81; fi
export RUN_ISAAC_PHYSICS=1
bash scripts/isaaclab_python.sh -m pytest -q tests/test_env_integration.py tests/test_front_guide_native.py tests/test_training_diagnostics_native.py > "$OUT/native_${LABEL}.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/native_${LABEL}.exit_code"
printf '{"actual_python_exit":%s,"source":"parent_shell_waited_for_child"}\n' "$rc" > "$OUT/native_${LABEL}_actual_exit.json"
exit "$rc"
