#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/lstm512_experiment_20261020"
cd "$ROOT" || exit 80
if [[ -e "$OUT/cpu_tests.log" ]]; then exit 81; fi
export CUDA_VISIBLE_DEVICES=""
unset RUN_ISAAC_PHYSICS
bash scripts/isaaclab_python.sh -m pytest -q tests > "$OUT/cpu_tests.log" 2>&1
tests_rc=$?
printf '%s\n' "$tests_rc" > "$OUT/cpu_tests.exit_code"
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts > "$OUT/compileall.log" 2>&1
compile_rc=$?
printf '%s\n' "$compile_rc" > "$OUT/compileall.exit_code"
bash -n "$OUT/prepare_experiment.sh" "$OUT/launch_experiment.sh" "$OUT/run_cpu_checks.sh" > "$OUT/bash_syntax.log" 2>&1
bash_rc=$?
printf '%s\n' "$bash_rc" > "$OUT/bash_syntax.exit_code"
printf '{"pytest_exit":%s,"compileall_exit":%s,"bash_syntax_exit":%s,"source":"parent_shell_waited_for_children"}\n' "$tests_rc" "$compile_rc" "$bash_rc" > "$OUT/cpu_actual_exit.json"
if [[ "$tests_rc" -ne 0 ]]; then exit "$tests_rc"; fi
if [[ "$compile_rc" -ne 0 ]]; then exit "$compile_rc"; fi
exit "$bash_rc"
