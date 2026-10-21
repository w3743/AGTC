#!/usr/bin/env bash
set -uo pipefail
cd '/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境' || exit 2
OUT='results/project_history/training_method_solutions_20261020'
export CUDA_VISIBLE_DEVICES=''
bash scripts/isaaclab_python.sh -m pytest -q tests > "$OUT/cpu_all_repaired_final.log" 2>&1
test_rc=$?
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests > "$OUT/compileall_repaired_final.log" 2>&1
compile_rc=$?
bash -n scripts/run_isaac_mastery.sh
shell_rc=$?
printf '{"pytest_actual_exit_code":%d,"compileall_actual_exit_code":%d,"bash_n_actual_exit_code":%d,"cpu_only":true}\n' "$test_rc" "$compile_rc" "$shell_rc" > "$OUT/repaired_final_verification_actual_exit.json"
tail -n 12 "$OUT/cpu_all_repaired_final.log"
cat "$OUT/repaired_final_verification_actual_exit.json"
if [[ $test_rc -ne 0 || $compile_rc -ne 0 || $shell_rc -ne 0 ]]; then exit 1; fi
