#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/isaac_field_diagnosis/drive_pi_v6_tests_20261020"
cd "$ROOT"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q > "$OUT/full_cpu_pytest.log" 2>&1
pytest_rc=$?
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests > "$OUT/compileall.log" 2>&1
compile_rc=$?
bash -n scripts/isaaclab_python.sh
wrapper_syntax_rc=$?
bash -n "$OUT/run_cpu_validation.sh"
runner_syntax_rc=$?
set -e
printf 'pytest_exit=%s\ncompileall_exit=%s\nwrapper_bash_n_exit=%s\nrunner_bash_n_exit=%s\n' "$pytest_rc" "$compile_rc" "$wrapper_syntax_rc" "$runner_syntax_rc" > "$OUT/validation_exit.txt"
if (( pytest_rc != 0 || compile_rc != 0 || wrapper_syntax_rc != 0 || runner_syntax_rc != 0 )); then exit 1; fi