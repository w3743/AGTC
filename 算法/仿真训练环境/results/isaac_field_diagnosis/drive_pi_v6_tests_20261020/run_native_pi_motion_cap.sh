#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/isaac_field_diagnosis/drive_pi_v6_tests_20261020"
cd "$ROOT"
set +e
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh -m pytest -q tests/test_env_integration.py::test_native_pi_drive_moves_and_effort_limit_caps_each_newton_physics_iteration > "$OUT/native_pi_motion_cap.log" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" > "$OUT/native_pi_motion_cap_exit.txt"
exit "$rc"