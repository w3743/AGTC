#!/usr/bin/env bash
set +e
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
cd "$ROOT"
./scripts/isaaclab_python.sh -m pytest -q tests/test_newton_drive.py tests/test_front_guide_cli.py
pytest_rc=$?
./scripts/isaaclab_python.sh -m compileall -q isaac_field/newton_drive.py isaac_field/env.py tests/test_newton_drive.py tests/test_front_guide_native.py tests/test_env_integration.py
compile_rc=$?
printf '{"pytest_exit_code":%d,"compileall_exit_code":%d,"actual_exit_code":%d}\n' "$pytest_rc" "$compile_rc" "$(( pytest_rc != 0 || compile_rc != 0 ))" > "$ROOT/results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt1_cpu/actual_exit_code.json"
if (( pytest_rc != 0 || compile_rc != 0 )); then exit 1; fi
exit 0