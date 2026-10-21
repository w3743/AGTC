#!/usr/bin/env bash
set +e
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
cd "$ROOT"
RUN_ISAAC_PHYSICS=1 ./scripts/isaaclab_python.sh -m pytest -q -s tests/test_front_guide_native.py -k passive_front > "$ROOT/results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt4_native_mapping_diagnosis/pytest.log" 2>&1
pytest_rc=$?
printf '{"command":"RUN_ISAAC_PHYSICS=1 ./scripts/isaaclab_python.sh -m pytest -q -s tests/test_front_guide_native.py -k passive_front","pytest_exit_code":%d,"actual_exit_code":%d}\n' "$pytest_rc" "$pytest_rc" > "$ROOT/results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt4_native_mapping_diagnosis/actual_exit_code.json"
cat "$ROOT/results/isaac_field_diagnosis/front_guide_coupled_20261020/attempt4_native_mapping_diagnosis/pytest.log"
exit "$pytest_rc"