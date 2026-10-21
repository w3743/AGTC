#!/usr/bin/env bash
set -uo pipefail

FIELD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="$FIELD_ROOT/results/isaac_field_viewer/minimal_inputs_20261020"
PYTEST_LOG="$OUT_DIR/native_physics_pytest.log"
PROBE_LOG="$OUT_DIR/native_feedback_probe.log"

if [[ "${RUN_ISAAC_PHYSICS:-}" != "1" ]]; then
    printf 'Refusing native acceptance without RUN_ISAAC_PHYSICS=1.\n' >&2
    exit 2
fi
if [[ -e "$PYTEST_LOG" || -e "$PROBE_LOG" ]]; then
    printf 'Refusing to overwrite existing acceptance logs in %s.\n' "$OUT_DIR" >&2
    exit 2
fi

cd -- "$FIELD_ROOT"
printf 'Native acceptance started: %s\n' "$(date -Is)" > "$PYTEST_LOG"
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh -m pytest -q \
    tests/test_env_integration.py::test_single_channel_depth_passes_official_sb3_wrapper_without_rescaling \
    tests/test_env_integration.py::test_native_gimbal_imu_and_drive_feedback_use_physical_signals \
    tests/test_env_integration.py::test_native_direct_track_commands_are_independent_limited_and_accelerated \
    tests/test_env_integration.py::test_reset_randomizes_clear_spawn_and_matches_live_plant_and_camera_state \
    tests/test_env_integration.py::test_two_env_autoreset_returns_fresh_first_observations \
    tests/test_env_integration.py::test_validated_goto_reset_keeps_the_exact_validated_start_goal_pair \
    >> "$PYTEST_LOG" 2>&1
pytest_exit=$?
printf '\npytest_exit=%s\n' "$pytest_exit" >> "$PYTEST_LOG"
tail -n 14 "$PYTEST_LOG"
if (( pytest_exit != 0 )); then
    printf 'Skipping follow-up probe because the native pytest selection failed.\n' >> "$PYTEST_LOG"
    exit "$pytest_exit"
fi

printf 'Native feedback probe started: %s\n' "$(date -Is)" > "$PROBE_LOG"
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh \
    results/isaac_field_viewer/minimal_inputs_20261020/native_feedback_probe.py \
    >> "$PROBE_LOG" 2>&1
probe_exit=$?
printf '\nprobe_exit=%s\n' "$probe_exit" >> "$PROBE_LOG"
tail -n 14 "$PROBE_LOG"
exit "$probe_exit"
