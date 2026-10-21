#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
LOG_PATH="$OUT_DIR/native_v5_regression_attempt1.log"
STATUS_PATH="$OUT_DIR/native_v5_regression_attempt1_exit.json"
if [[ -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite native v5 regression evidence\n' >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh -m pytest -q \
  tests/test_env_integration.py::test_native_drive_actuator_effort_limit_caps_each_newton_physics_iteration \
  >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"actual_exit_code":%s}\n' "$rc" >"$STATUS_PATH"
exit "$rc"
