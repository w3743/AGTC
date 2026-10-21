#!/usr/bin/env bash
set -uo pipefail
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$STAGE_DIR/../../../../" && pwd)"
LOG_PATH="$STAGE_DIR/v7_native_mimic_attempt2.log"
EXIT_PATH="$STAGE_DIR/v7_native_mimic_attempt2.exit.json"
if [[ -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior native evidence.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
unset CUDA_VISIBLE_DEVICES WARP_DISABLE_CUDA
export RUN_ISAAC_PHYSICS=1
set +e
scripts/isaaclab_python.sh -m pytest -q -s tests/test_env_integration.py::test_native_two_environment_same_side_mimic_isolated_and_partial_reset >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"device":"cuda","test":"test_native_two_environment_same_side_mimic_isolated_and_partial_reset","log_path":"%s"}\n' \
  "$rc" "$LOG_PATH" >"$EXIT_PATH")
exit "$rc"
