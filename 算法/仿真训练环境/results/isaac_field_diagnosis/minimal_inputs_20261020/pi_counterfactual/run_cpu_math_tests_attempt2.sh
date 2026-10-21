#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
LOG_PATH="$OUT_DIR/cpu_math_test_attempt2.log"
STATUS_PATH="$OUT_DIR/cpu_math_test_attempt2_exit.json"
if [[ -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite attempt-2 CPU validation evidence\n' >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q \
  "$OUT_DIR/test_pi_counterfactual_math.py" >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"actual_exit_code":%s}\n' "$rc" >"$STATUS_PATH"
exit "$rc"
