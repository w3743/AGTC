#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
LOG_PATH="$OUT_DIR/cpu_math_test.log"

if [[ -e "$LOG_PATH" ]]; then
  printf 'Refusing to overwrite %s\n' "$LOG_PATH" >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q \
  "$OUT_DIR/test_pi_counterfactual_math.py" >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
exit "$rc"
