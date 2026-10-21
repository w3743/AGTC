#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VALIDATOR="$SCRIPT_DIR/validate_side_pi_final_checkpoint.py"
RUNNER="$SCRIPT_DIR/run_final_checkpoint_validation.sh"
LOG="$SCRIPT_DIR/final_validator_preflight.log"
EXIT_FILE="$SCRIPT_DIR/final_validator_preflight.exit.txt"
if [[ -e "$LOG" || -e "$EXIT_FILE" ]]; then
  echo "refusing to overwrite preflight evidence in $SCRIPT_DIR" >&2
  exit 2
fi

cd "$ROOT"
set +e
(
  bash -n "$RUNNER"
  runner_rc=$?
  printf 'runner_bash_n_exit=%s\n' "$runner_rc"
  bash -n "$SCRIPT_DIR/run_final_validator_preflight.sh"
  preflight_rc=$?
  printf 'preflight_bash_n_exit=%s\n' "$preflight_rc"
  CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$VALIDATOR" --help
  validator_help_rc=$?
  printf 'validator_help_exit=%s\n' "$validator_help_rc"
  PYTHONPYCACHEPREFIX=/tmp/side_pi_final_validator_pycache CUDA_VISIBLE_DEVICES= \
    bash scripts/isaaclab_python.sh -m py_compile "$VALIDATOR"
  py_compile_rc=$?
  printf 'validator_py_compile_exit=%s\n' "$py_compile_rc"
  if [[ "$runner_rc" -ne 0 || "$preflight_rc" -ne 0 || "$validator_help_rc" -ne 0 || "$py_compile_rc" -ne 0 ]]; then
    exit 1
  fi
  exit 0
) > "$LOG" 2>&1
rc=$?
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" > "$EXIT_FILE"
exit "$rc"
