#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"
set +e
bash -n "$SCRIPT_DIR/run_checkpoint_validation.sh" > "$SCRIPT_DIR/static_preflight.log" 2>&1
bash_rc=$?
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$SCRIPT_DIR/validate_side_pi_checkpoint.py" --help > "$SCRIPT_DIR/validator_help.log" 2>&1
help_rc=$?
set -e
printf 'runner_bash_n_exit=%s\nvalidator_help_exit=%s\n' "$bash_rc" "$help_rc" > "$SCRIPT_DIR/static_preflight_exit.txt"
if (( bash_rc != 0 || help_rc != 0 )); then exit 1; fi
