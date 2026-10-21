#!/usr/bin/env bash
set -uo pipefail

FIELD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
attempt=1
while [[ -e "$OUT_DIR/diagnostic_run_attempt${attempt}.log" ]]; do
    attempt=$((attempt + 1))
done
LOG="$OUT_DIR/diagnostic_run_attempt${attempt}.log"

cd "$FIELD_ROOT"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh \
    "$OUT_DIR/diagnose_trial.py" >"$LOG" 2>&1
diagnostic_exit=$?
printf '\ndiagnostic_exit_code=%d\n' "$diagnostic_exit" >>"$LOG"
set -e
printf 'diagnostic_exit_code=%d log=%s\n' "$diagnostic_exit" "$LOG"
exit "$diagnostic_exit"
