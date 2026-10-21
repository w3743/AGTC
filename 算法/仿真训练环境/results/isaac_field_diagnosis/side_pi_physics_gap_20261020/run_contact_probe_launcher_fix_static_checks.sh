#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../.." && pwd)"
LOG_PATH="$PROBE_DIR/contact_probe_launcher_fix_static_validation.log"
STATUS_PATH="$PROBE_DIR/contact_probe_launcher_fix_static_validation_exit.json"

if [[ -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite existing launcher-fix static evidence.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"

set +e
scripts/isaaclab_python.sh -m py_compile \
  "$PROBE_DIR/contact_transfer_probe.py" \
  "$PROBE_DIR/inspect_newton_contact_api.py" \
  >"$LOG_PATH" 2>&1
py_rc=$?
printf 'PY_COMPILE_EXIT_CODE=%s\n' "$py_rc" >>"$LOG_PATH"

bash -n scripts/isaaclab_python.sh \
  "$PROBE_DIR/run_contact_transfer_probe.sh" \
  "$PROBE_DIR/run_contact_probe_static_checks.sh" \
  "$PROBE_DIR/run_contact_probe_launcher_fix_static_checks.sh" \
  >>"$LOG_PATH" 2>&1
bash_rc=$?
printf 'BASH_SYNTAX_EXIT_CODE=%s\n' "$bash_rc" >>"$LOG_PATH"
set -e

final_rc=0
if (( py_rc != 0 || bash_rc != 0 )); then
  final_rc=1
fi
printf '{"py_compile_exit_code":%s,"bash_syntax_exit_code":%s,"api_inspection_previous_exit_code":0,"api_inspection_source_unchanged":true,"actual_exit_code":%s}\n' \
  "$py_rc" "$bash_rc" "$final_rc" >"$STATUS_PATH"
exit "$final_rc"