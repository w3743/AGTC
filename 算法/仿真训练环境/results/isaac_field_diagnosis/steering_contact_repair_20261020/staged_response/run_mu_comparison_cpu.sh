#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
SCRIPT="$PROBE_DIR/summarize_mu_comparison.py"
LOG_PATH="$PROBE_DIR/mu_comparison.cpu.log"
EXIT_PATH="$PROBE_DIR/mu_comparison.exit.json"
if [[ $# -ne 0 ]]; then
  printf 'CPU comparison runner accepts no arguments.\n' >&2
  exit 2
fi
if [[ -e "$PROBE_DIR/mu_comparison.json" || -e "$PROBE_DIR/mu_comparison.md" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite an existing comparison artifact.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
set +e
python3 "$SCRIPT" >>"$LOG_PATH" 2>&1
script_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$script_rc" >>"$LOG_PATH"
script_sha="$(sha256sum "$SCRIPT" | cut -d' ' -f1)"
(set -o noclobber; printf '{"actual_exit_code":%s,"script_sha256":"%s","json":"%s/mu_comparison.json","markdown":"%s/mu_comparison.md","gpu_used":false}\n' \
  "$script_rc" "$script_sha" "$PROBE_DIR" "$PROBE_DIR" >"$EXIT_PATH")
exit "$script_rc"
