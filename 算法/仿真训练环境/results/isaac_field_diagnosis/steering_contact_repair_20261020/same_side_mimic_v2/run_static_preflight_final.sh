#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
LOG_PATH="$PROBE_DIR/static_cpu_preflight_final.log"
EXIT_PATH="$PROBE_DIR/static_cpu_preflight_final.exit.json"

if [[ -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior static preflight evidence.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
set +e
bash -n "$PROBE_DIR/run_same_side_mimic_v2.sh" >"$LOG_PATH" 2>&1
rc=$?
if [[ "$rc" -eq 0 ]]; then
  scripts/isaaclab_python.sh "$PROBE_DIR/contact_mimic_v2_runner.py" --static-check >>"$LOG_PATH" 2>&1
  rc=$?
fi
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"checks":["bash -n","Newton 1.6.1 public builder schema","CPU model fields","CPU SolverMuJoCo host and MJWarp neq","checked v1 canonical reset-state guard source"]}\n' \
  "$rc" >"$EXIT_PATH")
exit "$rc"
