#!/usr/bin/env bash
set -uo pipefail
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
LOG_PATH="$PROBE_DIR/static_cpu_preflight.log"
EXIT_PATH="$PROBE_DIR/static_cpu_preflight.exit.json"
if [[ -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite previous static preflight evidence\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
set +e
scripts/isaaclab_python.sh "$PROBE_DIR/contact_mimic_v2_dt1ms_runner.py" --static-check >"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"checks":["bash -n","Python compile","Newton 1.6.1 public builder schema","CPU model fields","CPU SolverMuJoCo host and MJWarp neq","1ms in-memory gate","fixed-contract and initial-state guards"]}\n' "$probe_rc" >"$EXIT_PATH")
exit "$probe_rc"
