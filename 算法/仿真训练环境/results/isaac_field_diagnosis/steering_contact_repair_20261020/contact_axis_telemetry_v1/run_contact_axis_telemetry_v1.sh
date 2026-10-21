#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
if [[ $# -ne 0 ]]; then
  printf 'This telemetry-only runner is pinned to the frozen v4 2 ms / elliptic / impratio 1 case and accepts no arguments.\n' >&2
  exit 2
fi
if [[ "${RUN_CONTACT_AXIS_TELEMETRY_V1:-}" != "1" ]]; then
  printf 'Runner requires RUN_CONTACT_AXIS_TELEMETRY_V1=1 after parent authorization.\n' >&2
  exit 2
fi

OUTPUT_ROOT="$PROBE_DIR/../contact_probe_runs_v4"
CASE_NAME="axis-telemetry-v1-attempt3__baseline__dt-2ms__cone-elliptic__impratio-1__mu-0.8__kf-1000"
OUTPUT_DIR="$OUTPUT_ROOT/$CASE_NAME"
LOG_PATH="$OUTPUT_ROOT/$CASE_NAME.log"
EXIT_PATH="$OUTPUT_ROOT/$CASE_NAME.exit.json"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior output/log/exit status: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
if ! mkdir -- "$OUTPUT_DIR"; then
  printf 'Could not create fresh output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
if ! (set -o noclobber; : >"$LOG_PATH") 2>/dev/null; then
  printf 'Refusing to overwrite prior log: %s\n' "$LOG_PATH" >&2
  rmdir "$OUTPUT_DIR" 2>/dev/null || true
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_CONTACT_AXIS_TELEMETRY_V1=1 RUN_CONTACT_FRICTION_V4=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_axis_telemetry_v1.py" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"physics_dt_s":0.002,"policy_dt_s":0.05,"cone":"elliptic","impratio":1,"mu":0.8,"kf":1000,"run_label":"authorized-telemetry-baseline","case":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$CASE_NAME" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
