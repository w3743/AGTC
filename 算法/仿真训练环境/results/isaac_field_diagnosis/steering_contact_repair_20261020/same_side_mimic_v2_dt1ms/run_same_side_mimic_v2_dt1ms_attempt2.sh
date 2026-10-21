#!/usr/bin/env bash
set -uo pipefail
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
OUTPUT_ROOT="$PROBE_DIR/contact_probe_runs_dt1ms"
CASE_NAME="dt-1ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt2"
OUTPUT_DIR="$OUTPUT_ROOT/$CASE_NAME"
LOG_PATH="$OUTPUT_ROOT/$CASE_NAME.log"
EXIT_PATH="$OUTPUT_ROOT/$CASE_NAME.exit.json"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior output/log/exit status: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
if ! (set -o noclobber; : >"$LOG_PATH") 2>/dev/null; then
  printf 'Refusing to overwrite prior log: %s\n' "$LOG_PATH" >&2
  exit 2
fi
cd "$ROOT_DIR"
set +e
RUN_CONTACT_MIMIC_V2_DT1MS=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_mimic_v2_dt1ms_runner_attempt2.py" \
  --physics-dt 0.001 \
  --cone elliptic \
  --impratio 1 \
  --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"attempt":2,"physics_dt_s":0.001,"decimation":50,"cone":"elliptic","impratio":1,"mu":0.8,"kf":1000,"eq_solref":[0.004,1],"eq_solimp":[0.999,0.9999,0.001,0.5,2],"output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
