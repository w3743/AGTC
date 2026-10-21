#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../../.." && pwd)"
PROBE="$PROBE_DIR/staged_motion_response_probe.py"
OUTPUT_ROOT="$PROBE_DIR/runs"
CASE_NAME="mu_0_4_5ms_pyramidal_1_kf1000"
OUTPUT_DIR="$OUTPUT_ROOT/$CASE_NAME"
LOG_PATH="$OUTPUT_ROOT/$CASE_NAME.log"
EXIT_PATH="$OUTPUT_ROOT/$CASE_NAME.exit.json"
DT="0.005"
CONE="pyramidal"
IMPRATIO="1"
SHAPE_MU="0.4"
EXPECTED_KF="1000"

if [[ "${RUN_STAGED_RESPONSE_MU:-}" != "1" ]]; then
  printf 'Requires RUN_STAGED_RESPONSE_MU=1 after parent authorizes the GPU run.\n' >&2
  exit 2
fi
if [[ $# -ne 0 ]]; then
  printf 'Pinned friction runner accepts no arguments.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to reuse output, log, or exit record: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
if ! mkdir -- "$OUTPUT_DIR"; then
  printf 'Could not reserve fresh output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
if ! (set -o noclobber; : >"$LOG_PATH") 2>/dev/null; then
  printf 'Refusing to overwrite prior log: %s\n' "$LOG_PATH" >&2
  rmdir "$OUTPUT_DIR" 2>/dev/null || true
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_STAGED_RESPONSE_PROBE=1 scripts/isaaclab_python.sh \
  "$PROBE" \
  --physics-dt "$DT" \
  --cone "$CONE" \
  --impratio "$IMPRATIO" \
  --shape-mu "$SHAPE_MU" \
  --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"physics_dt_s":%s,"policy_dt_s":0.05,"decimation":10,"cone":"%s","impratio":%s,"shape_mu":%s,"expected_kf":%s,"case":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$DT" "$CONE" "$IMPRATIO" "$SHAPE_MU" "$EXPECTED_KF" "$CASE_NAME" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
