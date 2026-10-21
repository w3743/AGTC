#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../.." && pwd)"
DT="${1:-}"
case "$DT" in
  0.005) CASE_NAME="newton_5ms_v2" ;;
  0.002) CASE_NAME="newton_2ms_v2" ;;
  0.001) CASE_NAME="newton_1ms_v2" ;;
  *) printf 'Usage: RUN_CONTACT_DT_CONVERGENCE=1 %s {0.005|0.002|0.001}\n' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac

if [[ "${RUN_CONTACT_DT_CONVERGENCE:-}" != "1" ]]; then
  printf 'Runner requires RUN_CONTACT_DT_CONVERGENCE=1 after parent authorization.\n' >&2
  exit 2
fi

OUTPUT_DIR="$PROBE_DIR/contact_probe_runs_v2/$CASE_NAME"
LOG_PATH="$PROBE_DIR/contact_probe_runs_v2/$CASE_NAME.log"
EXIT_PATH="$PROBE_DIR/contact_probe_runs_v2/$CASE_NAME.exit.json"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior output/log/exit status: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
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
RUN_CONTACT_DT_CONVERGENCE=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_dt_convergence_v2.py" \
  --physics-dt "$DT" \
  --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"physics_dt_s":%s,"case":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$DT" "$CASE_NAME" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
