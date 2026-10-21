#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../.." && pwd)"
DT="${1:-}"
CONE="${2:-pyramidal}"
IMPRATIO="${3:-1}"
RUN_LABEL="${4:-}"
case "$DT" in
  0.005) DT_TAG="5ms" ;;
  0.002) DT_TAG="2ms" ;;
  0.001) DT_TAG="1ms" ;;
  0.0005) DT_TAG="0p5ms" ;;
  *) printf 'Usage: RUN_CONTACT_FRICTION_V3=1 %s {0.005|0.002|0.001|0.0005} [pyramidal|elliptic] [positive-impratio] [unique-run-label]\n' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
case "$CONE" in
  pyramidal|elliptic) ;;
  *) printf 'Cone must be pyramidal or elliptic.\n' >&2; exit 2 ;;
esac
if [[ ! "$IMPRATIO" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || ! awk -v value="$IMPRATIO" 'BEGIN { exit !(value > 0) }'; then
  printf 'impratio must be a positive finite decimal.\n' >&2
  exit 2
fi
IMPRATIO_TAG="${IMPRATIO//./p}"
CASE_NAME="dt-${DT_TAG}__cone-${CONE}__impratio-${IMPRATIO_TAG}"
if [[ -n "$RUN_LABEL" ]]; then
  if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]]; then
    printf 'Run label must be 1-32 ASCII letters, digits, underscore, or hyphen.\n' >&2
    exit 2
  fi
  CASE_NAME="${CASE_NAME}__attempt-${RUN_LABEL}"
fi

if [[ "${RUN_CONTACT_FRICTION_V3:-}" != "1" ]]; then
  printf 'Runner requires RUN_CONTACT_FRICTION_V3=1 after parent authorization.\n' >&2
  exit 2
fi

OUTPUT_ROOT="$PROBE_DIR/contact_probe_runs_v3"
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
RUN_CONTACT_FRICTION_V3=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_friction_v3.py" \
  --physics-dt "$DT" \
  --cone "$CONE" \
  --impratio "$IMPRATIO" \
  --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"physics_dt_s":%s,"cone":"%s","impratio":%s,"run_label":"%s","case":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$DT" "$CONE" "$IMPRATIO" "$RUN_LABEL" "$CASE_NAME" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
