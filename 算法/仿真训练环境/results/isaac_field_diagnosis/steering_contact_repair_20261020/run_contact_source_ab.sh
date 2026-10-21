#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../.." && pwd)"
CONTACT_SOURCE="${1:-}"

if [[ "${RUN_CONTACT_SOURCE_AB:-}" != "1" ]]; then
  printf 'Prepared runner requires RUN_CONTACT_SOURCE_AB=1 after explicit authorization.\n' >&2
  exit 2
fi
if [[ "$CONTACT_SOURCE" != "newton" && "$CONTACT_SOURCE" != "mujoco" ]]; then
  printf 'Usage: RUN_CONTACT_SOURCE_AB=1 %s {newton|mujoco}\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

OUTPUT_DIR="$PROBE_DIR/contact_probe_runs/${CONTACT_SOURCE}_v1"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
if [[ "$OUTPUT_DIR" != "$PROBE_DIR/contact_probe_runs/${CONTACT_SOURCE}_v1" ]]; then
  printf 'Refusing unexpected output path: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
LOG_PATH="${OUTPUT_DIR}.log"
EXIT_PATH="$OUTPUT_DIR/contact_source_ab_exit.json"
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
RUN_CONTACT_SOURCE_AB=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_source_ab_probe.py" \
  --contact-source "$CONTACT_SOURCE" \
  --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"contact_source":"%s","output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$CONTACT_SOURCE" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
