#!/usr/bin/env bash
set -uo pipefail

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROBE_DIR/../../.." && pwd)"
OUTPUT_DIR="${CONTACT_TRANSFER_OUTPUT_DIR:-}"

if [[ "${RUN_CONTACT_TRANSFER_PROBE:-}" != "1" ]]; then
  printf 'Prepared only; wait for explicit parent authorization before native/GPU execution.\n' >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  printf 'Set CONTACT_TRANSFER_OUTPUT_DIR to a new directory under contact_probe_runs/.\n' >&2
  exit 2
fi

OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
if [[ "$OUTPUT_DIR" != "$ROOT_DIR/results/isaac_field_diagnosis/side_pi_physics_gap_20261020/contact_probe_runs/"* ]]; then
  printf 'Refusing output outside contact_probe_runs/: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
LOG_PATH="${OUTPUT_DIR}.log"
EXIT_PATH="$OUTPUT_DIR/contact_transfer_probe_exit.json"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior probe output/log/status: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
if ! mkdir -- "$OUTPUT_DIR"; then
  printf 'Could not create fresh probe output directory: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
if ! (set -o noclobber; : >"$LOG_PATH") 2>/dev/null; then
  printf 'Refusing to overwrite prior probe log: %s\n' "$LOG_PATH" >&2
  rmdir "$OUTPUT_DIR" 2>/dev/null || true
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_CONTACT_TRANSFER_PROBE=1 scripts/isaaclab_python.sh \
  "$PROBE_DIR/contact_transfer_probe.py" --output-dir "$OUTPUT_DIR" \
  >>"$LOG_PATH" 2>&1
probe_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$probe_rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"output_dir":"%s","log_path":"%s"}\n' \
  "$probe_rc" "$OUTPUT_DIR" "$LOG_PATH" >"$EXIT_PATH")
exit "$probe_rc"
