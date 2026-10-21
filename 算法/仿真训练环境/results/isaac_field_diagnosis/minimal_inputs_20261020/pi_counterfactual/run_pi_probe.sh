#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OUT_DIR/../../../.." && pwd)"
if [[ $# -ne 2 ]]; then
  printf 'usage: %s {capped_p_reference|pi_candidate} {0|0.1}\n' "$0" >&2
  exit 2
fi
label="$1"
ki="$2"
case "$label:$ki" in
  capped_p_reference:0) ;;
  pi_candidate:0.1) ;;
  *) printf 'unexpected probe label/Ki pair: %s %s\n' "$label" "$ki" >&2; exit 2 ;;
esac

OUTPUT_DIR="$OUT_DIR/$label"
LOG_PATH="$OUT_DIR/${label}.log"
STATUS_PATH="$OUT_DIR/${label}_exit.json"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" || -e "$STATUS_PATH" ]]; then
  printf 'Refusing to overwrite existing %s evidence\n' "$label" >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh \
  "$OUT_DIR/pi_counterfactual_probe.py" \
  --ki "$ki" --output-dir "$OUTPUT_DIR" >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"run_label":"%s","ki":%s,"actual_exit_code":%s}\n' \
  "$label" "$ki" "$rc" >"$STATUS_PATH"
exit "$rc"
