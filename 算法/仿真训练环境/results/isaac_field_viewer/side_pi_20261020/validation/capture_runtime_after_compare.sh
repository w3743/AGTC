#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUT="$ROOT/results/isaac_field_viewer/side_pi_20261020/runtime_8192"
CAPTURE="$ROOT/results/isaac_field_viewer/minimal_inputs_20261020/capture_frame_tensors.py"
COMPARE="$ROOT/results/isaac_field_viewer/minimal_inputs_20261020/compare_frame_tensors.py"
if [[ ! -f "$OUT/frame_before.json" || ! -f "$OUT/tensors_before.json" ]]; then
  echo "before capture is missing: $OUT" >&2
  exit 2
fi
if [[ -e "$OUT/frame_after.json" || -e "$OUT/tensors_after.json" || -e "$OUT/compare_before_after.json" ]]; then
  echo "refusing to overwrite after capture or comparison artifacts: $OUT" >&2
  exit 2
fi
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$CAPTURE" \
  --base-url http://127.0.0.1:8770 \
  --out-dir "$OUT" \
  --prefix after > "$OUT/capture_after.log" 2>&1
capture_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$capture_rc" > "$OUT/capture_after_exit.txt"
if [[ "$capture_rc" -ne 0 ]]; then
  exit "$capture_rc"
fi
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$COMPARE" \
  --before-dir "$OUT" --before-prefix before \
  --after-dir "$OUT" --after-prefix after \
  --output "$OUT/compare_before_after.json" > "$OUT/compare_before_after.log" 2>&1
compare_rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$compare_rc" > "$OUT/compare_before_after_exit.txt"
exit "$compare_rc"
