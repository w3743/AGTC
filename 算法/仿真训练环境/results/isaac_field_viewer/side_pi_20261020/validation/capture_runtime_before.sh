#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUT="$ROOT/results/isaac_field_viewer/side_pi_20261020/runtime_8192"
HELPER="$ROOT/results/isaac_field_viewer/minimal_inputs_20261020/capture_frame_tensors.py"
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing runtime capture directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
cd "$ROOT"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$HELPER" \
  --base-url http://127.0.0.1:8770 \
  --out-dir "$OUT" \
  --prefix before > "$OUT/capture_before.log" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" > "$OUT/capture_before_exit.txt"
exit "$rc"
