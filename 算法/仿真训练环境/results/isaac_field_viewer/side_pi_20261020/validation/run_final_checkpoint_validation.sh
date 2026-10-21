#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VALIDATOR="$SCRIPT_DIR/validate_side_pi_final_checkpoint.py"
PROTECTED_REFERENCE="$SCRIPT_DIR/protected_old_trial_reference.json"

if [[ $# != 3 ]]; then
  echo "usage: $0 <trial_dir> <final_checkpoint.zip> <output_stem>" >&2
  exit 2
fi

TRIAL_DIR="$(realpath -e "$1")"
CHECKPOINT="$(realpath -e "$2")"
OUTPUT_STEM="$3"
if [[ ! "$OUTPUT_STEM" =~ ^[A-Za-z0-9._-]+$ || "$OUTPUT_STEM" == "." || "$OUTPUT_STEM" == ".." ]]; then
  echo "invalid output stem: $OUTPUT_STEM" >&2
  exit 2
fi

BASELINE="$TRIAL_DIR/untrained_baseline.zip"
PROTOCOL="$TRIAL_DIR/protocol.json"
MANIFEST="$TRIAL_DIR/checkpoint_manifest.json"
RUN_STATUS="$TRIAL_DIR/run_status.json"
OUTPUT="$SCRIPT_DIR/$OUTPUT_STEM.json"
LOG="$SCRIPT_DIR/$OUTPUT_STEM.log"
EXIT_FILE="$SCRIPT_DIR/$OUTPUT_STEM.exit.txt"
for path in "$BASELINE" "$PROTOCOL" "$MANIFEST" "$RUN_STATUS" "$PROTECTED_REFERENCE" "$VALIDATOR"; do
  [[ -f "$path" ]] || { echo "required validation input is missing: $path" >&2; exit 2; }
done
for path in "$OUTPUT" "$LOG" "$EXIT_FILE"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite existing output: $path" >&2; exit 2; }
done

cd "$ROOT"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$VALIDATOR" \
  --repo-root "$ROOT" \
  --trial-dir "$TRIAL_DIR" \
  --protocol "$PROTOCOL" \
  --manifest "$MANIFEST" \
  --run-status "$RUN_STATUS" \
  --baseline "$BASELINE" \
  --checkpoint "$CHECKPOINT" \
  --expected-steps 196608 \
  --protected-reference "$PROTECTED_REFERENCE" \
  --output "$OUTPUT" > "$LOG" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" > "$EXIT_FILE"
exit "$rc"
