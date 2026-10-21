#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VALIDATOR="$SCRIPT_DIR/validate_side_pi_checkpoint.py"
PROTECTED_REFERENCE="$SCRIPT_DIR/protected_old_trial_reference.json"

if [[ $# != 6 ]]; then
  echo "usage: $0 <trial_dir> <baseline.zip> <checkpoint.zip|-> <protocol_sha256> <baseline_sha256> <output_stem>" >&2
  exit 2
fi

TRIAL_DIR="$(realpath -e "$1")"
BASELINE="$(realpath -e "$2")"
CHECKPOINT="$3"
EXPECTED_PROTOCOL_SHA256="$4"
EXPECTED_BASELINE_SHA256="$5"
OUTPUT_STEM="$6"

if [[ ! "$OUTPUT_STEM" =~ ^[A-Za-z0-9._-]+$ || "$OUTPUT_STEM" == "." || "$OUTPUT_STEM" == ".." ]]; then
  echo "invalid output stem: $OUTPUT_STEM" >&2
  exit 2
fi
if [[ ! "$EXPECTED_PROTOCOL_SHA256" =~ ^[A-Fa-f0-9]{64}$ ]]; then
  echo "expected protocol SHA-256 must contain exactly 64 hexadecimal characters" >&2
  exit 2
fi
if [[ ! "$EXPECTED_BASELINE_SHA256" =~ ^[A-Fa-f0-9]{64}$ ]]; then
  echo "expected baseline SHA-256 must contain exactly 64 hexadecimal characters" >&2
  exit 2
fi

PROTOCOL="$TRIAL_DIR/protocol.json"
MANIFEST="$TRIAL_DIR/checkpoint_manifest.json"
RUN_STATUS="$TRIAL_DIR/run_status.json"
OUTPUT="$SCRIPT_DIR/$OUTPUT_STEM.json"
LOG="$SCRIPT_DIR/$OUTPUT_STEM.log"
EXIT_FILE="$SCRIPT_DIR/$OUTPUT_STEM.exit.txt"

for path in "$PROTOCOL" "$MANIFEST" "$RUN_STATUS" "$PROTECTED_REFERENCE" "$VALIDATOR"; do
  [[ -f "$path" ]] || { echo "required validation input is missing: $path" >&2; exit 2; }
done
for path in "$OUTPUT" "$LOG" "$EXIT_FILE"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite existing output: $path" >&2; exit 2; }
done

ARGS=(
  --repo-root "$ROOT"
  --trial-dir "$TRIAL_DIR"
  --protocol "$PROTOCOL"
  --manifest "$MANIFEST"
  --run-status "$RUN_STATUS"
  --baseline "$BASELINE"
  --expected-protocol-sha256 "$EXPECTED_PROTOCOL_SHA256"
  --expected-baseline-sha256 "$EXPECTED_BASELINE_SHA256"
  --protected-reference "$PROTECTED_REFERENCE"
  --output "$OUTPUT"
)
if [[ "$CHECKPOINT" == "-" ]]; then
  ARGS+=(--baseline-only)
else
  CHECKPOINT="$(realpath -e "$CHECKPOINT")"
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

cd "$ROOT"
set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh "$VALIDATOR" "${ARGS[@]}" > "$LOG" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" > "$EXIT_FILE"
exit "$rc"
