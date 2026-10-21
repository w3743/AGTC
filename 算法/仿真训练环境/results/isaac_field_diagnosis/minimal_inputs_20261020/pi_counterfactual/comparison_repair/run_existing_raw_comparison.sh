#!/usr/bin/env bash
set -uo pipefail

REPAIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(cd "$REPAIR_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PI_DIR/../../../.." && pwd)"
LOG_PATH="$REPAIR_DIR/comparison_repair_attempt1.log"
STATUS_PATH="$REPAIR_DIR/comparison_repair_exit.json"
OUTPUT_PATH="$REPAIR_DIR/confirmation_pair_summary_repaired.json"
FREEZE_PATH="$REPAIR_DIR/comparison_only_source_freeze_sha256.txt"

for path in "$LOG_PATH" "$STATUS_PATH" "$OUTPUT_PATH"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to overwrite comparison repair evidence: %s\n' "$path" >&2
    exit 2
  fi
done
if [[ ! -f "$FREEZE_PATH" ]]; then
  printf 'Missing comparison-only freeze: %s\n' "$FREEZE_PATH" >&2
  exit 2
fi

cd "$ROOT_DIR"
set +e
python3 "$PI_DIR/compare_pi_counterfactual_runs.py" \
  --root "$PI_DIR" \
  --output "$OUTPUT_PATH" \
  --protocol-mode confirmation-v1 \
  --protocol "$PI_DIR/confirmation_protocol.json" \
  --source-freeze "$FREEZE_PATH" >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
printf '{"run_label":"comparison_repair_existing_raw_only","actual_exit_code":%s}\n' "$rc" \
  >"$STATUS_PATH"
exit "$rc"
