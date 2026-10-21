#!/usr/bin/env bash
set -uo pipefail

FIELD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVIDENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="$EVIDENCE_DIR/untrained_minimal_preview.zip"
OUTPUT="$EVIDENCE_DIR/netron_export_v6/untrained_minimal_preview.pt"

if [[ ! -f "$CHECKPOINT" || ! -f "$EVIDENCE_DIR/untrained_minimal_preview.json" ]]; then
    printf 'preview checkpoint and metadata must exist before export\n' >&2
    exit 2
fi

attempt=1
while [[ -e "$EVIDENCE_DIR/netron_export_validation_attempt${attempt}.log" ]]; do
    attempt=$((attempt + 1))
done
log="$EVIDENCE_DIR/netron_export_validation_attempt${attempt}.log"

cd "$FIELD_ROOT"
set +e
{
    sha256sum "$CHECKPOINT"
    CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh scripts/view_policy_netron.py \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT" \
        --export-only
    export_exit=$?
    printf 'netron_export_exit_code=%d\n' "$export_exit"
    exit "$export_exit"
} >"$log" 2>&1
result=$?
set -e
printf 'netron_export_exit_code=%d log=%s\n' "$result" "$log"
exit "$result"
