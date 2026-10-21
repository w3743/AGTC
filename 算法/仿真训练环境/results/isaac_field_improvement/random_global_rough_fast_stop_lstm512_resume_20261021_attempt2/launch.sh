#!/usr/bin/env bash
set -euo pipefail
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OUT/../../.." && pwd)"
cd "$ROOT"
if [[ -e "$OUT/supervisor.log" ]]; then
    printf 'Refusing to overwrite supervisor.log\n' >&2
    exit 1
fi
nohup bash scripts/isaaclab_python.sh scripts/resume_fast_stop.py \
    --launch --output "$OUT" --device cuda:0 \
    > "$OUT/supervisor.log" 2>&1 < /dev/null &
printf 'dispatched_supervisor_pid=%s\n' "$!"
