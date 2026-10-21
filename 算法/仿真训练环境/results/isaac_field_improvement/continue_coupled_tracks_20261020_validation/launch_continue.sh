#!/usr/bin/env bash
set -uo pipefail
VALIDATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$VALIDATION_DIR/../../.." && pwd)"
OUT="$ROOT_DIR/results/isaac_field_improvement/continue_coupled_tracks_20261020"
cd "$ROOT_DIR"
if [[ ! -f "$OUT/protocol.json" || -e "$OUT/launch.log" || -e "$OUT/process_exit.json" ]]; then
  printf 'Missing prepared protocol or existing launch evidence; refusing duplicate launch.\n' >&2
  exit 2
fi
launcher_pid=$$
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$launcher_pid" > "$OUT/launcher.pid"
printf '%s\n' "$started_utc" > "$OUT/launcher_started_utc.txt"
scripts/isaaclab_python.sh scripts/continue_isaac_foundation.py \
  --launch --output "$OUT" --device cuda:0 > >(tee -a "$OUT/launch.log") 2>&1 &
python_pid=$!
printf '%s\n' "$python_pid" > "$OUT/python.pid"
wait "$python_pid"
training_rc=$?
printf '%s\n' "$training_rc" > "$OUT/python.exit_code"
printf '{"launcher_pid":%s,"python_pid":%s,"exit_code":%s,"started_utc":"%s"}\n' \
  "$launcher_pid" "$python_pid" "$training_rc" "$started_utc" > "$OUT/process_exit.json"
exit "$training_rc"
