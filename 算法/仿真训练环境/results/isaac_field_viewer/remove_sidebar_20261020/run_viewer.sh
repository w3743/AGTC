#!/usr/bin/env bash
set -uo pipefail
VIEWER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$VIEWER_DIR/../../.." && pwd)"
cd "$ROOT_DIR"
CHECKPOINT="results/isaac_field_improvement/fresh_coupled_tracks_20261020/stage_04_turn90.zip"
if [[ ! -f "$CHECKPOINT" || -e "$VIEWER_DIR/viewer.log" ]]; then
  printf 'Missing final checkpoint or existing viewer log; refusing overwrite.\n' >&2
  exit 2
fi
scripts/isaaclab_python.sh -m isaac_field viewer \
  --mode goto --plants 16 --flat --seed 80003 --episode-seed 82001 \
  --max-time 20 --goal-distance 1 3 --goal-angle 60 --reward-profile protect_plants \
  --checkpoint "$CHECKPOINT" --output "$VIEWER_DIR/viewer_replay" --port 8769 \
  >"$VIEWER_DIR/viewer.log" 2>&1 &
viewer_pid=$!
printf '%s\n' "$viewer_pid" > "$VIEWER_DIR/viewer.pid"
wait "$viewer_pid"
viewer_rc=$?
printf '{"actual_exit_code":%s,"viewer_pid":%s}\n' "$viewer_rc" "$viewer_pid" > "$VIEWER_DIR/viewer.exit.json"
exit "$viewer_rc"
