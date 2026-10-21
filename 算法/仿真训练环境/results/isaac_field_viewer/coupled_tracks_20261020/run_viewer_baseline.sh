#!/usr/bin/env bash
set -uo pipefail

VIEWER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$VIEWER_DIR/../../.." && pwd)"
CHECKPOINT="results/isaac_field_improvement/fresh_coupled_tracks_20261020/untrained_baseline.zip"
LOG_PATH="$VIEWER_DIR/viewer_baseline.log"
EXIT_PATH="$VIEWER_DIR/viewer_baseline.exit.json"

if [[ "${RUN_COUPLED_VIEWER:-}" != 1 ]]; then
  printf 'Set RUN_COUPLED_VIEWER=1 to launch this verified baseline viewer.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
if [[ ! -f "$CHECKPOINT" || -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Missing formal baseline or existing viewer output; refusing overwrite.\n' >&2
  exit 2
fi

scripts/isaaclab_python.sh -m isaac_field viewer \
  --mode goto --plants 16 --flat --seed 80003 --episode-seed 82001 \
  --max-time 20 --goal-distance 1 3 --goal-angle 60 --reward-profile protect_plants \
  --checkpoint "$CHECKPOINT" \
  --output "$VIEWER_DIR/viewer_baseline_replay" --port 8769 \
  >"$LOG_PATH" 2>&1
viewer_rc=$?
printf '\nACTUAL_EXIT_CODE=%s\n' "$viewer_rc" >>"$LOG_PATH"
printf '{"actual_exit_code":%s,"checkpoint":"%s","log_path":"%s"}\n' \
  "$viewer_rc" "$CHECKPOINT" "$LOG_PATH" >"$EXIT_PATH"
exit "$viewer_rc"
