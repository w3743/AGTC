#!/usr/bin/env bash
set -u
set +e
bash scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --launch --output results/isaac_field_improvement/fresh_depth_radar_20261019_v2 --device cuda:0
task_training_exit=$?
printf '{"exit_code": %d, "completed_at_utc": "%s"}\n' "$task_training_exit" "$(date -u +%FT%TZ)" > results/isaac_field_improvement/fresh_depth_radar_20261019_v2/process_exit.json
exit "$task_training_exit"
