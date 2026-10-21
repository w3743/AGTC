#!/usr/bin/env bash
set -u
set +e
bash scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py \
  --launch --output results/isaac_field_improvement/fresh_direct_tracks_20261020 \
  --device cuda:0 \
  > results/isaac_field_improvement/fresh_direct_tracks_20261020/train_stdout.log \
  2> results/isaac_field_improvement/fresh_direct_tracks_20261020/train_stderr.log
task_training_exit=$?
printf '{"exit_code": %d, "completed_at_utc": "%s"}\n' "$task_training_exit" "$(date -u +%FT%TZ)" \
  > results/isaac_field_improvement/fresh_direct_tracks_20261020/process_exit.json
exit "$task_training_exit"
