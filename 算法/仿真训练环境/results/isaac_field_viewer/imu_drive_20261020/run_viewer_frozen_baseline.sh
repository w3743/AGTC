#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/isaaclab_python.sh -m isaac_field viewer \
  --mode goto --plants 16 --flat --seed 80003 --episode-seed 82001 \
  --max-time 20 --goal-distance 1 3 --goal-angle 60 \
  --reward-profile protect_plants \
  --checkpoint results/isaac_field_improvement/fresh_imu_drive_20261020/untrained_baseline.zip \
  --output results/isaac_field_viewer/imu_drive_20261020/viewer_frozen_baseline_replay \
  --port 8769 \
  > results/isaac_field_viewer/imu_drive_20261020/viewer_frozen_baseline_stdout.log \
  2> results/isaac_field_viewer/imu_drive_20261020/viewer_frozen_baseline_stderr.log
