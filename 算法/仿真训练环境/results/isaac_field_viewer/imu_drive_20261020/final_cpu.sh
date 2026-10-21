#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")/../../.."
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q > results/isaac_field_viewer/imu_drive_20261020/final_cpu_verified.log 2>&1
validation_exit=$?
printf '\npytest_exit=%s\n' "$validation_exit" >> results/isaac_field_viewer/imu_drive_20261020/final_cpu_verified.log
tail -n 7 results/isaac_field_viewer/imu_drive_20261020/final_cpu_verified.log
exit "$validation_exit"
