#!/usr/bin/env bash
set -eo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LAB_ROOT/scripts/env.sh"
set -u

# Keep this simulator and any later ROS bridge off the robotics_lab graph.
export ROS_DOMAIN_ID=73
export GZ_PARTITION="field_migration_$(id -u)_$(date +%s%N)_$$"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec /usr/bin/python3 "$LAB_ROOT/simulation/run_field_gazebo.py" "$@"
