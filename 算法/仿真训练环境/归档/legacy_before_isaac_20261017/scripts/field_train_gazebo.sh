#!/usr/bin/env bash
set -eo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LAB_ROOT/scripts/env.sh"
set -u

# Isolate training from the older lab graph and every independent Gazebo run.
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION="field_train_$(id -u)_$(date +%s%N)_$$"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# The training-only Python packages live in this project overlay.  Keep the
# ROS paths appended so Gazebo Transport remains importable.
export PYTHONPATH="$LAB_ROOT/.runtime-gazebo:$LAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
FIELD_TRAIN_PYTHON="$ROBOTICS_HOME/venv/learning/bin/python"
if [[ ! -x "$FIELD_TRAIN_PYTHON" ]]; then
  echo "Training Python environment not found: $FIELD_TRAIN_PYTHON" >&2
  exit 2
fi

cd "$LAB_ROOT"
exec "$FIELD_TRAIN_PYTHON" -m simulation.train_field_gazebo "$@"
