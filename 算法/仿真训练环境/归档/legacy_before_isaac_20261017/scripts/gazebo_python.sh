#!/usr/bin/env bash
set -eo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LAB_ROOT/scripts/env.sh"
export PYTHONPATH="$LAB_ROOT/.runtime-gazebo:$LAB_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$LAB_ROOT"
exec "$ROBOTICS_HOME/venv/learning/bin/python" "$@"
