#!/usr/bin/env bash
set -euo pipefail

FIELD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="$FIELD_ROOT/external/IsaacLab"
ISAACLAB_TOOLS_HOME="${ISAACLAB_TOOLS_HOME:-$HOME/robotics-lab/isaac-tools}"
export UV_PROJECT_ENVIRONMENT="${ISAACLAB_ENV:-$HOME/robotics-lab/venv/isaaclab-field}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FIELD_ROOT/.cache/uv}"
export UV_LINK_MODE=copy
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$FIELD_ROOT/.cache/warp}"
export PYTHONPATH="$FIELD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$FIELD_ROOT"
if [[ "${1:-}" == python ]]; then shift; fi
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"
