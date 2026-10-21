#!/usr/bin/env bash
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$STAGE_DIR/../../../.." && pwd)"
cd "$ROOT_DIR"
scripts/isaaclab_python.sh "$STAGE_DIR/probe_newton_api.py"
