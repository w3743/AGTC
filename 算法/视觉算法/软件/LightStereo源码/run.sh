#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT/runtime/vendor${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
mode=${1:-serve}
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
  serve) exec python3 -B "$ROOT/app/server.py" --host 0.0.0.0 "$@" ;;
  self-test) exec python3 -B "$ROOT/app/lightstereo_depth.py" --self-test "$@" ;;
  pointcloud) exec python3 -B "$ROOT/app/pointcloud.py" "$@" ;;
  *) echo 'Usage: sh run.sh [serve|self-test|pointcloud] [options]'; exit 2 ;;
esac
