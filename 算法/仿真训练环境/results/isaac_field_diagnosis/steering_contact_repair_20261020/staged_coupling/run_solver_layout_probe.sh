#!/usr/bin/env bash
set -uo pipefail
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$STAGE_DIR/../../../.." && pwd)"
LOG_PATH="$STAGE_DIR/solver_layout_probe.log"
EXIT_PATH="$STAGE_DIR/solver_layout_probe_runner.exit.json"
if [[ -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite prior solver-layout evidence.\n' >&2
  exit 2
fi
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=""
export WARP_DISABLE_CUDA=1
set +e
scripts/isaaclab_python.sh "$STAGE_DIR/probe_solver_layout.py" >"$LOG_PATH" 2>&1
rc=$?
set -e
printf 'ACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG_PATH"
(set -o noclobber; printf '{"actual_exit_code":%s,"device":"cpu","log_path":"%s"}\n' \
  "$rc" "$LOG_PATH" >"$EXIT_PATH")
exit "$rc"
