#!/usr/bin/env bash
set -euo pipefail
ROOT='/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境'
RUN_DIR="$ROOT/results/isaac_field_diagnosis/minimal_inputs_20261020"
OUT_DIR="$RUN_DIR/stage_pairing"
LOG="$RUN_DIR/stage_pairing_runner_attempt1.log"
cd "$ROOT"
if [[ -e "$LOG" ]]; then
  printf 'Refusing to overwrite existing log: %s\n' "$LOG" >&2
  exit 3
fi
if [[ -d "$OUT_DIR" ]] && [[ -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  printf 'Refusing to overwrite existing evaluation output: %s\n' "$OUT_DIR" >&2
  exit 3
fi
set +e
bash scripts/isaaclab_python.sh "$RUN_DIR/stage_pairing_eval.py" >"$LOG" 2>&1
rc=$?
set -e
printf '\nACTUAL_EXIT_CODE=%s\n' "$rc" >>"$LOG"
exit "$rc"
