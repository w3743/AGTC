#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
EVIDENCE="$ROOT/results/project_history/fast_stop_objective_20261020"
OUT="$ROOT/results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020"
cd "$ROOT" || exit 80
if [[ -e "$OUT" || -e "$EVIDENCE/prepare.log" ]]; then exit 81; fi
export CUDA_VISIBLE_DEVICES=""
unset RUN_ISAAC_PHYSICS
bash scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --prepare --profile random_rough_fast_stop --lstm-hidden-size 512 --output "$OUT" > "$EVIDENCE/prepare.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$EVIDENCE/prepare.exit_code"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi
bash scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --preflight --output "$OUT" > "$EVIDENCE/preflight.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$EVIDENCE/preflight.exit_code"
exit "$rc"
