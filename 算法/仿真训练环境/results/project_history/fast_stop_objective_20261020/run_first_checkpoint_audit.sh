#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/fast_stop_objective_20261020"
cd "$ROOT" || exit 80
if [[ -e "$OUT/first_checkpoint_cpu.log" ]]; then exit 81; fi
export CUDA_VISIBLE_DEVICES=""
unset RUN_ISAAC_PHYSICS
bash scripts/isaaclab_python.sh "$OUT/audit_checkpoint_cpu.py" \
  --trial results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020 \
  --checkpoint stage_1_global_random/checkpoints/field_policy_global_random_8192_steps.zip \
  --output "$OUT/first_checkpoint_cpu.json" > "$OUT/first_checkpoint_cpu.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/first_checkpoint_cpu.exit_code"
exit "$rc"
