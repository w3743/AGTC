#!/usr/bin/env bash
set -u
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUTPUT="$ROOT/results/isaac_field_improvement/random_global_rough_coupled_guides_20261020"
ATTEMPT="$ROOT/results/isaac_field_improvement/random_global_rough_coupled_guides_20261020_validation/prepare_preflight_attempt1"
if [[ -e "$OUTPUT" ]]; then
  printf '%s\n' 'refusing existing trial output' > "$ATTEMPT/runner_status.txt"
  exit 81
fi
unset RUN_ISAAC_PHYSICS
export CUDA_VISIBLE_DEVICES=""
printf 'started_utc=%s\nRUN_ISAAC_PHYSICS=unset\nCUDA_VISIBLE_DEVICES=<empty>\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ATTEMPT/runner_status.txt"
cd "$ROOT" || exit 82
./scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --prepare --profile random_rough --output "$OUTPUT" > "$ATTEMPT/prepare.log" 2>&1
prepare_rc=$?
printf '%s\n' "$prepare_rc" > "$ATTEMPT/prepare.exit_code"
printf 'prepare_exit=%s\n' "$prepare_rc" >> "$ATTEMPT/runner_status.txt"
if [[ "$prepare_rc" -ne 0 ]]; then
  printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ATTEMPT/runner_status.txt"
  exit "$prepare_rc"
fi
./scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --preflight --output "$OUTPUT" > "$ATTEMPT/preflight.log" 2>&1
preflight_rc=$?
printf '%s\n' "$preflight_rc" > "$ATTEMPT/preflight.exit_code"
printf 'preflight_exit=%s\n' "$preflight_rc" >> "$ATTEMPT/runner_status.txt"
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ATTEMPT/runner_status.txt"
if [[ "$preflight_rc" -ne 0 ]]; then exit "$preflight_rc"; fi
exit 0