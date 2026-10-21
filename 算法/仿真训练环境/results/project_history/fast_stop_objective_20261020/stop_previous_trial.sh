#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/fast_stop_objective_20261020"
TRIAL="$ROOT/results/isaac_field_improvement/random_global_rough_lstm512_20261020"
cd "$ROOT" || exit 80
if [[ -e "$OUT/stop_request.log" ]]; then exit 81; fi
pid=$(cat "$TRIAL/python.pid")
argv=$(ps -p "$pid" -o args=)
case "$argv" in
  *"python scripts/train_isaac_foundation_fresh.py --launch --device cuda:0 --output $TRIAL") ;;
  *) printf 'refused unexpected argv: %s\n' "$argv" > "$OUT/stop_request.log"; exit 82 ;;
esac
printf 'utc=%s\npid=%s\nargv=%s\nreason=user_approved_fast_arrival_and_strict_stop_objective\nsignal=SIGINT\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pid" "$argv" > "$OUT/stop_request.log"
kill -INT "$pid"
rc=$?
printf 'kill_actual_exit=%s\n' "$rc" >> "$OUT/stop_request.log"
exit "$rc"
