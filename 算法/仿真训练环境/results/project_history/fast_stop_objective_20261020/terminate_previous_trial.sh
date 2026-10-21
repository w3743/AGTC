#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/fast_stop_objective_20261020"
TRIAL="$ROOT/results/isaac_field_improvement/random_global_rough_lstm512_20261020"
pid=$(cat "$TRIAL/python.pid")
argv=$(ps -p "$pid" -o args=)
case "$argv" in
  *"python scripts/train_isaac_foundation_fresh.py --launch --device cuda:0 --output $TRIAL") ;;
  *) exit 82 ;;
esac
if [[ -e "$OUT/terminate_request.log" ]]; then exit 81; fi
printf 'utc=%s\npid=%s\nargv=%s\nreason=SIGINT_did_not_stop_background_child_user_changed_objective\nsignal=SIGTERM\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pid" "$argv" > "$OUT/terminate_request.log"
kill -TERM "$pid"
rc=$?
printf 'kill_actual_exit=%s\n' "$rc" >> "$OUT/terminate_request.log"
exit "$rc"
