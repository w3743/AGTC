#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/project_history/fast_stop_objective_20261020/pause_20261020"
TRIAL="$ROOT/results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020"
if [[ -e "$OUT/pause_record.json" || -e "$OUT/pause_request.log" ]]; then exit 81; fi
pid=$(cat "$TRIAL/python.pid")
argv=$(ps -p "$pid" -o args=)
case "$argv" in
  *"python scripts/train_isaac_foundation_fresh.py --launch --device cuda:0 --output $TRIAL") ;;
  *) printf 'refused unexpected argv: %s\n' "$argv" > "$OUT/pause_request.log"; exit 82 ;;
esac
printf 'requested_utc=%s\npid=%s\nargv=%s\nreason=explicit_user_request_to_pause_training\nsignal=SIGSTOP\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pid" "$argv" > "$OUT/pause_request.log"
kill -STOP "$pid"
rc=$?
printf 'signal_actual_exit=%s\n' "$rc" >> "$OUT/pause_request.log"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi
for attempt in {1..20}; do
  state=$(ps -p "$pid" -o stat=)
  case "$state" in
    T*)
      printf '{"status":"paused","python_pid":%s,"signal":"SIGSTOP","observed_process_state":"%s","verified_utc":"%s","resume_only_on_user_request":true,"process_exited":false,"pause_preserves_in_memory_state":true}\n' "$pid" "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/pause_record.json"
      cat "$OUT/pause_record.json"
      exit 0
      ;;
  esac
  sleep 0.1
done
printf 'failed_to_confirm_paused_state=%s\n' "$state" >> "$OUT/pause_request.log"
exit 83
