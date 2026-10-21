#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/isaac_field_improvement/random_global_rough_lstm512_20261020"
cd "$ROOT" || exit 80
if [[ ! -f "$OUT/protocol.json" || ! -f "$OUT/run_status.json" ]]; then exit 81; fi
if [[ -e "$OUT/launch.log" || -e "$OUT/process_exit.json" || -e "$OUT/untrained_baseline.zip" ]]; then exit 82; fi
if ! mkdir "$OUT/.launch_lock"; then exit 83; fi
printf '%s\n' "$$" > "$OUT/launcher.pid"
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$started_utc" > "$OUT/launcher_started_utc.txt"
unset RUN_ISAAC_PHYSICS
bash scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --launch --device cuda:0 --output "$OUT" > "$OUT/launch.log" 2>&1 &
python_pid=$!
printf '%s\n' "$python_pid" > "$OUT/python.pid"
wait "$python_pid"
rc=$?
printf '%s\n' "$rc" > "$OUT/python.exit_code"
printf '{"launcher_pid":%s,"python_pid":%s,"actual_exit_code":%s,"started_utc":"%s","finished_utc":"%s","source":"parent_shell_waited_for_child"}\n' \
  "$$" "$python_pid" "$rc" "$started_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/process_exit.json"
exit "$rc"
