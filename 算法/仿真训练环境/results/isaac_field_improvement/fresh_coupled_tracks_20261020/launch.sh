#!/usr/bin/env bash
set -uo pipefail
ROOT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$ROOT/results/isaac_field_improvement/fresh_coupled_tracks_20261020"
cd "$ROOT"
launcher_pid=$$
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$launcher_pid" > "$OUT/launcher.pid"
printf '%s\n' "$started_utc" > "$OUT/launcher_started_utc.txt"
scripts/isaaclab_python.sh scripts/train_isaac_foundation_fresh.py --launch --output "$OUT" --device cuda:0 > >(tee -a "$OUT/launch.log") 2>&1 &
python_pid=$!
printf '%s\n' "$python_pid" > "$OUT/python.pid"
set +e
wait "$python_pid"
rc=$?
printf '%s\n' "$rc" > "$OUT/python.exit_code"
printf '{"launcher_pid":%s,"python_pid":%s,"exit_code":%s,"started_utc":"%s"}\n' "$launcher_pid" "$python_pid" "$rc" "$started_utc" > "$OUT/process_exit.json"
exit "$rc"