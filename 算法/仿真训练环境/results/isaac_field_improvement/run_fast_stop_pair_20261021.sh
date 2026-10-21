#!/usr/bin/env bash
set -uo pipefail
cd '/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境' || exit 2
out='results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2_evaluation'
if [[ -e "$out" || -e "$out.launch.log" || -e "$out.process_exit.json" ]]; then
  printf 'Refusing to overwrite paired evaluation evidence.\n' >&2
  exit 2
fi
bash scripts/isaaclab_python.sh scripts/evaluate_fast_stop_resume.py --output "$out" --device cuda:0 > "$out.launch.log" 2>&1
task_rc=$?
printf '{"actual_exit_code":%d,"timestamp_utc":"%s","source":"parent_shell_waited_for_child"}\n' "$task_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out.process_exit.json"
exit "$task_rc"
