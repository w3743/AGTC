#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT="$PROJECT/results/isaac_field_diagnosis/minimal_inputs_20261020/effort_limit_smoke"
cd "$PROJECT"
if [[ -e "$OUT/effort_limit_summary.json" || -e "$OUT/effort_limit_substeps.jsonl" || -e "$OUT/python_process_exit.code" ]]; then
  echo "Refusing to overwrite existing effort-limit smoke outputs in $OUT" >&2
  exit 73
fi
set +e
RUN_ISAAC_PHYSICS=1 scripts/isaaclab_python.sh "$OUT/effort_limit_smoke.py" > "$OUT/python.stdout.log" 2>&1
python_exit=$?
set -e
printf '%s\n' "$python_exit" > "$OUT/python_process_exit.code"
printf '{"python_process_exit":%s,"execution_status":"%s"}\n' "$python_exit" "$(if [[ "$python_exit" -eq 0 ]]; then printf completed; else printf failed; fi)" > "$OUT/wrapper_exit.json"
exit "$python_exit"