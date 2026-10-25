#!/usr/bin/env bash
set +e
BASE="/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/repo_integration_before/cuda_graph_compare_v6b_20261025_204148"
OUT="$BASE/result"
PROTO="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
TRAIN="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train"
export PYTHONPATH="$PROTO:$TRAIN"
export AGTC_GRAPH_OUT="$OUT"
unset CUDA_VISIBLE_DEVICES
/opt/venv_mujoco315_smoke/bin/python "$BASE/cuda_graph_compare_v6.py" > "$BASE/stdout.log" 2> "$BASE/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$BASE/exit_code.txt"
exit "$rc"