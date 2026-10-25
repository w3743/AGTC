#!/usr/bin/env bash
set +e
OUT="/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/repo_integration_before/cuda_graph_compare_v5_20261025_202500"
PROTO="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
TRAIN="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train"
export PYTHONPATH="$PROTO:$TRAIN"
export AGTC_GRAPH_OUT="$OUT/run"
unset CUDA_VISIBLE_DEVICES
/opt/venv_mujoco315_smoke/bin/python "$OUT/cuda_graph_compare_v5.py"
rc=$?
printf '%s\n' "$rc" > "$OUT/run/exit_code.txt"
exit "$rc"