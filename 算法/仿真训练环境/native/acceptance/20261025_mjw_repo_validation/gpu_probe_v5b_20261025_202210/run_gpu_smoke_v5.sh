#!/usr/bin/env bash
set +e
OUT="/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/repo_integration_before/gpu_probe_v5b_20261025_202210"
PROTO="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
export PYTHONPATH="$PROTO"
unset CUDA_VISIBLE_DEVICES
/opt/venv_mujoco315_smoke/bin/python "$OUT/gpu_smoke_v5.py" --output "$OUT/run"
rc=$?
printf '%s\n' "$rc" > "$OUT/run/exit_code.txt"
exit "$rc"