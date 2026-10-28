#!/usr/bin/env bash
set +e
BASE="/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/repo_integration_before/warning_mask_smoke_v1b_20261025_220517"
OUT="$BASE/result"
PROTO="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
TRAIN="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROTO:$TRAIN"
export AGTC_NATIVE_PROTO="$PROTO"
export AGTC_WARN_OUT="$OUT"
unset CUDA_VISIBLE_DEVICES
/opt/venv_mujoco315_smoke/bin/python -u "$BASE/warn_mask_smoke.py" > "$BASE/stdout.log" 2> "$BASE/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$BASE/exit_code.txt"
exit "$rc"