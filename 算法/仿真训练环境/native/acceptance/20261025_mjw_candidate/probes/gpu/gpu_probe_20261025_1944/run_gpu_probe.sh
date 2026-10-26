#!/usr/bin/env bash
set +e
OUT='/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/mjw_candidate/gpu_probe_20261025_1944'
env CUDA_VISIBLE_DEVICES=0 /opt/venv_mujoco315_smoke/bin/python /mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/mjw_candidate/probe_dynamics.py --device cuda:0 --nworld 3 --steps 120 --output-dir "$OUT/result" > "$OUT/stdout.log" 2> "$OUT/stderr.log" &
child=$!
printf '%s\n' "$child" > "$OUT/wsl_python_pid.txt"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$OUT/exit_code.txt"
exit "$rc"