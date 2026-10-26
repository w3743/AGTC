#!/usr/bin/env bash
set +e
probe_root='/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/mjw_candidate/gpu_probe_v4b_20261025_200246'
echo $$ > "$probe_root/wsl_wrapper_pid.txt"
CUDA_VISIBLE_DEVICES=0 /opt/venv_mujoco315_smoke/bin/python /mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/mjw_candidate/probe_dynamics_v4b.py --device cuda:0 --nworld 3 --steps 120 --output-dir "$probe_root/result" > "$probe_root/stdout.log" 2> "$probe_root/stderr.log" &
py_pid=$!
echo "$py_pid" > "$probe_root/wsl_python_pid.txt"
wait "$py_pid"
rc=$?
echo "$rc" > "$probe_root/exit_code.txt"
exit "$rc"
