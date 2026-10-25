#!/usr/bin/env bash
set -uo pipefail
export PATH="/usr/lib/wsl/lib:${PATH:-}"
inherited_cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}
unset CUDA_VISIBLE_DEVICES
base='/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/acceptance/20261025_v3_diagnostic60'
run_out="$base/run"
mkdir -p "$run_out"
printf '{"cuda_visible_devices_inherited":"%s","cuda_visible_devices_for_training":"<unset>","path":"%s"}\n' "$inherited_cuda_visible_devices" "$PATH" > "$run_out/launch_environment.json"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv,noheader,nounits > "$run_out/gpu_before.csv"
/opt/venv/bin/python -B -u "$base/diagnostic_wrapper.py" \
  --num-envs 64 --num-steps 32 --iterations 60 --seed 0 --threads 8 --device cuda \
  --physics newton_xpbd --task-backend fast_stop_v2 --lidar warp --feedback vqf \
  --depth analytic --detections analytic --detections-warp \
  --sim-dt 0.001 --decimation 50 --policy lstm --lstm-hidden 512 \
  --nonfinite-limit 0 --save-interval 20 --out "$run_out" \
  > "$run_out/launcher.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$run_out/training.pid"
printf 'TRAINING_PID=%s\nRUN_OUT=%s\n' "$pid" "$run_out"
(
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv,noheader,nounits >> "$run_out/gpu_1hz.csv" 2>/dev/null || true
    sleep 1
  done
) &
sampler=$!
wait "$pid"
rc=$?
wait "$sampler" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv,noheader,nounits > "$run_out/gpu_after.csv" 2>/dev/null || true
printf '%s\n' "$rc" > "$run_out/exit_code.txt"
printf 'EXIT_CODE=%s\n' "$rc"
exit "$rc"