#!/bin/bash
# M3 GPU acceptance gate (run ONLY after Lead confirms the GPU is free).
#   M3a: GPU torch (encoder/PPO) + GPU warp lidar, physics still mujoco_cpu
#   M3b: GPU physics (newton XPBD) + GPU torch + GPU warp lidar
# Each run: 2 iterations, 64 env x 16 policy steps, JSONL + summary + nvidia-smi samples.
export PATH=/usr/lib/wsl/lib:$PATH
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
echo "PROTOTYPE_THROUGHPUT_ONLY: M3 GPU acceptance ${STAMP}"
echo "GPU identity:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

run_one() {
  local tag="$1"; shift
  local out="runs/m3_${tag}_${STAMP}"
  echo "==== ${tag}: $* -> ${out} ===="
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
      --format=csv,noheader -lms 100 > "runs/gpu_samples_m3_${tag}.csv" 2>&1 &
  local SP=$!
  /opt/venv/bin/python -u train_m0.py --num-envs 64 --num-steps 16 --iterations 2 \
      --seed 0 --threads 8 --device cuda --out "$out" "$@" 2>&1 | tee "runs/m3_${tag}.log"
  local rc=${PIPESTATUS[0]}
  kill "$SP" 2>/dev/null
  sleep 1
  echo "${tag}_exit=${rc}"
}

# M3a: GPU encoder/PPO + GPU lidar, CPU physics
run_one a --physics mujoco_cpu --lidar warp
# M3b: GPU physics (newton XPBD) + GPU encoder/PPO + GPU lidar
run_one b --physics newton_xpbd --lidar warp

echo "==== GPU sample stats ===="
/opt/venv/bin/python /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/scripts/gpu_sample_stats.sh 2>/dev/null | grep -E 'm3_' || true
echo "M3_GPU_ACCEPTANCE_DONE"
