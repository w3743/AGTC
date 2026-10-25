#!/bin/bash
# M3b at the Isaac-matched iteration size: 64 env x 64 steps = 4096 samples/iteration.
# For 88-96 s/iter (32 env x 128 steps = 4096 samples) wall-clock comparison.
# GPU WINDOW ONLY: run after Lead says "GPU 窗口开始"; expected ~60 s wall.
export PATH=/usr/lib/wsl/lib:$PATH
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
OUT=runs/m3b_4096
mkdir -p "$OUT"
echo "PROTOTYPE_THROUGHPUT_ONLY: M3b 4096-samples/iter ($(date -Is))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader -lms 100 > "$OUT/gpu_samples.csv" 2>&1 &
SP=$!
/opt/venv/bin/python -u train_m0.py \
    --num-envs 64 --num-steps 64 --iterations 3 --seed 0 --threads 8 \
    --device cuda --physics newton_xpbd --lidar warp \
    --out "$OUT" 2>&1 | tee "$OUT/run.log"
RC=${PIPESTATUS[0]}
kill "$SP" 2>/dev/null
sleep 1
echo "m3b_4096_exit=${RC}"
echo "M3B_4096_DONE $(date -Is)"
