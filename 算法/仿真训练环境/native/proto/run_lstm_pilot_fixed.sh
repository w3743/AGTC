#!/bin/bash
# 300-iter LSTM pilot rerun with non-finite guards (5ms/10 experimental physics).
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train || exit 1
OUT=runs/lstm_pilot_5ms_fixed
LOG=/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto/runs/lstm_pilot_5ms_fixed.log
mkdir -p "$OUT" /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto/runs
echo "PILOT_START $(date -Is) pid=$$" | tee "$LOG"
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,memory.used,temperature.gpu \
    --format=csv,noheader -lms 200 > /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto/runs/lstm_pilot_5ms_fixed_gpu.csv 2>&1 &
SAMP=$!
/opt/venv/bin/python -u train_wsl.py \
    --policy lstm --num-envs 64 --num-steps 32 --iterations 300 --seed 0 --threads 8 \
    --device cuda --physics newton_xpbd --lidar warp --feedback vqf \
    --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.005 --decimation 10 --profile --save-interval 20 \
    --nonfinite-limit 64 --out "$OUT" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
kill "$SAMP" 2>/dev/null
echo "PILOT_EXIT=$RC $(date -Is)" | tee -a "$LOG"
exit "$RC"
