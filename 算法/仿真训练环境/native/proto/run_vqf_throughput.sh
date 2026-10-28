#!/bin/bash
# CPU-only throughput impact of the VQF feedback backend (64 env, 8 steps).
export PATH=/usr/lib/wsl/lib:$PATH
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
for fb in direct vqf; do
  echo "==== feedback=${fb} ===="
  /opt/venv/bin/python -u train_m0.py --num-envs 64 --num-steps 8 --iterations 1 \
      --seed 0 --threads 8 --feedback "${fb}" --out "runs/thr_fb_${fb}" 2>&1 \
      | grep -E 'ITER|SUMMARY|Traceback|Error'
done
echo VQF_THROUGHPUT_DONE
