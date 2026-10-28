#!/bin/bash
# M0 acceptance: 64 env x 16 policy steps x 3 iterations, run twice with the same seed.
export PATH=/usr/lib/wsl/lib:$PATH
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
rm -rf runs/m0_rep_a runs/m0_rep_b "runs/m0_ 2" 2>/dev/null
for tag in rep_a rep_b; do
  echo "==== run ${tag} ===="
  /opt/venv/bin/python -u train_m0.py \
      --num-envs 64 --num-steps 16 --iterations 3 --seed 0 --threads 8 \
      --out "runs/m0_${tag}" 2>&1 | grep -E 'ITER |SUMMARY|OUT_DIR|Traceback|Error'
done
echo ACCEPTANCE_DONE
