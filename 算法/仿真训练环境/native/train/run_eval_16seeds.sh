#!/bin/bash
# Native evaluation: 16 dev seeds (87010-87025), 60 s / 1200 steps, 5ms/10 experimental physics.
# Usage: bash run_eval_16seeds.sh <checkpoint.pt> [out_dir] [seeds]
set -u
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train || exit 1
CKPT=${1:?usage: run_eval_16seeds.sh <checkpoint.pt> [out_dir] [seeds]}
OUT=${2:-runs/eval_native_$(date +%Y%m%d_%H%M%S)}
SEEDS=${3:-87010-87025}
/opt/venv/bin/python -u eval_native.py \
    --checkpoint "$CKPT" \
    --seeds "$SEEDS" \
    --max-steps 1200 \
    --device cuda --physics newton_xpbd \
    --lidar warp --feedback vqf \
    --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.005 --decimation 10 \
    --out "$OUT" 2>&1 | tee "$OUT/eval.log" | grep -E 'EVAL_SUMMARY|EVAL_OUT|Error|Traceback'
echo "EVAL_DIR=$OUT"
