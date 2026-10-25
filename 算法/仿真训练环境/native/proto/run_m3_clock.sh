#!/bin/bash
# One extra M3b iteration with SM-clock sampling (fills the "时钟" part of the GPU sample request).
export PATH=/usr/lib/wsl/lib:$PATH
unset CUDA_VISIBLE_DEVICES
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader -lms 100 > runs/gpu_samples_m3_clock.csv 2>&1 &
SP=$!
/opt/venv/bin/python -u train_m0.py --num-envs 64 --num-steps 16 --iterations 2 --seed 0 \
    --threads 8 --device cuda --physics newton_xpbd --lidar warp --out runs/m3_b_clock 2>&1 \
    | grep -E 'ITER |SUMMARY'
kill "$SP" 2>/dev/null
sleep 1
/opt/venv/bin/python - <<'PYEOF'
import re
sm=[]; util=[]; pw=[]
for line in open('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto/runs/gpu_samples_m3_clock.csv'):
    p=[x.strip() for x in line.strip().split(',')]
    if len(p)<6: continue
    try:
        util.append(int(p[1].rstrip('%'))); sm.append(int(re.sub(r'[^0-9]','',p[2])))
        pw.append(float(p[4].split()[0]))
    except: pass
print('CLOCK_STATS n=%d sm_mean=%.0f sm_max=%d sm_min=%d util_mean=%.1f util_max=%d power_mean=%.2f power_max=%.2f' % (
    len(sm), sum(sm)/len(sm), max(sm), min(sm), sum(util)/len(util), max(util), sum(pw)/len(pw), max(pw)))
PYEOF
echo M3_CLOCK_DONE
