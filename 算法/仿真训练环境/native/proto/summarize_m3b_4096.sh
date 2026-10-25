#!/bin/bash
# Summarise the M3b-4096 GPU window: sampling stats + confirm GPU/process release.
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
/opt/venv/bin/python - <<'PYEOF'
import csv, json, statistics as st
rows = []
with open('runs/m3b_4096/gpu_samples.csv', encoding='utf-8', errors='replace') as f:
    for r in csv.reader(f):
        if len(r) < 6 or r[0].strip().startswith('timestamp'):
            continue
        try:
            rows.append(dict(util=float(r[1].replace('%','').strip()),
                             sm=float(r[2].replace('MHz','').strip()),
                             mem=float(r[3].replace('MiB','').strip()),
                             power=r[4].strip(), temp=float(r[5].replace('C','').strip())))
        except Exception:
            continue
u = [x['util'] for x in rows]; s = [x['sm'] for x in rows]; m = [x['mem'] for x in rows]
print(json.dumps({
    'samples': len(rows),
    'util_mean_pct': round(st.mean(u),1), 'util_max_pct': max(u),
    'sm_clock_mean_mhz': round(st.mean(s),0), 'sm_clock_max_mhz': max(s),
    'mem_used_max_mib': max(m), 'mem_used_mean_mib': round(st.mean(m),0),
    'temp_max_c': max(x['temp'] for x in rows),
    'power_samples_tail': [x['power'] for x in rows[-6:]],
}))
fails = sum(1 for x in rows if x['util'] < 10)
print('samples_with_util<10pct', fails)
PYEOF
echo '--- residual processes ---'
ps -eo pid,args | grep -E 'train_m0|run_m3b|nvidia-smi' | grep -v grep || echo NONE
echo '--- gpu after ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
