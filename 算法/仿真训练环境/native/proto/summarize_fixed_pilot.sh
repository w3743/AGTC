#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/lstm_pilot_5ms_fixed || exit 1
ls -la | head -20
echo '--- failure.json ---'
cat failure.json 2>/dev/null
echo '--- metrics stats ---'
/opt/venv/bin/python - <<'PYEOF'
import json, statistics as st
rows = [json.loads(l) for l in open('metrics.jsonl', encoding='utf-8') if l.strip()]
print('iters', len(rows), 'last', rows[-1]['iter'])
nf = [r.get('nonfinite_env_resets', 0) for r in rows]
print('nonfinite total', sum(nf), 'mean/iter', round(st.mean(nf), 3), 'max/iter', max(nf),
      'iters_with_nf', [(r['iter'], v) for r, v in zip(rows, nf) if v])
early = [r for r in rows if 2 <= r['iter'] <= 75]
late = [r for r in rows if 76 <= r['iter'] <= 98]
def stats(rs, label):
    if not rs: return
    print(label, 'n', len(rs), 'wall mean', round(st.mean(r['wall_s'] for r in rs), 3),
          'samples/s mean', round(st.mean(r['end_to_end_samples_per_s'] for r in rs), 1),
          'collection mean', round(st.mean(r['collection_s'] for r in rs), 3),
          'env_step mean', round(st.mean(r['env_step_s'] for r in rs), 3))
stats(early, 'steady(2-75)')
stats(late, 'late(76-98)')
print('maxtotal wall_s', round(max(r['wall_s'] for r in rows), 3))
print('rss max', max(r['rss_mb'] for r in rows))
PYEOF
echo '--- gpu samples ---'
/opt/venv/bin/python - <<'PYEOF'
import csv, statistics as st
rows=[]
for r in csv.reader(open('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto/runs/lstm_pilot_5ms_fixed_gpu.csv', encoding='utf-8', errors='replace')):
    if len(r)<4 or r[0].strip().startswith('utilization'): continue
    try: rows.append((float(r[0].replace('%','').strip()), float(r[1].replace('MHz','').strip()), float(r[2].replace('MiB','').strip()), float(r[3].replace('C','').strip())))
    except Exception: pass
if rows:
    u=[x[0] for x in rows]; s=[x[1] for x in rows]; m=[x[2] for x in rows]
    print({'n':len(rows),'util_mean':round(st.mean(u),1),'util_max':max(u),'sm_mean':round(st.mean(s)),'sm_max':max(s),'mem_max':max(m),'temp_max':max(x[3] for x in rows)})
PYEOF
echo '--- log tail ---'
tail -3 /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto/runs/lstm_pilot_5ms_fixed.log
echo '--- residual ---'
ps -eo pid,args | grep -E 'train_wsl|nvidia-smi' | grep -v grep || echo NONE
/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
