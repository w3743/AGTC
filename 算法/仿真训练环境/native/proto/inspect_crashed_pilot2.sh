#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/lstm_pilot_5ms || exit 1
echo '--- files ---'
ls -la | head -20
echo '--- identity.json ---'
cat identity.json
echo '--- observation_contract head ---'
/opt/venv/bin/python - <<'PYEOF'
import json
c = json.load(open('observation_contract.json', encoding='utf-8'))
print({k: c.get(k) for k in ('physics_dt_s','decimation','policy_dt_s','physics_experimental','profile_enabled','run_id','phase')})
print('channels', {k: c['channels'][k].get('backend') for k in c['channels']})
print('policy sha', c['observation_contract_sha256'][:16])
PYEOF
echo '--- last metrics iter ---'
tail -1 metrics.jsonl | head -c 400
