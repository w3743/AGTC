#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
/opt/venv/bin/python - <<'PYEOF'
import json
import sys
sys.path.insert(0, '.')
from native_env import NativeFieldEnv

env = NativeFieldEnv(num_envs=1, seed=0, depth_backend='analytic', depth_device='cpu',
                     camera_backend='analytic')
c = env.observation_config()
ch = c['channels']
print(json.dumps({k: {'mode': ch[k]['mode'], 'backend': ch[k].get('backend'),
                      'verified_vs': ch[k].get('verified_vs')} for k in ch}, indent=1))
print('capability_metrics_allowed', c['capability_metrics_allowed'])
print('blockers', c['capability_blockers'])
print('provider_contract_sha256', env.provider.contract_sha256(env.policy_dt)[:16])
PYEOF
