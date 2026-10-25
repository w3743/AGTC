#!/bin/bash
# CPU check: contract SHA excludes run metadata; resume across different out dirs now works.
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train || exit 1
rm -rf runs/sha_scope_a runs/sha_scope_b
/opt/venv/bin/python -u train_wsl.py --num-envs 2 --num-steps 2 --iterations 1 \
    --device cpu --physics mujoco_cpu --lidar numpy --feedback direct --save-interval 1 \
    --out runs/sha_scope_a 2>&1 | grep -E 'OBS_CONTRACT_SHA' | head -1
/opt/venv/bin/python -u train_wsl.py --num-envs 2 --num-steps 2 --iterations 1 \
    --device cpu --physics mujoco_cpu --lidar numpy --feedback direct --save-interval 1 \
    --out runs/sha_scope_b 2>&1 | grep -E 'OBS_CONTRACT_SHA' | head -1
echo '--- resume across out dirs ---'
/opt/venv/bin/python -u train_wsl.py --num-envs 2 --num-steps 2 --iterations 1 \
    --device cpu --physics mujoco_cpu --lidar numpy --feedback direct --save-interval 1 \
    --resume runs/sha_scope_a/checkpoint_1.pt --out runs/sha_scope_b 2>&1 \
    | grep -E 'RESUMED|ITER|Error' | head -2
echo '--- sha_scope field ---'
/opt/venv/bin/python - <<'PYEOF'
import json
c = json.load(open('runs/sha_scope_a/observation_contract.json', encoding='utf-8'))
print('sha', c['observation_contract_sha256'][:16])
print('sha_scope', c['sha_scope'][:90] + '...')
print('has run_id/phase/note', all(k in c for k in ('run_id','phase','note')))
PYEOF
