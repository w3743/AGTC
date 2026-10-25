#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
echo '=== detections warp kernel vs numpy (mask + rows) ==='
/opt/venv/bin/python m1_detections_warp_verify.py 2>&1 | tail -2
echo '=== provider integration with analytic_warp (CPU device) ==='
/opt/venv/bin/python - <<'PYEOF'
import json, sys
sys.path.insert(0, '.')
import numpy as np
import detections_caster as dct
import depth_caster as dc
from native_env import NativeFieldEnv

n = 2
env = NativeFieldEnv(num_envs=n, seed=6, camera_backend='analytic_warp', camera_noise=False)
env.reset(seed=6)
scene = env.provider._det_scene
spawn = np.zeros((n, 4)); goals = np.zeros((n, 2))
for i in range(n):
    px, py = scene.xy[i]
    ang = i * (2 * np.pi / n) + 0.4
    spawn[i] = [px + 3.0 * np.cos(ang), py + 3.0 * np.sin(ang), ang + np.pi, 0.01]
    goals[i] = [50.0, 50.0]
env.reset_with_poses(spawn, goals)
action = np.zeros((n, 4))
for _ in range(16):
    env.step(action)
    if (env.provider.step[0] % 4) == 0:
        break
read = env._last_read
poses = np.stack([read["pos"][:, 0], read["pos"][:, 1], read["pos"][:, 2] + dc.CAMERA_HEIGHT,
                  np.asarray([np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)) for q in read["quat_wxyz"]]),
                  read["gimbal_q"][:, 0], read["gimbal_q"][:, 1]], axis=1)
masks, dev = dct.cast_mask_warp(poses, scene=scene, device='cpu')
center = dct.cast_center_distance(poses, scene=scene)
rows, _ = dct.observation_from_masks(masks, center, scene.leaf_dry, noise=False)
got = env.provider.det_latest
diff = np.abs(got.astype(np.float64) - rows.astype(np.float64))
print(json.dumps({'device': dev, 'max_abs_diff': float(diff.max()),
                  'active_rows': [int((r[:, 2] > 0).sum()) for r in rows],
                  'verdict': 'analytic_warp provider equals direct warp caster' if diff.max() == 0 else 'MISMATCH'}))
PYEOF
