"""Profile the M0 pieces separately to find the collection bottleneck."""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from native_env import NativeFieldEnv
from policy_m0 import ActorCritic

env = NativeFieldEnv(num_envs=64, seed=0)
print("ENV_READY")
act = np.zeros((64, 4), dtype=np.float64)
act[:, :2] = 0.5

# 1) physics only
t0 = time.perf_counter()
for _ in range(3):
    env.backend.step(act)
print("PHYSICS_3STEPS_S %.3f" % (time.perf_counter() - t0))

# 2) read only
t0 = time.perf_counter()
for _ in range(10):
    read = env.backend.read()
print("READ_10X_S %.4f" % (time.perf_counter() - t0))

# 3) observation only
t0 = time.perf_counter()
for _ in range(10):
    obs = env.provider.observe(read, actions=act)
print("OBSERVE_10X_S %.4f" % (time.perf_counter() - t0))

# 4) lidar cast only (every 2nd step)
origins = np.repeat(read["pos"][:, None, :], 1, axis=1)[:, 0, :] + np.array([0, 0, 0.40])
yaw = np.zeros(64)
t0 = time.perf_counter()
for _ in range(5):
    r = env.provider._lidar_cast(origins, yaw, np.zeros(64))
print("LIDAR_CAST_5X_S %.4f" % (time.perf_counter() - t0))

# 5) policy act
policy = ActorCritic(env.single_observation_space)
obs_t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}
t0 = time.perf_counter()
for _ in range(10):
    a, lp, v = policy.act(obs_t)
print("POLICY_ACT_10X_S %.4f" % (time.perf_counter() - t0))

# 6) full env.step
t0 = time.perf_counter()
for _ in range(2):
    obs, r, term, trunc, info = env.step(act.astype(np.float32))
print("ENV_STEP_2X_S %.4f" % (time.perf_counter() - t0))
