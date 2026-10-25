"""Where does env.step time go? (M0 prototype)"""
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from native_env import NativeFieldEnv

env = NativeFieldEnv(num_envs=64, seed=0)
rng = np.random.default_rng(0)
act = rng.uniform(-0.5, 0.5, (64, 4))
env.reset(seed=0)

# warm
for _ in range(2):
    env.step(act)

t_step = 0.0
t_phys = 0.0
t_read = 0.0
t_obs = 0.0
N = 8
for _ in range(N):
    t0 = time.perf_counter()
    env.backend.step(act)
    t_phys += time.perf_counter() - t0
    t0 = time.perf_counter()
    env.backend.read()
    t_read += time.perf_counter() - t0
    t0 = time.perf_counter()
    env.provider.observe(env.backend.read(), actions=act)
    t_obs += time.perf_counter() - t0
    t0 = time.perf_counter()
    env.step(act)
    t_step += time.perf_counter() - t0
print("N=%d per-step: backend.step=%.4f read=%.4f observe=%.4f env.step=%.4f"
      % (N, t_phys / N, t_read / N, t_obs / N, t_step / N))
