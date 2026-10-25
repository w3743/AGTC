"""Why is backend.step 2x slower with random poses/actions? Check solver iterations."""
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from physics_backend import MujocoCPUBackend

b = MujocoCPUBackend(nworld=64, seed=0)
# A: constant action, all worlds at origin
act_const = np.zeros((64, 4))
act_const[:, :2] = 0.5
b.reset()
for _ in range(3):
    b.step(act_const)
t0 = time.perf_counter()
for _ in range(10):
    b.step(act_const)
dt_a = time.perf_counter() - t0
niter_a = np.mean([d.solver_niter[0] for d in b.datas])
ncon_a = np.mean([d.ncon for d in b.datas])
print("A const_origin: env_steps/s=%.0f niter=%.1f ncon=%.1f" % (64 * 10 * 50 / dt_a, niter_a, ncon_a))

# B: random poses + random actions
b2 = MujocoCPUBackend(nworld=64, seed=0)
rng = np.random.default_rng(0)
poses = np.zeros((64, 4))
poses[:, 0:2] = rng.uniform(5, 95, (64, 2))
poses[:, 2] = rng.uniform(-np.pi, np.pi, 64)
b2.reset(poses=poses)
act_rand = rng.uniform(-0.5, 0.5, (64, 4))
for _ in range(3):
    b2.step(act_rand)
t0 = time.perf_counter()
for _ in range(10):
    b2.step(act_rand)
dt_b = time.perf_counter() - t0
niter_b = np.mean([d.solver_niter[0] for d in b2.datas])
ncon_b = np.mean([d.ncon for d in b2.datas])
print("B rand_pose_rand_act: env_steps/s=%.0f niter=%.1f ncon=%.1f" % (64 * 10 * 50 / dt_b, niter_b, ncon_b))

# C: random poses, constant action
b3 = MujocoCPUBackend(nworld=64, seed=0)
b3.reset(poses=poses)
for _ in range(3):
    b3.step(act_const)
t0 = time.perf_counter()
for _ in range(10):
    b3.step(act_const)
dt_c = time.perf_counter() - t0
print("C rand_pose_const_act: env_steps/s=%.0f" % (64 * 10 * 50 / dt_c))
