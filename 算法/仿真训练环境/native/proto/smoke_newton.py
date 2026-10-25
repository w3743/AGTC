"""CPU validation of the Newton XPBD backend (no GPU: CUDA_VISIBLE_DEVICES='')."""
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from newton_backend import NewtonXPBDGBackend

b = NewtonXPBDGBackend(nworld=2, seed=0, device="cpu")
print("IDENTITY", b.identity())
poses = np.array([[10.0, 20.0, 0.3, 0.0], [30.0, 40.0, -1.0, 0.0]])
b.reset(poses=poses)
act = np.zeros((2, 4))
act[:, :2] = 0.3
for _ in range(2):
    b.step(act)
r = b.read()
print("READ pos", np.round(r["pos"], 4).tolist())
print("READ wheel_qd", np.round(r["wheel_qd"][0, :3], 3).tolist())
print("finite", bool(np.isfinite(r["pos"]).all() and np.isfinite(r["wheel_qd"]).all()))
t0 = time.perf_counter()
for _ in range(3):
    b.step(act)
dt = time.perf_counter() - t0
print("CPU_VALIDATE env_steps/s=%.0f (2 worlds x 50 substeps x 3 steps)" % (2 * 50 * 3 / dt))
