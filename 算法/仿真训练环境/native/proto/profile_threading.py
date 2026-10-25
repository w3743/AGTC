"""Does MuJoCo's Python binding release the GIL? Can envs be stepped in threads?"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from physics_backend import MujocoCPUBackend

b = MujocoCPUBackend(nworld=64, seed=0)
act = np.zeros((64, 4))
act[:, :2] = 0.3


def step_range(datas, start, end, substeps=10):
    for _ in range(substeps):
        for d in datas[start:end]:
            d.ctrl[:14] = 0.5
            d.ctrl[14:] = 0.0
            import mujoco
            mujoco.mj_step(b.mjm, d)


# sequential baseline: 64 envs x 10 substeps
for _ in range(2):
    step_range(b.datas, 0, 64)
t0 = time.perf_counter()
for _ in range(5):
    step_range(b.datas, 0, 64)
seq = time.perf_counter() - t0
print("SEQUENTIAL 64x10 substeps: %.4f s" % seq)

for nthreads in (2, 4, 8, 16):
    chunks = [(i * 64 // nthreads, (i + 1) * 64 // nthreads) for i in range(nthreads)]
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        for _ in range(2):
            list(ex.map(lambda c: step_range(b.datas, c[0], c[1]), chunks))
        t0 = time.perf_counter()
        for _ in range(5):
            list(ex.map(lambda c: step_range(b.datas, c[0], c[1]), chunks))
        th = time.perf_counter() - t0
    print("THREADS=%d: %.4f s  speedup=%.2fx" % (nthreads, th, seq / th))
