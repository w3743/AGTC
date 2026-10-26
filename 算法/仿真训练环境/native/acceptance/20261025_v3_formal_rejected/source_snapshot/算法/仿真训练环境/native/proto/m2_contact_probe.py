"""M2.1: contact behaviour at spawn and under drive (single world, small arrays)."""
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from physics_backend import MujocoCPUBackend

b = MujocoCPUBackend(nworld=1, seed=0)
d = b.datas[0]
act = np.zeros((1, 4))
act[0, :2] = 0.5

rows = []
for step in range(40):
    b.step(act)
    rows.append((step, int(d.ncon), round(float(d.qpos[2]), 5), round(float(d.qvel[0]), 3),
                 round(float(d.solver_niter[0]), 2)))
print("step,ncon,z,vel_x,niter (every 5th)")
for r in rows[::5]:
    print(r)
print("ncon_mean", float(np.mean([r[1] for r in rows])), "ncon_min", min(r[1] for r in rows))
print("z_min", min(r[2] for r in rows), "z_max", max(r[2] for r in rows))

# static spawn: zero action for 20 steps
b2 = MujocoCPUBackend(nworld=1, seed=1)
d2 = b2.datas[0]
zero = np.zeros((1, 4))
for _ in range(20):
    b2.step(zero)
print("zero-action ncon", int(d2.ncon), "z", round(float(d2.qpos[2]), 6),
      "x", round(float(d2.qpos[0]), 6), "y", round(float(d2.qpos[1]), 6))
# spawn clearance variant: 1 cm above ground (isaac_field env.py spawn convention)
b3 = MujocoCPUBackend(nworld=1, seed=2)
b3.reset(poses=np.array([[50.0, 50.0, 0.3, 0.01]]))
d3 = b3.datas[0]
for _ in range(20):
    b3.step(zero)
print("+1cm spawn ncon", int(d3.ncon), "z", round(float(d3.qpos[2]), 6),
      "x", round(float(d3.qpos[0]), 5), "y", round(float(d3.qpos[1]), 5))
