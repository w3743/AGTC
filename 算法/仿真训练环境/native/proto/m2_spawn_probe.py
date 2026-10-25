"""M2.1 probe: why ncon=0 at spawn (solver sees no wheel-ground contact)."""
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
import mujoco
from physics_backend import MujocoCPUBackend, WHEELS_PER_SIDE, WHEEL_RADIUS, WHEEL_Z

b = MujocoCPUBackend(nworld=1, seed=0)
d = b.datas[0]
print("initial qpos z", round(float(d.qpos[2]), 6), "expected wheel bottom",
      round(float(d.qpos[2]) + WHEEL_Z - WHEEL_RADIUS, 6))
print("total mass", float(b.mjm.body_mass.sum()))
contacts = []
zs = []
for step in range(60):
    d.ctrl[:] = 0.0
    mujoco.mj_step(b.mjm, d)
    contacts.append(int(d.ncon))
    zs.append(float(d.qpos[2]))
print("ncon per step:", contacts)
print("z min/max:", round(min(zs), 6), round(max(zs), 6))
print("final ncon", int(d.ncon), "niter", int(d.solver_niter[0]))
if d.ncon:
    for i in range(min(3, d.ncon)):
        print("contact", i, "dist", round(float(d.contact[i].dist), 8),
              "geom1", mujoco.mj_id2name(b.mjm, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1),
              "geom2", mujoco.mj_id2name(b.mjm, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2))
print("wheel geom count", sum(1 for g in range(b.mjm.ngeom)
                              if (mujoco.mj_id2name(b.mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_geom")))
