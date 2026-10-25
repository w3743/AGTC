"""Find where the Newton XPBD backend NaNs."""
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from newton_backend import NewtonXPBDGBackend

b = NewtonXPBDGBackend(nworld=1, seed=0, device="cpu")
print("base_coord", b.base_coord, "gimbal_coords", b.gimbal_q_coord,
      "wheel_target", b.wheel_target["left"][:3], "wheel_dof", b.wheel_dof["left"][:3])
print("target_qd shape", b.control.joint_target_qd.shape, b.control.joint_target_q.shape)
print("joint_target_q_start", b.model.joint_target_q_start.numpy()[:10])
print("joints labels", [str(x) for x in b.model.joint_label][:6])

# state before stepping
r = b.read()
print("after reset finite:", bool(np.isfinite(r["pos"]).all()), "pos", np.round(r["pos"], 4).tolist())

# step 1 substep with zero control
b.control.joint_target_qd.assign(np.zeros(b.control.joint_target_qd.shape, dtype=np.float32))
b.control.joint_target_q.assign(np.zeros(b.control.joint_target_q.shape, dtype=np.float32))
b.solver.step(b.state_0, b.state_1, b.control, None, 0.001)
b.state_0, b.state_1 = b.state_1, b.state_0
r = b.read()
print("1 zero-control substep finite:", bool(np.isfinite(r["pos"]).all()), "pos", np.round(r["pos"], 4).tolist())

# now set a single wheel velocity target and step once
qd = np.zeros(b.control.joint_target_qd.shape, dtype=np.float32)
qd[b.wheel_target["left"][0]] = 5.0
b.control.joint_target_qd.assign(qd)
b.solver.step(b.state_0, b.state_1, b.control, None, 0.001)
b.state_0, b.state_1 = b.state_1, b.state_0
r = b.read()
print("1 wheel-target substep finite:", bool(np.isfinite(r["pos"]).all()), "pos", np.round(r["pos"], 4).tolist(),
      "wheel_qd0", round(float(r["wheel_qd"][0, 0]), 3))
