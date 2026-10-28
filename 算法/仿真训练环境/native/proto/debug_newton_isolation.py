"""Isolate the Newton XPBD NaN: control / mimic / substep count / iterations."""
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from newton_backend import NewtonXPBDGBackend


def trial(name, *, wheel_v=0.0, gimbal=0.0, substeps=50, iterations=10):
    import newton
    b = NewtonXPBDGBackend(nworld=1, seed=0, device="cpu")
    b.solver = newton.solvers.SolverXPBD(b.model, iterations=iterations)
    qd = np.zeros(b.control.joint_target_qd.shape, dtype=np.float32)
    qp = np.zeros(b.control.joint_target_q.shape, dtype=np.float32)
    for side in ("left", "right"):
        for ti in b.wheel_target_dof[side]:
            qd[ti] = wheel_v
    for c in b.gimbal_q_coord:
        qp[c] = gimbal
    b.control.joint_target_qd.assign(qd)
    b.control.joint_target_q.assign(qp)
    bad = None
    for k in range(substeps):
        b.solver.step(b.state_0, b.state_1, b.control, None, 0.001)
        b.state_0, b.state_1 = b.state_1, b.state_0
        if k % 10 == 9:
            r = b.read()
            ok = np.isfinite(r["pos"]).all() and np.isfinite(r["wheel_qd"]).all()
            if not ok:
                bad = k + 1
                break
    print("TRIAL %-28s wheel_v=%.1f gimbal=%.2f substeps=%d iters=%d -> %s"
          % (name, wheel_v, gimbal, substeps, iterations,
             "NaN at substep %s" % bad if bad else "FINITE"))


trial("zero control 50 substeps", wheel_v=0.0, gimbal=0.0, substeps=50)
trial("wheel 5 rad/s 50 substeps", wheel_v=5.0, gimbal=0.0, substeps=50)
trial("wheel 5 gimbal 0.1 50", wheel_v=5.0, gimbal=0.1, substeps=50)
trial("wheel 5 iterations=20", wheel_v=5.0, gimbal=0.0, substeps=50, iterations=20)
trial("wheel 21.7 (full cmd)", wheel_v=21.7, gimbal=0.0, substeps=50)
