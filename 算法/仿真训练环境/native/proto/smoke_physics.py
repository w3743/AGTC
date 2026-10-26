"""Smoke test: physics backend only (MuJoCo CPU, no observations)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from physics_backend import MujocoCPUBackend

backend = MujocoCPUBackend(nworld=64, seed=0)
print("IDENTITY " + json.dumps(backend.identity()))
backend.reset()
rng = np.random.default_rng(0)
actions = np.zeros((64, 4), dtype=np.float64)
actions[:, :2] = 0.5
# warmup 3 policy steps
for _ in range(3):
    backend.step(actions)
t0 = time.perf_counter()
nsteps = 10
for _ in range(nsteps):
    backend.step(actions)
dt = time.perf_counter() - t0
read = backend.read()
policy_steps = nsteps * 64
env_steps = policy_steps * backend.decimation
print("RESULT " + json.dumps({
    "policy_steps": policy_steps, "env_steps": env_steps,
    "wall_s": round(dt, 4),
    "policy_steps_per_s": round(policy_steps / dt, 2),
    "env_steps_per_s": round(env_steps / dt, 1),
    "finite": bool(np.isfinite(read["pos"]).all() and np.isfinite(read["wheel_qd"]).all()),
    "mean_x_m": round(float(read["pos"][:, 0].mean()), 4),
    "mean_track_cmd": round(float(read["track_command"][:, 0].mean()), 4),
    "mean_wheel_qd": round(float(read["wheel_qd"].mean()), 3),
    "mean_effort_nm": round(float(np.abs(read["effort"]).mean()), 3),
}))
