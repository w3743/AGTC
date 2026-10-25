"""CPU tests for the non-finite state guards (env/provider/train_wsl failure path)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
WSL = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train"
for p in (HERE, WSL):
    if p not in sys.path:
        sys.path.insert(0, p)

from native_env import NativeFieldEnv  # noqa: E402
from observation_provider import ObservationProvider  # noqa: E402


def test_env_recovers_injected_nan():
    env = NativeFieldEnv(num_envs=4, seed=3, physics="mujoco_cpu", lidar_backend="numpy",
                         feedback_backend="direct", depth_backend="constant",
                         camera_backend="constant", auto_reset=True)
    env.reset(seed=3)
    action = np.zeros((4, 4), np.float32)
    env.step(action)
    real_read = env.backend.read
    state = {"active": True, "calls": 0}

    def patched_read():
        read = real_read()
        state["calls"] += 1
        if state["active"]:
            read["pos"] = np.array(read["pos"], copy=True)
            read["pos"][0, 0] = np.nan  # simulate one diverged env
        return read

    env.backend.read = patched_read
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    finally:
        env.backend.read = real_read
        state["active"] = False
    assert np.isfinite(obs["state"]).all(), "state must stay finite"
    assert np.isfinite(obs["depth"]).all() and np.isfinite(obs["detections"]).all()
    assert bool(info["nonfinite"][0]) is True
    diag = env.consume_nonfinite()
    assert diag["count"] >= 1 and diag["events"], "non-finite event must be recorded"
    # the diverged env was reset by the guard; the following step stays finite
    obs2, *_ = env.step(action)
    for key, value in obs2.items():
        assert np.isfinite(value).all(), f"post-recovery {key} non-finite"
    return {"injected_env": 0, "events": len(diag["events"]), "count": diag["count"],
            "sensor_frames": diag["sensor_frames"], "recovered": True}


def test_provider_guards_nonfinite_frames():
    provider = ObservationProvider(2, seed=1, lidar_backend="numpy", feedback_backend="direct",
                                   depth_backend="analytic", camera_backend="analytic")
    pos = np.zeros((2, 3), np.float32)
    pos[:, 2] = 0.15
    pos[1, 0] = np.nan  # second env is diverged
    read = {
        "pos": pos, "quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (2, 1)),
        "lin_vel": np.zeros((2, 3), np.float32), "ang_vel": np.zeros((2, 3), np.float32),
        "wheel_qd": np.zeros((2, 14), np.float32), "gimbal_q": np.zeros((2, 2), np.float32),
        "gimbal_qd": np.zeros((2, 2), np.float32), "effort": np.zeros((2, 14), np.float32),
        "track_command": np.zeros((2, 2), np.float32),
    }
    provider.reset(goal_xy=np.array([[10.0, 10.0], [20.0, 20.0]]))
    obs = provider.observe(read, actions=np.zeros((2, 4), np.float32))
    for key, value in obs.items():
        assert np.isfinite(value).all(), f"{key} non-finite"
    assert obs["state"][1].sum() == 0.0 and obs["feedback"][1].sum() == 0.0
    assert provider.nonfinite_frames >= 1
    return {"channels": {k: list(v.shape) for k, v in obs.items()},
            "nonfinite_frames": int(provider.nonfinite_frames)}


def test_train_wsl_failure_path():
    """Resume a checkpoint whose policy weights are NaN: must write failure.json and exit 3."""
    tmp = Path(tempfile.mkdtemp(prefix="wsl_nan_test_"))
    env_cpu = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    base = tmp / "base"
    common = [str(Path("/opt/venv/bin/python")), str(Path(WSL) / "train_wsl.py"),
              "--policy", "lstm", "--num-envs", "1", "--num-steps", "2", "--iterations", "1",
              "--device", "cpu", "--physics", "mujoco_cpu", "--lidar", "numpy",
              "--feedback", "direct", "--save-interval", "1"]
    base_proc = subprocess.run(common + ["--out", str(base)], capture_output=True, text=True,
                               env=env_cpu, timeout=300)
    assert base_proc.returncode == 0, base_proc.stderr[-800:]
    ckpt = torch.load(base / "checkpoint_1.pt", map_location="cpu", weights_only=False)
    state = ckpt["policy"]
    key = "actor_mean.2.weight" if "actor_mean.2.weight" in state else "actor_mean.0.weight"
    state[key] = torch.full_like(state[key], float("nan"))
    nan_ckpt = tmp / "nan_policy.pt"
    ckpt["policy"] = state
    torch.save(ckpt, nan_ckpt)
    out = tmp / "run"
    proc = subprocess.run(common + ["--resume", str(nan_ckpt), "--out", str(out)],
                          capture_output=True, text=True, env=env_cpu, timeout=300)
    failure = out / "failure.json"
    payload = json.loads(failure.read_text(encoding="utf-8")) if failure.exists() else None
    ok = (proc.returncode == 3 and payload is not None
          and payload.get("reason") in ("non_finite_parameter", "non_finite_action", "policy_forward_error")
          and payload.get("last_checkpoint") is not None)
    return {"returncode": proc.returncode, "failure_json": bool(failure),
            "reason": None if payload is None else payload.get("reason"),
            "last_checkpoint": None if payload is None else payload.get("last_checkpoint"),
            "pass": ok, "stdout_tail": proc.stdout.strip().splitlines()[-2:]}


def main():
    results = {
        "env_injected_nan": test_env_recovers_injected_nan(),
        "provider_nonfinite_frames": test_provider_guards_nonfinite_frames(),
        "train_wsl_failure_path": test_train_wsl_failure_path(),
    }
    ok = (results["env_injected_nan"]["recovered"]
          and results["train_wsl_failure_path"]["pass"])
    with open(f"{HERE}/nonfinite_guard_tests.json", "w", encoding="utf-8") as f:
        json.dump({"pass": bool(ok), "results": results}, f, indent=2, ensure_ascii=False)
    print("NONFINITE_GUARDS " + json.dumps({"pass": bool(ok), "results": results}))


if __name__ == "__main__":
    main()
