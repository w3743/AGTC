"""CPU diagnostics: does 5 ms/10 diverge under the pilot policy while 1 ms/50 stays finite?

Uses the crashed pilot's checkpoint (LSTM) with MuJoCo CPU physics (fast enough for CPU),
same seed / episode length as evaluation, and records the first non-finite state.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
WSL = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train"
if WSL not in sys.path:
    sys.path.insert(0, WSL)

from native_env import NativeFieldEnv  # noqa: E402
from eval_native import load_policy  # noqa: E402
from train_m0 import to_torch  # noqa: E402

CKPT = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/lstm_pilot_5ms/checkpoint_160.pt"
FIELDS = ("pos", "quat_wxyz", "lin_vel", "ang_vel", "wheel_qd", "gimbal_q")


def run_case(name, *, dt, decimation, physics, steps, seed, zero_action=False, ckpt=CKPT):
    # auto_reset=True mirrors training: episodes end, envs reset, and the policy keeps
    # acting; we count non-finite recoveries instead of stopping at the first episode end.
    env = NativeFieldEnv(num_envs=1, seed=seed, physics=physics, dt=dt, decimation=decimation,
                         lidar_backend="numpy", feedback_backend="direct",
                         depth_backend="constant", camera_backend="constant", auto_reset=True)
    policy, meta = load_policy(ckpt, env.single_observation_space)
    policy.eval()
    obs, _ = env.reset(seed=seed)
    obs_t = to_torch(obs, torch.device("cpu"))
    actions = []
    first_bad = None
    nf_events = 0
    t0 = time.perf_counter()
    for step in range(steps):
        with torch.no_grad():
            mean, _value, _hidden = policy.forward(obs_t, None) if hasattr(policy, "init_hidden") \
                else (*policy(obs_t), None)
        action = np.zeros((1, 4), np.float32) if zero_action else mean.cpu().numpy()
        actions.append(action[0].copy())
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:
            first_bad = {"step": step + 1, "error": f"{type(exc).__name__}: {exc}"}
            break
        read = env._last_read
        bad = {f: not np.isfinite(np.asarray(read[f])).all() for f in FIELDS if f in read}
        bad = {k: v for k, v in bad.items() if v}
        if int(np.sum(info.get("nonfinite", False))):
            nf_events += int(np.sum(info.get("nonfinite", False)))
            if first_bad is None:
                first_bad = {"step": step + 1, "env_nonfinite": True,
                             "fields": list(bad.keys()) if bad else None,
                             "action": action[0].tolist()}
        if bad and first_bad is None:
            first_bad = {"step": step + 1, "fields": list(bad), "pos": read["pos"][0].tolist()}
        if not np.isfinite(reward).all():
            first_bad = first_bad or {"step": step + 1, "reward_nonfinite": True}
            break
        obs_t = to_torch(obs, torch.device("cpu"))
    elapsed = time.perf_counter() - t0
    arr = np.asarray(actions) if actions else np.zeros((1, 4))
    return {
        "case": name, "physics": physics, "dt": dt, "decimation": decimation,
        "steps_run": len(actions), "wall_s": round(elapsed, 2),
        "nonfinite_env_events": int(nf_events),
        "first_nonfinite": first_bad,
        "action_absmax": round(float(np.abs(arr).max()), 4),
        "action_mean_abs": np.round(np.abs(arr).mean(axis=0), 4).tolist(),
        "action_saturation_ratio": round(float((np.abs(arr) > 0.99).mean()), 4),
        "policy": meta.get("policy_type"),
    }


def main():
    out = []
    out.append(run_case("5ms_zero", dt=0.005, decimation=10, physics="mujoco_cpu",
                        steps=400, seed=87010, zero_action=True))
    out.append(run_case("5ms_policy", dt=0.005, decimation=10, physics="mujoco_cpu",
                        steps=500, seed=87010))
    out.append(run_case("1ms_policy", dt=0.001, decimation=50, physics="mujoco_cpu",
                        steps=500, seed=87010))
    out.append(run_case("5ms_policy_seed11", dt=0.005, decimation=10, physics="mujoco_cpu",
                        steps=500, seed=87011))
    # newton XPBD CPU: same solver family as the crashed GPU pilot (small step budget on CPU)
    try:
        out.append(run_case("5ms_newton_cpu", dt=0.005, decimation=10, physics="newton_xpbd",
                            steps=120, seed=87010))
        out.append(run_case("1ms_newton_cpu", dt=0.001, decimation=50, physics="newton_xpbd",
                            steps=120, seed=87010))
    except Exception as exc:
        out.append({"case": "newton_cpu", "error": f"{type(exc).__name__}: {exc}"})
    result = {"checkpoint": CKPT, "cases": out}
    with open(f"{HERE}/diag_5ms_divergence.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    for row in out:
        print("CASE " + json.dumps({k: row.get(k) for k in
              ("case", "physics", "steps_run", "wall_s", "nonfinite_env_events",
               "first_nonfinite", "action_absmax", "action_saturation_ratio")}))


if __name__ == "__main__":
    main()
