"""Record a native-stack rollout to JSON for visualization.

Runs one fixed seed deterministically (mean actions) on the native env with the
pre-protocol LSTM pilot checkpoint and records per-step state so an offline
viewer can animate it. Constant depth/detections are used only to keep the
recording fast; physics/observation other than those match the pilot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "proto"))
sys.path.insert(0, os.path.join(ROOT, "train"))

from native_env import NativeFieldEnv  # noqa: E402
from eval_native import load_policy  # noqa: E402
from train_m0 import to_torch  # noqa: E402
from reward_fast_stop import pitch_roll_from_quat_wxyz  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=87010)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--sim-dt", type=float, default=1e-3)
    parser.add_argument("--decimation", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lidar", type=str, default="warp")
    parser.add_argument("--feedback", type=str, default="vqf")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if abs(args.sim_dt * args.decimation - 0.05) > 1e-9:
        parser.error("sim_dt*decimation must be 0.05")
    device = torch.device(args.device)
    env = NativeFieldEnv(
        num_envs=1,
        seed=args.seed,
        dt=args.sim_dt,
        decimation=args.decimation,
        physics="newton_xpbd",
        lidar_backend=args.lidar,
        device="cuda:0" if args.device == "cuda" else "cpu",
        feedback_backend=args.feedback,
        feedback_noise=True,
        depth_backend="analytic",
        depth_device="cuda:0" if args.device == "cuda" else "cpu",
        camera_backend="analytic_warp",
        camera_device="cuda:0" if args.device == "cuda" else "cpu",
        auto_reset=False,
    )
    policy, meta = load_policy(args.checkpoint, env.single_observation_space)
    policy.to(device)
    obs, _ = env.reset(seed=args.seed)
    is_lstm = hasattr(policy, "init_hidden")
    hidden = policy.init_hidden(1, device) if is_lstm else None

    frames = []
    goal = env.goal_xy[0].astype(float)
    for step in range(args.max_steps):
        obs_t = to_torch(obs, device)
        with torch.no_grad():
            if is_lstm:
                mean, _value, hidden = policy.forward(obs_t, hidden)
            else:
                mean, _value = policy(obs_t)
        action = mean.cpu().numpy()[0]
        obs, _reward, terminated, truncated, info = env.step(action[None, :].astype(np.float32))
        read = env._last_read
        pos = read["pos"][0].astype(float)
        quat = read["quat_wxyz"][0].astype(float)
        roll, pitch = pitch_roll_from_quat_wxyz(quat)
        yaw = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
        distance = float(np.linalg.norm(goal - pos[:2]))
        frames.append({
            "t": round((step + 1) * 0.05, 3),
            "p": [round(float(v), 4) for v in pos],
            "q": [round(float(v), 5) for v in quat],
            "yaw": round(float(yaw), 4),
            "roll": round(float(roll), 4),
            "pitch": round(float(pitch), 4),
            "speed": round(float(np.linalg.norm(read["lin_vel"][0, :2])), 4),
            "ang_speed": round(float(np.linalg.norm(read["ang_vel"][0])), 4),
            "dist": round(distance, 4),
            "a": [round(float(v), 4) for v in action],
            "flipped": bool(info["flipped"][0]),
            "success": bool(info["success"][0]),
        })
        if is_lstm:
            done_t = torch.tensor([bool(terminated[0]) or bool(truncated[0])], device=device)
            hidden = policy.reset_hidden(hidden, done_t)
        if bool(terminated[0]) or bool(truncated[0]):
            break

    out = args.out or os.path.join(HERE, f"rollout_seed{args.seed}.json")
    payload = {
        "banner": "NATIVE_VIZ_ROLLOUT",
        "note": "pre-protocol LSTM pilot checkpoint; mean actions; native Newton XPBD; not a capability run",
        "checkpoint": args.checkpoint,
        "policy": meta,
        "seed": args.seed,
        "sim_dt": args.sim_dt,
        "decimation": args.decimation,
        "policy_dt": 0.05,
        "goal": [round(float(v), 4) for v in goal],
        "frames": frames,
    }
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    print(f"WROTE {out} frames={len(frames)} final_dist={frames[-1]['dist'] if frames else None} "
          f"flipped={frames[-1]['flipped'] if frames else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
