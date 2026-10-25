"""Publish native WSL physics state as a live-state packet for the existing Viser viewer.

Runs the native Newton/MuJoCo-free env (1 world), reads per-body poses from the
Newton state, and writes the same schema-v1 packet consumed by
`scripts/view_training_state_web.py` (the viewer reused from the Isaac pipeline).
Four virtual environments are emitted (same motion, offset origins) because the
viewer expects the four-environment layout.

Body names are mapped from Newton labels (w0_/base, ...) to the USD robot prim
names (base_link, ...); native-only front-guide bodies are emitted as names
without USD prims, which the viewer renders as proxy boxes.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROTO = os.path.join(ROOT, "proto")
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, REPO)

from native_env import NativeFieldEnv  # noqa: E402
from eval_native import load_policy  # noqa: E402
from train_m0 import to_torch  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "isaac_field_live_state", os.path.join(REPO, "isaac_field", "live_state.py"))
_live_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_live_state)
LiveStateExporter = _live_state.LiveStateExporter

def _viewer_path(path: Path) -> str:
    """Windows viewers need C:\\... while WSL python sees /mnt/c/..."""
    text = str(path)
    if text.startswith("/mnt/c/"):
        return "C:\\" + text[len("/mnt/c/"):].replace("/", "\\")
    return text


ROBOT_USD = _viewer_path(Path(__file__).resolve().parent / "field_robot_19.usda")
TERRAIN_USD = _viewer_path(Path(REPO) / "results" / "isaac_field" / "assets" / "010a1c5428ffe08e" / "terrain.usda")


def usd_name(native: str) -> str:
    raw = native.split("_/", 1)[-1]
    if raw == "base":
        return "base_link"
    if raw == "gimbal_yaw":
        return "gimbal_yaw_link"
    if raw == "gimbal_pitch":
        return "gimbal_pitch_link"
    if raw.startswith("left_wheel_") or raw.startswith("right_wheel_"):
        return raw.replace("_link", "_wheel_link")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=87010)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--real-time", action="store_true", default=True)
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spacing", type=float, default=2.5)
    args = parser.parse_args()

    env = NativeFieldEnv(
        num_envs=1, seed=args.seed, dt=1e-3, decimation=50,
        physics="newton_xpbd", lidar_backend="warp", device="cuda:0",
        feedback_backend="vqf", feedback_noise=True,
        depth_backend="analytic", depth_device="cuda:0",
        camera_backend="analytic_warp", camera_device="cuda:0",
        auto_reset=False,
    )
    backend = env.backend
    model = backend.model
    native_names = [str(label) for label in model.body_label]
    body_names = [usd_name(name) for name in native_names]
    if len(body_names) != 19:
        raise RuntimeError(f"expected 19 bodies for the viewer, got {len(body_names)}")
    print("body_names", body_names, flush=True)

    policy, meta = None, {}
    if args.checkpoint:
        policy, meta = load_policy(args.checkpoint, env.single_observation_space)
        policy.to(torch.device(args.device))
        policy.eval()
    is_lstm = policy is not None and hasattr(policy, "init_hidden")
    device = torch.device(args.device)
    hidden = policy.init_hidden(1, device) if is_lstm else None
    obs, _ = env.reset(seed=args.seed)
    goal = env.goal_xy[0].astype(np.float64)
    origins = np.array([[0.0, 0.0, 0.0], [args.spacing, 0.0, 0.0],
                        [0.0, args.spacing, 0.0], [args.spacing, args.spacing, 0.0]])
    exporter = LiveStateExporter(args.state_file, robot_usd_path=ROBOT_USD,
                                 terrain_usd_path=TERRAIN_USD, body_names=body_names)
    exporter.run_id = "native_" + (os.path.basename(args.checkpoint) if args.checkpoint else "zero")
    print("run_id", exporter.run_id, flush=True)

    t0 = time.monotonic()
    sim_time = 0.0
    for step in range(args.max_steps):
        if policy is not None:
            obs_t = to_torch(obs, device)
            with torch.no_grad():
                if is_lstm:
                    mean, _value, hidden = policy.forward(obs_t, hidden)
                else:
                    mean, _value = policy(obs_t)
            action = mean.cpu().numpy()
        else:
            action = np.zeros((1, 4), dtype=np.float32)
        obs, _reward, terminated, truncated, info = env.step(action.astype(np.float32))
        sim_time += env.policy_dt
        if is_lstm:
            done_t = torch.tensor([bool(terminated[0]) or bool(truncated[0])], device=device)
            hidden = policy.reset_hidden(hidden, done_t)

        body_q = backend.state_0.body_q.numpy()  # (19, 7) px py pz qx qy qz qw
        base = int(backend.base_body[0])
        if not np.isfinite(body_q[base]).all() or body_q[base, 2] < 0.0 or step % 400 == 399:
            obs, _ = env.reset(seed=args.seed + step + 1)
            if is_lstm:
                hidden = policy.init_hidden(1, device)
            continue
        root_pose = np.concatenate([body_q[base, 0:3], body_q[base, 3:7]])
        base_xy = body_q[base, 0:2].copy()
        origins = np.array([
            [base_xy[0], base_xy[1], 0.0],
            [base_xy[0] + args.spacing, base_xy[1], 0.0],
            [base_xy[0], base_xy[1] + args.spacing, 0.0],
            [base_xy[0] + args.spacing, base_xy[1] + args.spacing, 0.0],
        ])
        body_link_pose_w = np.zeros((4, len(body_names), 7))
        for e in range(4):
            for b in range(len(body_names)):
                pose = body_q[b].copy()
                pose[0] += origins[e, 0] - base_xy[0]
                pose[1] += origins[e, 1] - base_xy[1]
                body_link_pose_w[e, b] = pose
        root_rows = body_link_pose_w[:, base, :]
        goals_local_xy = np.tile(goal - base_xy, (4, 1))
        task_times = np.full(4, sim_time)
        exporter.write(root_link_pose_w=root_rows, body_link_pose_w=body_link_pose_w,
                       env_origins_w=origins, goals_local_xy=goals_local_xy,
                       task_times_s=task_times, sim_time_s=sim_time)
        if args.real_time:
            target = t0 + (step + 1) * env.policy_dt
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if bool(terminated[0]) or bool(truncated[0]):
            obs, _ = env.reset(seed=args.seed + step + 1)
            if is_lstm:
                hidden = policy.init_hidden(1, device)
    print("DONE steps", args.max_steps, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
