"""M1-depth provider integration check: env-produced depth channel == direct caster output.

Runs 2 MuJoCo-CPU envs with depth_backend="analytic" (CPU warp device, noise disabled),
drives until a depth frame is due, then recomputes the expected frame from the same
pose with depth_caster and compares (mask + metric depth + normalised channel).
"""
from __future__ import annotations

import json
import sys

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import depth_caster as dc  # noqa: E402
from native_env import NativeFieldEnv  # noqa: E402


def main(n_envs: int = 2, seed: int = 5) -> dict:
    env = NativeFieldEnv(num_envs=n_envs, seed=seed, depth_backend="analytic",
                         depth_device="cpu", depth_noise=False)
    env.reset(seed=seed)
    action = np.zeros((n_envs, 4))
    provider = env.provider
    frames = 0
    for _ in range(provider.depth_latency_steps + 9):
        env.step(action)
        if provider.step[0] % 8 == 0:
            frames += 1
            break
    ids = np.flatnonzero((provider.step % 8) == 0)
    read = env._last_read
    poses = np.stack([read["pos"][ids, 0], read["pos"][ids, 1],
                      read["pos"][ids, 2] + dc.CAMERA_HEIGHT,
                      np.asarray([np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                             1 - 2 * (q[2] ** 2 + q[3] ** 2)) for q in read["quat_wxyz"][ids]]),
                      read["gimbal_q"][ids, 0], read["gimbal_q"][ids, 1]], axis=1)
    dep, val, _dev = dc.cast_warp(poses, scene=provider._depth_scene, intrinsics=provider._depth_intr,
                                  camera_xy=provider._depth_cam_xy, device="cpu")
    expected = np.where(val, dep, 0.0) / dc.DEPTH_RANGE
    got = provider.depth_latest[ids]
    diff = np.abs(got - expected)
    return {
        "envs": int(n_envs),
        "depth_frames_rendered": int(frames),
        "channel_shape": list(got.shape),
        "obs_shape": list(env.provider.observation_shapes()["depth"]),
        "valid_ratio": round(float(val.mean()), 4),
        "mask_mismatch_pixels": int(((got > 0) != val).sum()),
        "max_abs_diff": round(float(diff.max()), 8),
        "mean_abs_diff": round(float(diff.mean()), 8),
        "all_zero_or_one": bool(np.all((expected >= 0.0) & (expected <= 1.0))),
        "verdict": ("provider depth equals the analytic caster"
                    if float(diff.max()) <= 1e-6 and int(((got > 0) != val).sum()) == 0
                    else "MISMATCH - inspect"),
    }


if __name__ == "__main__":
    result = main()
    with open(f"{HERE}/depth_integration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DEPTH_INTEGRATION " + json.dumps(result))
