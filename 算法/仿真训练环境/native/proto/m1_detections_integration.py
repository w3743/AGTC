"""M1 detections provider integration check (CPU).

2 MuJoCo-CPU envs with camera_backend="analytic" (noise off); drives until a detector
frame is due, recomputes the expected detections from the same 6-column pose with
detections_caster, and compares the (16, 8) channel element-wise.
"""
from __future__ import annotations

import json
import sys

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import detections_caster as dct  # noqa: E402
import depth_caster as dc  # noqa: E402
from native_env import NativeFieldEnv  # noqa: E402


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main(n_envs: int = 2, seed: int = 6) -> dict:
    env = NativeFieldEnv(num_envs=n_envs, seed=seed, camera_backend="analytic",
                         camera_noise=False)
    env.reset(seed=seed)
    provider = env.provider
    action = np.zeros((n_envs, 4))
    # spawn next to plants and face them so detections actually exist (random field spawns
    # usually see no leaf in the 64x64 detector FOV)
    scene = provider._det_scene
    spawn = np.zeros((n_envs, 4))
    goals = np.zeros((n_envs, 2))
    for i in range(n_envs):
        px, py = scene.xy[i]
        ang = i * (2 * np.pi / n_envs) + 0.4
        spawn[i] = [px + 3.0 * np.cos(ang), py + 3.0 * np.sin(ang), ang + np.pi, 0.01]
        goals[i] = [50.0, 50.0]
    env.reset_with_poses(spawn, goals)
    for _ in range(16):
        env.step(action)
        if (provider.step[0] % 4) == 0:
            break
    ids = np.flatnonzero((provider.step % 4) == 0)
    read = env._last_read
    poses = np.stack([read["pos"][ids, 0], read["pos"][ids, 1],
                      read["pos"][ids, 2] + dc.CAMERA_HEIGHT,
                      np.asarray([yaw_from_quat(q) for q in read["quat_wxyz"][ids]]),
                      read["gimbal_q"][ids, 0], read["gimbal_q"][ids, 1]], axis=1)
    masks = dct.cast_mask(poses, scene=provider._det_scene)
    center = dct.cast_center_distance(poses, scene=provider._det_scene)
    rows, dets = dct.observation_from_masks(masks, center, provider._det_scene.leaf_dry,
                                            rng=None, noise=False)
    got = provider.det_latest[ids]
    diff = np.abs(got.astype(np.float64) - rows.astype(np.float64))
    return {
        "envs": int(n_envs),
        "channel_shape": list(got.shape),
        "obs_shape": list(env.provider.observation_shapes()["detections"]),
        "active_rows": [int(np.count_nonzero(r[:, 2] > 0)) for r in rows],
        "max_abs_diff": round(float(diff.max()), 9),
        "mean_abs_diff": round(float(diff.mean()), 9),
        "verdict": ("provider detections equal the analytic caster"
                    if float(diff.max()) <= 1e-7 else "MISMATCH - inspect"),
    }


if __name__ == "__main__":
    result = main()
    with open(f"{HERE}/detections_integration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DETECTIONS_INTEGRATION " + json.dumps(result))
