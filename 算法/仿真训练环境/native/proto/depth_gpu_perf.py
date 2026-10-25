"""GPU performance: analytic depth cast for 64 camera poses (per-frame and batch)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import depth_caster as dc  # noqa: E402


def build_poses(scene, n=64):
    rng = np.random.default_rng(123)
    poses = np.zeros((n, 6))
    terrain = scene.terrain
    for i in range(n):
        p = scene.xy[rng.integers(0, len(scene.xy))]
        ang = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(1.5, 4.0)
        x, y = p[0] + dist * np.cos(ang), p[1] + dist * np.sin(ang)
        z = float(terrain.surface_height(np.array([[x, y]]))[0]) + dc.ROOT_OFFSET + dc.CAMERA_HEIGHT
        poses[i] = [x, y, z, ang + np.pi + rng.uniform(-0.3, 0.3),
                    rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.2)]
    return poses


def main():
    scene = dc.build_reference_scene()
    intr, cam_xy = dc.camera_model()
    poses = build_poses(scene, 64)
    dev = "cuda:0"
    # warmup + JIT compile
    t0 = time.perf_counter()
    w_depth, w_valid, w_dev = dc.cast_warp(poses[:4], scene=scene, intrinsics=intr,
                                           camera_xy=cam_xy, device=dev)
    compile_s = time.perf_counter() - t0
    reps = 3
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        depth, valid, d = dc.cast_warp(poses, scene=scene, intrinsics=intr,
                                       camera_xy=cam_xy, device=dev)
        times.append(time.perf_counter() - t0)
    best = min(times)
    result = {
        "device": d,
        "poses": int(poses.shape[0]),
        "pixels_per_frame": int(dc.DEPTH_HEIGHT * dc.DEPTH_WIDTH),
        "rays_per_batch": int(poses.shape[0] * dc.DEPTH_HEIGHT * dc.DEPTH_WIDTH),
        "compile_warmup_s": round(compile_s, 3),
        "batch_s": [round(t, 4) for t in times],
        "best_batch_s": round(best, 4),
        "ms_per_frame": round(best / poses.shape[0] * 1e3, 3),
        "ms_per_1000_rays": round(best / (poses.shape[0] * dc.DEPTH_HEIGHT * dc.DEPTH_WIDTH) * 1e6, 4),
        "frames_per_s": round(poses.shape[0] / best, 1),
        "valid_ratio": round(float(valid.mean()), 4),
        "target_ms_per_frame": [4.0, 6.0],
        "meets_target": bool(4.0 <= best / poses.shape[0] * 1e3 <= 6.0),
    }
    with open(f"{HERE}/runs/depth_gpu_perf.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DEPTH_GPU_PERF " + json.dumps(result))


if __name__ == "__main__":
    main()
