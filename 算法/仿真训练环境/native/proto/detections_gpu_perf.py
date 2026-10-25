"""GPU performance: warp detector mask for 64 camera poses (64x64 = 4,096 rays/pose)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import detections_caster as dct  # noqa: E402
from m1_detections_warp_verify import build_poses  # noqa: E402
import depth_caster as dc  # noqa: E402


def main():
    scene = dc.build_reference_scene()
    poses = build_poses(scene, 64)
    dev = "cuda:0"
    t0 = time.perf_counter()
    masks, d = dct.cast_mask_warp(poses[:2], scene=scene, device=dev)
    compile_s = time.perf_counter() - t0
    reps = 3
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        masks, d = dct.cast_mask_warp(poses, scene=scene, device=dev)
        times.append(time.perf_counter() - t0)
    best = min(times)
    # single-launch batched version
    t0 = time.perf_counter()
    bmasks, bd = dct.cast_mask_warp_batch(poses[:2], scene=scene, device=dev)
    bcompile = time.perf_counter() - t0
    btimes = []
    for _ in range(reps):
        t0 = time.perf_counter()
        bmasks, bd = dct.cast_mask_warp_batch(poses, scene=scene, device=dev)
        btimes.append(time.perf_counter() - t0)
    bbest = min(btimes)
    same = bool(np.array_equal(masks, bmasks))
    result = {
        "device": d,
        "poses": int(poses.shape[0]),
        "pixels_per_frame": int(dct.DETECTOR_SIZE ** 2),
        "compile_warmup_s": round(compile_s, 3),
        "batch_s": [round(t, 5) for t in times],
        "best_batch_s": round(best, 5),
        "ms_per_frame": round(best / poses.shape[0] * 1e3, 4),
        "frames_per_s": round(poses.shape[0] / best, 1),
        "batch_kernel": {
            "compile_warmup_s": round(bcompile, 3),
            "batch_s": [round(t, 5) for t in btimes],
            "best_batch_s": round(bbest, 5),
            "ms_per_frame": round(bbest / poses.shape[0] * 1e3, 4),
            "frames_per_s": round(poses.shape[0] / bbest, 1),
            "masks_equal_per_pose_grid": same,
            "target_batch_ms": 30.0,
            "meets_target": bool(bbest * 1e3 <= 30.0),
        },
        "leaf_hit_ratio": round(float((masks >= 0).mean()), 4),
        "distinct_instances": int(len(np.unique(masks[masks >= 0]))),
        "target_batch_ms_per_pose_launch": 30.0,
        "meets_target_per_pose_launch": bool(best * 1e3 <= 30.0),
    }
    with open(f"{HERE}/runs/detections_gpu_perf.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DETECTIONS_GPU_PERF " + json.dumps(result))


if __name__ == "__main__":
    main()
