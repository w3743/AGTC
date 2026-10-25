"""M1 detections verification: analytic mask/box/field semantics vs isaac_field reference packing.

Reference side: `FieldTask._observation()` camera-detection packing (task.py:1540-1580),
fed with the analytic detection dicts on identical inputs. Checks field order, scaling,
range/valid handling, clipping and truncation element-wise. The mask->box rules themselves
come from `isaac_field/env.py:_capture(kind='detector')` and are re-implemented here.
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import detections_caster as dct  # noqa: E402
import depth_caster as dc  # noqa: E402


def peak_rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def reference_pack(detections_list, scene, n_envs):
    """Feed analytic detections through the reference FieldTask._observation packing."""
    from isaac_field.feedback import FEEDBACK_DIM, FeedbackSnapshot
    from isaac_field.task import ABSOLUTE_GOAL_INPUT_SCHEMA, Config, FieldTask

    cfg = Config(size=100.0, plants=0, dry_fraction=0.0, terrain_enabled=False,
                 max_time=60.0, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
    task = FieldTask(cfg, "goto")
    task._pose_noise = np.zeros(2)
    task.goal = np.array([10.0, 10.0])
    task.time = 0.0
    task.last_action = np.zeros(4)
    task._new = {"lidar": 0.0, "depth": 0.0, "camera": 0.0}
    task._feedback_snapshot = None
    rows = []
    for dets in detections_list:
        task.frames = {
            "lidar": {"capture": 0.0, "estimated_pose": np.zeros(5), "valid": np.zeros(400, bool),
                      "ranges": np.zeros(400, np.float32)},
            "depth": {"capture": 0.0, "estimated_pose": np.zeros(5), "valid": np.zeros((160, 88), bool),
                      "values": np.zeros((160, 88), np.float32)},
            "camera": {"capture": 0.0, "estimated_pose": np.zeros(5), "detections": dets},
        }
        rows.append(task._observation()["detections"])
    return np.stack(rows, axis=0)


def center_locked_pose(scene):
    """Pose whose centre pixel (32,32) hits a leaf instance (exercises the range branch)."""
    leaf = 0
    center = np.asarray(scene.centers[leaf], np.float64)
    terrain = scene.terrain
    for dist in (1.2, 1.5, 2.0):
        for dyaw in np.linspace(-0.2, 0.2, 9):
            for dpitch in np.linspace(-0.2, 0.2, 9):
                x, y = center[0] - dist, center[1]
                z = float(terrain.surface_height(np.array([[x, y]]))[0]) + dc.ROOT_OFFSET + dc.CAMERA_HEIGHT
                delta = center - np.array([x, y, z])
                yaw = math.atan2(delta[1], delta[0]) + dyaw
                gp = math.asin(max(-0.99, min(0.99, delta[2] / np.linalg.norm(delta)))) + dpitch
                pose = np.array([[x, y, z, yaw, 0.0, gp]])
                mask = dct.cast_mask(pose, scene=scene)
                if mask[0, 32, 32] >= 0:
                    return pose
    return np.array([[center[0] - 1.5, center[1], center[2] + 0.2, math.pi, 0.0, 0.0]])


def main(n_envs=6):
    scene = dc.build_reference_scene()
    intr, cam_xy = dc.camera_model()
    poses = np.zeros((n_envs, 6))
    terrain = scene.terrain
    for i in range(n_envs):
        px, py = scene.xy[i]
        ang = i * (2 * np.pi / n_envs) + 0.4
        x, y = px + 3.0 * np.cos(ang), py + 3.0 * np.sin(ang)
        z = float(terrain.surface_height(np.array([[x, y]]))[0]) + dc.ROOT_OFFSET + dc.CAMERA_HEIGHT
        poses[i] = [x, y, z, ang + np.pi, 0.0, 0.03]
    # extra pose locked onto a leaf so the centre pixel (32,32) hits an instance and the
    # reference-only "range from the centre instance" branch is exercised
    poses = np.concatenate([poses, center_locked_pose(scene)], axis=0)
    n_envs = poses.shape[0]

    t0 = time.perf_counter()
    masks = dct.cast_mask(poses, scene=scene)
    center = dct.cast_center_distance(poses, scene=scene)
    t_mask = time.perf_counter() - t0
    t1 = time.perf_counter()
    rows, dets = dct.observation_from_masks(masks, center, scene.leaf_dry, noise=False)
    t_assemble = time.perf_counter() - t1
    ref = reference_pack(dets, scene, n_envs)

    diff = np.abs(rows.astype(np.float64) - ref.astype(np.float64))
    counts = [len(d) for d in dets]
    all_boxes = np.array([d["box"] for ds in dets for d in ds]) if any(counts) else np.zeros((0, 4))
    valid_ranges = [d["range"] is not None for ds in dets for d in ds]
    result = {
        "phase": "M1_detections_analytic",
        "poses": int(n_envs),
        "detector": {"size": dct.DETECTOR_SIZE, "focal": round(dct.DETECTOR_FOCAL, 4),
                     "cx": dct.DETECTOR_CX, "cone_radius_px": 32,
                     "clip_m": [dct.DETECTOR_CLIP_MIN, dct.DETECTOR_CLIP_MAX]},
        "masks": {"leaf_hit_ratio": round(float((masks >= 0).mean()), 4),
                  "distinct_instances_total": int(len(np.unique(masks[masks >= 0])))},
        "detections_per_frame": counts,
        "active_rows_total": int(sum(counts)),
        "rows_max_abs_error_vs_reference_packing": round(float(diff.max()), 9),
        "rows_mean_abs_error": round(float(diff.mean()), 9),
        "field_order_check": ["cx", "cy", "w", "h", "class_id", "confidence", "range/5", "range_valid"],
        "box_checks": {"min_w": round(float(all_boxes[:, 2].min()), 4) if len(all_boxes) else None,
                       "min_h": round(float(all_boxes[:, 3].min()), 4) if len(all_boxes) else None,
                       "max_w": round(float(all_boxes[:, 2].max()), 4) if len(all_boxes) else None,
                       "all_positive": bool(np.all(all_boxes[:, 2] > 0) and np.all(all_boxes[:, 3] > 0))
                       if len(all_boxes) else True,
                       "in_unit_range": bool(np.all(all_boxes >= 0.0) and np.all(all_boxes <= 1.0))
                       if len(all_boxes) else True},
        "range_field": {"detections_with_range": int(sum(valid_ranges)),
                        "detections_total": int(len(valid_ranges))},
        "performance": {"mask_render_s_per_frame": round(t_mask / n_envs, 4),
                        "assemble_s_per_frame": round(t_assemble / n_envs, 4),
                        "note": "CPU only; GPU/warp kernel for the 64x64 mask is the next optimisation if needed"},
        "peak_rss_mb": peak_rss(),
        "verdict": ("detections packing matches the reference FieldTask element-wise"
                    if float(diff.max()) <= 1e-7 else "MISMATCH - inspect"),
    }
    with open(f"{HERE}/detections_verification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DETECTIONS_SUMMARY " + json.dumps({k: result[k] for k in
          ("poses", "masks", "detections_per_frame", "active_rows_total",
           "rows_max_abs_error_vs_reference_packing", "box_checks", "range_field",
           "performance", "verdict")}))
    print("DETECTIONS_DONE")


if __name__ == "__main__":
    main()
