"""Compare captured Gazebo depth points with the source field's analytic terrain.

This is an offline check: it never launches or controls Gazebo.  It expects a
runner capture directory, its original field_sim scene JSON, and (optionally)
the matching state/manifest when they are not stored beside the scene/capture.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


NATIVE_WIDTH = 352
NATIVE_HEIGHT = 640
NATIVE_K = np.array([
    [393.29918354972216, 0.0, 177.54501597086588],
    [0.0, 393.29918354972216, 323.73144276936847],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _paired_state(scene_path: Path) -> Path | None:
    name = scene_path.name
    if name.endswith("_scene.json"):
        candidate = scene_path.with_name(name[:-11] + "_state.json")
        if candidate.is_file():
            return candidate
    return None


def _finite_float(value, default=math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _depth_meta(doc: dict) -> dict:
    summary = doc.get("export_summary", doc)
    return (summary.get("vehicle", {}).get("sensors", {}).get("depth", {})
            if isinstance(summary, dict) else {})


def _intrinsics(capture: Path, scene: dict, state: dict | None,
                manifest_path: Path | None) -> tuple[np.ndarray, str, dict]:
    cfg = scene.get("config", {})
    width = int(cfg.get("depth_width", 88))
    height = int(cfg.get("depth_height", 160))
    manifest_doc = None
    candidate = manifest_path or (capture / "acceptance.json")
    if candidate.is_file():
        manifest_doc = _json(candidate)
    manifest_meta = _depth_meta(manifest_doc or {})
    source_k = None
    if state and state.get("depth", {}).get("intrinsics") is not None:
        source_k = np.asarray(state["depth"]["intrinsics"], dtype=np.float64)
    manifest_k = manifest_meta.get("intrinsics_K")
    if manifest_k is not None:
        matrix = np.asarray(manifest_k, dtype=np.float64)
        origin = f"capture manifest: {candidate}"
    elif source_k is not None:
        matrix = source_k
        origin = "matching source state depth.intrinsics"
    else:
        sx, sy = width / NATIVE_WIDTH, height / NATIVE_HEIGHT
        matrix = NATIVE_K.copy()
        matrix[0, 0] *= sx
        matrix[0, 2] = (matrix[0, 2] + 0.5) * sx - 0.5
        matrix[1, 1] *= sy
        matrix[1, 2] = (matrix[1, 2] + 0.5) * sy - 0.5
        origin = "scaled LightStereo native intrinsics fallback"
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid depth intrinsics from {origin}")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"nonpositive focal length from {origin}")
    agreement = None
    if source_k is not None and manifest_k is not None:
        agreement = float(np.max(np.abs(source_k - np.asarray(manifest_k, dtype=np.float64))))
    return matrix, origin, {"source_state_manifest_max_abs_difference": agreement}


def _pose_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for row in rows:
        item = [_finite_float(row.get(k)) for k in
                ("sim_time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")]
        if all(math.isfinite(v) for v in item):
            q = np.asarray(item[4:8], dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm > 1e-8:
                values.append([*item[:4], *(q / norm)])
    if not values:
        return np.empty(0), np.empty((0, 7))
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values[:, 0], kind="stable")
    return values[order, 0], values[order, 1:]


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _nearest(times: np.ndarray, values: np.ndarray, stamp: float):
    if not len(times) or not math.isfinite(stamp):
        return None, math.inf
    index = int(np.searchsorted(times, stamp))
    candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
    best = min(candidates, key=lambda i: abs(float(times[i]) - stamp))
    return values[best], abs(float(times[best]) - stamp)


def _joint_series(rows: list[dict], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for row in rows:
        if row.get("joint") != name or str(row.get("axis", "1")) != "1":
            continue
        t = _finite_float(row.get("sim_time_s"))
        p = _finite_float(row.get("position_rad"))
        v = _finite_float(row.get("velocity"))
        if math.isfinite(t) and math.isfinite(p):
            values.append((t, p, v))
    if not values:
        return np.empty(0), np.empty(0), np.empty(0)
    data = np.asarray(values, dtype=np.float64)
    data = data[np.argsort(data[:, 0], kind="stable")]
    # Keep the last sample if a callback wrote more than one row at one sim time.
    _, reverse_indices = np.unique(data[::-1, 0], return_index=True)
    data = data[np.sort(len(data) - 1 - reverse_indices)]
    return data[:, 0], data[:, 1], data[:, 2]


def _distribution(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0}
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(len(a)), "min": float(np.min(a)), "max": float(np.max(a)),
        "mean": float(np.mean(a)), "std": float(np.std(a)),
        "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
        "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
    }


def _highpass_stats(times: np.ndarray, positions: np.ndarray, window_s: float) -> dict:
    if len(times) < 5 or times[-1] - times[0] < window_s:
        return {"window_s": window_s, "count": 0, "reason": "insufficient time span"}
    unwrapped = np.unwrap(positions)
    cumulative = np.r_[0.0, np.cumsum(unwrapped)]
    half = window_s / 2
    left = np.searchsorted(times, times - half, side="left")
    right = np.searchsorted(times, times + half, side="right")
    counts = np.maximum(1, right - left)
    smooth = (cumulative[right] - cumulative[left]) / counts
    residual = unwrapped - smooth
    interior = (times >= times[0] + half) & (times <= times[-1] - half)
    residual = residual[interior]
    if not len(residual):
        return {"window_s": window_s, "count": 0, "reason": "no complete windows"}
    return {
        "window_s": window_s, "count": int(len(residual)),
        "rms_rad": float(np.sqrt(np.mean(residual ** 2))),
        "abs_p95_rad": float(np.percentile(np.abs(residual), 95)),
        "abs_max_rad": float(np.max(np.abs(residual))),
    }


def _gimbal_report(joint_rows: list[dict], hf_window_s: float) -> dict:
    result = {}
    for axis_name, joint_name in (("yaw", "gimbal_yaw_joint"), ("pitch", "gimbal_pitch_joint")):
        times, position, velocity = _joint_series(joint_rows, joint_name)
        if not len(times):
            result[axis_name] = {"joint": joint_name, "count": 0, "available": False}
            continue
        unwrapped = np.unwrap(position)
        dt = np.diff(times)
        delta = np.diff(unwrapped)
        good = dt > 1e-9
        diff_rate = delta[good] / dt[good]
        report = {
            "joint": joint_name, "available": True, "count": int(len(times)),
            "time_span_s": float(times[-1] - times[0]),
            "position_rad": _distribution(position),
            "sample_delta_abs_rad": _distribution(np.abs(delta)),
            "finite_difference_abs_rate_rad_s": _distribution(np.abs(diff_rate)),
            "reported_velocity_abs_rad_s": _distribution(np.abs(velocity)),
            "high_frequency_position_residual": _highpass_stats(times, position, hf_window_s),
        }
        result[axis_name] = report
    return result


def _terrain_z(x: np.ndarray, y: np.ndarray, terrain: dict) -> np.ndarray:
    base = float(terrain.get("base_height", 0.0))
    if not terrain.get("enabled", False):
        return np.full_like(x, base, dtype=np.float64)
    waves = np.asarray(terrain.get("wave_vectors4x2", []), dtype=np.float64)
    amplitudes = np.asarray(terrain.get("wave_amplitudes4", []), dtype=np.float64)
    phases = np.asarray(terrain.get("wave_phases4", []), dtype=np.float64)
    if waves.shape != (4, 2) or amplitudes.shape != (4,) or phases.shape != (4,):
        raise ValueError("scene terrain must include four wave vectors, amplitudes, and phases")
    z = np.full_like(x, base, dtype=np.float64)
    for wave, amplitude, phase in zip(waves, amplitudes, phases):
        z += amplitude * np.sin(x * wave[0] + y * wave[1] + phase)
    return z


def validate(capture: Path, scene_path: Path, state_path: Path | None,
             manifest_path: Path | None, threshold: float, pixel_stride: int,
             pose_gap: float, joint_gap: float, hf_window_s: float) -> dict:
    scene = _json(scene_path)
    state = _json(state_path) if state_path and state_path.is_file() else None
    acceptance_path = manifest_path or capture / "acceptance.json"
    acceptance = _json(acceptance_path) if acceptance_path.is_file() else {}
    capture_scene_id = (acceptance.get("export_summary", {}).get(
        "scene_id", acceptance.get("scene_id")) if acceptance else None)
    scene_id = scene.get("scene_id")
    if capture_scene_id is not None and scene_id is not None and int(capture_scene_id) != int(scene_id):
        raise ValueError(f"scene id mismatch: capture={capture_scene_id}, source={scene_id}")

    depth_path = capture / "depth.npy"
    stamp_path = capture / "depth_sim_timestamps.npy"
    pose_path = capture / "world_trajectory.csv"
    joints_path = capture / "joint_states.csv"
    for required in (depth_path, stamp_path, pose_path, joints_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    depth = np.load(depth_path, mmap_mode="r")
    stamps = np.asarray(np.load(stamp_path), dtype=np.float64).reshape(-1)
    if depth.ndim != 3 or len(depth) != len(stamps):
        raise ValueError(f"depth/timestamp shape mismatch: {depth.shape} vs {stamps.shape}")
    intrinsics, intrinsic_source, intrinsic_check = _intrinsics(
        capture, scene, state, manifest_path)
    fx, fy, cx, cy = (float(intrinsics[0, 0]), float(intrinsics[1, 1]),
                      float(intrinsics[0, 2]), float(intrinsics[1, 2]))
    if (depth.shape[2], depth.shape[1]) != (int(scene.get("config", {}).get("depth_width", depth.shape[2])),
                                                   int(scene.get("config", {}).get("depth_height", depth.shape[1]))):
        raise ValueError("captured depth resolution differs from source scene config")
    if state and state.get("depth", {}).get("resolution"):
        if tuple(map(int, state["depth"]["resolution"])) != (depth.shape[2], depth.shape[1]):
            raise ValueError("source state depth resolution differs from capture")

    cfg = scene.get("config", {})
    camera_height = float(cfg.get("camera_height", 0.4))
    depth_meta = _depth_meta(acceptance)
    ranges = depth_meta.get("range_m", [cfg.get("depth_min_range", 0.0),
                                          cfg.get("depth_range", 100.0)])
    min_depth, max_depth = float(ranges[0]), float(ranges[1])
    size = float(scene["size"])
    terrain = scene.get("terrain", {})
    time_pose, poses = _pose_rows(_csv(pose_path))
    joints = _csv(joints_path)
    yaw_t, yaw_p, _ = _joint_series(joints, "gimbal_yaw_joint")
    pitch_t, pitch_p, _ = _joint_series(joints, "gimbal_pitch_joint")

    residuals = []
    frame_summaries = []
    skipped = {"invalid_timestamp": 0, "pose_unavailable_or_stale": 0,
               "gimbal_unavailable_or_stale": 0, "no_valid_depth": 0}
    pose_deltas, gimbal_deltas = [], []
    sampled_pixels = 0
    valid_depth_pixels = 0
    matched_valid_depth_pixels = 0
    in_bounds_points = 0
    inliers = 0
    frame_h, frame_w = depth.shape[1:]
    stride = max(1, int(pixel_stride))
    us = np.arange(0, frame_w, stride, dtype=np.float64)
    vs = np.arange(0, frame_h, stride, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(us, vs, indexing="xy")

    for frame_index, stamp in enumerate(stamps):
        image = np.asarray(depth[frame_index, ::stride, ::stride], dtype=np.float64)
        sampled_pixels += int(image.size)
        valid = np.isfinite(image) & (image >= min_depth) & (image <= max_depth)
        valid_depth_pixels += int(np.count_nonzero(valid))
        if not math.isfinite(float(stamp)):
            skipped["invalid_timestamp"] += 1
            continue
        pose, pose_dt = _nearest(time_pose, poses, float(stamp))
        yaw, yaw_dt = _nearest(yaw_t, yaw_p, float(stamp))
        pitch, pitch_dt = _nearest(pitch_t, pitch_p, float(stamp))
        if pose is None or pose_dt > pose_gap:
            skipped["pose_unavailable_or_stale"] += 1
            continue
        if yaw is None or pitch is None or yaw_dt > joint_gap or pitch_dt > joint_gap:
            skipped["gimbal_unavailable_or_stale"] += 1
            continue
        pose_deltas.append(pose_dt)
        gimbal_deltas.extend((yaw_dt, pitch_dt))
        if not np.any(valid):
            skipped["no_valid_depth"] += 1
            frame_summaries.append({"frame": int(frame_index), "sim_time_s": float(stamp),
                                    "valid_depth_pixels": 0, "ground_inliers": 0})
            continue
        d = image[valid]
        matched_valid_depth_pixels += int(len(d))
        u = u_grid[valid]
        v = v_grid[valid]
        xcam = (u - cx) * d / fx
        yup = (cy - v) * d / fy

        # Sensor axes in the body frame; image rows increase downwards.
        gy, gp = float(yaw), float(pitch)
        forward = np.array([math.cos(gp) * math.cos(gy),
                            math.cos(gp) * math.sin(gy), math.sin(gp)])
        right = np.array([math.sin(gy), -math.cos(gy), 0.0])
        up = np.cross(right, forward)
        rotation = _quat_matrix(pose[3:7])
        origin = pose[:3] + rotation @ np.array([0.0, 0.0, camera_height])
        f_world, r_world, u_world = rotation @ forward, rotation @ right, rotation @ up
        points = (origin[None, :] + d[:, None] * f_world[None, :]
                  + xcam[:, None] * r_world[None, :] + yup[:, None] * u_world[None, :])
        source_x = points[:, 0] + size / 2.0
        source_y = points[:, 1] + size / 2.0
        inside = ((source_x >= 0.0) & (source_x <= size)
                  & (source_y >= 0.0) & (source_y <= size))
        if not np.any(inside):
            frame_summaries.append({"frame": int(frame_index), "sim_time_s": float(stamp),
                                    "valid_depth_pixels": int(len(d)), "in_field_points": 0,
                                    "ground_inliers": 0})
            continue
        in_bounds_points += int(np.count_nonzero(inside))
        expected_z = _terrain_z(source_x[inside], source_y[inside], terrain)
        frame_resid = points[inside, 2] - expected_z
        residuals.append(frame_resid)
        frame_inliers = int(np.count_nonzero(np.abs(frame_resid) < threshold))
        inliers += frame_inliers
        frame_summaries.append({
            "frame": int(frame_index), "sim_time_s": float(stamp),
            "pose_nearest_delta_s": float(pose_dt),
            "gimbal_yaw_rad": gy, "gimbal_pitch_rad": gp,
            "valid_depth_pixels": int(len(d)), "in_field_points": int(len(frame_resid)),
            "ground_inliers": frame_inliers,
            "ground_inlier_fraction": frame_inliers / len(frame_resid) if len(frame_resid) else 0.0,
            "residual_median_m": float(np.median(frame_resid)) if len(frame_resid) else None,
        })

    all_residuals = np.concatenate(residuals) if residuals else np.empty(0)
    gimbal_report = _gimbal_report(joints, hf_window_s)
    report = {
        "kind": "gazebo_depth_ground_alignment",
        "capture": str(capture.resolve()), "scene_json": str(scene_path.resolve()),
        "scene_id": scene_id, "capture_scene_id": capture_scene_id,
        "coordinate_mapping": "source_xy = Gazebo_world_xy + field_size/2; source_z retained",
        "camera_model": {
            "depth_convention": "camera_z", "camera_height_m": camera_height,
            "intrinsics_K": intrinsics.tolist(), "intrinsics_source": intrinsic_source,
            "intrinsics_crosscheck": intrinsic_check,
            "body_frame_basis": "forward=[cos(p)cos(y),cos(p)sin(y),sin(p)]; right=[sin(y),-cos(y),0]; up=right x forward",
            "image_vertical": "row increases down; backprojection uses +camera-up*(cy-v)*depth/fy",
            "timestamp_association": "nearest world pose and actual joint sample",
        },
        "terrain_model": {
            "enabled": bool(terrain.get("enabled", False)),
            "base_height_m": float(terrain.get("base_height", 0.0)),
            "wave_count": len(terrain.get("wave_amplitudes4", [])),
            "inlier_threshold_abs_m": float(threshold),
        },
        "counts": {
            "depth_frames": int(len(depth)), "frames_used": len(frame_summaries) - skipped["no_valid_depth"],
            "sampled_pixels": sampled_pixels, "valid_depth_pixels": valid_depth_pixels,
            "valid_depth_pixels_with_pose_and_gimbal": matched_valid_depth_pixels,
            "valid_depth_fraction_sampled": valid_depth_pixels / sampled_pixels if sampled_pixels else 0.0,
            "in_field_world_points": in_bounds_points,
            "ground_inliers": inliers,
            "ground_inlier_fraction_of_in_field_points": inliers / in_bounds_points if in_bounds_points else 0.0,
            "ground_inlier_fraction_of_valid_depth_pixels": inliers / valid_depth_pixels if valid_depth_pixels else 0.0,
            "skipped_frames": skipped,
        },
        "pose_timestamp_delta_s": _distribution(np.asarray(pose_deltas)),
        "gimbal_timestamp_delta_s": _distribution(np.asarray(gimbal_deltas)),
        "terrain_z_residual_m": _distribution(all_residuals),
        "has_ground_alignment_evidence": bool(inliers >= 10),
        "gimbal_actual": gimbal_report,
        "frames": frame_summaries,
        "notes": [
            "Residuals include terrain, plants, rocks, and vehicle returns; not every valid pixel should match ground.",
            "Ground inliers are captured points within the configured absolute z-residual threshold of the original four-wave terrain.",
            "This checks actual captured depth and pose/joint streams; it does not reuse exported heightmap pixels.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path,
                        help="Gazebo runner capture directory")
    parser.add_argument("--scene-json", required=True, type=Path,
                        help="original field_sim scene payload JSON")
    parser.add_argument("--state-json", type=Path,
                        help="matching source state JSON (auto-detected from *_scene.json)")
    parser.add_argument("--manifest", type=Path,
                        help="capture acceptance.json or exported manifest.json")
    parser.add_argument("--output", type=Path,
                        help="output JSON (default: <capture>/ground_alignment.json)")
    parser.add_argument("--threshold", type=float, default=0.08,
                        help="ground inlier absolute z residual in metres")
    parser.add_argument("--pixel-stride", type=int, default=1,
                        help="sample every Nth pixel in each image dimension")
    parser.add_argument("--max-pose-gap", type=float, default=0.1,
                        help="maximum nearest-pose time difference in seconds")
    parser.add_argument("--max-joint-gap", type=float, default=0.1,
                        help="maximum nearest-gimbal-sample time difference in seconds")
    parser.add_argument("--gimbal-hf-window", type=float, default=0.25,
                        help="running-mean window for high-frequency gimbal residual")
    args = parser.parse_args()
    capture = args.capture.resolve()
    scene_path = args.scene_json.resolve()
    state_path = args.state_json.resolve() if args.state_json else _paired_state(scene_path)
    output = args.output.resolve() if args.output else capture / "ground_alignment.json"
    if args.threshold <= 0 or args.pixel_stride < 1:
        parser.error("--threshold must be > 0 and --pixel-stride must be >= 1")
    report = validate(capture, scene_path, state_path, args.manifest, args.threshold,
                      args.pixel_stride, args.max_pose_gap, args.max_joint_gap,
                      args.gimbal_hf_window)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    counts = report["counts"]
    print(json.dumps({
        "output": str(output), "scene_id": report["scene_id"],
        "frames_used": counts["frames_used"], "valid_depth_pixels": counts["valid_depth_pixels"],
        "in_field_world_points": counts["in_field_world_points"],
        "ground_inliers": counts["ground_inliers"],
        "ground_inlier_fraction": counts["ground_inlier_fraction_of_in_field_points"],
        "residual_median_m": report["terrain_z_residual_m"].get("p50"),
        "has_ground_alignment_evidence": report["has_ground_alignment_evidence"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
