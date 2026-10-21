#!/usr/bin/env python3
"""Run an already-exported field world and record a bounded sensor smoke run.

This uses Gazebo Transport directly and deliberately does not import ROS, Gym,
the field environment, or a learning framework. It is a scene / sensor / physics
smoke test, not a policy training loop.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zlib
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = ROOT / "results" / "gazebo_field" / "current" / "world.sdf"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "field_gazebo"
os.environ["ROS_DOMAIN_ID"] = "73"
os.environ["GZ_PARTITION"] = os.environ.get(
    "FIELD_GZ_PARTITION", f"field_migration_{os.getpid()}_{uuid.uuid4().hex[:8]}"
)

TOPIC_DEFAULTS = {
    "cmd_vel": "/field/cmd_vel",
    "odom": "/field/odom",
    "joint_states": "/field/joint_states",
    "lidar": "/field/lidar/scan",
    "depth": "/field/depth/image",
    "world_pose": "/world/field_training/dynamic_pose/info",
    "imu": "/field/imu/data",
    "magnetometer": "/field/magnetometer/data",
    "gimbal_yaw": "/field/gimbal/yaw/cmd_pos",
    "gimbal_pitch": "/field/gimbal/pitch/cmd_pos",
}


def _topic_map(path: Path | None) -> dict[str, str]:
    topics = dict(TOPIC_DEFAULTS)
    if path is None or not path.is_file():
        return topics
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw.get("topics"), dict):
        raw = raw["topics"]
    aliases = {
        "cmd_vel": ("cmd_vel", "command_velocity", "command_topic"),
        "odom": ("odom", "odometry", "odom_topic"),
        "joint_states": ("joint_states", "joint_state", "joint_state_topic"),
        "lidar": ("lidar", "lidar_scan", "lidar_topic"),
        "depth": ("depth", "depth_image", "depth_topic"),
        "world_pose": ("world_pose", "dynamic_pose", "dynamic_pose_topic"),
        "imu": ("imu", "imu_data", "imu_topic"),
        "magnetometer": ("magnetometer", "magnetometer_data", "magnetometer_topic"),
        "gimbal_yaw": ("gimbal_yaw", "gimbal_yaw_cmd", "gimbal_yaw_topic"),
        "gimbal_pitch": ("gimbal_pitch", "gimbal_pitch_cmd", "gimbal_pitch_topic"),
    }
    for key, names in aliases.items():
        for name in names:
            value = raw.get(name)
            if isinstance(value, str):
                topics[key] = value
                break
            if isinstance(value, dict) and isinstance(value.get("topic"), str):
                topics[key] = value["topic"]
                break
    return topics


def _message_type(module_names: tuple[str, ...], class_name: str):
    errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(f"gz.msgs10.{module_name}_pb2")
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError(f"Gazebo message type {class_name} is unavailable ({'; '.join(errors)})")


def _seconds(message) -> float | None:
    """Read either a Gazebo Time/Duration or a finite numeric timestamp."""
    if message is None:
        return None
    if isinstance(message, (float, int)):
        value = float(message)
        return value if math.isfinite(value) else None
    for sec_name, nsec_name in (("sec", "nsec"), ("seconds", "nanoseconds")):
        sec = getattr(message, sec_name, None)
        nsec = getattr(message, nsec_name, None)
        if sec is not None:
            try:
                value = float(sec) + (float(nsec or 0) * 1e-9)
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                pass
    return None


def _stamp(message) -> float | None:
    header = getattr(message, "header", None)
    if header is not None and _has_field(message, "header") is not False:
        stamp_message = getattr(header, "stamp", None)
        if stamp_message is not None and _has_field(header, "stamp") is not False:
            stamp = _seconds(stamp_message)
        else:
            stamp = None
        if stamp is not None:
            return stamp
    time_message = getattr(message, "time", None)
    if time_message is not None and _has_field(message, "time") is not False:
        return _seconds(time_message)
    return None


def _load_world_name(world_path: Path) -> str:
    root = ET.parse(world_path).getroot()
    worlds = root.findall(".//world")
    if not worlds:
        raise ValueError(f"No <world> element in SDF: {world_path}")
    return worlds[0].get("name") or "default"


def _write_unlimited_copy(world_path: Path) -> Path:
    """Create a neighboring temporary SDF so relative asset URIs keep working."""
    tree = ET.parse(world_path)
    physics = tree.find(".//world/physics")
    if physics is None:
        physics = tree.find(".//physics")
    if physics is None:
        raise ValueError("--unlimited requested, but SDF contains no <physics> element")
    factor = physics.find("real_time_factor")
    if factor is None:
        factor = ET.SubElement(physics, "real_time_factor")
    factor.text = "0"
    fd, name = tempfile.mkstemp(prefix=world_path.stem + "_unlimited_", suffix=".sdf",
                                dir=world_path.parent)
    os.close(fd)
    temp_path = Path(name)
    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _write_depth_preview(path: Path, depth: np.ndarray, max_range: float = 5.0) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    x = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0) / max_range, 0.0, 1.0)
    # Compact jet-like pseudo-colour; invalid pixels remain black.
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    rgb[~valid] = 0
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    payload = (b"\x89PNG\r\n\x1a\n"
               + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
               + _png_chunk(b"IEND", b""))
    path.write_bytes(payload)


class Capture:
    def __init__(self, max_frames: int = 2000):
        self.lock = threading.Lock()
        self.max_frames = max_frames
        self.latest_stats: dict | None = None
        self.items = {
            key: deque(maxlen=max_frames) for key in (
                "depth", "lidar", "odom", "world_pose", "imu", "magnetometer",
                "joint_states", "uwb")
        }
        self.counts = {key: 0 for key in self.items}
        self.truncated = {key: 0 for key in self.items}
        self.measured_counts = {key: 0 for key in self.items}
        self.measured_unique_counts = {key: 0 for key in self.items}
        self.last_measured_stamp = {key: None for key in self.items}
        self.measured_start_sim: float | None = None
        self.measured_end_sim: float | None = None
        self.callback_errors: list[str] = []

    def add(self, kind: str, message, sim_fallback: float | None = None,
            extra: dict | None = None) -> None:
        wall = time.monotonic()
        sim_time = _stamp(message)
        if sim_time is None:
            sim_time = sim_fallback
        row = {"wall": wall, "sim": sim_time, "message": message}
        if extra:
            row.update(extra)
        with self.lock:
            self.counts[kind] += 1
            if (sim_time is not None and self.measured_start_sim is not None
                    and sim_time >= self.measured_start_sim - 1e-6
                    and (self.measured_end_sim is None
                         or sim_time <= self.measured_end_sim + 1e-6)):
                self.measured_counts[kind] += 1
                previous = self.last_measured_stamp[kind]
                if previous is None or sim_time > previous + 1e-9:
                    self.measured_unique_counts[kind] += 1
                    self.last_measured_stamp[kind] = sim_time
            if len(self.items[kind]) >= self.max_frames:
                self.truncated[kind] += 1
            self.items[kind].append(row)

    def start_measurement(self, sim_time: float) -> None:
        with self.lock:
            self.measured_start_sim = sim_time

    def finish_measurement(self, sim_time: float | None) -> None:
        if sim_time is None:
            return
        with self.lock:
            self.measured_end_sim = sim_time

    def add_stats(self, message) -> None:
        sim_time = _seconds(getattr(message, "sim_time", None))
        real_time = _seconds(getattr(message, "real_time", None))
        rtf = float(getattr(message, "real_time_factor", 0.0))
        iterations = int(getattr(message, "iterations", 0))
        snapshot = {"wall": time.monotonic(), "sim": sim_time,
                    "real": real_time, "rtf": rtf, "iterations": iterations}
        with self.lock:
            self.latest_stats = snapshot


def _subscribe(node, topic: str, msg_type, callback):
    # The packaged gz-transport13 binding is protobuf-first and returns None
    # on success. `advertise`, by contrast, takes topic before message type.
    result = node.subscribe(msg_type, topic, callback)
    if result is False:
        raise RuntimeError(f"Could not subscribe to {topic}")
    return result


def _advertise(node, topic: str, msg_type):
    pub = node.advertise(topic, msg_type)
    if pub is None or pub is False:
        raise RuntimeError(f"Could not advertise {topic}")
    return pub


def _safe_attr(obj, name: str, default=None):
    try:
        return getattr(obj, name)
    except (AttributeError, ValueError):
        return default


def _has_field(message, name: str) -> bool | None:
    try:
        return bool(message.HasField(name))
    except (AttributeError, ValueError):
        return None


def _protobuf_enum_name(message, field_name: str) -> str | None:
    try:
        field = message.DESCRIPTOR.fields_by_name[field_name]
        number = int(getattr(message, field_name))
        enum_value = field.enum_type.values_by_number.get(number)
        return enum_value.name if enum_value is not None else None
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _uwb_id_from_link(link_name: str, sensor_kind: str) -> str | None:
    """Read the numeric anchor/tag ID encoded by the local UWB plugin's link names."""
    base = str(link_name).rsplit("::", 1)[-1]
    prefix = f"uwb_{sensor_kind}_"
    if not base.startswith(prefix):
        return None
    sensor_id = base[len(prefix):]
    return sensor_id if sensor_id.isdigit() else None


def _uwb_pair_topics(uwb_manifest: dict) -> list[dict]:
    """Derive every measured topic from manifest link IDs (source plugin convention)."""
    anchors = uwb_manifest.get("anchors", [])
    tag_id = _uwb_id_from_link(uwb_manifest.get("tag_link", "uwb_tag_1"), "tag")
    if not isinstance(anchors, list) or tag_id is None:
        return []
    base = str(uwb_manifest.get(
        "range_topic_base", uwb_manifest.get("distance_topic_base", "/uwb_gz_simulator/distances")
    )).rstrip("/")
    pairs = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        anchor_id = _uwb_id_from_link(anchor.get("link", ""), "anchor")
        if anchor_id is None:
            continue
        pairs.append({
            "anchor_id": anchor_id,
            "tag_id": tag_id,
            "anchor_model": anchor.get("model"),
            "anchor_link": anchor.get("link"),
            "topic": f"{base}/a{anchor_id}t{tag_id}",
        })
    return pairs


def _manifest_summary(manifest: dict) -> dict:
    terrain = manifest.get("terrain", {})
    return {
        "schema": manifest.get("schema"),
        "scene_id": manifest.get("scene_id"),
        "vehicle": manifest.get("vehicle", {}),
        "geometry_counts": manifest.get("geometry_counts", {}),
        "terrain": {key: terrain.get(key) for key in (
            "enabled", "source", "resolution_px", "horizontal_size_m",
            "heightmap_size_z_m", "heightmap_pos_z_m", "max_sample_quantization_error_m",
            "rock_count", "scene_seed") if key in terrain},
        "uwb": manifest.get("uwb", {}),
    }


def _preserve_renderer_log(out: Path, process_started_epoch: float) -> dict:
    source = Path.home() / ".gz" / "rendering" / "ogre2.log"
    if not source.is_file():
        return {"status": "missing", "source": str(source)}
    modified = source.stat().st_mtime
    detail = {"source": str(source), "mtime_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified))}
    if modified < process_started_epoch - 1.0:
        detail["status"] = "stale"
        return detail
    destination = out / "renderer.log"
    shutil.copy2(source, destination)
    lines = destination.read_text(encoding="utf-8", errors="replace").splitlines()
    for key in ("GL_RENDERER", "GL_VERSION"):
        match = next((line.strip() for line in lines if key in line), None)
        if match:
            detail[key] = match
    detail.update({"status": "fresh", "captured_path": str(destination)})
    return detail


def _save_capture(out: Path, capture: Capture, stats_start: dict | None,
                  stats_end: dict | None, measured_wall: float, measured_sim: float,
                  topics: dict[str, str], args, world_path: Path,
                  manifest: dict, model_name: str) -> dict:
    report = {
        "kind": "gazebo_scene_sensor_smoke",
        "training_throughput_claim": False,
        "world": str(world_path),
        "world_name": _load_world_name(world_path),
        "robot_model": model_name,
        "export_summary": _manifest_summary(manifest),
        "partition": os.environ["GZ_PARTITION"],
        "ros_domain_id": int(os.environ["ROS_DOMAIN_ID"]),
        "unlimited": bool(args.unlimited),
        "motion_enabled": not args.no_motion,
        "target_sim_seconds": float(args.seconds),
        "measured_wall_seconds": measured_wall,
        "measured_sim_seconds": measured_sim,
        "topics": topics,
        "sensor_frames": {},
        "range_valid_fraction": {},
        "physics": {},
        "errors": list(capture.callback_errors),
    }
    uwb_manifest = manifest.get("uwb", {})
    report["uwb_measurement"] = {
        "enabled": bool(uwb_manifest.get("enabled", False)),
        "message_type": "gz.msgs.Double",
        "raw_unit": "cm",
        "converted_unit": "m",
        "timestamp_source": (
            "latest WorldStatistics.sim_time observed by the callback; estimated simulation "
            "time, not a UWB sensor capture timestamp"
        ),
        "topic_base": uwb_manifest.get(
            "range_topic_base", uwb_manifest.get("distance_topic_base", "/uwb_gz_simulator/distances")),
        "nlos_compatibility_preset": uwb_manifest.get("nlos_compatibility_preset"),
        "model_limitations": uwb_manifest.get(
            "note", "NLOS reliability depends on the exported preset and terrain-AABB limitations"),
    }
    if stats_start and stats_end:
        iterations = max(0, stats_end["iterations"] - stats_start["iterations"])
        report["physics"] = {
            "start_iterations": stats_start["iterations"],
            "end_iterations": stats_end["iterations"],
            "iterations": iterations,
            "iterations_per_wall_second": iterations / measured_wall if measured_wall > 0 else 0.0,
            "reported_rtf_start": stats_start["rtf"],
            "reported_rtf_end": stats_end["rtf"],
            "measured_rtf_sim_over_wall": measured_sim / measured_wall if measured_wall > 0 else 0.0,
        }

    with capture.lock:
        stored = {key: list(value) for key, value in capture.items.items()}
        counts = dict(capture.counts)
        truncated = dict(capture.truncated)
        measured_counts = dict(capture.measured_counts)
        measured_unique_counts = dict(capture.measured_unique_counts)
        capture_measure_start = capture.measured_start_sim
        capture_measure_end = capture.measured_end_sim
    measured_start = (stats_start.get("sim") if stats_start else None)
    measured_end = (stats_end.get("sim") if stats_end else None)

    def during_measurement(row):
        stamp = row["sim"]
        if stamp is None or measured_start is None:
            return True
        if stamp < measured_start - 1e-6:
            return False
        return measured_end is None or stamp <= measured_end + 1e-6

    measured_rows = {}
    for kind, rows in stored.items():
        measured = [r for r in rows if during_measurement(r)]
        measured_rows[kind] = measured
        stamps = [r["sim"] for r in measured if r["sim"] is not None]
        unique_stamps = sorted(set(round(x, 9) for x in stamps))
        unique_count = (measured_unique_counts[kind] if capture_measure_start is not None
                        else len(unique_stamps))
        denominator = measured_sim if measured_sim > 0 else 0.0
        report["sensor_frames"][kind] = {
            "received_total": counts[kind],
            "saved_retained": len(rows),
            "measured_saved": len(measured),
            "measured_received": measured_counts[kind],
            "unique_sim_timestamps": unique_count,
            "new_hz_sim": unique_count / denominator if denominator else 0.0,
            "wall_hz": measured_counts[kind] / measured_wall if measured_wall > 0 else 0.0,
            "retention_dropped": truncated[kind],
            "measured_timestamp_span_sim_seconds": (
                max(0.0, capture_measure_end - capture_measure_start)
                if capture_measure_start is not None and capture_measure_end is not None else 0.0
            ),
            "retained_timestamp_span_sim_seconds": (unique_stamps[-1] - unique_stamps[0]
                                                      if len(unique_stamps) > 1 else 0.0),
        }

    depth_rows = [r for r in measured_rows["depth"] if r["message"] is not None]
    depth_arrays = []
    depth_stamps = []
    depth_valid = 0
    depth_total = 0
    depth_valid_fractions = []
    depth_dimensions = set()
    depth_formats = set()
    depth_metadata = None
    depth_meta = manifest.get("vehicle", {}).get("sensors", {}).get("depth", {})
    depth_limits = depth_meta.get("range_m", [0.0, 5.0])
    depth_min, depth_max = float(depth_limits[0]), float(depth_limits[1])
    for row in depth_rows:
        msg = row["message"]
        width, height = int(msg.width), int(msg.height)
        format_name = (_protobuf_enum_name(msg, "pixel_format_type")
                       or str(_safe_attr(msg, "pixel_format_type", "")))
        depth_dimensions.add((width, height))
        depth_formats.add(format_name)
        if depth_metadata is None:
            depth_metadata = {
                "resolution_wh": [width, height],
                "format": format_name,
                "frame_id": str(_safe_attr(_safe_attr(msg, "header"), "frame_id", "")),
            }
        data = bytes(msg.data)
        step = int(getattr(msg, "step", 0)) or (width * 4)
        if len(data) < height * step or step < width * 4:
            capture.callback_errors.append(
                f"depth payload has {len(data)} bytes for {width}x{height}, step={step}; expected float32"
            )
            continue
        image = np.ndarray((height, width), dtype="<f4", buffer=data,
                           strides=(step, 4)).copy()
        valid = np.isfinite(image) & (image >= depth_min) & (image <= depth_max)
        depth_valid += int(valid.sum())
        depth_total += int(valid.size)
        depth_valid_fractions.append(float(valid.mean()))
        depth_arrays.append(image)
        depth_stamps.append(row["sim"] if row["sim"] is not None else math.nan)
    if depth_arrays:
        np.save(out / "depth.npy", np.stack(depth_arrays).astype(np.float32, copy=False))
        np.save(out / "depth_sim_timestamps.npy", np.asarray(depth_stamps, dtype=np.float64))
        _write_depth_preview(out / "depth_preview.png", depth_arrays[-1])
    else:
        np.save(out / "depth.npy", np.empty((0, 0, 0), dtype=np.float32))
        np.save(out / "depth_sim_timestamps.npy", np.empty((0,), dtype=np.float64))
    report["range_valid_fraction"]["depth"] = depth_valid / depth_total if depth_total else 0.0
    if depth_metadata:
        report["sensor_frames"]["depth"].update(depth_metadata)
    expected_depth_wh = depth_meta.get("resolution_wh", [88, 160])
    report["sensor_frames"]["depth"].update({
        "expected_resolution_wh": expected_depth_wh,
        "expected_format": depth_meta.get("format", "R_FLOAT32"),
        "observed_formats": sorted(depth_formats),
        "shape_and_format_valid": bool(
            depth_dimensions == {(int(expected_depth_wh[0]), int(expected_depth_wh[1]))}
            and depth_formats == {str(depth_meta.get("format", "R_FLOAT32"))}
        ),
        "last_frame_valid_fraction": depth_valid_fractions[-1] if depth_valid_fractions else 0.0,
        "median_frame_valid_fraction": float(np.median(depth_valid_fractions))
        if depth_valid_fractions else 0.0,
    })

    lidar_rows = [r for r in measured_rows["lidar"] if r["message"] is not None]
    lidar_arrays = [np.asarray(getattr(r["message"], "ranges", []), dtype=np.float32)
                    for r in lidar_rows]
    lidar_valid_fractions = []
    lidar_ray_counts = {len(values) for values in lidar_arrays}
    expected_lidar_rays = int(manifest.get("vehicle", {}).get("sensors", {}).get(
        "lidar", {}).get("horizontal_samples", 400))
    if lidar_arrays:
        max_len = max(map(len, lidar_arrays))
        ranges = np.full((len(lidar_arrays), max_len), np.nan, dtype=np.float32)
        for i, values in enumerate(lidar_arrays):
            ranges[i, :len(values)] = values
        lidar_stamps = np.asarray([r["sim"] if r["sim"] is not None else math.nan
                                   for r in lidar_rows], dtype=np.float64)
        np.savez_compressed(out / "lidar.npz", ranges=ranges, sim_timestamps=lidar_stamps)
        min_ranges = np.asarray([float(_safe_attr(r["message"], "range_min", 0.0))
                                 for r in lidar_rows], dtype=np.float32)[:, None]
        max_ranges = np.asarray([float(_safe_attr(r["message"], "range_max", math.inf))
                                 for r in lidar_rows], dtype=np.float32)[:, None]
        valid = np.isfinite(ranges) & (ranges >= min_ranges) & (ranges <= max_ranges)
        report["range_valid_fraction"]["lidar"] = int(valid.sum()) / int(valid.size) if valid.size else 0.0
        lidar_valid_fractions = [float(np.mean(np.isfinite(values)
                                               & (values >= float(_safe_attr(row["message"], "range_min", 0.0)))
                                               & (values <= float(_safe_attr(row["message"], "range_max", math.inf)))))
                                 for row, values in zip(lidar_rows, lidar_arrays)]
        first_scan = lidar_rows[0]["message"]
        report["sensor_frames"]["lidar"].update({
            "ray_count": int(_safe_attr(first_scan, "count", len(lidar_arrays[0]))),
            "observed_range_lengths": sorted(lidar_ray_counts),
            "expected_ray_count": expected_lidar_rays,
            "ray_count_valid": lidar_ray_counts == {expected_lidar_rays},
            "range_min_m": float(_safe_attr(first_scan, "range_min", 0.0)),
            "range_max_m": float(_safe_attr(first_scan, "range_max", math.inf)),
            "angle_min_rad": float(_safe_attr(first_scan, "angle_min", math.nan)),
            "angle_max_rad": float(_safe_attr(first_scan, "angle_max", math.nan)),
            "angle_step_rad": float(_safe_attr(first_scan, "angle_step", math.nan)),
            "last_scan_valid_fraction": lidar_valid_fractions[-1] if lidar_valid_fractions else 0.0,
            "median_scan_valid_fraction": float(np.median(lidar_valid_fractions))
            if lidar_valid_fractions else 0.0,
        })
    else:
        np.savez_compressed(out / "lidar.npz", ranges=np.empty((0, 0), dtype=np.float32),
                            sim_timestamps=np.empty((0,), dtype=np.float64))
        report["range_valid_fraction"]["lidar"] = 0.0

    odom_trajectory = []
    for row in measured_rows["odom"]:
        msg = row["message"]
        pos = _safe_attr(msg, "position")
        rot = _safe_attr(msg, "orientation")
        if pos is None:
            # Harmonic Odometry releases have exposed both Pose and
            # PoseWithCovariance-style wrappers across protobuf versions.
            pose_field = _safe_attr(msg, "pose")
            nested_pose = _safe_attr(pose_field, "pose")
            odom_pose = nested_pose if nested_pose is not None else pose_field
            pos = _safe_attr(odom_pose, "position")
            rot = _safe_attr(odom_pose, "orientation")
        if pos is None:
            continue
        values = [float(_safe_attr(pos, k, math.nan)) for k in ("x", "y", "z")]
        values.extend(float(_safe_attr(rot, k, math.nan)) for k in ("x", "y", "z", "w"))
        stamp = row["sim"] if row["sim"] is not None else math.nan
        if all(math.isfinite(x) for x in values) and math.isfinite(stamp):
            odom_trajectory.append([stamp, *values])
    with (out / "odom_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
        writer.writerows(odom_trajectory)

    world_trajectory = []
    for row in measured_rows["world_pose"]:
        msg = row["message"]
        poses = _safe_attr(msg, "pose", [])
        pose = next((p for p in poses
                     if str(_safe_attr(p, "name", "")) == model_name
                     or str(_safe_attr(p, "name", "")).endswith(f"::{model_name}")), None)
        if pose is None:
            continue
        pos = _safe_attr(pose, "position")
        rot = _safe_attr(pose, "orientation")
        if pos is None or rot is None:
            continue
        values = [float(_safe_attr(pos, key, math.nan)) for key in ("x", "y", "z")]
        values.extend(float(_safe_attr(rot, key, math.nan)) for key in ("x", "y", "z", "w"))
        stamp = row["sim"] if row["sim"] is not None else math.nan
        if all(math.isfinite(x) for x in values) and math.isfinite(stamp):
            world_trajectory.append([stamp, *values])
    with (out / "world_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
        writer.writerows(world_trajectory)
    report["trajectory_samples_finite"] = len(world_trajectory)
    report["physical_world_displacement_m"] = (
        math.dist(world_trajectory[0][1:4], world_trajectory[-1][1:4])
        if len(world_trajectory) > 1 else 0.0
    )
    report["odom_integrated_displacement_m"] = (
        math.dist(odom_trajectory[0][1:4], odom_trajectory[-1][1:4])
        if len(odom_trajectory) > 1 else 0.0
    )

    joint_samples = []
    valid_joint_names = set()
    for row in measured_rows["joint_states"]:
        model = row["message"]
        for joint in _safe_attr(model, "joint", []):
            name = str(_safe_attr(joint, "name", ""))
            axes = [(1, "axis1", _safe_attr(joint, "axis1")),
                    (2, "axis2", _safe_attr(joint, "axis2"))]
            legacy_axes = _safe_attr(joint, "axis", [])
            if legacy_axes:
                axes = [(index, "axis", axis)
                        for index, axis in enumerate(legacy_axes, start=1)]
            for axis_index, axis_field, axis in axes:
                if axis is None:
                    continue
                if _has_field(joint, axis_field) is False:
                    continue
                if _has_field(axis, "position") is False:
                    continue
                position = float(_safe_attr(axis, "position", math.nan))
                velocity = float(_safe_attr(axis, "velocity", math.nan))
                force = float(_safe_attr(axis, "force", math.nan))
                joint_samples.append([
                    row["sim"] if row["sim"] is not None else math.nan,
                    name, axis_index, position, velocity, force,
                ])
                if math.isfinite(position) and math.isfinite(velocity):
                    valid_joint_names.add(name)
    with (out / "joint_states.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", "joint", "axis", "position_rad", "velocity", "force"])
        writer.writerows(joint_samples)
    finite_joint_position_samples = sum(
        1 for row in joint_samples if math.isfinite(row[3]) and math.isfinite(row[4]))
    required_gimbal_joints = {"gimbal_yaw_joint", "gimbal_pitch_joint"}
    report["sensor_frames"]["joint_states"].update({
        "finite_joint_axis_samples": len(joint_samples),
        "finite_position_samples": finite_joint_position_samples,
        "joint_names_with_finite_positions": sorted(valid_joint_names),
        "gimbal_positions_valid": required_gimbal_joints.issubset(valid_joint_names),
    })

    imu_rows = []
    valid_imu_rows = []
    for row in measured_rows["imu"]:
        msg = row["message"]
        av = _safe_attr(msg, "angular_velocity")
        la = _safe_attr(msg, "linear_acceleration")
        ori = _safe_attr(msg, "orientation")
        if av is None or la is None:
            continue
        values = [row["sim"] if row["sim"] is not None else math.nan]
        for obj, names in ((ori, ("x", "y", "z", "w")),
                           (av, ("x", "y", "z")), (la, ("x", "y", "z"))):
            values.extend(float(_safe_attr(obj, key, math.nan)) for key in names)
        if all(math.isfinite(v) for v in values):
            imu_rows.append(values)
            quat_norm = math.sqrt(sum(float(_safe_attr(ori, key, 0.0)) ** 2
                                      for key in ("x", "y", "z", "w")))
            if 0.5 <= quat_norm <= 1.5:
                valid_imu_rows.append(values)
    with (out / "imu.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"])
        writer.writerows(imu_rows)

    report["sensor_frames"]["imu"].update({
        "saved_finite": len(imu_rows),
        "valid_orientation_samples": len(valid_imu_rows),
        "orientation_valid": len(valid_imu_rows) > 0,
    })
    uwb_rows = []
    uwb_counts = {}
    finite_uwb_pairs = set()
    for row in measured_rows["uwb"]:
        topic = str(row.get("topic", "unknown"))
        value_cm = float(_safe_attr(row["message"], "data", math.nan))
        uwb_counts[topic] = uwb_counts.get(topic, 0) + 1
        if math.isfinite(value_cm):
            pair_id = topic.rsplit("/", 1)[-1]
            finite_uwb_pairs.add(pair_id)
            uwb_rows.append([
                row["sim"] if row["sim"] is not None else math.nan,
                topic, value_cm, value_cm / 100.0,
            ])
    with (out / "uwb.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", "topic", "range_raw_cm", "range_m"])
        writer.writerows(uwb_rows)
    uwb_enabled = bool(manifest.get("uwb", {}).get("enabled", False))
    min_finite_uwb_pairs = 3
    uwb_pair_gate_passed = (not uwb_enabled or len(finite_uwb_pairs) >= min_finite_uwb_pairs)
    report["sensor_frames"]["uwb"].update({
        "received_by_topic": uwb_counts,
        "finite_samples": len(uwb_rows),
        "finite_anchor_pairs": sorted(finite_uwb_pairs),
        "finite_anchor_pair_count": len(finite_uwb_pairs),
        "minimum_finite_anchor_pairs_for_smoke": min_finite_uwb_pairs if uwb_enabled else 0,
        "message_type": "gz.msgs.Double",
        "raw_unit": "cm",
        "converted_unit": "m",
    })
    report["errors"] = list(capture.callback_errors)
    terrain = manifest.get("terrain", {})
    terrain_low = float(terrain.get("sampled_wave_min_m",
                                   terrain.get("heightmap_pos_z_m", -1.0)))
    terrain_high = float(terrain.get("sampled_wave_max_m",
                                    terrain_low + terrain.get("heightmap_size_z_m", 2.0)))
    allowed_z_band = [terrain_low - 0.5, terrain_high + 1.0]
    world_z = [row[3] for row in world_trajectory]
    vertical_span = max(world_z) - min(world_z) if world_z else math.inf
    vertical_drift = abs(world_z[-1] - world_z[0]) if len(world_z) > 1 else math.inf
    z_in_band = bool(world_z) and all(allowed_z_band[0] <= z <= allowed_z_band[1]
                                      for z in world_z)
    physical_displacement = report["physical_world_displacement_m"]
    expected_depth_wh = depth_meta.get("resolution_wh", [88, 160])
    depth_shape_valid = report["sensor_frames"]["depth"].get("shape_and_format_valid", False)
    depth_ranges_valid = bool(
        report["range_valid_fraction"]["depth"] >= 0.05
        and report["sensor_frames"]["depth"].get("last_frame_valid_fraction", 0.0) >= 0.05
    )
    lidar_ranges_valid = bool(
        report["range_valid_fraction"]["lidar"] >= 0.01
        and report["sensor_frames"]["lidar"].get("last_scan_valid_fraction", 0.0) >= 0.01
    )
    lidar_geometry_valid = bool(report["sensor_frames"]["lidar"].get("ray_count_valid", False))
    motion_expected = not args.no_motion
    report["physical_validity"] = {
        "terrain_allowed_root_z_band_m": allowed_z_band,
        "world_pose_samples_finite": bool(world_trajectory),
        "robot_z_all_in_terrain_band": z_in_band,
        "robot_z_vertical_span_m": vertical_span if math.isfinite(vertical_span) else None,
        "robot_z_start_to_end_drift_m": vertical_drift if math.isfinite(vertical_drift) else None,
        "vertical_span_le_2m": vertical_span <= 2.0,
        "vertical_drift_le_2m": vertical_drift <= 2.0,
        "horizontal_displacement_m": physical_displacement,
        "motion_expected": motion_expected,
        "horizontal_motion_valid": not motion_expected or physical_displacement >= 0.1,
    }
    report["sensor_validity"] = {
        "depth_shape_and_format_valid": depth_shape_valid,
        "depth_resolution_expected_wh": expected_depth_wh,
        "depth_format_expected": depth_meta.get("format", "R_FLOAT32"),
        "depth_in_range_fraction_valid": depth_ranges_valid,
        "depth_min_valid_fraction": 0.05,
        "lidar_ray_count_valid": lidar_geometry_valid,
        "lidar_in_range_fraction_valid": lidar_ranges_valid,
        "lidar_min_valid_fraction": 0.01,
        "imu_orientation_valid": report["sensor_frames"]["imu"].get("orientation_valid", False),
        "gimbal_joint_positions_valid": report["sensor_frames"]["joint_states"].get(
            "gimbal_positions_valid", False),
        "uwb_finite_anchor_pairs_valid": uwb_pair_gate_passed,
        "uwb_finite_anchor_pair_count": len(finite_uwb_pairs),
        "uwb_min_finite_anchor_pairs": min_finite_uwb_pairs if uwb_enabled else 0,
    }
    report["rate_targets_hz"] = {
        "lidar_expected_min": 9.0,
        "depth_expected_min": 2.25,
        "lidar_met": report["sensor_frames"]["lidar"]["new_hz_sim"] >= 9.0,
        "depth_met": report["sensor_frames"]["depth"]["new_hz_sim"] >= 2.25,
    }
    report["performance_targets_met"] = bool(
        report["rate_targets_hz"]["lidar_met"] and report["rate_targets_hz"]["depth_met"]
    )
    report["smoke_passed"] = bool(
        report["sensor_frames"]["depth"]["measured_saved"] > 0
        and report["sensor_frames"]["lidar"]["measured_saved"] > 0
        and report["sensor_frames"]["odom"]["measured_saved"] > 0
        and report["sensor_frames"]["world_pose"]["measured_saved"] > 0
        and len(world_trajectory) > 0
        and report["physics"].get("iterations", 0) > 0
        and z_in_band
        and vertical_span <= 2.0
        and vertical_drift <= 2.0
        and (not motion_expected or physical_displacement >= 0.1)
        and depth_shape_valid
        and depth_ranges_valid
        and lidar_geometry_valid
        and lidar_ranges_valid
        and report["sensor_validity"]["imu_orientation_valid"]
        and report["sensor_validity"]["gimbal_joint_positions_valid"]
        and uwb_pair_gate_passed
        and report["performance_targets_met"]
    )
    report["notes"] = [
        "Scene/sensor smoke only; no Gym step, policy inference, reward, or training throughput was measured.",
        "Measured sensor rates use distinct message simulation timestamps over measured simulation seconds.",
        "IMU rows are copied from Gazebo messages; no synthetic gyro or pose correction is applied.",
        "Physical world trajectory comes from SceneBroadcaster dynamic_pose; odometry is saved separately and may be command-integrated.",
        "Sensor performance targets are reported independently from the scene smoke status.",
        "UWB gz.msgs.Double values are raw centimetres; uwb.csv also stores converted metres. Their simulation timestamps fall back to the latest WorldStatistics.sim_time observed at callback time, not a UWB sensor capture timestamp.",
        "The local UWB proxy's NLOS reliability depends on the exported preset and terrain-AABB limitations.",
    ]
    return report


def _load_export_metadata(world_path: Path) -> tuple[dict[str, str], dict, str]:
    topics_path = world_path.parent / "world.topics.json"
    manifest_path = world_path.parent / "manifest.json"
    topics_doc = {}
    manifest = {}
    if topics_path.is_file():
        topics_doc = json.loads(topics_path.read_text(encoding="utf-8"))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    topics = _topic_map(topics_path)
    world_name = _load_world_name(world_path)
    # The scene broadcaster path includes the world name; model names may be
    # prefixed by the optional local UWB plugin integration.
    model_name = (manifest.get("vehicle", {}).get("model_name")
                  or topics_doc.get("model")
                  or manifest.get("topics", {}).get("model")
                  or "field_robot")
    topics["world_pose"] = f"/world/{world_name}/dynamic_pose/info"
    return topics, manifest, str(model_name)


def _stop_owned_process(process: subprocess.Popen, timeout: float = 8.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run(args) -> int:
    out = Path(args.output).expanduser().resolve() if args.output else None
    if out is None:
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        out = DEFAULT_OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S") / f"run_{os.getpid()}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"--output must be a new empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    world_path = (Path(args.world).expanduser().resolve() if args.world
                  else DEFAULT_WORLD.resolve())
    if not world_path.is_file():
        raise FileNotFoundError(
            f"World SDF not found: {world_path}. Export it first with simulation.field_world "
            "or pass --world <exported world.sdf>."
        )
    wall_timeout = args.wall_timeout or max(120.0, args.seconds * 2.0 + 30.0)
    temp_world = None
    source_world_path = world_path
    topics, manifest, model_name = _load_export_metadata(source_world_path)
    world_name = _load_world_name(source_world_path)
    stats_topic = f"/world/{world_name}/stats"

    # Transport and protobuf modules must be imported after the partition and
    # domain variables above have been fixed for this run.
    from gz.transport13 import Node

    types = {
        "stats": _message_type(("world_stats", "world_statistics"), "WorldStatistics"),
        "image": _message_type(("image",), "Image"),
        "lidar": _message_type(("laserscan", "laser_scan"), "LaserScan"),
        "odom": _message_type(("odometry",), "Odometry"),
        "pose_v": _message_type(("pose_v",), "Pose_V"),
        "imu": _message_type(("imu",), "IMU"),
        "magnetometer": _message_type(("magnetometer",), "Magnetometer"),
        "joint": _message_type(("model",), "Model"),
        "twist": _message_type(("twist",), "Twist"),
        "double": _message_type(("double",), "Double"),
    }
    capture = Capture()
    node = Node()
    subscriptions = []
    last_stats_lock = capture.lock

    def stats_callback(message, *_):
        capture.add_stats(message)

    def sensor_callback(kind, extra=None):
        def cb(message, *_):
            with last_stats_lock:
                fallback = capture.latest_stats["sim"] if capture.latest_stats else None
            try:
                capture.add(kind, message, fallback, extra=extra)
            except Exception as exc:  # transport callbacks must not terminate the runner
                with capture.lock:
                    capture.callback_errors.append(f"{kind}: {type(exc).__name__}: {exc}")
        return cb

    subscriptions.append(_subscribe(node, stats_topic, types["stats"], stats_callback))
    subscriptions.append(_subscribe(node, topics["depth"], types["image"], sensor_callback("depth")))
    subscriptions.append(_subscribe(node, topics["lidar"], types["lidar"], sensor_callback("lidar")))
    subscriptions.append(_subscribe(node, topics["odom"], types["odom"], sensor_callback("odom")))
    subscriptions.append(_subscribe(node, topics["world_pose"], types["pose_v"], sensor_callback("world_pose")))
    subscriptions.append(_subscribe(node, topics["imu"], types["imu"], sensor_callback("imu")))
    subscriptions.append(_subscribe(node, topics["magnetometer"], types["magnetometer"],
                                    sensor_callback("magnetometer")))
    subscriptions.append(_subscribe(node, topics["joint_states"], types["joint"], sensor_callback("joint_states")))

    optional_topics = {}
    uwb_pairs = _uwb_pair_topics(manifest.get("uwb", {}))
    if manifest.get("uwb", {}).get("enabled"):
        if not uwb_pairs:
            optional_topics["manifest.uwb.anchors"] = "missing usable anchor/tag link IDs"
        for pair in uwb_pairs:
            topic = pair["topic"]
            try:
                subscriptions.append(_subscribe(node, topic, types["double"],
                                                sensor_callback("uwb", {"topic": topic})))
                optional_topics[topic] = "subscribed"
            except Exception as exc:
                optional_topics[topic] = f"unavailable: {type(exc).__name__}: {exc}"
    else:
        uwb_pairs = []

    cmd_pub = _advertise(node, topics["cmd_vel"], types["twist"])
    yaw_pub = _advertise(node, topics["gimbal_yaw"], types["double"])
    pitch_pub = _advertise(node, topics["gimbal_pitch"], types["double"])

    run_world_path = source_world_path
    if args.unlimited:
        # Keep the copy next to its source so relative asset URIs remain working.
        temp_world = _write_unlimited_copy(source_world_path)
        run_world_path = temp_world

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    command = ["gz", "sim", "-r", "-v", str(args.verbosity)]
    if not args.gui:
        command.extend(["-s", "--headless-rendering"])
    command.append(str(run_world_path))
    log_path = out / "gazebo.log"
    log = log_path.open("wb")
    process_started_epoch = time.time()
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        if temp_world and temp_world.exists():
            temp_world.unlink()
        raise
    wall_start = time.monotonic()
    ready_deadline = wall_start + wall_timeout
    stats_start = None
    stats_end = None
    warmup_wall = None
    measured_wall = 0.0
    measured_sim = 0.0
    command_count = 0
    last_command_sim = None
    capture_start_sim = None
    failure = None
    interrupted = False
    try:
        while time.monotonic() < ready_deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Gazebo exited during startup with code {process.returncode}; see {log_path}")
            with capture.lock:
                stats = dict(capture.latest_stats) if capture.latest_stats else None
                depth_rows = list(capture.items["depth"])
            first_depth_sim = next(
                (float(row["sim"]) for row in depth_rows if row["sim"] is not None),
                None,
            )
            # Let initial contact settle and allow renderer/sensor startup frames
            # to pass before the measured interval begins.
            warmup_origin_sim = (first_depth_sim if first_depth_sim is not None
                                 else (float(stats["sim"]) if stats and stats.get("sim") is not None
                                       else None))
            if (stats and stats.get("sim") is not None and len(depth_rows) >= 2
                    and warmup_origin_sim is not None
                    and float(stats["sim"]) >= warmup_origin_sim + 1.0):
                stats_start = stats
                capture_start_sim = float(stats["sim"])
                capture.start_measurement(capture_start_sim)
                warmup_wall = time.monotonic()
                break
            time.sleep(0.02)
        if stats_start is None:
            raise TimeoutError("No ready world stats + depth frame before wall timeout")

        deadline = time.monotonic() + wall_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Gazebo exited during capture with code {process.returncode}; see {log_path}")
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("World did not reach requested simulation duration before wall timeout")
            with capture.lock:
                stats = dict(capture.latest_stats) if capture.latest_stats else None
            if not stats or stats.get("sim") is None:
                time.sleep(0.01)
                continue
            sim_elapsed = float(stats["sim"]) - capture_start_sim
            if not args.no_motion and sim_elapsed >= 0:
                if last_command_sim is None:
                    last_command_sim = capture_start_sim
                sent_this_loop = 0
                while (float(stats["sim"]) - last_command_sim >= 0.05
                       and sent_this_loop < 5):
                    action_time = max(0.0, last_command_sim - capture_start_sim)
                    twist = types["twist"]()
                    twist.linear.x = 0.3
                    twist.angular.z = 0.12 * math.sin(2.0 * math.pi * action_time / 8.0)
                    cmd_pub.publish(twist)
                    yaw = types["double"]()
                    yaw.data = 0.20 * math.sin(2.0 * math.pi * action_time / 6.0)
                    yaw_pub.publish(yaw)
                    pitch = types["double"]()
                    pitch.data = 0.15 + 0.08 * math.sin(2.0 * math.pi * action_time / 7.0)
                    pitch_pub.publish(pitch)
                    command_count += 1
                    sent_this_loop += 1
                    last_command_sim += 0.05
            if sim_elapsed >= args.seconds:
                stats_end = stats
                capture.finish_measurement(float(stats["sim"]))
                measured_sim = sim_elapsed
                measured_wall = max(1e-9, time.monotonic() - warmup_wall)
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        _stop_owned_process(process)
        log.close()
        if temp_world and temp_world.exists():
            temp_world.unlink()

    if stats_end is None:
        with capture.lock:
            stats_end = dict(capture.latest_stats) if capture.latest_stats else None
    if stats_end and stats_end.get("sim") is not None:
        capture.finish_measurement(float(stats_end["sim"]))
    if stats_end is None:
        stats_end = None
    if stats_end is not None and stats_start and stats_start.get("sim") is not None and stats_end.get("sim") is not None:
        measured_sim = max(measured_sim, stats_end["sim"] - stats_start["sim"])
        if failure is not None or interrupted:
            measured_wall = max(0.0, time.monotonic() - (warmup_wall or wall_start))
    report = _save_capture(out, capture, stats_start, stats_end, measured_wall,
                           measured_sim, topics, args, source_world_path, manifest, model_name)
    report["renderer_evidence"] = _preserve_renderer_log(out, process_started_epoch)
    report["optional_topics"] = optional_topics
    report["uwb_measurement"]["subscribed_pair_topics"] = [pair["topic"] for pair in uwb_pairs]
    report["uwb_measurement"]["subscribed_anchor_count"] = len(uwb_pairs)
    report["commands_published"] = command_count
    report["failure"] = failure
    report["interrupted"] = interrupted
    report["status"] = ("interrupted" if interrupted else
                        "passed" if report["smoke_passed"] and failure is None else "failed")
    (out / "acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out),
                      "failure": failure, "sensor_frames": report["sensor_frames"],
                      "physics": report["physics"], "rate_targets_hz": report["rate_targets_hz"]},
                     indent=2, ensure_ascii=False))
    return 130 if interrupted else (0 if report["status"] == "passed" else 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD,
                        help="Existing exported world SDF (default: results/gazebo_field/current/world.sdf)")
    parser.add_argument("--output", type=Path, help="New empty output directory")
    parser.add_argument("--seconds", type=float, default=10.0, help="Measured simulation seconds after depth/stats warmup")
    parser.add_argument("--wall-timeout", type=float, default=None,
                        help="Wall-clock timeout (default scales with --seconds; Ctrl+C saves partial capture)")
    parser.add_argument("--gui", action="store_true", help="Open Gazebo GUI; headless is the default")
    parser.add_argument("--unlimited", action="store_true", help="Set real_time_factor=0 in a temporary SDF copy")
    parser.add_argument("--no-motion", action="store_true", help="Do not publish velocity/gimbal commands")
    parser.add_argument("--verbosity", type=int, default=3)
    args = parser.parse_args(argv)
    if args.seconds <= 0 or (args.wall_timeout is not None and args.wall_timeout <= 0):
        parser.error("seconds and wall-timeout must be positive")
    try:
        return _run(args)
    except Exception as exc:
        print(f"field Gazebo runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
