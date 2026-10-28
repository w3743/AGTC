"""Serve a lightweight WebGL view of the latest Isaac Field state packet.

This process only reads the atomic JSON state file. It never imports Isaac Sim,
Isaac Lab, the training environment, PyTorch, or policy/checkpoint code.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from view_training_state import (
    packet_age_seconds,
    state_is_stale,
    validate_packet as _validate_source_packet,
)

ENV_COUNT = 4
BODY_COUNT = 19
STALE_AFTER_S = 5.0
_LANGUAGE = "en"
POLL_INTERVAL_S = 0.1
STATUS_INTERVAL_S = 1.0
FORBIDDEN_MODULE_PREFIXES = ("isaacsim", "isaaclab", "isaac_field", "omni", "torch")
INITIAL_CAMERA_OFFSET = (2.4, -2.8, 1.6)


def _finite_vec(values: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} must contain {size} values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_rows(values: Any, rows: int, columns: int, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(values, list) or len(values) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    return tuple(_finite_vec(row, columns, f"{name}[{index}]") for index, row in enumerate(values))


def _is_absolute_asset_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def validate_packet(raw: Any, env_index: int = 0) -> dict[str, Any]:
    """Validate and select one environment from live-state schema v1."""
    if isinstance(env_index, bool) or not isinstance(env_index, int) or not 0 <= env_index < ENV_COUNT:
        raise ValueError(f"env_index must be between 0 and {ENV_COUNT - 1}")
    if not isinstance(raw, dict):
        raise ValueError("live-state packet must be a JSON object")
    selected = _validate_source_packet(raw, env_index)
    if len(selected["body_names"]) != BODY_COUNT:
        raise ValueError(f"body_names must contain {BODY_COUNT} bodies")
    if not selected["run_id"].strip():
        raise ValueError("run_id must be a non-empty string")
    if not _is_absolute_asset_path(selected["robot_usd_path"]) or not _is_absolute_asset_path(selected["terrain_usd_path"]):
        raise ValueError("robot and terrain USD paths must be absolute")
    goals_world = _finite_rows(raw.get("goals_world_xy"), ENV_COUNT, 2, "goals_world_xy")
    goal_world = goals_world[env_index]
    origin = selected["origin"]
    if "base_link" in selected["body_names"]:
        base_pose = selected["body_poses"][selected["body_names"].index("base_link")]
        if any(abs(left - right) > 1e-4
               for left, right in zip(selected["root_pose"], base_pose, strict=True)):
            raise ValueError("root_link_pose_w must match the base_link body pose")
    selected["goal_local_xy"] = (goal_world[0] - origin[0], goal_world[1] - origin[1])
    return selected


def pose_to_local_wxyz(pose_xyzw: Sequence[float], origin: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a world position/quaternion to env-local position and Viser wxyz."""
    pose = _finite_vec(pose_xyzw, 7, "pose")
    offset = _finite_vec(origin, 3, "origin")
    x, y, z, qx, qy, qz, qw = pose
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("pose quaternion must be non-zero")
    wxyz = (qw / norm, qx / norm, qy / norm, qz / norm)
    local_position = (x - offset[0], y - offset[1], z - offset[2])
    return local_position, wxyz


def nearest_body_ancestor(prim_path: str, body_paths: Mapping[str, str]) -> str | None:
    """Return the closest mapped body path that contains a USD geometry prim."""
    matches = [name for name, path in body_paths.items()
               if prim_path == path or prim_path.startswith(path.rstrip("/") + "/")]
    return max(matches, key=lambda name: len(body_paths[name])) if matches else None


def group_geometry_by_body(
    body_paths: Mapping[str, str], geometry_paths: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    """Group USD shapes under their nearest articulation body, without name guessing."""
    grouped: dict[str, list[str]] = {name: [] for name in body_paths}
    for path in geometry_paths:
        body = nearest_body_ancestor(path, body_paths)
        if body is not None:
            grouped[body].append(path)
    return {name: tuple(paths) for name, paths in grouped.items()}


@dataclass(frozen=True)
class MeshData:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class RobotGeometry:
    by_body: Mapping[str, tuple[MeshData, ...]]
    terrain: MeshData
    proxy_bodies: tuple[str, ...]


@dataclass(frozen=True)
class NativeViewerContext:
    """Verified static CPU geometry plus identity from its live training trial."""
    trial_name: str
    engine: str
    backend_revision: str
    task_identity_sha256: str
    training_env_count: int
    model_sha256: str
    model_json_path: str
    trial_dir: str
    source_commit: str
    backend_source_sha256: str
    train_wsl_source_sha256: str
    protocol_sha256: str
    body_names: tuple[str, ...]
    geometry: RobotGeometry
    publish_interval_steps: int
    sim_dt_s: float
    decimation: int
    policy_dt_s: float


def _triangulate_face_counts(counts: Sequence[int], indices: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    faces: list[tuple[int, int, int]] = []
    cursor = 0
    for count_value in counts:
        count = int(count_value)
        if count < 3 or cursor + count > len(indices):
            raise ValueError("USD mesh contains invalid face topology")
        polygon = [int(index) for index in indices[cursor:cursor + count]]
        faces.extend((polygon[0], polygon[index], polygon[index + 1])
                     for index in range(1, count - 1))
        cursor += count
    if cursor != len(indices):
        raise ValueError("USD mesh contains unreferenced face indices")
    return tuple(faces)


def _apply_gf_transform(points: Iterable[Any], matrix: Any, Gf: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in matrix.Transform(Gf.Vec3d(point))) for point in points)


def _box_mesh(size: float, matrix: Any, Gf: Any, color: tuple[int, int, int]) -> MeshData:
    half = size / 2.0
    points = [(-half, -half, -half), (half, -half, -half), (half, half, -half), (-half, half, -half),
              (-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half)]
    faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    return MeshData(_apply_gf_transform(points, matrix, Gf), faces, color)


def _cylinder_mesh(radius: float, height: float, axis: str, matrix: Any, Gf: Any,
                   color: tuple[int, int, int], segments: int = 24) -> MeshData:
    if radius <= 0.0 or height <= 0.0 or axis not in {"X", "Y", "Z"}:
        raise ValueError(f"unsupported USD cylinder dimensions/axis: radius={radius}, height={height}, axis={axis}")
    points: list[tuple[float, float, float]] = []
    for sign in (-1.0, 1.0):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            a, b = radius * math.cos(angle), radius * math.sin(angle)
            if axis == "X":
                point = (sign * height / 2.0, a, b)
            elif axis == "Y":
                point = (a, sign * height / 2.0, b)
            else:
                point = (a, b, sign * height / 2.0)
            points.append(point)
    bottom_center = len(points)
    points.append(tuple(-height / 2.0 if index == "XYZ".index(axis) else 0.0 for index in "XYZ"))
    top_center = len(points)
    points.append(tuple(height / 2.0 if index == "XYZ".index(axis) else 0.0 for index in "XYZ"))

    faces: list[tuple[int, int, int]] = []
    reverse_side = axis == "Y"
    for index in range(segments):
        nxt = (index + 1) % segments
        lo, lo_next = index, nxt
        hi, hi_next = segments + index, segments + nxt
        side_faces = ((lo, lo_next, hi_next), (lo, hi_next, hi))
        if reverse_side:
            side_faces = tuple(tuple(reversed(face)) for face in side_faces)
        faces.extend(side_faces)
        if axis == "Y":
            faces.extend(((bottom_center, lo, lo_next), (top_center, hi_next, hi)))
        else:
            faces.extend(((bottom_center, lo_next, lo), (top_center, hi, hi_next)))
    return MeshData(_apply_gf_transform(points, matrix, Gf), tuple(faces), color)


NATIVE_GROUND_DISPLAY_HALF_EXTENT_M = 200.0
NATIVE_BODY_LABELS = (
    "base_link",
    "gimbal_yaw_link",
    "gimbal_pitch_link",
    *(f"left_wheel_{index}_wheel_link" for index in range(7)),
    *(f"right_wheel_{index}_wheel_link" for index in range(7)),
    "left_front_guide_link",
    "right_front_guide_link",
)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _rotate_wxyz(vector: Sequence[float], quaternion: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a local vector by a MuJoCo scalar-first quaternion."""
    x, y, z = _finite_vec(vector, 3, "geometry vertex")
    w, qx, qy, qz = _finite_vec(quaternion, 4, "local_quat_wxyz")
    norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("local_quat_wxyz must be non-zero")
    w, qx, qy, qz = w / norm, qx / norm, qy / norm, qz / norm
    # v' = v + 2*w*(q x v) + 2*(q x (q x v)); this preserves the wxyz order.
    cx, cy, cz = qy * z - qz * y, qz * x - qx * z, qx * y - qy * x
    dx, dy, dz = qy * cz - qz * cy, qz * cx - qx * cz, qx * cy - qy * cx
    return (x + 2.0 * (w * cx + dx),
            y + 2.0 * (w * cy + dy),
            z + 2.0 * (w * cz + dz))


def _native_geom_color(rgba: Any, name: str) -> tuple[int, int, int]:
    channels = _finite_vec(rgba, 4, f"{name}.rgba")
    if any(channel < 0.0 or channel > 1.0 for channel in channels):
        raise ValueError(f"{name}.rgba channels must be between 0 and 1")
    return tuple(round(channel * 255.0) for channel in channels[:3])


def _native_geom_mesh(geom: Mapping[str, Any]) -> MeshData:
    """Build one exact MuJoCo box or cylinder mesh in its owning body frame."""
    name = str(geom.get("name", "geometry"))
    shape = geom.get("type")
    size_raw = geom.get("size_m")
    position = _finite_vec(geom.get("local_pos_xyz_m"), 3, f"{name}.local_pos_xyz_m")
    quaternion = _finite_vec(geom.get("local_quat_wxyz"), 4, f"{name}.local_quat_wxyz")
    color = _native_geom_color(geom.get("rgba"), name)

    if shape == "box":
        half = _finite_vec(size_raw, 3, f"{name}.size_m")
        if any(value <= 0.0 for value in half):
            raise ValueError(f"{name} box half-sizes must be positive")
        sx, sy, sz = half
        vertices = (
            (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
            (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
        )
        faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                 (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    elif shape == "cylinder":
        dimensions = _finite_vec(size_raw, 3, f"{name}.size_m")
        radius, half_height = dimensions[:2]
        if radius <= 0.0 or half_height <= 0.0:
            raise ValueError(f"{name} cylinder radius and half-height must be positive")
        segments = 24
        vertices_list: list[tuple[float, float, float]] = []
        for z in (-half_height, half_height):
            vertices_list.extend((radius * math.cos(2.0 * math.pi * i / segments),
                                  radius * math.sin(2.0 * math.pi * i / segments), z)
                                 for i in range(segments))
        bottom_center, top_center = len(vertices_list), len(vertices_list) + 1
        vertices_list.extend(((0.0, 0.0, -half_height), (0.0, 0.0, half_height)))
        faces_list: list[tuple[int, int, int]] = []
        for i in range(segments):
            nxt = (i + 1) % segments
            lo, hi = i, segments + i
            lo_next, hi_next = nxt, segments + nxt
            faces_list.extend(((lo, lo_next, hi_next), (lo, hi_next, hi),
                               (bottom_center, lo_next, lo), (top_center, hi, hi_next)))
        vertices = tuple(vertices_list)
        faces = tuple(faces_list)
    else:
        raise ValueError(f"unsupported native MuJoCo geom type {shape!r} at {name}")

    transformed = []
    for vertex in vertices:
        rotated = _rotate_wxyz(vertex, quaternion)
        transformed.append(tuple(rotated[index] + position[index] for index in range(3)))
    return MeshData(tuple(transformed), tuple(faces), color)


def _native_ground_mesh(geom: Mapping[str, Any]) -> MeshData:
    """Render the exported horizontal z=0 physical plane with a display crop."""
    if geom.get("type") != "plane" or geom.get("body_name") != "world":
        raise ValueError("native ground geometry must be the exported world plane")
    position = _finite_vec(geom.get("local_pos_xyz_m"), 3, "ground.local_pos_xyz_m")
    quaternion = _finite_vec(geom.get("local_quat_wxyz"), 4, "ground.local_quat_wxyz")
    if abs(position[2]) > 1e-8:
        raise ValueError("native physical plane must lie on z=0")
    normal = _rotate_wxyz((0.0, 0.0, 1.0), quaternion)
    if abs(normal[0]) > 1e-8 or abs(normal[1]) > 1e-8 or abs(abs(normal[2]) - 1.0) > 1e-8:
        raise ValueError("native physical plane must be horizontal")
    extent = NATIVE_GROUND_DISPLAY_HALF_EXTENT_M
    vertices = []
    for x, y in ((-extent, -extent), (extent, -extent), (extent, extent), (-extent, extent)):
        rotated = _rotate_wxyz((x, y, 0.0), quaternion)
        vertices.append(tuple(rotated[index] + position[index] for index in range(3)))
    return MeshData(tuple(vertices), ((0, 1, 2), (0, 2, 3)), _native_geom_color(geom.get("rgba"), "ground"))


def _native_geometry_from_payload(payload: Mapping[str, Any]) -> tuple[RobotGeometry, tuple[str, ...]]:
    """Fail closed unless the static export maps every live body to real geometry."""
    if payload.get("schema") != "agtc.native_view_geometry.v1":
        raise ValueError("unsupported native geometry schema")
    model = payload.get("mujoco_static_model")
    newton = payload.get("newton_static_model_world0")
    if not isinstance(model, dict) or not isinstance(newton, dict):
        raise ValueError("native geometry is missing the MuJoCo/Newton model records")
    mapping = model.get("body_mapping_to_newton")
    verification = model.get("body_mapping_verification")
    if (not isinstance(mapping, list) or not isinstance(verification, dict)
            or verification.get("one_to_one") is not True):
        raise ValueError("native 19-body Newton/MuJoCo mapping is not verified one-to-one")
    body_names = newton.get("viewer_body_names_live_order")
    if (not isinstance(body_names, list) or len(body_names) != BODY_COUNT
            or len(set(body_names)) != BODY_COUNT or any(not isinstance(name, str) or not name for name in body_names)):
        raise ValueError("static export must provide 19 unique live viewer body names")
    body_names_tuple = tuple(body_names)
    if body_names_tuple != NATIVE_BODY_LABELS:
        raise ValueError("static export live body order differs from the native trainer packet")
    mapped_names = [entry.get("viewer_name") for entry in mapping if isinstance(entry, dict)]
    if (len(mapping) != BODY_COUNT or len(mapped_names) != BODY_COUNT
            or len(set(mapped_names)) != BODY_COUNT or set(mapped_names) != set(body_names_tuple)):
        raise ValueError("static body mapping does not cover the 19 native live bodies exactly once")
    if verification.get("mujoco_nonworld_bodies") != BODY_COUNT or verification.get("newton_bodies") != BODY_COUNT:
        raise ValueError("static body mapping counts do not match the 19-body live topology")
    if newton.get("body_count") != BODY_COUNT:
        raise ValueError("Newton static export body_count does not equal 19")
    if newton.get("shape_count") != BODY_COUNT + 1:
        raise ValueError("Newton static export shape_count must include 19 robot geoms and the ground plane")

    mapping_by_name: dict[str, Mapping[str, Any]] = {}
    mapping_newton_ids: set[int] = set()
    for entry in mapping:
        body_name = entry.get("mujoco_body_name")
        if not isinstance(body_name, str) or body_name in mapping_by_name:
            raise ValueError("static MuJoCo-to-Newton body names must be unique")
        newton_body_id = _strict_int(entry.get("newton_body_id"), "mapping.newton_body_id")
        if newton_body_id in mapping_newton_ids:
            raise ValueError("static Newton body mapping must be one-to-one")
        mapping_newton_ids.add(newton_body_id)
        mapping_by_name[body_name] = entry
    if mapping_newton_ids != set(range(BODY_COUNT)):
        raise ValueError("static Newton body IDs must cover 0 through 18 exactly once")

    body_records = model.get("bodies")
    if not isinstance(body_records, list) or len(body_records) != BODY_COUNT + 1:
        raise ValueError("MuJoCo body records are missing")
    body_map: dict[str, str] = {}
    body_ids: dict[str, int] = {}
    seen_body_ids: set[int] = set()
    world_count = 0
    for body in body_records:
        if not isinstance(body, dict):
            raise ValueError("MuJoCo body record must be an object")
        body_id = _strict_int(body.get("mujoco_body_id"), "mujoco_body_id")
        if body_id in seen_body_ids:
            raise ValueError("MuJoCo body IDs must be unique")
        seen_body_ids.add(body_id)
        name = body.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("MuJoCo body name is missing")
        if body_id == 0:
            if name != "world":
                raise ValueError("MuJoCo body id 0 must be world")
            world_count += 1
            continue
        viewer_name = body.get("viewer_name")
        if viewer_name not in body_names_tuple or viewer_name in body_map.values():
            raise ValueError("MuJoCo body-to-viewer mapping is missing or duplicated")
        mapping_entry = mapping_by_name.get(name)
        if (mapping_entry is None or mapping_entry.get("mujoco_body_id") != body_id
                or mapping_entry.get("viewer_name") != viewer_name):
            raise ValueError("MuJoCo body record disagrees with the verified Newton body map")
        body_map[name] = viewer_name
        body_ids[name] = body_id
    if (world_count != 1 or seen_body_ids != set(range(BODY_COUNT + 1))
            or len(body_map) != BODY_COUNT or set(body_map.values()) != set(body_names_tuple)):
        raise ValueError("static geometry must map all 19 non-world bodies")

    all_geoms = model.get("geometries")
    grounds = model.get("ground_plane_geometries")
    if not isinstance(all_geoms, list) or not isinstance(grounds, list):
        raise ValueError("MuJoCo geom arrays are missing")
    if (_strict_int(model.get("nbody"), "mujoco_static_model.nbody") != BODY_COUNT + 1
            or _strict_int(model.get("ngeom"), "mujoco_static_model.ngeom") != BODY_COUNT + 1):
        raise ValueError("MuJoCo static topology must contain world + 19 bodies and 19 robot geoms + one plane")
    if len(all_geoms) != BODY_COUNT + 1 or len(grounds) != 1:
        raise ValueError("static export must contain 19 robot geoms and exactly one physical ground plane")

    ground = grounds[0]
    ground_id = _strict_int(ground.get("geom_id"), "ground.geom_id")
    if (ground_id < 0 or _strict_int(ground.get("body_id"), "ground.body_id") != 0
            or ground.get("type") != "plane"):
        raise ValueError("ground plane record has invalid body/type identity")
    _finite_vec(ground.get("size_m"), 3, "ground.size_m")
    ground_mesh = _native_ground_mesh(ground)
    seen_geom_ids: set[int] = set()
    geometry_by_body: dict[str, list[MeshData]] = {name: [] for name in body_names_tuple}
    for geom in all_geoms:
        if not isinstance(geom, dict):
            raise ValueError("MuJoCo geometry record must be an object")
        geom_id = _strict_int(geom.get("geom_id"), "geom_id")
        if geom_id in seen_geom_ids:
            raise ValueError("MuJoCo geom_id values must be unique")
        seen_geom_ids.add(geom_id)
        if geom_id == ground_id:
            if geom.get("type") != "plane" or geom.get("body_name") != "world":
                raise ValueError("the exported ground plane does not match its geometry record")
            continue
        body_name = geom.get("body_name")
        viewer_name = geom.get("viewer_name")
        body_id = _strict_int(geom.get("body_id"), "geom.body_id")
        newton_body_id = _strict_int(geom.get("newton_body_id"), "geom.newton_body_id")
        mapping_entry = mapping_by_name.get(body_name)
        if (body_name not in body_map or body_map[body_name] != viewer_name
                or body_ids.get(body_name) != body_id or mapping_entry is None
                or mapping_entry.get("newton_body_id") != newton_body_id):
            raise ValueError(f"geometry {geom.get('name')!r} does not map to its verified Newton body")
        geometry_by_body[viewer_name].append(_native_geom_mesh(geom))
    if len(seen_geom_ids) != len(all_geoms):
        raise ValueError("MuJoCo geom ids are incomplete")
    missing_geometry = [name for name, meshes in geometry_by_body.items() if not meshes]
    if missing_geometry:
        raise ValueError(f"native geometry is missing for 19-body links: {missing_geometry}")
    return RobotGeometry({name: tuple(meshes) for name, meshes in geometry_by_body.items()},
                         ground_mesh, ()), body_names_tuple


def _collapse_shared_values(value: Any) -> Any:
    """Collapse arrays repeated identically for each world to their shared value."""
    if isinstance(value, Mapping):
        ignored = {"device", "nworld", "naconmax_total", "runtime_diagnostics"}
        return {key: _collapse_shared_values(item) for key, item in value.items()
                if key not in ignored and not key.endswith("_array_shape")}
    if isinstance(value, (list, tuple)):
        items = [_collapse_shared_values(item) for item in value]
        if not items:
            return []
        if all(_same_identity_value(items[0], item) for item in items[1:]):
            return items[0]
        return items
    return value


def _same_identity_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-8)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (left.keys() == right.keys()
                and all(_same_identity_value(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(_same_identity_value(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def _physics_identity_view(physics: Mapping[str, Any], world_count: int) -> dict[str, Any]:
    result = dict(physics)
    for key in ("dof", "joints", "bodies"):
        value = _strict_int(result.get(key), f"physics.{key}")
        if value % world_count:
            raise ValueError(f"physics.{key} is not an integer per-world topology")
        result[key] = value // world_count
    return _collapse_shared_values(result)


def _require_native_task_match(static: Mapping[str, Any], identity: Mapping[str, Any],
                               contract: Mapping[str, Any]) -> tuple[str, str, float, int, float, int]:
    """Match the static CPU export to the frozen GPU trial without equating their SHA scope."""
    provenance = static.get("provenance")
    backend = static.get("backend_identity_cpu")
    if not isinstance(provenance, dict) or not isinstance(backend, dict):
        raise ValueError("native geometry export is missing provenance/backend identity")
    identity_env = identity.get("env")
    identity_physics = identity_env.get("physics") if isinstance(identity_env, dict) else None
    contract_physics = contract.get("physics")
    if not isinstance(identity_physics, dict) or not isinstance(contract_physics, dict):
        raise ValueError("native trial contract is missing its physics identity")

    task_sha = identity.get("task_identity_sha256")
    if (not isinstance(task_sha, str) or not task_sha
            or contract.get("task_identity_sha256") != task_sha
            or identity_env.get("task_identity_sha256") != task_sha
            or provenance.get("formal_gpu_run_task_identity_sha256_read_from_run_contract") != task_sha):
        raise ValueError("native trial and static export task contract SHA do not match")
    if identity.get("physics") != "newton_mujoco_warp" or identity_physics.get("engine") != "newton_mujoco_warp":
        raise ValueError("native trial task contract must use newton_mujoco_warp")
    if contract_physics.get("engine") != "newton_mujoco_warp":
        raise ValueError("observation contract physics engine is not newton_mujoco_warp")
    engine = backend.get("engine")
    revision = backend.get("backend_revision")
    if engine != "newton_mujoco_warp" or not isinstance(revision, str) or not revision:
        raise ValueError("static CPU model engine/revision is invalid")

    trial_worlds = _strict_int(identity.get("num_envs"), "identity.num_envs")
    cpu_worlds = _strict_int(backend.get("nworld"), "backend_identity_cpu.nworld")
    if trial_worlds < ENV_COUNT or cpu_worlds != 1:
        raise ValueError(f"native trial must train at least {ENV_COUNT} environments; geometry export must use one CPU world")
    if identity_physics.get("backend_revision") != revision or contract_physics.get("backend_revision") != revision:
        raise ValueError("native trial and CPU geometry backend revisions do not match")
    formal_physics = provenance.get("formal_run_physics")
    if not isinstance(formal_physics, dict):
        raise ValueError("static export does not record the formal GPU physics identity")
    if (formal_physics.get("engine") != engine
            or formal_physics.get("backend_revision") != revision
            or _strict_int(formal_physics.get("nworld"), "provenance.formal_run_physics.nworld") != trial_worlds):
        raise ValueError("static export formal GPU engine/revision/world count differs from the trial")

    cpu_physics_view = _physics_identity_view(backend, cpu_worlds)
    live_physics_view = _physics_identity_view(identity_physics, trial_worlds)
    # The zero-step CPU export records geometry/solver identity but not gravity;
    # the separately frozen formal GPU identity still checks gravity below.
    if "gravity_mps2" not in backend:
        live_physics_view.pop("gravity_mps2", None)
    if cpu_physics_view != live_physics_view:
        raise ValueError("static CPU and live GPU physics topology/options differ after per-world normalization")
    if _physics_identity_view(formal_physics, trial_worlds) != _physics_identity_view(identity_physics, trial_worlds):
        raise ValueError("static export's recorded formal GPU physics does not match the live trial")

    sim_dt = float(backend.get("dt_s"))
    decimation = _strict_int(backend.get("decimation"), "backend_identity_cpu.decimation")
    policy_dt = float(backend.get("policy_dt_s"))
    if (not all(math.isfinite(v) for v in (sim_dt, policy_dt)) or sim_dt <= 0.0
            or decimation <= 0 or not math.isclose(sim_dt * decimation, policy_dt, rel_tol=1e-7, abs_tol=1e-9)):
        raise ValueError("native static physics/policy timestep relation is invalid")
    for physics in (identity_physics, contract_physics):
        if (not math.isclose(float(physics.get("dt_s")), sim_dt, rel_tol=1e-9, abs_tol=1e-12)
                or _strict_int(physics.get("decimation"), "physics.decimation") != decimation):
            raise ValueError("native trial timestep differs from the static CPU export")
    if not math.isclose(float(identity_env.get("policy_dt_s")), policy_dt, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("native trial policy timestep differs from the static CPU export")
    return engine, revision, sim_dt, decimation, policy_dt, trial_worlds


def load_native_viewer_context(model_json: Path, trial_dir: Path) -> NativeViewerContext:
    """Validate the CPU geometry snapshot, frozen source hashes, and live trial contract."""
    model_path = model_json.resolve()
    trial_path = trial_dir.resolve()
    if not model_path.is_file() or not trial_path.is_dir():
        raise ValueError("native model JSON or trial directory does not exist")
    model = _read_json_object(model_path, "native static geometry export")
    if model.get("schema") != "agtc.native_view_geometry.v1":
        raise ValueError("unsupported native static geometry schema")
    trial_name = trial_path.name
    provenance = model.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("static export provenance is missing")
    if PureWindowsPath(str(provenance.get("formal_run_dir", ""))).name != trial_name:
        raise ValueError("static geometry export was captured for a different training trial")
    identity = _read_json_object(trial_path / "identity.json", "native trial identity.json")
    contract = _read_json_object(trial_path / "observation_contract.json", "native trial observation_contract.json")
    engine, revision, sim_dt, decimation, policy_dt, training_env_count = _require_native_task_match(
        model, identity, contract)
    formal_protocol_sha = provenance.get("formal_protocol_sha256")
    protocol_file = _read_json_object(trial_path / "PROTOCOL_NATIVE_V2.sha256", "trial protocol hash")
    if protocol_file.get("protocol_sha256") != formal_protocol_sha:
        raise ValueError("static geometry export and training trial use different frozen V2 protocols")

    source_snapshot = provenance.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("static export source snapshot identity is missing")
    if (source_snapshot.get("relevant_sources_dirty") is not False
            or source_snapshot.get("missing") != [] or source_snapshot.get("mismatches") != []
            or source_snapshot.get("native_source_count") != source_snapshot.get("native_snapshot_files_verified")):
        raise ValueError("static export source snapshot is incomplete or dirty")
    expected_paths = {
        "backend_relative_path": "算法/仿真训练环境/native/proto/newton_mujoco_backend.py",
        "train_wsl_relative_path": "算法/仿真训练环境/native/train/train_wsl.py",
    }
    repo_root = Path(__file__).resolve().parents[3]
    for key, relative_path in expected_paths.items():
        if source_snapshot.get(key) != relative_path:
            raise ValueError(f"static export has an unexpected frozen source path: {key}")
    for path_key, hash_key in (("backend_relative_path", "backend_sha256"),
                               ("train_wsl_relative_path", "train_wsl_sha256")):
        source_path = repo_root.joinpath(*source_snapshot[path_key].split("/"))
        expected_hash = source_snapshot.get(hash_key)
        if not isinstance(expected_hash, str) or _sha256_file(source_path) != expected_hash:
            raise ValueError(f"current frozen source bytes do not match static export: {source_snapshot[path_key]}")
    if provenance.get("constructed_backend_source_sha256") != source_snapshot.get("backend_sha256"):
        raise ValueError("constructed backend source hash disagrees with the frozen snapshot")

    quaternion_conventions = model.get("quaternion_conventions")
    if (not isinstance(quaternion_conventions, dict)
            or quaternion_conventions.get("mujoco_geom_quat_order") != "wxyz"
            or quaternion_conventions.get("newton_live_state_body_q_order") != "xyz + xyzw"):
        raise ValueError("static export quaternion conventions are missing or unsupported")

    construction = provenance.get("construction")
    if (not isinstance(construction, dict) or construction.get("device") != "cpu"
            or construction.get("step_calls") != 0 or construction.get("policy_steps") != 0):
        raise ValueError("static geometry must come from a zero-step CPU construction")
    model_geometry, body_names = _native_geometry_from_payload(model)

    cli_path = trial_path / "cli_args.txt"
    try:
        argv = shlex.split(cli_path.read_text(encoding="utf-8"), posix=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read native trial launch arguments: {exc}") from exc
    intervals = []
    outputs = []
    for index, token in enumerate(argv):
        if token == "--live-state-interval" and index + 1 < len(argv):
            intervals.append(argv[index + 1])
        elif token.startswith("--live-state-interval="):
            intervals.append(token.split("=", 1)[1])
        if token == "--out" and index + 1 < len(argv):
            outputs.append(argv[index + 1])
        elif token.startswith("--out="):
            outputs.append(token.split("=", 1)[1])
    if len(intervals) != 1 or len(outputs) != 1 or Path(outputs[0]).name != trial_name:
        raise ValueError("trial CLI must identify this output and one live-state interval")
    try:
        publish_interval = int(intervals[0])
    except ValueError as exc:
        raise ValueError("live-state publish interval must be an integer") from exc
    if publish_interval <= 0:
        raise ValueError("live-state publish interval must be positive")

    return NativeViewerContext(
        trial_name=trial_name, engine=engine, backend_revision=revision,
        task_identity_sha256=str(identity["task_identity_sha256"]), training_env_count=training_env_count,
        model_sha256=_sha256_file(model_path),
        model_json_path=str(model_path), trial_dir=str(trial_path),
        source_commit=str(source_snapshot.get("git_commit", "unknown")),
        backend_source_sha256=str(source_snapshot["backend_sha256"]),
        train_wsl_source_sha256=str(source_snapshot["train_wsl_sha256"]),
        protocol_sha256=str(formal_protocol_sha),
        body_names=body_names, geometry=model_geometry,
        publish_interval_steps=publish_interval, sim_dt_s=sim_dt,
        decimation=decimation, policy_dt_s=policy_dt,
    )


def validate_native_packet(raw: Any, env_index: int,
                           context: NativeViewerContext) -> dict[str, Any]:
    """Select the native trainer's raw body/goal/time data without consulting USD."""
    if isinstance(env_index, bool) or not isinstance(env_index, int) or not 0 <= env_index < ENV_COUNT:
        raise ValueError(f"env_index must be between 0 and {ENV_COUNT - 1}")
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("expected native trainer live-state schema_version 1")
    if raw.get("quaternion_order") != "xyzw":
        raise ValueError("native trainer packet quaternion_order must be xyzw")
    run_id = raw.get("run_id")
    if run_id != context.trial_name:
        raise ValueError("native live packet run_id does not match trial directory name")
    source_pid = _strict_int(raw.get("source_pid"), "source_pid")
    sequence = _strict_int(raw.get("sequence"), "sequence")
    timestamp = _strict_int(raw.get("timestamp_unix_ns"), "timestamp_unix_ns")
    monotonic_timestamp = _strict_int(raw.get("timestamp_monotonic_ns"), "timestamp_monotonic_ns")
    source_start_time = _strict_int(raw.get("source_start_time_unix_ns"), "source_start_time_unix_ns")
    if (source_pid <= 0 or sequence < 0 or timestamp <= 0 or monotonic_timestamp <= 0
            or source_start_time <= 0):
        raise ValueError("native live packet PID/sequence/timestamps are invalid")
    if raw.get("publish_status") != "ok":
        raise ValueError("native live packet reports a failed publish")
    if _strict_int(raw.get("failed_publish_count"), "failed_publish_count") != 0:
        raise ValueError("native live packet reports prior publish failures")
    names = raw.get("body_names")
    if not isinstance(names, list) or tuple(names) != context.body_names:
        raise ValueError("native live packet body topology differs from the frozen geometry map")
    if len(names) != BODY_COUNT:
        raise ValueError(f"native live packet must contain all {BODY_COUNT} bodies")
    nested = raw.get("body_link_pose_w")
    if not isinstance(nested, list) or len(nested) != ENV_COUNT:
        raise ValueError(f"body_link_pose_w must contain {ENV_COUNT} environments")
    all_body_poses = []
    for row_index, row in enumerate(nested):
        if not isinstance(row, list) or len(row) != BODY_COUNT:
            raise ValueError(f"body_link_pose_w[{row_index}] must contain {BODY_COUNT} bodies")
        all_body_poses.append(tuple(
            _finite_vec(pose, 7, f"body_link_pose_w[{row_index}][{body_index}]")
            for body_index, pose in enumerate(row)
        ))
    body_poses = tuple(all_body_poses[env_index])
    roots = _finite_rows(raw.get("root_link_pose_w"), ENV_COUNT, 7, "root_link_pose_w")
    origins = _finite_rows(raw.get("env_origins_w"), ENV_COUNT, 3, "env_origins_w")
    goals_local = _finite_rows(raw.get("goals_local_xy"), ENV_COUNT, 2, "goals_local_xy")
    goals_world = _finite_rows(raw.get("goals_world_xy"), ENV_COUNT, 2, "goals_world_xy")
    for index in range(ENV_COUNT):
        expected_world = (goals_local[index][0] + origins[index][0],
                          goals_local[index][1] + origins[index][1])
        if any(abs(actual - expected) > 1e-4
               for actual, expected in zip(goals_world[index], expected_world, strict=True)):
            raise ValueError(f"native goal/world-origin mismatch for environment {index}")
    task_times = raw.get("task_times_s")
    if not isinstance(task_times, list) or len(task_times) != ENV_COUNT:
        raise ValueError(f"task_times_s must contain {ENV_COUNT} values")
    task_time = float(task_times[env_index])
    sim_time = float(raw.get("sim_time_s"))
    if not math.isfinite(task_time) or not math.isfinite(sim_time) or task_time < 0.0 or sim_time < 0.0:
        raise ValueError("native sim/episode times must be finite and non-negative")
    base_index = names.index("base_link")
    if any(abs(a - b) > 1e-4 for a, b in zip(roots[env_index], body_poses[base_index], strict=True)):
        raise ValueError("native root pose must match the base_link body pose")
    packet_task_sha = raw.get("task_identity_sha256")
    if packet_task_sha is not None and packet_task_sha != context.task_identity_sha256:
        raise ValueError("native live packet task contract SHA does not match its trial")
    packet_engine = raw.get("engine")
    if packet_engine is not None and packet_engine != context.engine:
        raise ValueError("native live packet engine differs from its trial contract")
    packet_revision = raw.get("backend_revision")
    if packet_revision is not None and packet_revision != context.backend_revision:
        raise ValueError("native live packet backend revision differs from static geometry")
    origin = origins[env_index]
    return {
        "source": "native Newton MuJoCo-Warp trainer",
        "source_pid": source_pid, "source_start_time_unix_ns": source_start_time,
        "sequence": sequence, "timestamp_unix_ns": timestamp,
        "timestamp_monotonic_ns": monotonic_timestamp,
        "run_id": run_id, "sim_time_s": sim_time,
        "episode_record_time_s": task_time, "task_identity_sha256": context.task_identity_sha256,
        "engine": context.engine, "backend_revision": context.backend_revision,
        "body_names": tuple(names), "body_poses": body_poses,
        "root_pose": roots[env_index], "origin": origin,
        "goal_local_xy": goals_local[env_index],
        "sample_phase": str(raw.get("sample_phase", "unknown")),
    }


def load_usd_geometry(robot_path: str, terrain_path: str, body_names: Sequence[str]) -> RobotGeometry:
    """Read this scene's USDA primitives with pxr only; no Kit/SimulationApp is used."""
    from pxr import Gf, Usd, UsdGeom

    robot_stage = Usd.Stage.Open(robot_path)
    terrain_stage = Usd.Stage.Open(terrain_path)
    if not robot_stage or not terrain_stage:
        raise ValueError("could not open a robot or terrain USD asset")
    for name, stage in (("robot", robot_stage), ("terrain", terrain_stage)):
        if UsdGeom.GetStageUpAxis(stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
            raise ValueError(f"{name} USD must use Z-up and meters")

    root = robot_stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise ValueError("robot USD has no valid default prim")
    prims = tuple(Usd.PrimRange(root))
    body_matches: dict[str, list[Any]] = {name: [] for name in body_names}
    for prim in prims:
        if prim.GetName() in body_matches:
            body_matches[prim.GetName()].append(prim)
    missing = [name for name, matches in body_matches.items() if not matches]
    ambiguous = [name for name, matches in body_matches.items() if len(matches) > 1]
    if missing or ambiguous:
        raise ValueError(f"robot USD body mapping mismatch; missing={missing}, ambiguous={ambiguous}")
    body_prims = {name: matches[0] for name, matches in body_matches.items()}
    body_paths = {name: str(prim.GetPath()) for name, prim in body_prims.items()}
    geometry_prims = [prim for prim in prims if prim.IsA(UsdGeom.Gprim)]
    grouped_prims: dict[str, list[Any]] = {name: [] for name in body_names}
    for prim in geometry_prims:
        body = nearest_body_ancestor(str(prim.GetPath()), body_paths)
        if body is not None:
            grouped_prims[body].append(prim)

    cache = UsdGeom.XformCache()
    body_data: dict[str, tuple[MeshData, ...]] = {}
    for body_name in body_names:
        body_matrix = cache.GetLocalToWorldTransform(body_prims[body_name])
        inverse_body = body_matrix.GetInverse()
        meshes: list[MeshData] = []
        for prim in grouped_prims[body_name]:
            relative = cache.GetLocalToWorldTransform(prim) * inverse_body
            shape = str(prim.GetTypeName())
            if shape == "Cube":
                usd_shape = UsdGeom.Cube(prim)
                size = float(usd_shape.GetSizeAttr().Get())
                meshes.append(_box_mesh(size, relative, Gf, (90, 112, 132)))
            elif shape == "Cylinder":
                usd_shape = UsdGeom.Cylinder(prim)
                axis = str(usd_shape.GetAxisAttr().Get())
                meshes.append(_cylinder_mesh(float(usd_shape.GetRadiusAttr().Get()),
                                             float(usd_shape.GetHeightAttr().Get()), axis,
                                             relative, Gf,
                                             (48, 54, 64) if "wheel" in body_name else (214, 132, 48)))
            elif shape == "Mesh":
                raise ValueError(f"robot mesh {prim.GetPath()} is not supported by the primitive viewer")
            else:
                raise ValueError(f"unsupported robot USD shape {shape!r} at {prim.GetPath()}")
        body_data[body_name] = tuple(meshes)

    terrain_meshes = [prim for prim in Usd.PrimRange(terrain_stage.GetDefaultPrim())
                      if prim.IsA(UsdGeom.Mesh)]
    if len(terrain_meshes) != 1:
        raise ValueError(f"expected one terrain mesh, found {len(terrain_meshes)}")
    terrain_prim = terrain_meshes[0]
    mesh = UsdGeom.Mesh(terrain_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        raise ValueError("terrain USD mesh is empty")
    terrain_transform = UsdGeom.XformCache()
    terrain_matrix = terrain_transform.GetLocalToWorldTransform(terrain_prim)
    terrain = MeshData(
        _apply_gf_transform(points, terrain_matrix, Gf),
        _triangulate_face_counts(counts, indices),
        (116, 100, 76),
    )
    proxy_bodies = tuple(name for name in body_names if not body_data[name])
    return RobotGeometry(body_data, terrain, proxy_bodies)


class LatestPacketReader:
    """Stat each poll; retain the last accepted state across missing/torn files."""
    def __init__(self, path: Path):
        self.path = path
        self.signature: tuple[int, int] | None = None
        self.error = ""
        self.last_good: Any | None = None

    def poll(self) -> Any | None:
        try:
            info = self.path.stat()
        except OSError as exc:
            self.error = str(exc)
            return None
        signature = (info.st_mtime_ns, info.st_size)
        if signature == self.signature:
            return None
        try:
            packet = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.error = str(exc)
            return None
        self.signature = signature
        self.error = ""
        self.last_good = packet
        return packet


class PacketIdentity:
    """Lock source/run/assets once and reject resets or packets from other runs."""
    def __init__(self, expected_run_id: str = "", expected_source_pid: int | None = None):
        self.expected_run_id = expected_run_id
        self.expected_source_pid = expected_source_pid
        self.run_id: str | None = expected_run_id or None
        self.source_pid: int | None = expected_source_pid
        self.asset_paths: tuple[str, str] | None = None
        self.body_names: tuple[str, ...] | None = None
        self.sequence: int | None = None

    def accept(self, packet: Mapping[str, Any]) -> bool:
        run_id = packet["run_id"]
        source_pid = packet["source_pid"]
        paths = (packet["robot_usd_path"], packet["terrain_usd_path"])
        names = tuple(packet["body_names"])
        if self.run_id is not None and run_id != self.run_id:
            raise ValueError("packet run_id does not match the locked source")
        if self.source_pid is not None and source_pid != self.source_pid:
            raise ValueError("packet source_pid does not match the locked source")
        if self.asset_paths is not None and paths != self.asset_paths:
            raise ValueError("USD asset paths changed during the viewer session")
        if self.body_names is not None and names != self.body_names:
            raise ValueError("body_names changed during the viewer session")
        if self.sequence is not None and packet["sequence"] <= self.sequence:
            if packet["sequence"] == self.sequence:
                return False
            raise ValueError("source sequence moved backwards")
        self.run_id = run_id
        self.source_pid = source_pid
        self.asset_paths = paths
        self.body_names = names
        self.sequence = packet["sequence"]
        return True


class NativePacketIdentity:
    """Lock a native raw-state stream to its verified trial and static model."""
    def __init__(self, context: NativeViewerContext, expected_source_pid: int | None = None):
        self.context = context
        self.expected_source_pid = expected_source_pid
        self.source_pid: int | None = expected_source_pid
        self.source_start_time_unix_ns: int | None = None
        self.sequence: int | None = None

    def accept(self, packet: Mapping[str, Any]) -> bool:
        if packet["run_id"] != self.context.trial_name:
            raise ValueError("native packet run_id changed from the locked trial")
        if packet["task_identity_sha256"] != self.context.task_identity_sha256:
            raise ValueError("native packet task identity changed from the locked trial")
        if packet["engine"] != self.context.engine or packet["backend_revision"] != self.context.backend_revision:
            raise ValueError("native packet engine/revision changed from the locked geometry")
        if tuple(packet["body_names"]) != self.context.body_names:
            raise ValueError("native body topology changed during the viewer session")
        source_pid = packet["source_pid"]
        source_start = packet["source_start_time_unix_ns"]
        if self.source_pid is not None and source_pid != self.source_pid:
            raise ValueError("native packet source_pid does not match the locked trainer")
        if self.source_start_time_unix_ns is not None and source_start != self.source_start_time_unix_ns:
            raise ValueError("native trainer process start time changed during the viewer session")
        if self.sequence is not None and packet["sequence"] <= self.sequence:
            if packet["sequence"] == self.sequence:
                return False
            raise ValueError("native source sequence moved backwards")
        self.source_pid = source_pid
        self.source_start_time_unix_ns = source_start
        self.sequence = packet["sequence"]
        return True


def forbidden_modules_loaded(module_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(name for name in module_names
                        if any(name == prefix or name.startswith(prefix + ".")
                               for prefix in FORBIDDEN_MODULE_PREFIXES)))


def _state_packet_age_s(packet: Mapping[str, Any] | None, now_unix_ns: int) -> float | None:
    """Return the age of a validated state packet, or None before one is accepted."""
    if packet is None:
        return None
    return packet_age_seconds(packet["timestamp_unix_ns"], now_unix_ns)


def _write_status(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _proxy_box() -> MeshData:
    half = 0.035
    vertices = tuple((x, y, z) for x, y, z in (
        (-half, -half, -half), (half, -half, -half), (half, half, -half), (-half, half, -half),
        (-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half),
    ))
    faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    return MeshData(vertices, faces, (224, 142, 50))


def _load_scene(server: Any, packet: Mapping[str, Any], env_index: int) -> tuple[RobotGeometry, dict[str, Any], Any, Any]:
    import numpy as np

    robot_path, terrain_path = packet["robot_usd_path"], packet["terrain_usd_path"]
    if not Path(robot_path).is_file() or not Path(terrain_path).is_file():
        raise ValueError("a USD asset path from the state packet does not exist")
    geometry = load_usd_geometry(robot_path, terrain_path, packet["body_names"])
    server.scene.add_mesh_simple("/terrain/mesh", np.asarray(geometry.terrain.vertices, dtype=np.float32),
                                 np.asarray(geometry.terrain.faces, dtype=np.uint32),
                                 color=geometry.terrain.color, side="double", flat_shading=True,
                                 cast_shadow=False, receive_shadow=True)
    server.scene.add_grid("/terrain/grid", width=12.0, height=12.0, plane="xy",
                          position=(6.0, 6.0, 0.002), cell_size=1.0, section_size=1.0,
                          cell_color=(184, 184, 184), section_color=(138, 138, 138),
                          plane_opacity=0.0)

    frames: dict[str, Any] = {}
    proxy = _proxy_box()
    for body_name in packet["body_names"]:
        frame_path = f"/robot/{body_name}"
        frames[body_name] = server.scene.add_frame(frame_path, show_axes=False)
        meshes = geometry.by_body[body_name]
        if not meshes:
            meshes = (proxy,)
        for index, mesh in enumerate(meshes):
            server.scene.add_mesh_simple(f"{frame_path}/shape_{index}",
                                         np.asarray(mesh.vertices, dtype=np.float32),
                                         np.asarray(mesh.faces, dtype=np.uint32),
                                         color=mesh.color, side="double", flat_shading=True,
                                         cast_shadow=True, receive_shadow=True)

    goal = server.scene.add_icosphere("/goal/marker", radius=0.14, subdivisions=2,
                                      color=(50, 210, 130), side="double")
    goal_label = server.scene.add_label("/goal/label", _t("Goal", "目标"), font_size_mode="scene",
                                        font_scene_height=0.2, position=(0.0, 0.0, 0.25))
    return geometry, frames, goal, goal_label


def _load_native_scene(server: Any, context: NativeViewerContext,
                       packet: Mapping[str, Any]) -> tuple[RobotGeometry, dict[str, Any], Any, Any]:
    """Add only the verified MuJoCo static geometry and live Newton state."""
    import numpy as np

    geometry = context.geometry
    server.scene.add_mesh_simple(
        "/native/physical_flat_plane_z0",
        np.asarray(geometry.terrain.vertices, dtype=np.float32),
        np.asarray(geometry.terrain.faces, dtype=np.uint32),
        color=geometry.terrain.color, side="double", flat_shading=True,
        cast_shadow=False, receive_shadow=True,
    )
    server.scene.add_grid(
        "/native/grid", width=400.0, height=400.0, plane="xy",
        position=(0.0, 0.0, 0.002), cell_size=1.0, section_size=5.0,
        cell_color=(104, 112, 120), section_color=(144, 150, 156),
        plane_opacity=0.0,
    )
    frames: dict[str, Any] = {}
    for body_name in context.body_names:
        frame_path = f"/native/robot/{body_name}"
        frames[body_name] = server.scene.add_frame(frame_path, show_axes=False)
        meshes = geometry.by_body[body_name]
        if not meshes:
            raise ValueError(f"native static model has no drawable geometry for {body_name}")
        for index, mesh in enumerate(meshes):
            server.scene.add_mesh_simple(
                f"{frame_path}/geom_{index}",
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.uint32),
                color=mesh.color, side="double", flat_shading=True,
                cast_shadow=True, receive_shadow=True,
            )
    goal = server.scene.add_icosphere("/native/goal/marker", radius=0.14, subdivisions=2,
                                      color=(50, 210, 130), side="double")
    goal_label = server.scene.add_label("/native/goal/label", _t("Trainer goal", "训练目标"),
                                        font_size_mode="scene", font_scene_height=0.2,
                                        position=(0.0, 0.0, 0.25))
    return geometry, frames, goal, goal_label


def _update_scene(packet: Mapping[str, Any], frames: Mapping[str, Any], goal: Any, goal_label: Any) -> None:
    for name, pose in zip(packet["body_names"], packet["body_poses"]):
        position, wxyz = pose_to_local_wxyz(pose, packet["origin"])
        frames[name].position = position
        frames[name].wxyz = wxyz
    gx, gy = packet["goal_local_xy"]
    goal.position = (gx, gy, 0.08)
    goal_label.position = (gx, gy, 0.30)


def _timestamp_utc_text(timestamp_unix_ns: int) -> str:
    seconds, nanos = divmod(timestamp_unix_ns, 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{stamp.strftime('%Y-%m-%d %H:%M:%S')}.{nanos:09d}Z"


def _native_panel_text(packet: Mapping[str, Any], age: float | None, stale: bool,
                       env_index: int, context: NativeViewerContext, error: str) -> str:
    state = _t("STALE · holding last pose", "过期 · 保持最后位姿") if stale else _t("LIVE", "实时")
    if error and not stale:
        state = _t("UPDATE UNAVAILABLE · holding last pose", "更新不可用 · 保持最后位姿")
    age_text = _t("unknown", "未知") if age is None else f"{age:.2f} s"
    base_index = packet["body_names"].index("base_link")
    base_pose = packet["body_poses"][base_index]
    local_position, body_wxyz = pose_to_local_wxyz(base_pose, packet["origin"])
    w, x, y, z = body_wxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    gx, gy = packet["goal_local_xy"]
    goal_distance = math.hypot(gx - local_position[0], gy - local_position[1])
    sample_interval_s = context.publish_interval_steps * context.policy_dt_s
    content = _t(
        f"**Training state:** {state}  \n"
        f"**Training environments:** {context.training_env_count} · first {ENV_COUNT} exported · viewing {env_index + 1} / {ENV_COUNT} · PID {packet['source_pid']}  \n"
        f"**Scene sampling:** every {sample_interval_s:.2f} simulation seconds ({context.publish_interval_steps} control steps)  \n"
        f"**State time:** {_timestamp_utc_text(packet['timestamp_unix_ns'])} · **Age:** {age_text}  \n"
        f"**Episode time:** {packet['episode_record_time_s']:.2f} s  \n"
        f"**Vehicle pose:** x {local_position[0]:.2f} m · y {local_position[1]:.2f} m · z {local_position[2]:.2f} m · yaw {math.degrees(yaw):.1f}°  \n"
        f"**Distance to goal:** {goal_distance:.2f} m  \n"
        "**Physical model:** 19 / 19 collision links mapped · flat ground at z=0  \n"
        "**Perception:** sensor data not shown · capability metrics false.",
        f"**训练状态：** {state}  \n"
        f"**训练环境：** {context.training_env_count} 个 · 实时导出前 {ENV_COUNT} 个 · 当前查看第 {env_index + 1} 个（{env_index + 1}/{ENV_COUNT}）· PID {packet['source_pid']}  \n"
        f"**画面采样步距：** {sample_interval_s:.2f} 秒仿真时间（每 {context.publish_interval_steps} 个控制步）  \n"
        f"**状态时间：** {_timestamp_utc_text(packet['timestamp_unix_ns'])} · **延迟：** {age_text}  \n"
        f"**本回合时间：** {packet['episode_record_time_s']:.2f} s  \n"
        f"**车体位姿：** x {local_position[0]:.2f} m · y {local_position[1]:.2f} m · z {local_position[2]:.2f} m · yaw {math.degrees(yaw):.1f}°  \n"
        f"**目标距离：** {goal_distance:.2f} m  \n"
        "**物理模型：** 19 / 19 个真实碰撞几何已映射 · z=0 平面地面  \n"
        "**感知：** 未显示传感器数据 · capability metrics=false。",
    )
    if error and not stale:
        content += f"  \n{html.escape(error[:160])}"
    return content


def _native_details_text(packet: Mapping[str, Any] | None, age: float | None,
                         context: NativeViewerContext) -> str:
    if packet is None:
        live_summary = _t("Waiting for the live trainer packet.", "等待训练进程实时数据包。")
    else:
        age_text = _t("unknown", "未知") if age is None else f"{age:.2f} s"
        live_summary = _t(
            f"Live PID {packet['source_pid']} · sequence {packet['sequence']} · {_timestamp_utc_text(packet['timestamp_unix_ns'])} · age {age_text}.",
            f"实时 PID {packet['source_pid']} · 序号 {packet['sequence']} · {_timestamp_utc_text(packet['timestamp_unix_ns'])} · 延迟 {age_text}。",
        )
    return _t(
        f"**Run:** {html.escape(context.trial_name)}  \n"
        f"**Engine / revision:** {context.engine} / {html.escape(context.backend_revision)}  \n"
        f"**Task identity SHA-256:** {context.task_identity_sha256}  \n"
        f"**Packet cadence:** {context.publish_interval_steps} policy steps · physics {context.sim_dt_s:g} s × {context.decimation} = {context.policy_dt_s:g} s/policy  \n"
        f"**Static model:** {html.escape(context.model_json_path)}  \n"
        f"**Static model SHA-256:** {context.model_sha256}  \n"
        f"**Verified source commit:** {context.source_commit}  \n"
        f"**Newton backend source SHA-256:** {context.backend_source_sha256}  \n"
        f"**train_wsl source SHA-256:** {context.train_wsl_source_sha256}  \n"
        f"**Frozen protocol SHA-256:** {context.protocol_sha256}  \n"
        f"**Live trainer trial:** {html.escape(context.trial_dir)}  \n"
        f"{live_summary}  \n"
        "Static geometry was built on CPU from the verified frozen backend source with zero simulation steps; no GPU geometry arrays were read. Dynamic body poses, goal, and episode time come from the live trainer packet. The physical plane is infinite at z=0; its viewer mesh is clipped to ±200 m. Sensor channels are not exported, capability metrics are false, and this is not RTX or real-vehicle validation.",
        f"**训练 run：** {html.escape(context.trial_name)}  \n"
        f"**引擎 / 版本：** {context.engine} / {html.escape(context.backend_revision)}  \n"
        f"**Task identity SHA-256：** {context.task_identity_sha256}  \n"
        f"**数据发布：** 每 {context.publish_interval_steps} 个 policy step · 物理 {context.sim_dt_s:g} s × {context.decimation} = {context.policy_dt_s:g} s/policy  \n"
        f"**静态模型：** {html.escape(context.model_json_path)}  \n"
        f"**静态模型 SHA-256：** {context.model_sha256}  \n"
        f"**已验证源码提交：** {context.source_commit}  \n"
        f"**Newton 后端源码 SHA-256：** {context.backend_source_sha256}  \n"
        f"**train_wsl 源码 SHA-256：** {context.train_wsl_source_sha256}  \n"
        f"**冻结协议 SHA-256：** {context.protocol_sha256}  \n"
        f"**实时训练目录：** {html.escape(context.trial_dir)}  \n"
        f"{live_summary}  \n"
        "静态几何来自经验证冻结后端源码的 CPU 零步构建；未读取 GPU 几何数组。车体位姿、目标和 episode 时间来自训练进程实时数据包。物理平面在 z=0 无限延伸，查看器网格裁剪到 ±200 m。此视图未导出传感器数据，capability metrics=false，不代表 RTX 或实车验收。",
    )


def _initial_camera_pose(packet: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Focus the initial camera on the selected vehicle at roughly four meters."""
    if "base_link" in packet["body_names"]:
        base_index = packet["body_names"].index("base_link")
        world_position = packet["body_poses"][base_index][:3]
    else:
        world_position = packet["root_pose"][:3]
    target = tuple(float(world_position[index]) - packet["origin"][index] for index in range(3))
    position = tuple(target[index] + INITIAL_CAMERA_OFFSET[index] for index in range(3))
    return position, target


def _t(english: str, chinese: str) -> str:
    """Return the requested UI language label (default English, unchanged behavior)."""
    return chinese if _LANGUAGE == "zh" else english


def _panel_text(packet: Mapping[str, Any] | None, age: float | None, stale: bool,
                env_index: int, proxy_count: int, error: str,
                native_context: NativeViewerContext | None = None) -> str:
    if packet is None:
        message = _t("Waiting for valid state data", "等待有效状态数据")
        if error:
            message += f"  ·  {html.escape(error[:160])}"
        if native_context is not None:
            return _t(
                f"**Physics-state view:** Native Newton MuJoCo-Warp  \n"
                f"**Backend revision:** {html.escape(native_context.backend_revision)}  \n"
                f"**Static geometry:** verified zero-step CPU export from {html.escape(native_context.source_commit[:12])}  \n"
                f"**Trial:** {html.escape(native_context.trial_name)} · waiting for raw trainer state  \n"
                "**Ground:** physical flat plane z=0; view clipped to ±200 m.  \n"
                "Perception channels are not exported; capability metrics false.",
                f"**物理状态视图：** Native Newton MuJoCo-Warp  \n"
                f"**后端版本：** {html.escape(native_context.backend_revision)}  \n"
                f"**静态几何：** 经验证的零步 CPU 导出，source {html.escape(native_context.source_commit[:12])}  \n"
                f"**训练 trial：** {html.escape(native_context.trial_name)} · 等待训练原始状态  \n"
                "**地面：** 物理平面 z=0；显示裁剪到 ±200 m。  \n"
                "此视图未导出感知通道；capability metrics=false。",
            )
        return _t(f"**Training state**  \n{message}", f"**训练状态**  \n{message}")
    if native_context is not None:
        return _native_panel_text(packet, age, stale, env_index, native_context, error)
    state = _t("STALE · holding last pose", "过期 · 保持最后位姿") if stale else _t("LIVE", "实时")
    if error and not stale:
        state = _t("UPDATE UNAVAILABLE · holding last pose", "更新不可用 · 保持最后位姿")
    age_text = _t("unknown", "未知") if age is None else f"{age:.1f} s"
    content = _t(
        f"**Training source:** {html.escape(packet['source'])}  \n"
        f"**Environment:** {env_index + 1} / {ENV_COUNT}  \n"
        f"**Sequence:** {packet['sequence']}  \n"
        f"**State age:** {age_text}  ·  **{state}**",
        f"**训练来源：** {html.escape(packet['source'])}  \n"
        f"**环境：** {env_index + 1} / {ENV_COUNT}  \n"
        f"**序列号：** {packet['sequence']}  \n"
        f"**状态延迟：** {age_text}  ·  **{state}**",
    )
    if proxy_count:
        content += _t(f"  \n{proxy_count} link markers use simple geometry proxies.",
                      f"  \n{proxy_count} 个连杆使用简单几何代理。")
    if error and not stale:
        content += f"  \n{html.escape(error[:160])}"
    return content


def _status_record(args: argparse.Namespace, packet: Mapping[str, Any] | None,
                   age: float | None, stale: bool, model_loaded: bool,
                   forbidden: tuple[str, ...], error: str, server_url: str,
                   native_context: NativeViewerContext | None = None) -> dict[str, Any]:
    record = {
        "server_pid": os.getpid(),
        "serverpid": os.getpid(),
        "source_pid": None if packet is None else packet["source_pid"],
        "sourcepid": None if packet is None else packet["source_pid"],
        "sequence": None if packet is None else packet["sequence"],
        "run_id": None if packet is None else packet["run_id"],
        "env_index": args.env_index,
        "state_age_s": age,
        "stale": stale,
        "model_loaded": model_loaded,
        "no_simulation": not forbidden,
        "consumer_lightweight": True,
        "consumerlightweight": True,
        "isaacsim_loaded": any(name == "isaacsim" or name.startswith("isaacsim.") for name in forbidden),
        "simulation_app_loaded": any(name == "isaacsim" or name.startswith("isaacsim.")
                                      for name in forbidden),
        "simulation_context_loaded": any(name == "isaaclab" or name.startswith("isaaclab.")
                                          for name in forbidden),
        "training_environment_loaded": any(name.startswith("isaac_field") or name.startswith("isaaclab")
                                            for name in forbidden),
        "torch_loaded": any(name == "torch" or name.startswith("torch.") for name in forbidden),
        "forbidden_modules_loaded": list(forbidden),
        "url": server_url,
        "state_file": str(Path(args.state_file)),
        "error": error[:500] or None,
        "updated_at_unix_ns": time.time_ns(),
    }
    if native_context is not None:
        record.update({
            "viewer_mode": "native_physics_state",
            "engine": native_context.engine,
            "backend_revision": native_context.backend_revision,
            "task_identity_sha256": native_context.task_identity_sha256,
            "native_trial_name": native_context.trial_name,
            "native_model_json": str(Path(args.native_model_json)),
            "native_model_sha256": native_context.model_sha256,
            "static_geometry_source": "verified_zero_step_cpu_export",
            "static_source_commit": native_context.source_commit,
            "backend_source_sha256": native_context.backend_source_sha256,
            "train_wsl_source_sha256": native_context.train_wsl_source_sha256,
            "protocol_sha256": native_context.protocol_sha256,
            "publish_interval_policy_steps": native_context.publish_interval_steps,
            "physics_dt_s": native_context.sim_dt_s,
            "decimation": native_context.decimation,
            "policy_dt_s": native_context.policy_dt_s,
            "perception_channels_exported": False,
            "capability_metrics_allowed": False,
        })
    return record


def run(args: argparse.Namespace) -> None:
    global _LANGUAGE
    _LANGUAGE = getattr(args, "language", "en")
    native_context = None
    if getattr(args, "native_model_json", None) and getattr(args, "native_trial_dir", None):
        native_context = load_native_viewer_context(Path(args.native_model_json), Path(args.native_trial_dir))
        if args.expected_run_id and args.expected_run_id != native_context.trial_name:
            raise ValueError("--expected-run-id does not match --native-trial-dir")
    from viser import ViserServer
    import numpy as np

    server = ViserServer(host=args.host, port=args.port)
    server_url = f"http://{args.host}:{server.get_port()}"
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")
    background = np.empty((2, 2, 3), dtype=np.uint8)
    background[:] = (18, 24, 32)
    server.scene.set_background_image(background, format="png")
    if native_context is None:
        initial_panel = _t("**Training state**  \nWaiting for state data",
                           "**训练状态**  \n等待状态数据")
    else:
        initial_panel = _t("**Native Newton MuJoCo-Warp physics-state view**  \nWaiting for the verified trainer packet",
                           "**Native Newton MuJoCo-Warp 物理状态视图**  \n等待已验证的训练数据包")
    panel = server.gui.add_markdown(initial_panel)
    detail_panel = None
    if native_context is not None:
        with server.gui.add_folder(_t("Source and calibration", "来源与校准"),
                                   expand_by_default=False):
            detail_panel = server.gui.add_markdown(_native_details_text(None, None, native_context))

    reader = LatestPacketReader(Path(args.state_file))
    identity = (NativePacketIdentity(native_context, args.expected_source_pid)
                if native_context is not None
                else PacketIdentity(args.expected_run_id, args.expected_source_pid))
    packet: dict[str, Any] | None = None
    geometry: RobotGeometry | None = None
    frames: dict[str, Any] = {}
    goal = goal_label = None
    accepted_at = 0.0
    last_status_at = 0.0
    last_panel = ""
    last_panel_update_at = 0.0
    last_panel_sequence: int | None = None
    last_panel_stale: bool | None = None
    last_panel_error: str | None = None
    last_panel_proxy_count: int | None = None
    last_detail_panel = ""
    last_error = ""
    camera_initialized = False
    status_path = Path(args.output) / "viewer_status.json"
    forbidden = forbidden_modules_loaded(sys.modules)

    print(f"Web state viewer: {server_url}", flush=True)
    try:
        while True:
            raw = reader.poll()
            if raw is not None:
                try:
                    candidate = (validate_native_packet(raw, args.env_index, native_context)
                                 if native_context is not None else validate_packet(raw, args.env_index))
                    if identity.accept(candidate):
                        if geometry is None:
                            if native_context is None:
                                geometry, frames, goal, goal_label = _load_scene(server, candidate, args.env_index)
                            else:
                                geometry, frames, goal, goal_label = _load_native_scene(server, native_context,
                                                                                        candidate)
                        packet = candidate
                        _update_scene(packet, frames, goal, goal_label)
                        if not camera_initialized:
                            camera_position, camera_target = _initial_camera_pose(packet)
                            server.initial_camera.position = camera_position
                            server.initial_camera.look_at = camera_target
                            camera_initialized = True
                        accepted_at = time.monotonic()
                        last_error = ""
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    last_error = str(exc)
            elif reader.error:
                last_error = f"state read error: {reader.error}"

            now_mono = time.monotonic()
            now_unix_ns = time.time_ns()
            age = _state_packet_age_s(packet, now_unix_ns)
            stale = state_is_stale(age, now_mono - accepted_at if packet is not None else math.inf)
            proxy_count = 0 if geometry is None else len(geometry.proxy_bodies)
            panel_sequence = None if packet is None else packet["sequence"]
            panel_event = (panel_sequence != last_panel_sequence or stale != last_panel_stale
                           or last_error != last_panel_error or proxy_count != last_panel_proxy_count)
            panel_refresh_due = now_mono - last_panel_update_at >= STATUS_INTERVAL_S
            if panel_event or panel_refresh_due:
                panel_text = _panel_text(packet, age, stale, args.env_index, proxy_count, last_error,
                                         native_context)
                if panel_text != last_panel:
                    panel.content = panel_text
                    last_panel = panel_text
                if native_context is not None and detail_panel is not None:
                    detail_text = _native_details_text(packet, age, native_context)
                    if detail_text != last_detail_panel:
                        detail_panel.content = detail_text
                        last_detail_panel = detail_text
                last_panel_update_at = now_mono
                last_panel_sequence = panel_sequence
                last_panel_stale = stale
                last_panel_error = last_error
                last_panel_proxy_count = proxy_count

            if now_mono - last_status_at >= STATUS_INTERVAL_S:
                forbidden = forbidden_modules_loaded(sys.modules)
                try:
                    _write_status(status_path, _status_record(args, packet, age, stale,
                                                              geometry is not None, forbidden,
                                                              last_error, server_url, native_context))
                except OSError as exc:
                    print(f"viewer status write failed: {exc}", file=sys.stderr, flush=True)
                last_status_at = now_mono
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", required=True, help="Atomic live-state JSON from the training process")
    parser.add_argument("--output", required=True, help="Directory for viewer_status.json")
    parser.add_argument("--native-model-json", help="Verified zero-step CPU MuJoCo geometry export for native viewing")
    parser.add_argument("--native-trial-dir", help="Identity/contract directory for the live native trainer")
    parser.add_argument("--expected-run-id", default="", help="Reject packets from other runs")
    parser.add_argument("--expected-source-pid", type=int, help="Reject packets from other source processes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--language", default="en", choices=("en", "zh"),
                        help="viewer UI language for panel labels (default en)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if bool(args.native_model_json) != bool(args.native_trial_dir):
        raise SystemExit("--native-model-json and --native-trial-dir must be supplied together")
    if not 0 <= args.env_index < ENV_COUNT:
        raise SystemExit(f"--env-index must be between 0 and {ENV_COUNT - 1}")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.expected_source_pid is not None and args.expected_source_pid <= 0:
        raise SystemExit("--expected-source-pid must be positive")
    try:
        run(args)
    except ValueError as exc:
        raise SystemExit(f"viewer identity preflight failed: {exc}") from exc


if __name__ == "__main__":
    main()
