"""Serve a lightweight WebGL view of the latest Isaac Field state packet.

This process only reads the atomic JSON state file. It never imports Isaac Sim,
Isaac Lab, the training environment, PyTorch, or policy/checkpoint code.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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
    goal_label = server.scene.add_label("/goal/label", "Goal", font_size_mode="scene",
                                        font_scene_height=0.2, position=(0.0, 0.0, 0.25))
    return geometry, frames, goal, goal_label


def _update_scene(packet: Mapping[str, Any], frames: Mapping[str, Any], goal: Any, goal_label: Any) -> None:
    for name, pose in zip(packet["body_names"], packet["body_poses"]):
        position, wxyz = pose_to_local_wxyz(pose, packet["origin"])
        frames[name].position = position
        frames[name].wxyz = wxyz
    gx, gy = packet["goal_local_xy"]
    goal.position = (gx, gy, 0.08)
    goal_label.position = (gx, gy, 0.30)


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


def _panel_text(packet: Mapping[str, Any] | None, age: float | None, stale: bool,
                env_index: int, proxy_count: int, error: str) -> str:
    if packet is None:
        message = "Waiting for valid state data"
        if error:
            message += f"  ·  {html.escape(error[:160])}"
        return f"**Training state**  \n{message}"
    state = "STALE · holding last pose" if stale else "LIVE"
    if error and not stale:
        state = "UPDATE UNAVAILABLE · holding last pose"
    age_text = "unknown" if age is None else f"{age:.1f} s"
    content = (
        f"**Training source:** {html.escape(packet['source'])}  \n"
        f"**Environment:** {env_index + 1} / {ENV_COUNT}  \n"
        f"**Sequence:** {packet['sequence']}  \n"
        f"**State age:** {age_text}  ·  **{state}**"
    )
    if proxy_count:
        content += f"  \n{proxy_count} link markers use simple geometry proxies."
    if error and not stale:
        content += f"  \n{html.escape(error[:160])}"
    return content


def _status_record(args: argparse.Namespace, packet: Mapping[str, Any] | None,
                   age: float | None, stale: bool, model_loaded: bool,
                   forbidden: tuple[str, ...], error: str, server_url: str) -> dict[str, Any]:
    return {
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


def run(args: argparse.Namespace) -> None:
    from viser import ViserServer
    import numpy as np

    server = ViserServer(host=args.host, port=args.port)
    server_url = f"http://{args.host}:{server.get_port()}"
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")
    background = np.empty((2, 2, 3), dtype=np.uint8)
    background[:] = (18, 24, 32)
    server.scene.set_background_image(background, format="png")
    panel = server.gui.add_markdown("**Training state**  \nWaiting for state data")

    reader = LatestPacketReader(Path(args.state_file))
    identity = PacketIdentity(args.expected_run_id, args.expected_source_pid)
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
                    candidate = validate_packet(raw, args.env_index)
                    if identity.accept(candidate):
                        if geometry is None:
                            geometry, frames, goal, goal_label = _load_scene(server, candidate, args.env_index)
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
                panel_text = _panel_text(packet, age, stale, args.env_index, proxy_count, last_error)
                if panel_text != last_panel:
                    panel.content = panel_text
                    last_panel = panel_text
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
                                                              last_error, server_url))
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
    parser.add_argument("--expected-run-id", default="", help="Reject packets from other runs")
    parser.add_argument("--expected-source-pid", type=int, help="Reject packets from other source processes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--env-index", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0 <= args.env_index < ENV_COUNT:
        raise SystemExit(f"--env-index must be between 0 and {ENV_COUNT - 1}")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.expected_source_pid is not None and args.expected_source_pid <= 0:
        raise SystemExit("--expected-source-pid must be positive")
    run(args)


if __name__ == "__main__":
    main()
