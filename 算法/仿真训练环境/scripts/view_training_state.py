"""Render the newest Isaac Lab state packet in an independent, stopped Kit."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PORTABLE_ROOT = Path("C:/pioneer/isaac-training/state-viewer")
FULL_KIT = "C:/pioneer/isaac-training/.venv/Lib/site-packages/isaacsim/apps/isaacsim.exp.full.kit"
STALE_AFTER_S = 5.0
FRAME_PERIOD_S = 1.0 / 30.0


def packet_age_seconds(timestamp_unix_ns: int, now_unix_ns: int) -> float:
    """Return non-negative wall-clock age of a published packet."""
    return max(0.0, (int(now_unix_ns) - int(timestamp_unix_ns)) / 1_000_000_000.0)


def state_is_stale(packet_age_s: float | None, sequence_stall_s: float,
                   threshold_s: float = STALE_AFTER_S) -> bool:
    """Mark the source stale when packets age or the source sequence stops."""
    return packet_age_s is None or packet_age_s > threshold_s or sequence_stall_s > threshold_s


def usd_pose_matrix_xyzw(pose: Any, origin: Any) -> tuple[tuple[float, ...], ...]:
    """Build a Gf-compatible row-major world pose relative to an env origin.

    Isaac Lab packets explicitly carry xyzw quaternions. Callers apply this
    matrix with ``SetResetXformStack(True)`` so nested articulation links keep
    their mesh children but do not multiply the parent's world transform again.
    """
    if len(pose) != 7 or len(origin) != 3:
        raise ValueError("pose/origin must contain 7/3 values")
    values = tuple(float(v) for v in pose)
    offset = tuple(float(v) for v in origin)
    if not all(math.isfinite(v) for v in (*values, *offset)):
        raise ValueError("pose/origin values must be finite")
    x, y, z, qx, qy, qz, qw = values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("pose quaternion must be non-zero")
    qx, qy, qz, qw = (q / norm for q in (qx, qy, qz, qw))
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw), 0.0),
        (2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw), 0.0),
        (2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy), 0.0),
        (x - offset[0], y - offset[1], z - offset[2], 1.0),
    )


def _vec(values: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_packet(raw: Any, env_index: int = 0) -> dict[str, Any]:
    """Validate and select one environment from the live-state schema v1."""
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("expected live-state schema_version 1")
    if raw.get("quaternion_order") != "xyzw":
        raise ValueError("packet quaternion_order must be xyzw")
    if not 0 <= env_index < 4:
        raise ValueError("env_index must be between 0 and 3")
    names = raw.get("body_names")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("body_names must be a non-empty list of names")
    if len(set(names)) != len(names):
        raise ValueError("body_names must be unique")
    body_rows = raw.get("body_link_pose_w")
    root_rows = raw.get("root_link_pose_w")
    origins = raw.get("env_origins_w")
    if not isinstance(body_rows, list) or len(body_rows) != 4 or not isinstance(root_rows, list) or len(root_rows) != 4:
        raise ValueError("root/body poses must contain four environments")
    if not isinstance(origins, list) or len(origins) != 4:
        raise ValueError("env_origins_w must contain four environments")
    row = body_rows[env_index]
    if not isinstance(row, list) or len(row) != len(names):
        raise ValueError("body pose count does not match body_names")
    tasks = raw.get("task_times_s")
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise ValueError("task_times_s must contain four values")
    episode_time = float(tasks[env_index])
    sim_time = float(raw["sim_time_s"])
    if not math.isfinite(sim_time) or not math.isfinite(episode_time):
        raise ValueError("sim/episode time must be finite")
    robot_path, terrain_path = raw.get("robot_usd_path"), raw.get("terrain_usd_path")
    if not isinstance(robot_path, str) or not isinstance(terrain_path, str):
        raise ValueError("robot_usd_path and terrain_usd_path are required")
    source_pid, sequence = int(raw["source_pid"]), int(raw["sequence"])
    timestamp = int(raw["timestamp_unix_ns"])
    if source_pid <= 0 or sequence < 0 or timestamp <= 0:
        raise ValueError("source_pid, sequence, and timestamp are invalid")
    return {
        "source": str(raw.get("source", "unknown")), "source_pid": source_pid,
        "sequence": sequence, "timestamp_unix_ns": timestamp,
        "timestamp_monotonic_ns": int(raw["timestamp_monotonic_ns"]),
        "run_id": str(raw.get("run_id", "")), "sim_time_s": sim_time,
        "episode_record_time_s": episode_time,
        "body_names": tuple(names), "root_pose": _vec(root_rows[env_index], 7, "root pose"),
        "body_poses": tuple(_vec(p, 7, "body pose") for p in row),
        "origin": _vec(origins[env_index], 3, "environment origin"),
        "robot_usd_path": robot_path, "terrain_usd_path": terrain_path,
        "publish_status": str(raw.get("publish_status", "ok")),
        "failed_publish_count": int(raw.get("failed_publish_count", 0)),
        "last_write_error": str(raw.get("last_write_error", "")),
    }


class LatestPacketReader:
    """Stat on each poll; open the JSON only after mtime or size changes."""
    def __init__(self, path: Path):
        self.path = path
        self.signature: tuple[int, int] | None = None
        self.error = ""

    def poll(self) -> Any | None:
        try:
            info = self.path.stat()
        except OSError as exc:
            self.error = str(exc)
            return None
        signature = (info.st_mtime_ns, info.st_size)
        if signature == self.signature:
            return None
        self.signature = signature
        try:
            packet = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.error = str(exc)
            return None
        self.error = ""
        return packet


def _build_scene(context: Any, packet: dict[str, Any],
                 UsdGeom: Any, UsdLux: Any, env_index: int) -> tuple[Any, Any]:
    """Create the independent render stage once, with references to configured assets."""
    context.new_stage()
    stage = context.get_stage()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    env_path = f"/World/Env{env_index}"
    UsdGeom.Xform.Define(stage, env_path)  # Terrain assets are already in env-local coordinates.
    terrain = UsdGeom.Xform.Define(stage, f"{env_path}/Terrain").GetPrim()
    robot = UsdGeom.Xform.Define(stage, f"{env_path}/Robot").GetPrim()
    terrain.GetReferences().AddReference(packet["terrain_usd_path"])
    robot.GetReferences().AddReference(packet["robot_usd_path"])
    dome = UsdLux.DomeLight.Define(stage, "/World/EnvironmentLight")
    dome.CreateIntensityAttr(1500.0)
    return stage, robot


def _find_body_prims(Usd: Any, robot_prim: Any, names: tuple[str, ...]) -> dict[str, Any]:
    wanted = set(names)
    found: dict[str, Any] = {}
    duplicates: set[str] = set()
    for prim in Usd.PrimRange(robot_prim):
        name = prim.GetName()
        if name in wanted:
            if name in found:
                duplicates.add(name)
            else:
                found[name] = prim
    if duplicates or set(found) != wanted:
        return {}
    return found


def _apply_poses(Gf: Any, UsdGeom: Any, body_prims: dict[str, Any], packet: dict[str, Any]) -> None:
    for name, pose in zip(packet["body_names"], packet["body_poses"]):
        xformable = UsdGeom.Xformable(body_prims[name])
        matrix_op = xformable.MakeMatrixXform()
        xformable.SetResetXformStack(True)
        matrix_op.Set(Gf.Matrix4d(usd_pose_matrix_xyzw(pose, packet["origin"])))


def _write_status(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temp, path)


def _run_app(app: Any, args: argparse.Namespace, timeline: Any, context: Any,
             ui: Any, Gf: Any, Usd: Any, UsdGeom: Any, UsdLux: Any) -> None:
    timeline.stop()
    window = ui.Window("Live training", width=470, height=118)
    with window.frame:
        with ui.VStack(spacing=5):
            source_label = ui.Label(f"Live training · Environment {args.env_index + 1}")
            detail_label = ui.Label("Waiting for state data…")
    state_path, output = Path(args.state_file), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "viewer_status.json"
    reader = LatestPacketReader(state_path)
    packet: dict[str, Any] | None = None
    asset_paths: tuple[str, str] | None = None
    stage = robot_prim = None
    body_prims: dict[str, Any] = {}
    last_sequence: int | None = None
    last_sequence_at = time.monotonic()
    next_map_retry = 0.0
    camera_set = False
    last_status_at = 0.0
    last_labels: tuple[str, str] | None = None
    expected_run_id = args.expected_run_id
    expected_pid = args.expected_source_pid

    while app.is_running():
        loop_start = time.perf_counter()
        if timeline.is_playing():
            timeline.stop()
        raw = reader.poll()
        if raw is not None:
            try:
                candidate = validate_packet(raw, args.env_index)
                if expected_run_id and candidate["run_id"] != expected_run_id:
                    raise ValueError("packet run_id does not match --expected-run-id")
                if expected_pid and candidate["source_pid"] != expected_pid:
                    raise ValueError("packet source_pid does not match --expected-source-pid")
                paths = (candidate["robot_usd_path"], candidate["terrain_usd_path"])
                if not all(Path(p).is_file() for p in paths):
                    raise ValueError("packet USD asset path is missing")
                if asset_paths is not None and paths != asset_paths:
                    raise ValueError("USD asset paths changed; restart viewer to keep the first scene fixed")
                if packet is not None and candidate["body_names"] != packet["body_names"]:
                    raise ValueError("body_names changed during viewer session")
                if packet is None:
                    asset_paths = paths
                    stage, robot_prim = _build_scene(context, candidate, UsdGeom, UsdLux, args.env_index)
                if last_sequence is None or candidate["sequence"] != last_sequence:
                    last_sequence_at = time.monotonic()
                packet = candidate
                reader.error = ""
            except (KeyError, TypeError, ValueError, OSError) as exc:
                reader.error = str(exc)

        now_mono = time.monotonic()
        if packet is not None and not body_prims and now_mono >= next_map_retry:
            body_prims = _find_body_prims(Usd, robot_prim, packet["body_names"])
            next_map_retry = now_mono + 0.25
        rootprimvalid = bool(packet is not None and body_prims and len(body_prims) == len(packet["body_names"]))
        if rootprimvalid and packet["sequence"] != last_sequence:
            _apply_poses(Gf, UsdGeom, body_prims, packet)
            last_sequence = packet["sequence"]
        if rootprimvalid and not camera_set:
            from isaacsim.core.utils.viewports import set_camera_view
            import numpy as np
            center = np.array([6.0, 6.0, 0.0])
            set_camera_view(eye=np.array([6.0, -13.0, 13.0]), target=center)
            camera_set = True

        packet_age = None if packet is None else packet_age_seconds(packet["timestamp_unix_ns"], time.time_ns())
        sequence_stall = now_mono - last_sequence_at
        stale = state_is_stale(packet_age, sequence_stall)
        if packet is None:
            source_text = f"Live training · Environment {args.env_index + 1}"
            detail_text = "Waiting for valid state data…" if reader.error else "Waiting for state data…"
        else:
            source_text = f"Live training · Environment {args.env_index + 1}"
            age_text = "unknown" if packet_age is None else f"{packet_age:.1f}s"
            detail_text = f"State age {age_text} | Simulation time {packet['sim_time_s']:.2f}s"
            if stale:
                detail_text += " | Timed out; holding last pose"
            elif reader.error:
                detail_text += " | Update unavailable; holding last pose"
        labels = (source_text, detail_text)
        if labels != last_labels:
            source_label.text, detail_label.text = labels
            last_labels = labels

        if time.monotonic() - last_status_at >= 1.0:
            status = {
                "pid": os.getpid(), "source": None if packet is None else packet["source"],
                "state_file": str(state_path),
                "source_pid": None if packet is None else packet["source_pid"],
                "sequence": None if packet is None else packet["sequence"],
                "rootprimvalid": rootprimvalid, "timelineplaying": bool(timeline.is_playing()),
                "state_age_s": packet_age, "stale": stale,
                "sim_time_s": None if packet is None else packet["sim_time_s"],
                "episode_record_time_s": None if packet is None else packet["episode_record_time_s"],
                "run_id": None if packet is None else packet["run_id"],
                "publish_status": None if packet is None else packet["publish_status"],
                "failed_publish_count": None if packet is None else packet["failed_publish_count"],
                "last_write_error": None if packet is None else packet["last_write_error"],
                "error": reader.error[:500] or None,
            }
            try:
                _write_status(status_path, status)
            except OSError as exc:
                print(f"viewer status write failed: {exc}", file=sys.stderr, flush=True)
            last_status_at = time.monotonic()

        app.update()
        delay = FRAME_PERIOD_S - (time.perf_counter() - loop_start)
        if delay > 0:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", required=True, help="Atomic latest-state JSON written by training")
    parser.add_argument("--output", required=True, help="Directory for viewer_status.json")
    parser.add_argument("--expected-run-id", default="")
    parser.add_argument("--expected-source-pid", type=int)
    parser.add_argument("--env-index", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    sys.argv.extend([
        "--portable-root", str(PORTABLE_ROOT),
        "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
        "--/isaac/startup/create_new_stage=false", "--/app/window/hideUi=0",
    ])
    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": False, "hide_ui": None, "width": 960, "height": 540,
        "window_width": 1500, "window_height": 900, "multi_gpu": False,
        "active_gpu": 0, "physics_gpu": 0, "limit_cpu_threads": 16,
        "sync_loads": False, "renderer": "RaytracedLighting",
        "anti_aliasing": 2, "create_new_stage": False,
    }, experience=FULL_KIT)
    try:
        import omni.timeline
        import omni.ui as ui
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux
        _run_app(app, args, omni.timeline.get_timeline_interface(), omni.usd.get_context(),
                 ui, Gf, Usd, UsdGeom, UsdLux)
    finally:
        app.close()


if __name__ == "__main__":
    main()
