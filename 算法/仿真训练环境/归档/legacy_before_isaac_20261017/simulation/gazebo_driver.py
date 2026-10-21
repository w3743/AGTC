"""Paused, step-synchronous Gazebo Harmonic driver for field_sim.

This module owns only the Gazebo process it starts.  It changes a temporary
copy of an exported SDF, keeps the world paused between calls, and exposes raw
sensor frames with their simulation-time pose / gimbal context.  Gym reward,
observation construction, and sensor latency remain the caller's job.
"""
from __future__ import annotations

from collections import deque
import importlib
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = ROOT / "results" / "gazebo_field" / "current" / "world.sdf"
DEFAULT_TOPICS = {
    "cmd_vel": "/field/cmd_vel",
    "joint_states": "/field/joint_states",
    "lidar": "/field/lidar/scan",
    "depth": "/field/depth/image",
    "imu": "/field/imu/data",
    "magnetometer": "/field/magnetometer/data",
    "gimbal_yaw": "/field/gimbal/yaw/cmd_pos",
    "gimbal_pitch": "/field/gimbal/pitch/cmd_pos",
}


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


def _seconds(message) -> float | None:
    if message is None:
        return None
    if isinstance(message, (float, int)):
        value = float(message)
        return value if math.isfinite(value) else None
    for sec_name, nsec_name in (("sec", "nsec"), ("seconds", "nanoseconds")):
        sec = _safe_attr(message, sec_name)
        nsec = _safe_attr(message, nsec_name, 0)
        if sec is not None:
            try:
                value = float(sec) + float(nsec or 0) * 1e-9
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                pass
    return None


def _stamp(message) -> float | None:
    """Return a present protobuf header/time stamp; absent means use a clock fallback."""
    header = _safe_attr(message, "header")
    if header is not None and _has_field(message, "header") is not False:
        stamp = _safe_attr(header, "stamp")
        if stamp is not None and _has_field(header, "stamp") is not False:
            value = _seconds(stamp)
            if value is not None:
                return value
    time_field = _safe_attr(message, "time")
    if time_field is not None and _has_field(message, "time") is not False:
        return _seconds(time_field)
    return None


def _message_type(module_name: str, class_name: str):
    _ensure_ros_protobuf()
    module = importlib.import_module(f"gz.msgs10.{module_name}_pb2")
    return getattr(module, class_name)


def _ensure_ros_protobuf() -> None:
    """Expose Ubuntu's ROS protobuf package to an isolated learning venv if needed."""
    try:
        import google.protobuf  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if system_dist_packages.is_dir() and str(system_dist_packages) not in sys.path:
        sys.path.append(str(system_dist_packages))
    try:
        import google.protobuf  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gazebo protobuf Python bindings need google.protobuf; activate the WSL ROS Jazzy environment"
        ) from exc


def _load_native_control(partition: str, service: str):
    """Load a tiny C++ RPC client so blocking Gazebo requests release the GIL."""
    library_path = ROOT / ".runtime-gazebo" / "libfield_gz_control.so"
    source_path = ROOT / "simulation" / "native_gazebo_control.cc"
    build_script = ROOT / "scripts" / "build_gazebo_control.sh"
    if (not library_path.is_file() or
            library_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns):
        completed = subprocess.run(
            ["bash", str(build_script)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "could not build the local Gazebo control client:\n" + completed.stdout[-4000:]
            )
    library = ctypes.CDLL(str(library_path))
    library.field_gz_control_create.argtypes = (
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    )
    library.field_gz_control_create.restype = ctypes.c_void_p
    library.field_gz_control_request.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.c_size_t,
    )
    library.field_gz_control_request.restype = ctypes.c_int
    library.field_gz_control_destroy.argtypes = (ctypes.c_void_p,)
    library.field_gz_control_destroy.restype = None
    error = ctypes.create_string_buffer(512)
    handle = library.field_gz_control_create(
        partition.encode("utf-8"), service.encode("utf-8"), error, len(error),
    )
    if not handle:
        raise RuntimeError(f"could not create native Gazebo control client: {error.value.decode(errors='replace')}")
    return library, handle


def _pose7(pose) -> list[float] | None:
    position = _safe_attr(pose, "position")
    orientation = _safe_attr(pose, "orientation")
    if position is None or orientation is None:
        return None
    values = [float(_safe_attr(position, key, math.nan)) for key in ("x", "y", "z")]
    values.extend(float(_safe_attr(orientation, key, math.nan))
                  for key in ("x", "y", "z", "w"))
    return values if all(math.isfinite(value) for value in values) else None


def _find_robot_pose(message, model_name: str) -> list[float] | None:
    poses = _safe_attr(message, "pose", ())
    named = [(str(_safe_attr(pose, "name", "")), pose) for pose in poses]
    for name, pose in named:
        if name == model_name or name.endswith(f"::{model_name}"):
            value = _pose7(pose)
            if value is not None:
                return value
    # Some PosePublisher builds include links but omit the model pose.  The
    # base link is the closest available physical pose in that case.
    for name, pose in named:
        if name.endswith(f"{model_name}::base_link") or name.endswith("::base_link"):
            value = _pose7(pose)
            if value is not None:
                return value
    return None


def _joint_angles(message, yaw_joint: str = "gimbal_yaw_joint",
                  pitch_joint: str = "gimbal_pitch_joint") -> list[float] | None:
    found = {}
    for joint in _safe_attr(message, "joint", ()):
        name = str(_safe_attr(joint, "name", ""))
        target = ("yaw" if name == yaw_joint or name.endswith(f"::{yaw_joint}") else
                  "pitch" if name == pitch_joint or name.endswith(f"::{pitch_joint}") else None)
        if target is None:
            continue
        axis = _safe_attr(joint, "axis1")
        if axis is None or _has_field(joint, "axis1") is False:
            legacy = _safe_attr(joint, "axis", ())
            axis = legacy[0] if legacy else None
        if axis is None:
            continue
        position = _safe_attr(axis, "position")
        if position is None or _has_field(axis, "position") is False:
            continue
        value = float(position)
        if math.isfinite(value):
            found[target] = value
    if set(found) != {"yaw", "pitch"}:
        return None
    return [found["yaw"], found["pitch"]]


def _decode_depth(message):
    import numpy as np

    width, height = int(message.width), int(message.height)
    step = int(_safe_attr(message, "step", 0)) or width * 4
    data = bytes(message.data)
    if width <= 0 or height <= 0 or step < width * 4 or len(data) < height * step:
        raise ValueError(f"invalid depth payload: {width}x{height}, step={step}, bytes={len(data)}")
    return np.ndarray((height, width), dtype="<f4", buffer=data,
                      strides=(step, 4)).copy()


def _match_context(stamp: float, pose_history, gimbal_history) -> dict:
    def nearest(history):
        if not history:
            return None, None
        sample = min(history, key=lambda row: abs(float(row["sim_time"]) - stamp))
        return sample, abs(float(sample["sim_time"]) - stamp)

    pose, pose_error = nearest(pose_history)
    gimbal, gimbal_error = nearest(gimbal_history)
    return {
        "pose": None if pose is None else list(pose["pose"]),
        "gimbal": None if gimbal is None else list(gimbal["gimbal"]),
        "pose_match_error_s": pose_error,
        "gimbal_match_error_s": gimbal_error,
    }


def _write_runtime_sdf(source: Path, model_name: str, step_size: float,
                       gimbal_rates: tuple[float, float]) -> Path:
    tree = ET.parse(source)
    root = tree.getroot()
    physics = root.find(".//world/physics")
    if physics is None:
        raise ValueError(f"SDF has no world physics element: {source}")
    max_step = physics.find("max_step_size")
    if max_step is None:
        max_step = ET.SubElement(physics, "max_step_size")
    max_step.text = format(step_size, ".12g")
    rtf = physics.find("real_time_factor")
    if rtf is None:
        rtf = ET.SubElement(physics, "real_time_factor")
    rtf.text = "0"
    update_rate = physics.find("real_time_update_rate")
    if update_rate is not None:
        update_rate.text = "0"

    world = root.find(".//world")
    if world is None:
        raise ValueError(f"SDF has no world element: {source}")
    robot = next((model for model in world.findall("model")
                  if model.get("name") == model_name), None)
    if robot is None:
        raise ValueError(f"SDF has no inline model named {model_name!r}")

    for plugin in robot.findall("plugin"):
        joint_name = (plugin.findtext("joint_name") or "").strip()
        max_rate = (gimbal_rates[0] if joint_name == "gimbal_yaw_joint" else
                    gimbal_rates[1] if joint_name == "gimbal_pitch_joint" else None)
        if max_rate is not None:
            cmd_max = plugin.find("cmd_max")
            if cmd_max is None:
                cmd_max = ET.SubElement(plugin, "cmd_max")
            cmd_max.text = format(max_rate, ".12g")

    pose_publisher_name = "gz::sim::systems::PosePublisher"
    has_pose_publisher = any(
        plugin.get("name") in (pose_publisher_name, "ignition::gazebo::systems::PosePublisher")
        for plugin in robot.findall("plugin")
    )
    if not has_pose_publisher:
        plugin = ET.SubElement(robot, "plugin", {
            "filename": "gz-sim-pose-publisher-system", "name": pose_publisher_name,
        })
        ET.SubElement(plugin, "publish_model_pose").text = "true"
        ET.SubElement(plugin, "publish_link_pose").text = "true"
        ET.SubElement(plugin, "publish_collision_pose").text = "false"
        ET.SubElement(plugin, "publish_visual_pose").text = "false"
        ET.SubElement(plugin, "publish_nested_model_pose").text = "false"
        ET.SubElement(plugin, "use_pose_vector_msg").text = "true"

    fd, name = tempfile.mkstemp(prefix=f".{source.stem}_driver_", suffix=".sdf",
                                dir=source.parent)
    os.close(fd)
    target = Path(name)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target


def _stop_owned_process(process: subprocess.Popen, timeout: float = 3.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class GazeboDriver:
    """Own a paused Gazebo process and advance it in exact physics-step batches."""

    def __init__(self, world_path=DEFAULT_WORLD, log_dir="results/gazebo_field_runs/driver",
                 dt: float = 0.05, step_size: float = 0.005, gui: bool = False,
                 gimbal_rates=(1.8, 1.2)):
        self.world_path = Path(world_path).expanduser().resolve()
        if not self.world_path.is_file():
            raise FileNotFoundError(self.world_path)
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dt = float(dt)
        self.step_size = float(step_size)
        self.gui = bool(gui)
        self.gimbal_rates = tuple(float(value) for value in gimbal_rates)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive")
        if len(self.gimbal_rates) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.gimbal_rates):
            raise ValueError("gimbal_rates must contain two finite positive values")

        # Keep the partition local to this Node and its child process.  Do not
        # mutate process-wide transport settings shared by other driver instances.
        self.partition = f"field_driver_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._condition = threading.Condition()
        self._closed = False
        self._warmed = False
        self._callback_errors: list[str] = []
        self._request_counter = 0
        self._latest_clock: float | None = None
        self._latest_stats: float | None = None
        self._pose_history = deque(maxlen=20000)
        self._gimbal_history = deque(maxlen=10000)
        self._sensors = {key: deque(maxlen=32) for key in ("lidar", "depth", "imu", "uwb")}
        self._uwb_latest_by_topic = {}
        self._sensor_sequences = {key: 0 for key in self._sensors}
        self._last_snapshot_stamps = {key: None for key in self._sensors}
        self._last_uwb_topic_stamps = {}
        self._uwb_pair_topics: list[str] = []
        self._topics, self._manifest = self._load_metadata()
        self.model_name = str((self._manifest.get("vehicle") or {}).get("model_name", "field_robot"))
        self.world_name = self._world_name(self.world_path)
        self.pose_topic = f"/model/{self.model_name}/pose"
        self.control_service = f"/world/{self.world_name}/control"
        self._sensor_rates = self._load_sensor_rates()
        self._time_tolerance = self.step_size * 0.51
        self._runtime_sdf = _write_runtime_sdf(self.world_path, self.model_name,
                                               self.step_size, self.gimbal_rates)
        self._log_file = None
        self.process = None
        self._native_control_lib = None
        self._native_control_handle = None

        try:
            self._native_control_lib, self._native_control_handle = _load_native_control(
                self.partition, self.control_service,
            )
            from gz.transport13 import Node, NodeOptions

            self._types = {
                "clock": _message_type("clock", "Clock"),
                "stats": _message_type("world_stats", "WorldStatistics"),
                "pose_v": _message_type("pose_v", "Pose_V"),
                "model": _message_type("model", "Model"),
                "image": _message_type("image", "Image"),
                "lidar": _message_type("laserscan", "LaserScan"),
                "imu": _message_type("imu", "IMU"),
                "double": _message_type("double", "Double"),
                "twist": _message_type("twist", "Twist"),
                "world_control": _message_type("world_control", "WorldControl"),
                "boolean": _message_type("boolean", "Boolean"),
            }
            node_options = NodeOptions()
            node_options.partition = self.partition
            self._node = Node(node_options)
            self._subscriptions = []
            self._subscribe("/clock", self._types["clock"], self._on_clock)
            self._subscribe(f"/world/{self.world_name}/stats", self._types["stats"], self._on_stats)
            self._subscribe(self.pose_topic, self._types["pose_v"], self._on_pose)
            self._subscribe(self._topics["joint_states"], self._types["model"], self._on_joints)
            self._subscribe(self._topics["lidar"], self._types["lidar"], self._sensor_callback("lidar"))
            self._subscribe(self._topics["depth"], self._types["image"], self._sensor_callback("depth"))
            self._subscribe(self._topics["imu"], self._types["imu"], self._sensor_callback("imu"))
            self._cmd_pub = self._advertise(self._topics["cmd_vel"], self._types["twist"])
            self._yaw_pub = self._advertise(self._topics["gimbal_yaw"], self._types["double"])
            self._pitch_pub = self._advertise(self._topics["gimbal_pitch"], self._types["double"])
            self._enable_optional_uwb()
            self._start_process()
            self._wait_for_service(timeout_s=60.0)
        except BaseException:
            self.close()
            raise

    def _load_metadata(self):
        manifest_path = self.world_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        topics = dict(DEFAULT_TOPICS)
        topics_path = self.world_path.parent / "world.topics.json"
        if topics_path.is_file():
            raw = json.loads(topics_path.read_text(encoding="utf-8"))
            raw = raw.get("topics", raw)
            aliases = {
                "cmd_vel": ("cmd_vel",),
                "joint_states": ("joint_states",),
                "lidar": ("lidar",),
                "depth": ("depth",),
                "imu": ("imu",),
                "magnetometer": ("magnetometer",),
                "gimbal_yaw": ("gimbal_yaw_cmd", "gimbal_yaw"),
                "gimbal_pitch": ("gimbal_pitch_cmd", "gimbal_pitch"),
            }
            for key, names in aliases.items():
                for name in names:
                    value = raw.get(name)
                    if isinstance(value, str):
                        topics[key] = value
                        break
        return topics, manifest

    def _load_sensor_rates(self):
        sensors = ((self._manifest.get("vehicle") or {}).get("sensors") or {})
        rates = {}
        for kind, default in (("lidar", 10.0), ("depth", 2.5), ("imu", 100.0)):
            try:
                rate = float((sensors.get(kind) or {}).get("update_rate_hz", default))
            except (TypeError, ValueError):
                rate = default
            rates[kind] = rate if math.isfinite(rate) and rate > 0 else default
        return rates

    @staticmethod
    def _world_name(world_path: Path) -> str:
        root = ET.parse(world_path).getroot()
        world = root.find(".//world")
        if world is None:
            raise ValueError(f"SDF has no world element: {world_path}")
        return world.get("name") or "default"

    def _subscribe(self, topic, msg_type, callback):
        result = self._node.subscribe(msg_type, topic, callback)
        if result is False:
            raise RuntimeError(f"could not subscribe to Gazebo topic {topic}")
        self._subscriptions.append(topic)

    def _advertise(self, topic, msg_type):
        publisher = self._node.advertise(topic, msg_type)
        if publisher is None or publisher is False:
            raise RuntimeError(f"could not advertise Gazebo topic {topic}")
        return publisher

    def _enable_optional_uwb(self):
        uwb = self._manifest.get("uwb") or {}
        if not uwb.get("enabled"):
            return
        base = uwb.get("range_topic_base", "/uwb_gz_simulator/distances").rstrip("/")
        tag_ids = {1}
        anchors = uwb.get("anchors", [])
        for anchor in anchors:
            try:
                anchor_id = int(anchor.get("anchor_id", anchor.get("id")))
                for tag_id in tag_ids:
                    topic = f"{base}/a{anchor_id}t{tag_id}"
                    self._subscribe(topic, self._types["double"], self._sensor_callback("uwb", topic))
                    self._uwb_pair_topics.append(topic)
            except (AttributeError, TypeError, ValueError):
                continue
        if not self._uwb_pair_topics:
            raise ValueError("UWB is enabled in manifest but has no usable anchor IDs")

    def _start_process(self):
        child_env = os.environ.copy()
        child_env["GZ_PARTITION"] = self.partition
        child_env["ROS_DOMAIN_ID"] = "73"
        child_env["OMP_NUM_THREADS"] = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        if (self._manifest.get("uwb") or {}).get("enabled"):
            plugin_dir = os.environ.get("FIELD_UWB_PLUGIN_PATH")
            if not plugin_dir:
                plugin_dir = str(Path.home() / "robotics-lab" / "build" / "uwb")
            if Path(plugin_dir).is_dir():
                current = child_env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
                child_env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(
                    part for part in (plugin_dir, current) if part
                )
        command = ["gz", "sim", "-v", "2"]
        if not self.gui:
            command.extend(("-s", "--headless-rendering"))
        # Do not pass -r: the server starts paused and advances only on control requests.
        command.append(str(self._runtime_sdf))
        self._log_file = (self.log_dir / "gazebo.log").open("wb")
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=child_env, stdout=self._log_file,
            stderr=subprocess.STDOUT, start_new_session=True,
        )

    def _request_control(self, multi_step: int, timeout_ms: int = 30000):
        self._request_counter += 1
        request_id = self._request_counter
        if self._native_control_handle is None:
            raise RuntimeError("native Gazebo control client is unavailable")
        error = ctypes.create_string_buffer(512)
        ok = self._native_control_lib.field_gz_control_request(
            self._native_control_handle, int(multi_step), int(timeout_ms),
            error, len(error),
        )
        if not ok:
            raise TimeoutError(self._request_failure_detail(
                request_id, multi_step, error.value.decode(errors="replace") or "native request failed",
            ))
        return True

    def _request_failure_detail(self, request_id: int, multi_step: int, reason: str) -> str:
        with self._condition:
            clock_time = self._latest_clock
            stats_time = self._latest_stats
            pose_time = self._pose_history[-1]["sim_time"] if self._pose_history else None
            pose = self._pose_history[-1]["pose"] if self._pose_history else None
            joint_time = self._gimbal_history[-1]["sim_time"] if self._gimbal_history else None
            gimbal = self._gimbal_history[-1]["gimbal"] if self._gimbal_history else None
            errors = self._callback_errors[-5:]
        return (
            f"WorldControl request #{request_id} (multi_step={multi_step}) to "
            f"{self.control_service} failed: {reason}; process_poll="
            f"{None if self.process is None else self.process.poll()}, clock={clock_time}, "
            f"stats={stats_time}, model_pose_time={pose_time}, model_pose={pose}, "
            f"joint_time={joint_time}, gimbal={gimbal}, callback_errors={errors}"
        )

    def _wait_for_service(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Gazebo exited during startup (code {self.process.returncode}); see {self.log_dir / 'gazebo.log'}")
            try:
                # A zero-step paused control request only discovers/validates the service.
                self._request_control(0, timeout_ms=500)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        raise TimeoutError(f"WorldControl service was not ready: {last_error}")

    def _latest_sim_locked(self) -> float:
        if self._pose_history:
            return float(self._pose_history[-1]["sim_time"])
        if self._latest_clock is not None:
            return self._latest_clock
        if self._latest_stats is not None:
            return self._latest_stats
        return 0.0

    def _fallback_stamp_locked(self) -> float | None:
        if self._latest_clock is not None:
            return self._latest_clock
        if self._pose_history:
            return float(self._pose_history[-1]["sim_time"])
        return self._latest_stats

    def _on_clock(self, message, *_):
        sim = _seconds(_safe_attr(message, "sim"))
        if sim is None:
            return
        with self._condition:
            self._latest_clock = sim
            self._condition.notify_all()

    def _on_stats(self, message, *_):
        sim = _seconds(_safe_attr(message, "sim_time"))
        if sim is None:
            return
        with self._condition:
            self._latest_stats = sim
            self._condition.notify_all()

    def _on_pose(self, message, *_):
        try:
            pose = _find_robot_pose(message, self.model_name)
            stamp = _stamp(message)
            with self._condition:
                if stamp is None:
                    stamp = self._fallback_stamp_locked()
                if pose is not None and stamp is not None:
                    if not self._pose_history or stamp >= self._pose_history[-1]["sim_time"] - 1e-9:
                        self._pose_history.append({"sim_time": float(stamp), "pose": pose})
                self._condition.notify_all()
        except Exception as exc:
            self._record_error("pose", exc)

    def _on_joints(self, message, *_):
        try:
            gimbal = _joint_angles(message)
            stamp = _stamp(message)
            with self._condition:
                if stamp is None:
                    stamp = self._fallback_stamp_locked()
                if gimbal is not None and stamp is not None:
                    if not self._gimbal_history or stamp >= self._gimbal_history[-1]["sim_time"] - 1e-9:
                        self._gimbal_history.append({"sim_time": float(stamp), "gimbal": gimbal})
                self._condition.notify_all()
        except Exception as exc:
            self._record_error("joint states", exc)

    def _sensor_callback(self, kind: str, topic: str | None = None):
        def callback(message, *_):
            try:
                with self._condition:
                    stamp = _stamp(message)
                    source = "message_header"
                    if stamp is None:
                        stamp = self._fallback_stamp_locked()
                        source = "latest_sim_clock_estimate"
                if stamp is None:
                    return
                frame = {"sim_time": float(stamp), "stamp_source": source}
                if kind == "lidar":
                    import numpy as np
                    frame.update({
                        "ranges": np.asarray(message.ranges, dtype=np.float32).copy(),
                        "range_min": float(_safe_attr(message, "range_min", 0.0)),
                        "range_max": float(_safe_attr(message, "range_max", math.inf)),
                    })
                elif kind == "depth":
                    frame["values"] = _decode_depth(message)
                    frame["format"] = str(_safe_attr(message, "pixel_format_type", ""))
                    frame["width"] = int(message.width)
                    frame["height"] = int(message.height)
                elif kind == "imu":
                    frame["orientation"] = _vector4(_safe_attr(message, "orientation"))
                    frame["angular_velocity"] = _vector3(_safe_attr(message, "angular_velocity"))
                    frame["linear_acceleration"] = _vector3(_safe_attr(message, "linear_acceleration"))
                elif kind == "uwb":
                    raw_cm = float(_safe_attr(message, "data", math.nan))
                    frame.update({"topic": topic, "range_raw_cm": raw_cm,
                                  "range_m": raw_cm / 100.0})
                with self._condition:
                    self._sensor_sequences[kind] += 1
                    frame["sequence"] = self._sensor_sequences[kind]
                    self._sensors[kind].append(frame)
                    if kind == "uwb" and topic is not None:
                        self._uwb_latest_by_topic[topic] = dict(frame)
                    self._condition.notify_all()
            except Exception as exc:
                self._record_error(kind, exc)
        return callback

    def _record_error(self, source: str, exc: Exception):
        with self._condition:
            self._callback_errors.append(f"{source}: {type(exc).__name__}: {exc}")
            if len(self._callback_errors) > 100:
                del self._callback_errors[:-100]
            self._condition.notify_all()

    def _publish_action(self, v: float, w: float, yaw: float, pitch: float):
        twist = self._types["twist"]()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self._cmd_pub.publish(twist)
        yaw_msg = self._types["double"]()
        yaw_msg.data = float(yaw)
        self._yaw_pub.publish(yaw_msg)
        pitch_msg = self._types["double"]()
        pitch_msg.data = float(pitch)
        self._pitch_pub.publish(pitch_msg)

    def _wait_for_target(self, target: float, timeout_s: float, *, require_sensors=()):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Gazebo exited while stepping (code {self.process.returncode})")
            with self._condition:
                pose_ok = bool(self._pose_history and
                               self._pose_history[-1]["sim_time"] >= target - self._time_tolerance)
                gimbal_ok = bool(self._gimbal_history and
                                 self._gimbal_history[-1]["sim_time"] >= target - self._time_tolerance)
                clock_ok = (self._latest_clock is None or
                            self._latest_clock >= target - self._time_tolerance)
                sensors_ok = all(predicate() for predicate in require_sensors)
                if pose_ok and gimbal_ok and clock_ok and sensors_ok:
                    return
                self._condition.wait(timeout=0.02)
        with self._condition:
            pose_time = self._pose_history[-1]["sim_time"] if self._pose_history else None
            joint_time = self._gimbal_history[-1]["sim_time"] if self._gimbal_history else None
            clock_time = self._latest_clock
        raise TimeoutError(
            f"Gazebo did not reach sim_time {target:.6f}; model pose={pose_time}, "
            f"gimbal joints={joint_time}, clock={clock_time}"
        )

    def _request_steps(self, num_steps: int, timeout_s: float):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        with self._condition:
            start = self._latest_sim_locked()
        target = start + num_steps * self.step_size
        request_timeout_ms = max(1000, int(timeout_s * 1000))
        self._request_control(num_steps, timeout_ms=request_timeout_ms)
        self._wait_for_target(target, timeout_s)
        return start, target

    def _sensor_due_predicate(self, kind: str, target: float):
        period = 1.0 / self._sensor_rates[kind]
        with self._condition:
            frames = list(self._sensors[kind])
        if not frames:
            return lambda: False
        last_stamp = float(frames[-1]["sim_time"])
        if last_stamp >= target - self._time_tolerance:
            return lambda: True
        due_count = max(0, int(math.floor((target - last_stamp + self._time_tolerance) / period)))
        if due_count <= 0:
            return lambda: True
        required_stamp = last_stamp + due_count * period

        def reached():
            with self._condition:
                return bool(self._sensors[kind] and
                            self._sensors[kind][-1]["sim_time"] >= required_stamp - self._time_tolerance)
        return reached

    def _snapshot(self, trajectory=None):
        import numpy as np

        with self._condition:
            pose_history = list(self._pose_history)
            gimbal_history = list(self._gimbal_history)
            sensor_frames = {kind: (dict(self._sensors[kind][-1]) if self._sensors[kind] else None)
                             for kind in ("lidar", "depth", "imu")}
            uwb_frames = {topic: dict(frame) for topic, frame in self._uwb_latest_by_topic.items()}
            sim_time = self._latest_sim_locked()
            errors = list(self._callback_errors)
            new_flags = {}
            for kind, frame in sensor_frames.items():
                stamp = None if frame is None else frame["sim_time"]
                new_flags[kind] = stamp is not None and stamp != self._last_snapshot_stamps[kind]
                if stamp is not None:
                    self._last_snapshot_stamps[kind] = stamp
            uwb_new_flags = {}
            for topic, frame in uwb_frames.items():
                stamp = float(frame["sim_time"])
                uwb_new_flags[topic] = stamp != self._last_uwb_topic_stamps.get(topic)
                self._last_uwb_topic_stamps[topic] = stamp

        latest_pose = pose_history[-1] if pose_history else None
        latest_gimbal = gimbal_history[-1] if gimbal_history else None
        result = {
            "sim_time": float(sim_time),
            "pose": None if latest_pose is None else list(latest_pose["pose"]),
            "gimbal": None if latest_gimbal is None else list(latest_gimbal["gimbal"]),
            "trajectory": list(trajectory or []),
            "callback_errors": errors,
        }
        for kind, frame in sensor_frames.items():
            if frame is None:
                result[kind] = None
                continue
            stamp = float(frame["sim_time"])
            frame.update(_match_context(stamp, pose_history, gimbal_history))
            frame["new"] = bool(new_flags[kind])
            frame["age_s"] = max(0.0, float(sim_time) - stamp)
            frame["values"] = (np.asarray(frame["values"], dtype=np.float32).copy()
                               if "values" in frame else None)
            frame["ranges"] = (np.asarray(frame["ranges"], dtype=np.float32).copy()
                               if "ranges" in frame else None)
            result[kind] = frame
        uwb_result = {}
        for topic, frame in uwb_frames.items():
            stamp = float(frame["sim_time"])
            frame.update(_match_context(stamp, pose_history, gimbal_history))
            frame["new"] = bool(uwb_new_flags[topic])
            frame["age_s"] = max(0.0, float(sim_time) - stamp)
            uwb_result[topic] = frame
        result["uwb"] = uwb_result
        return result

    def warmup(self, seconds: float = 1.2, timeout_s: float = 90.0):
        """Settle physics and wait for actual initial lidar/depth frames."""
        if self._closed:
            raise RuntimeError("driver is closed")
        if self._warmed:
            return self.get_snapshot()
        steps = self._duration_steps(seconds)
        with self._condition:
            lidar_start = self._sensor_sequences["lidar"]
            depth_start = self._sensor_sequences["depth"]
        start, target = self._request_steps(steps, timeout_s)

        def warmup_sensors_ready():
            with self._condition:
                lidar_rows = [row for row in self._sensors["lidar"] if row["sim_time"] > start]
                depth_rows = [row for row in self._sensors["depth"] if row["sim_time"] > start]
                return (self._sensor_sequences["lidar"] > lidar_start
                        and self._sensor_sequences["depth"] >= depth_start + 2
                        and len(lidar_rows) >= 1 and len(depth_rows) >= 2
                        and max(row["sim_time"] for row in depth_rows) >= target - 1.0 / self._sensor_rates["depth"] - self._time_tolerance)

        self._wait_for_target(target, timeout_s, require_sensors=(warmup_sensors_ready,))
        self._warmed = True
        return self._snapshot()

    def _duration_steps(self, duration: float) -> int:
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        steps_float = duration / self.step_size
        steps = int(round(steps_float))
        if steps < 1 or abs(steps_float - steps) > 1e-8:
            raise ValueError("duration must be a positive integer multiple of step_size")
        return steps

    def step(self, v: float, w: float, gy: float, gp: float, duration: float | None = None):
        """Apply physical drive commands and absolute gimbal targets for one interval.

        ``v`` is m/s, ``w`` is rad/s, and ``gy`` / ``gp`` are absolute target
        angles in radians. The environment adapter owns action clipping,
        acceleration slew, and gimbal integration.
        """
        if self._closed:
            raise RuntimeError("driver is closed")
        if not self._warmed:
            raise RuntimeError("call warmup() before step()")
        values = tuple(float(value) for value in (v, w, gy, gp))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("step commands must be finite")
        action_duration = self.dt if duration is None else float(duration)
        steps = self._duration_steps(action_duration)
        with self._condition:
            start = self._latest_sim_locked()
            due_waits = tuple(self._sensor_due_predicate(kind, start + steps * self.step_size)
                              for kind in ("lidar", "depth"))
            previous_pose = self._pose_history[-1] if self._pose_history else None
        self._publish_action(*values)
        timeout_s = max(15.0, action_duration * 20.0 + 5.0)
        actual_start, target = self._request_steps(steps, timeout_s)
        if abs(actual_start - start) > self._time_tolerance:
            start = actual_start
        self._wait_for_target(target, timeout_s, require_sensors=due_waits)
        with self._condition:
            trajectory_rows = [row for row in self._pose_history
                               if start - self._time_tolerance <= row["sim_time"] <= target + self._time_tolerance]
            if previous_pose is not None and (not trajectory_rows or
                    trajectory_rows[0]["sim_time"] > start + self._time_tolerance):
                trajectory_rows.insert(0, previous_pose)
            trajectory = [{"sim_time": float(row["sim_time"]), "pose": list(row["pose"])}
                          for row in trajectory_rows]
        return self._snapshot(trajectory=trajectory)

    def get_snapshot(self):
        if self._closed:
            raise RuntimeError("driver is closed")
        return self._snapshot()

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "process", None)
        if process is not None:
            _stop_owned_process(process)
        native_handle = getattr(self, "_native_control_handle", None)
        native_lib = getattr(self, "_native_control_lib", None)
        if native_handle is not None and native_lib is not None:
            native_lib.field_gz_control_destroy(native_handle)
            self._native_control_handle = None
            self._native_control_lib = None
        node = getattr(self, "_node", None)
        if node is not None:
            for topic in getattr(self, "_subscriptions", ()):
                try:
                    node.unsubscribe(topic)
                except Exception:
                    pass
            self._subscriptions = []
            self._cmd_pub = None
            self._yaw_pub = None
            self._pitch_pub = None
            self._node = None
        log_file = getattr(self, "_log_file", None)
        if log_file is not None and not log_file.closed:
            log_file.close()
        runtime_sdf = getattr(self, "_runtime_sdf", None)
        if runtime_sdf is not None:
            try:
                runtime_sdf.unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _vector3(value):
    if value is None:
        return None
    result = [float(_safe_attr(value, key, math.nan)) for key in ("x", "y", "z")]
    return result if all(math.isfinite(item) for item in result) else None


def _vector4(value):
    if value is None:
        return None
    result = [float(_safe_attr(value, key, math.nan)) for key in ("x", "y", "z", "w")]
    return result if all(math.isfinite(item) for item in result) else None
