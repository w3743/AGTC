"""LASSC consumer bridge; no camera, CAN, serial or simulator is started here.

Depth decoding is reused from the RK implementation, never forked. Sensor frame
dictionaries and observation packing reuse FieldTask. All actions are returned
as data; the application owns arming, drive sessions and physical safety limits.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.feedback import FeedbackSnapshot
from isaac_field.task import Config, FieldTask
from isaac_field.telemetry import encode_speed_command


def load_depth_modules(app: str | Path | None = None):
    """Import only the existing pure RK wire/adapter modules, not its server."""
    path = (Path(app) if app else ROOT.parent / "视觉算法/软件/数值深度接口候选/app").resolve()
    for name in ("depth_adapter.py", "depth_wire.py"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    name = "_lassc_depth_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return (importlib.import_module(name + ".depth_adapter"),
            importlib.import_module(name + ".depth_wire"))


def depth_source_hashes(app: str | Path | None = None) -> dict[str, str]:
    adapter, wire = load_depth_modules(app)
    return {str(Path(module.__file__).resolve()): hashlib.sha256(
        Path(module.__file__).read_bytes()).hexdigest() for module in (adapter, wire)}


def fetch_depth(url: str, *, timeout_s: float = 2.0) -> tuple[dict, bytes]:
    """Read one atomic /depth.bin response, with bounded body size."""
    _, wire = load_depth_modules()
    with urlopen(url, timeout=timeout_s) as response:
        metadata = wire.parse_depth_metadata(response.headers["X-Depth-Metadata"])
        payload = response.read(62_881)
    # The reused wire validator rejects missing/extra bytes and malformed headers.
    wire.decode_depth_payload(payload, json.dumps(metadata, ensure_ascii=True))
    return metadata, payload


@dataclass(frozen=True)
class SourcePose:
    """Estimated XY/heading and measured gimbal at exactly the source timestamp.

    This is supplied from telemetry history, not from a command or current pose.
    Clock ID/session/epoch/timestamp must match the depth record exactly.
    """
    session_id: str
    pipeline_epoch: int
    clock_id: str
    source_mono_ns: int
    pose: tuple[float, float, float, float, float]
    measured_gimbal: bool

    def __post_init__(self):
        if (not isinstance(self.session_id, str) or not self.session_id
                or not isinstance(self.clock_id, str) or not self.clock_id
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in (self.pipeline_epoch, self.source_mono_ns))):
            raise ValueError("source pose requires valid session, epoch, clock and exact integer time")
        if len(self.pose) != 5 or not np.isfinite(self.pose).all():
            raise ValueError("source pose must contain five finite measurements")
        if not isinstance(self.measured_gimbal, bool):
            raise ValueError("measured_gimbal must be bool")
        object.__setattr__(self, "pose", tuple(float(v) for v in self.pose))


class PoseHistory:
    """Bounded exact-time telemetry history; never substitutes current pose.

    A clock mapping must first align acquisition times externally. No unsafe
    interpolation across clock/session epochs is inferred by this container.
    """
    def __init__(self, capacity: int = 512):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("positive history capacity required")
        self.records = deque(maxlen=capacity)

    def append(self, *, session_id: str, pipeline_epoch: int, clock_id: str,
               source_mono_ns: int, estimated_xy, feedback: FeedbackSnapshot):
        if not feedback.pose_valid:
            raise ValueError("pose history requires measured attitude and gimbal")
        xy = np.asarray(estimated_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("estimated XY must be finite")
        pose = SourcePose(session_id, pipeline_epoch, clock_id, source_mono_ns,
            (*xy, feedback.estimated_heading_rad, *feedback.estimated_gimbal_q_rad), True)
        self.records.append(pose)
        return pose

    def lookup(self, metadata: dict) -> SourcePose | None:
        key = (metadata.get("session_id"), metadata.get("pipeline_epoch"),
               metadata.get("monotonic_clock_id"), metadata.get("source_mono_ns"))
        for item in reversed(self.records):
            if (item.session_id, item.pipeline_epoch, item.clock_id, item.source_mono_ns) == key:
                return item
        return None


@dataclass(frozen=True)
class ObservationContext:
    episode_time: float
    estimated_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    previous_action: tuple[float, float, float, float]
    feedback: FeedbackSnapshot
    lidar: dict
    camera: dict


@dataclass(frozen=True)
class ConsumerResult:
    ready: bool
    reasons: tuple[str, ...]
    new_frame: bool = False
    reset_recurrent: bool = False
    age_s: float | None = None
    observation: dict | None = None
    action: np.ndarray | None = None


class _MeasuredObservationTask(FieldTask):
    """Use the original packing code with a measured pose, no simulated UWB."""
    def _estimated_pose(self):
        return self.measured_pose.copy()


class LASSCConsumer:
    """Transactional bridge: rejected packets never advance accepted state.

    Geometry registrations and clock mappings are independently provisioned by
    the caller, never copied from untrusted frame metadata. Age limits are an
    explicit deployment/test configuration, not asserted hardware safe values.
    Repeated identical fresh frames may be consumed at control rate with
    new_frame=False. Session/epoch switches reset recurrent state on acceptance.
    """
    def __init__(self, adapter, *, cfg: Config, max_depth_age_s: float,
                 max_other_age_s: float, runner=None):
        for age in (max_depth_age_s, max_other_age_s):
            if not math.isfinite(age) or age <= 0:
                raise ValueError("explicit finite positive age limits required")
        if (cfg.depth_height, cfg.depth_width, cfg.depth_range) != (160, 88, 5.0):
            raise ValueError("legacy depth profile must be (160,88), 5 metres")
        target_k = getattr(adapter, "target_k", load_depth_modules()[0].TARGET_K)
        if not np.allclose(target_k, load_depth_modules()[0].TARGET_K, rtol=0, atol=1e-6):
            raise ValueError("57344 consumer requires the centered Isaac Camera target K")
        self.adapter = adapter
        self.cfg = cfg
        self.max_depth_age_s = max_depth_age_s
        self.max_other_age_s = max_other_age_s
        self.runner = runner
        self.accepted = 0
        self.stream = None
        self.retired_streams = set()
        self.last_identity = None
        self.last_digest = None
        self.epoch_watermark = {}
        # Official task reset establishes serial counters and all bookkeeping.
        # The packing-only template has no terrain or plants; it never steps or
        # supplies synthetic sensor values. Restore original normalization cfg.
        packing_cfg = replace(cfg, plants=0, terrain_enabled=False,
                              rock_density=0, validate_scene=False)
        self.task_template = _MeasuredObservationTask(packing_cfg, mode="goto")
        self.task_template.prepare(seed=0, options={"random_start": False})
        self.task_template.cfg = cfg

    def _observation(self, depth, pose, context):
        c = context
        if not isinstance(c, ObservationContext):
            raise ValueError("ObservationContext required")
        if (not math.isfinite(c.episode_time) or c.episode_time < 0
                or not np.isfinite(c.estimated_xy).all() or len(c.estimated_xy) != 2
                or not np.isfinite(c.goal_xy).all() or len(c.goal_xy) != 2
                or np.asarray(c.previous_action).shape != (4,)
                or not np.isfinite(c.previous_action).all()
                or np.max(np.abs(c.previous_action)) > 1):
            raise ValueError("invalid observation context")
        if (not isinstance(c.feedback, FeedbackSnapshot) or not c.feedback.pose_valid
                or abs(c.feedback.sample_time - c.episode_time) > 1e-6):
            raise ValueError("current measured feedback/pose is unavailable or mistimed")
        if depth.age > c.episode_time + 1e-9:
            raise ValueError("source precedes episode; supply consistent elapsed time")
        task = copy.copy(self.task_template)
        task._serial = self.task_template._serial.copy()
        task.time = c.episode_time
        task.goal = np.asarray(c.goal_xy, dtype=np.float64)
        task.last_action = np.asarray(c.previous_action, dtype=np.float32)
        task._feedback_snapshot = c.feedback
        task.measured_pose = np.array([*c.estimated_xy,
            c.feedback.estimated_heading_rad, *c.feedback.estimated_gimbal_q_rad], dtype=np.float32)
        frames = {"lidar": dict(c.lidar), "camera": dict(c.camera), "depth": {
            "values": depth.target_z_m.copy(), "valid": depth.target_valid.copy(),
            "capture": c.episode_time - depth.age, "estimated_pose": np.asarray(pose.pose),
            "intrinsics": load_depth_modules()[0].TARGET_K.copy(),
            "seq": depth.metadata["seq"],
        }}
        new = {"depth": depth.new_frame}
        for name in ("lidar", "camera"):
            f = frames[name]
            if "capture" not in f or "estimated_pose" not in f or "new_frame" not in f:
                raise ValueError(name + " requires capture, estimated_pose, new_frame")
            age = c.episode_time - float(f["capture"])
            if (not math.isfinite(age) or age < 0 or age > self.max_other_age_s
                    or np.asarray(f["estimated_pose"]).shape != (5,)
                    or not np.isfinite(f["estimated_pose"]).all()
                    or not isinstance(f["new_frame"], bool)):
                raise ValueError(name + " is stale, future or malformed")
            new[name] = f["new_frame"]
        # Reuse capture validation, including the explicit depth valid mask BEFORE
        # FieldTask's historical clipping. No duplicate sensor normalizer exists.
        task.attach_sensors(lambda kind: frames[kind])
        task.frames = {name: task._capture(name) for name in ("lidar", "depth", "camera")}
        task._new = new
        observation = task._observation()
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise ValueError("non-finite observation")
        return observation

    def consume(self, metadata: dict, payload: bytes, *, context: ObservationContext,
                source_pose: SourcePose | None, consume_mono_ns: int,
                consume_clock_id: str, producer_to_consumer=None,
                source_mask=None) -> ConsumerResult:
        module, _ = load_depth_modules()
        try:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            session, epoch = metadata.get("session_id"), metadata.get("pipeline_epoch")
            if (not isinstance(session, str) or not session
                    or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
                raise ValueError("invalid session or pipeline epoch")
            stream = (session, epoch)
            if (stream in self.retired_streams
                    or epoch < self.epoch_watermark.get(session, epoch)):
                return ConsumerResult(False, ("retired_session_or_epoch",))
            identity = (*stream, metadata.get("seq"))
            digest = hashlib.sha256(payload + json.dumps(metadata, sort_keys=True,
                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
            if identity == self.last_identity and digest != self.last_digest:
                raise ValueError("conflicting duplicate depth record")
            evidence = None
            if source_pose is not None:
                if not isinstance(source_pose, SourcePose):
                    raise ValueError("SourcePose history record required")
                if (source_pose.session_id != stream[0]
                        or source_pose.pipeline_epoch != stream[1]):
                    raise ValueError("source pose session/epoch mismatch")
                evidence = module.SourceStateEvidence(source_pose.source_mono_ns,
                    source_pose.clock_id, source_pose.pipeline_epoch, True,
                    source_pose.measured_gimbal, True)
            trial = copy.deepcopy(self.adapter)
            depth = trial.decode_frame(metadata, payload, consume_mono_ns=consume_mono_ns,
                consume_clock_id=consume_clock_id, consume_pipeline_epoch=stream[1],
                producer_to_consumer=producer_to_consumer, source_state=evidence,
                source_mask=source_mask)
            if not depth.legacy_ready:
                return ConsumerResult(False, depth.readiness_missing)
            # Conservative upper bound includes cross-host clock uncertainty.
            if depth.age + (depth.age_error_bound_ns or 0) / 1e9 > self.max_depth_age_s:
                return ConsumerResult(False, ("depth_stale",), age_s=depth.age)
            observation = self._observation(depth, source_pose, context)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return ConsumerResult(False, (str(exc),))
        reset = self.stream is not None and self.stream != stream
        action = None
        if self.runner is not None:
            state = copy.deepcopy(self.runner.state)
            starts = self.runner.episode_start.copy()
            try:
                if reset:
                    self.runner.reset()
                action = np.asarray(self.runner.action(observation), dtype=np.float32).reshape(-1)
                if action.shape != (4,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
                    raise ValueError("policy action must be four finite values in [-1,1]")
                if self.runner.state is None or not all(np.isfinite(s).all() for s in self.runner.state):
                    raise ValueError("policy recurrent state is missing or non-finite")
            except Exception:
                self.runner.state, self.runner.episode_start = state, starts
                raise
        if reset:
            self.retired_streams.add(self.stream)
        self.adapter = trial
        self.stream, self.last_identity, self.last_digest = stream, identity, digest
        self.epoch_watermark[session] = epoch
        self.accepted += 1
        return ConsumerResult(True, (), depth.new_frame, reset, depth.age, observation, action)


def action_commands(action, *, session_id: int, sequence: int, ttl_10ms: int,
                    cfg: Config, drive_radius_m: float = .03,
                    reduction: float = 4.0) -> dict:
    """Pure command conversion using physical sprocket radius, NOT proxy radius.

    Returns candidate requests, not applied/measured RPM, and never transmits.
    Gimbal output remains typed rad/s; no undocumented serial codec is invented.
    """
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1:
        raise ValueError("action must be four finite normalized values")
    if (not math.isfinite(drive_radius_m) or drive_radius_m <= 0
            or not math.isfinite(reduction) or reduction <= 0):
        raise ValueError("positive drive geometry required")
    rpm = np.rint(a[:2] * cfg.max_speed / (2 * math.pi * drive_radius_m)
                  * 60 * reduction).astype(np.int64)
    packets = [encode_speed_command(side, session_id=session_id, sequence=sequence,
        target_motor_rpm=int(value), ttl_10ms=ttl_10ms)
        for side, value in zip(("left", "right"), rpm)]
    return {"motor_request_rpm": rpm.tolist(), "drive_packets": packets,
            "gimbal_request_rad_s": [float(a[2]*cfg.gimbal_yaw_rate),
                                     float(a[3]*cfg.gimbal_pitch_rate)],
            "transmitted": False}
