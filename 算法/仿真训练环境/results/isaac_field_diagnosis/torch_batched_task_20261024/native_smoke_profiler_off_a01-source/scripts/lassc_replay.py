"""CPU-only replay QA for the LASSC policy consumer.

This harness uses the RK depth wire/adapter, production consumer, task sensor
packer, telemetry assembler and the real recurrent checkpoint. It never opens a
camera/CAN/serial device, starts a simulator, or transmits a command.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from collections.abc import Mapping
from typing import Any
from urllib.error import URLError

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.feedback import (
    ABSOLUTE_GOAL_INPUT_SCHEMA,
    INPUT_SCHEMA,
    FeedbackSample,
    FeedbackSnapshot,
)
from isaac_field.policy_contract import ACTION_SHAPE, OBSERVATION_SHAPES, validate_policy_spaces
from isaac_field.task import Config
from isaac_field.telemetry import (
    ATKIMU901StreamDecoder,
    TelemetryAssembler,
    drive_crc16_ccitt_false,
    make_hardware_feedback_estimator,
)
from scripts.lassc_consumer import (
    LASSCConsumer,
    ObservationContext,
    PoseHistory,
    SourcePose,
    action_commands,
    depth_source_hashes,
    fetch_depth,
    load_depth_modules,
)


CAN_GOLDEN_PATH = ROOT / "tests/fixtures/foc_drive_feedback_v1_golden_vectors.json"
CAN_GOLDEN_PROVENANCE_PATH = ROOT / "tests/fixtures/foc_drive_feedback_v1_golden_vectors.provenance.json"
DEPTH_CONTRACT_PATH = ROOT.parent / "视觉算法/深度接口契约.md"
SYNTHETIC_SESSION = "lassc-replay-fixture-session"
SYNTHETIC_CLOCK = "lassc-replay-fixture-monotonic"
SYNTHETIC_SOURCE_K = np.array(
    [[147.487194, 0.0, 65.766881],
     [0.0, 147.487194, 121.086791],
     [0.0, 0.0, 1.0]], dtype=np.float64,
)
FIXTURE_CALIBRATION = "qa-only-synthetic-calibration-v1"
FIXTURE_EXTRINSICS = "qa-only-measured-pose-history-v1"
FIXTURE_PREPROCESSING = "qa-only-depth-z-fixture-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atk_packet(frame_id: int, payload: bytes) -> bytes:
    """Encode the documented ATK-IMU901 framing used by the checked-in tests."""
    packet = b"\x55\x55" + bytes((frame_id, len(payload))) + payload
    return packet + bytes((sum(packet) & 0xFF,))


def _i16s(values: tuple[int, ...]) -> bytes:
    return b"".join(struct.pack("<h", value) for value in values)


def _golden_vectors() -> tuple[dict[str, dict[str, Any]], str]:
    raw = CAN_GOLDEN_PATH.read_bytes()
    provenance = json.loads(CAN_GOLDEN_PROVENANCE_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest().upper()
    if digest != provenance["sha256"]:
        raise ValueError("checked-in C golden CAN fixture provenance hash mismatch")
    fixture = json.loads(raw)
    if fixture.get("protocol") != "foc-drive-feedback-v1":
        raise ValueError("unexpected CAN golden fixture protocol")
    return {item["name"]: item for item in fixture["vectors"]}, digest


def build_telemetry_snapshot(
    episode_time: float,
    host_monotonic: float,
    *,
    estimator,
    gimbal_q: tuple[float, float] = (0.10, 0.20),
    gimbal_qd: tuple[float, float] = (0.01, -0.02),
    attitude_yaw_rad: float = 0.0,
) -> tuple[FeedbackSnapshot, dict[str, Any]]:
    """Feed real protocol decoders with deterministic checked-in QA payloads."""
    vectors, golden_sha = _golden_vectors()
    assembler = TelemetryAssembler(sensor_timeout_s=0.5, motor_timeout_s=0.1)

    for name in ("boot_epoch_left", "state_left", "quality_left"):
        vector = vectors[name]
        result = assembler.feed_can_frame(
            int(vector["id"], 16), bytes.fromhex(vector["data_hex"]),
            received_monotonic=host_monotonic,
        )
        if name in ("state_left", "quality_left") and result is None:
            raise ValueError(f"golden CAN vector {name} was not decoded")

    # These are the exact ATK frame layouts covered by test_telemetry.py. A
    # parameterized yaw quaternion keeps the QA pose measured through the real
    # decoder/fusion path while allowing the absolute-goal semantic check to
    # distinguish world coordinates from heading-relative goal coordinates.
    if not math.isfinite(attitude_yaw_rad):
        raise ValueError("attitude_yaw_rad must be finite")
    half_yaw = attitude_yaw_rad / 2.0
    quaternion_wxyz = (
        int(round(math.cos(half_yaw) * 32767)), 0, 0,
        int(round(math.sin(half_yaw) * 32767)),
    )
    atk_bytes = b"".join((
        _atk_packet(0x02, _i16s(quaternion_wxyz)),
        _atk_packet(0x03, _i16s((0, 0, 8192, 0, 0, 0))),
        _atk_packet(0x04, _i16s((100, -50, 200, 2500))),
    ))
    decoder = ATKIMU901StreamDecoder()
    decoded_atk = decoder.feed(atk_bytes, received_monotonic=host_monotonic)
    if {frame["group"] for frame in decoded_atk} != {"attitude", "accel", "gyro", "mag_temperature"}:
        raise ValueError("ATK fixture did not decode into the expected telemetry groups")
    assembler.feed_imu_bytes(atk_bytes, received_monotonic=host_monotonic)
    gimbal = (
        assembler.update_gimbal_axis("yaw", gimbal_q[0], gimbal_qd[0], host_monotonic),
        assembler.update_gimbal_axis("pitch", gimbal_q[1], gimbal_qd[1], host_monotonic),
    )

    sample = assembler.assemble(episode_time, now_monotonic=host_monotonic)
    if not isinstance(sample, FeedbackSample):
        raise TypeError("TelemetryAssembler did not return FeedbackSample")
    snapshot = estimator.update_sample(sample)
    if not isinstance(snapshot, FeedbackSnapshot):
        raise TypeError("hardware feedback estimator did not return FeedbackSnapshot")
    if not snapshot.pose_valid or not np.isfinite(snapshot.feedback).all():
        raise ValueError("measured telemetry fixture did not produce a finite valid pose")
    return snapshot, {
        "can_golden_sha256": golden_sha,
        "can_vectors_fed": ["boot_epoch_left", "state_left", "quality_left"],
        "atk_groups_decoded": sorted(frame["group"] for frame in decoded_atk),
        "gimbal_units": {axis["source"]: dict(axis["units"]) for axis in gimbal},
        "feedback_sample_type": type(sample).__name__,
        "snapshot_type": type(snapshot).__name__,
        "pose_valid": bool(snapshot.pose_valid),
        "valid_groups": {key: bool(value) for key, value in sample.valid.items()},
    }


def make_depth_metadata(*, seq: int, pipeline_epoch: int, source_mono_ns: int,
                        session_id: str = SYNTHETIC_SESSION) -> dict[str, Any]:
    """Metadata for QA-only frames; registrations are explicitly synthetic."""
    return {
        "schema": "depth-z-u16le-v1",
        "session_id": session_id,
        "pipeline_epoch": int(pipeline_epoch),
        "seq": int(seq),
        "width": 131,
        "height": 240,
        "encoding": "16UC1",
        "byte_order": "little",
        "unit": "mm",
        "convention": "camera_z",
        "frame_id": "left_rectified_optical",
        "K": SYNTHETIC_SOURCE_K.reshape(-1).tolist(),
        "calibration_id": FIXTURE_CALIBRATION,
        "extrinsics_id": FIXTURE_EXTRINSICS,
        "preprocessing_id": FIXTURE_PREPROCESSING,
        "min_mm": 300,
        "max_mm": 5000,
        "left_pts_ns": int(source_mono_ns),
        "right_pts_ns": int(source_mono_ns + 1_000_000),
        "publish_mono_ns": int(source_mono_ns + 5_000_000),
        "source_mono_ns": int(source_mono_ns),
        "clock_mapping_id": "qa-direct-monotonic-v1",
        "clock_quality": "verified",
        "monotonic_clock_id": SYNTHETIC_CLOCK,
        "source_time_reference": "left_pts",
    }


def _fixture_raster(kind: str) -> np.ndarray:
    if kind == "constant":
        return np.full((240, 131), 2.0, dtype=np.float32, order="C")
    if kind == "checker":
        yy, xx = np.indices((240, 131))
        return np.ascontiguousarray(np.where((xx + yy) % 2 == 0, 0.8, 4.2).astype(np.float32))
    if kind == "invalid":
        # The wire codec maps NaN/out-of-profile values to the invalid 0 mm code.
        return np.full((240, 131), np.nan, dtype=np.float32, order="C")
    raise ValueError(f"unknown synthetic depth fixture {kind!r}")


def build_depth_fixture(kind: str, *, seq: int = 1, pipeline_epoch: int = 0,
                        source_mono_ns: int = 100_750_000_000,
                        session_id: str = SYNTHETIC_SESSION):
    adapter_module, wire = load_depth_modules()
    raster = _fixture_raster(kind)
    payload, width, height = wire.encode_depth_payload(raster)
    if (width, height) != (131, 240):
        raise ValueError("depth wire codec returned an unexpected source profile")
    metadata = make_depth_metadata(seq=seq, pipeline_epoch=pipeline_epoch,
                                   source_mono_ns=source_mono_ns, session_id=session_id)
    record = wire.build_depth_record(payload, metadata)
    return record.metadata, record.payload, raster, adapter_module, wire


def make_qa_adapter(adapter_module):
    """Use explicit fixture-only geometry registrations; never trust frame IDs."""
    return adapter_module.DepthAdapter(
        expected_calibration_id=FIXTURE_CALIBRATION,
        expected_preprocessing_id=FIXTURE_PREPROCESSING,
        expected_extrinsics_id=FIXTURE_EXTRINSICS,
        expected_source_k=SYNTHETIC_SOURCE_K,
        expected_frame_id="left_rectified_optical",
    )


def _space_only_env(observation_space, action_space, *, observation_config=None,
                    action_config=None, input_schema=None):
    import gymnasium as gym

    class SpaceOnlyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, observation_space, action_space, observation_config,
                     action_config, input_schema):
            super().__init__()
            self.observation_space = observation_space
            self.action_space = action_space
            self.observation_config = copy.deepcopy(observation_config)
            self.action_config = copy.deepcopy(action_config)
            self.input_schema = input_schema
            self.num_envs = 1

        def reset(self, *, seed=None, options=None):
            raise NotImplementedError("checkpoint space adapter does not simulate environment reset")

        def step(self, action):
            raise NotImplementedError("checkpoint space adapter does not simulate environment steps")

    return SpaceOnlyEnv(observation_space, action_space, observation_config,
                        action_config, input_schema)


def _rsl_spaces_from_contract(contract):
    """Rebuild the RSL checkpoint's read-only Gym spaces from its field contract."""
    import gymnasium as gym

    if not isinstance(contract, dict) or contract.get("backend") != "rsl_rl":
        raise ValueError("checkpoint does not contain an RSL-RL field policy contract")
    input_schema = contract.get("input_schema")
    if input_schema != ABSOLUTE_GOAL_INPUT_SCHEMA:
        raise ValueError(
            "production RSL replay requires the absolute-goal input schema; "
            "use an explicitly supported checkpoint or select SB3 legacy mode"
        )
    observation_config = contract.get("observation_config")
    action_config = contract.get("action_config")
    declared_shapes = contract.get("observation_shapes")
    expected_shapes = {key: list(shape) for key, shape in OBSERVATION_SHAPES.items()}
    if declared_shapes != expected_shapes:
        raise ValueError("RSL checkpoint observation_shapes do not match the field policy contract")

    bounds = {
        "lidar": (0.0, 1.0),
        "depth": (-1.0, 1.0),
        "detections": (-1.0, 1.0),
        "state": (-1.0, 1.0),
        "feedback": (-np.inf, np.inf),
    }
    observation_space = gym.spaces.Dict({
        key: gym.spaces.Box(low=low, high=high, shape=OBSERVATION_SHAPES[key], dtype=np.float32)
        for key, (low, high) in bounds.items()
    })
    action_space = gym.spaces.Box(-1.0, 1.0, shape=ACTION_SHAPE, dtype=np.float32)
    validate_policy_spaces(
        observation_space, action_space,
        input_schema=input_schema,
        expected_observation_config=observation_config,
    )
    env = _space_only_env(
        observation_space, action_space,
        observation_config=observation_config,
        action_config=action_config,
        input_schema=input_schema,
    )
    # Keep the exact source contract available to the replay constructor. The
    # space-only adapter is still read-only and does not infer task defaults.
    env.field_policy_contract = copy.deepcopy(contract)
    env.task_config = copy.deepcopy(contract.get("task_config"))
    return env


def _rsl_config_from_contract(contract) -> Config:
    """Create the LASSC task packer from the checkpoint's complete task contract."""
    if not isinstance(contract, Mapping):
        raise ValueError("RSL replay requires a field_policy_contract mapping")
    task_config = contract.get("task_config")
    if not isinstance(task_config, Mapping):
        raise ValueError(
            "RSL replay requires field_policy_contract.task_config; refusing to infer observation normalization"
        )
    try:
        cfg = Config.from_checkpoint_config(task_config, require_complete=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RSL task_config is invalid for this Config version: {exc}") from exc
    schema = contract.get("input_schema")
    if cfg.input_schema != schema:
        raise ValueError(
            f"RSL task_config input_schema {cfg.input_schema!r} does not match checkpoint {schema!r}"
        )
    observation_config = contract.get("observation_config")
    if not isinstance(observation_config, Mapping):
        raise ValueError("RSL checkpoint observation_config is required to validate task normalization")
    expected_shapes = {
        "lidar": [1, cfg.lidar_rays],
        "depth": [1, cfg.depth_height, cfg.depth_width],
        "detections": [cfg.max_detections, 8],
        "state": [33],
        "feedback": [37],
    }
    declared_shapes = observation_config.get("observation_shapes")
    if not isinstance(declared_shapes, Mapping):
        raise ValueError("RSL observation_config must declare observation_shapes")
    mismatched_shapes = {
        key: (declared_shapes.get(key), expected)
        for key, expected in expected_shapes.items()
        if declared_shapes.get(key) != expected
    }
    if mismatched_shapes:
        raise ValueError(f"RSL task_config dimensions disagree with observation_config: {mismatched_shapes}")
    if schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        coordinates = observation_config.get("absolute_goal_coordinates")
        if (not isinstance(coordinates, Mapping)
                or coordinates.get("field_size_m") != cfg.size):
            raise ValueError(
                "RSL task_config size does not match absolute_goal_coordinates.field_size_m"
            )
    return cfg


def _policy_env(checkpoint: Path, *, legacy_sb3: bool = False):
    """Expose checkpoint-bound spaces without creating or stepping an environment.

    RSL is the production path. SB3 ZIP inspection remains available only when
    the caller explicitly opts into its legacy compatibility branch.
    """
    checkpoint = Path(checkpoint)
    suffix = checkpoint.suffix.lower()
    if suffix == ".zip":
        if not legacy_sb3:
            raise ValueError("SB3 .zip replay requires the explicit --legacy-sb3 option")
        from stable_baselines3.common.save_util import load_from_zip_file

        data, _params, _pytorch_variables = load_from_zip_file(str(checkpoint), device="cpu")
        observation_space = data.get("observation_space")
        action_space = data.get("action_space")
        if observation_space is None or action_space is None:
            raise ValueError("checkpoint does not serialize observation_space and action_space")
        validate_policy_spaces(observation_space, action_space, input_schema=INPUT_SCHEMA)
        return _space_only_env(
            observation_space, action_space, input_schema=INPUT_SCHEMA,
        )
    if legacy_sb3:
        raise ValueError("--legacy-sb3 accepts only an SB3 .zip checkpoint")
    if suffix != ".pt":
        raise ValueError(f"RSL replay requires a .pt checkpoint, got {suffix or '<no extension>'}")

    from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

    payload = load_rsl_checkpoint_payload(checkpoint, map_location="cpu")
    try:
        contract = payload["field_policy_contract"]
        return _rsl_spaces_from_contract(contract)
    finally:
        del payload


def _state_copy(runner):
    return None if runner.state is None else tuple(np.asarray(value).copy() for value in runner.state)


def _state_finite(runner) -> bool:
    return runner.state is not None and all(np.isfinite(value).all() for value in runner.state)


def _state_equal(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


def _policy_context(cfg: Config, episode_time: float, feedback: FeedbackSnapshot,
                    estimated_xy: tuple[float, float], previous_action) -> ObservationContext:
    pose = np.asarray([*estimated_xy, feedback.estimated_heading_rad,
                       *feedback.estimated_gimbal_q_rad], dtype=np.float32)
    return ObservationContext(
        episode_time=float(episode_time),
        estimated_xy=estimated_xy,
        goal_xy=(3.0, 2.0),
        previous_action=tuple(float(v) for v in previous_action),
        feedback=feedback,
        lidar={"ranges": np.full(cfg.lidar_rays, 4.0, dtype=np.float32),
               "capture": episode_time - 0.02,
               "estimated_pose": pose.copy(), "new_frame": True},
        camera={"detections": [], "capture": episode_time - 0.03,
                "estimated_pose": pose.copy(), "new_frame": True},
    )


def _decode_invalid_fixture(adapter_module, metadata, payload,
                            source_pose, consume_mono_ns: int):
    adapter = make_qa_adapter(adapter_module)
    depth = adapter.decode_frame(
        metadata, payload,
        consume_mono_ns=consume_mono_ns,
        consume_clock_id=SYNTHETIC_CLOCK,
        consume_pipeline_epoch=metadata["pipeline_epoch"],
        source_state=adapter_module.SourceStateEvidence(
            source_pose.source_mono_ns, source_pose.clock_id,
            source_pose.pipeline_epoch, True, True, True,
        ),
    )
    return depth


def _action_codec_check(cfg: Config) -> dict[str, Any]:
    vectors, _ = _golden_vectors()
    rpm_per_action = cfg.max_speed / (2.0 * math.pi * 0.03) * 60.0 * 4.0
    action = np.asarray([1200.0 / rpm_per_action, -1500.0 / rpm_per_action, 0.25, -0.5],
                        dtype=np.float64)
    command = action_commands(action, session_id=0x1234, sequence=0x7E,
                              ttl_10ms=10, cfg=cfg, drive_radius_m=0.03, reduction=4.0)
    by_name = vectors
    left = command["drive_packets"][0]
    right = command["drive_packets"][1]
    # The left output is byte-for-byte the checked-in C golden speed vector.
    left_match = (left["can_id"] == int(by_name["speed_left_forward"]["id"], 16)
                  and left["data"] == bytes.fromhex(by_name["speed_left_forward"]["data_hex"]))
    right_bytes = right["data"]
    parsed_right = {
        "session_id": int.from_bytes(right_bytes[0:2], "little"),
        "sequence": right_bytes[2],
        "target_motor_rpm": int.from_bytes(right_bytes[3:5], "little", signed=True),
        "ttl_10ms": right_bytes[5],
    }
    right_crc = drive_crc16_ccitt_false(right["can_id"], right["dlc"], right_bytes[:6])
    standalone_right = __import__("isaac_field.telemetry", fromlist=["encode_speed_command"]).encode_speed_command(
        "right", session_id=0xBEEF, sequence=0, target_motor_rpm=-1500, ttl_10ms=5)
    right_golden_match = (standalone_right["can_id"] == int(by_name["speed_right_reverse"]["id"], 16)
                          and standalone_right["data"]
                          == bytes.fromhex(by_name["speed_right_reverse"]["data_hex"]))
    ok = bool(
        command["motor_request_rpm"] == [1200, -1500]
        and left_match
        and parsed_right == {"session_id": 0x1234, "sequence": 0x7E,
                             "target_motor_rpm": -1500, "ttl_10ms": 10}
        and right_crc == int.from_bytes(right_bytes[6:8], "little")
        and right_golden_match
        and np.allclose(command["gimbal_request_rad_s"], [0.45, -0.6])
        and command["transmitted"] is False
    )
    return {
        "ok": ok,
        "input_action": action.tolist(),
        "mechanical_sprocket_radius_m": 0.03,
        "reduction": 4.0,
        "motor_request_rpm": command["motor_request_rpm"],
        "drive_packets": [{**{k: v for k, v in packet.items() if k != "data"}}
                           for packet in command["drive_packets"]],
        "left_matches_c_golden": bool(left_match),
        "right_crc_valid": bool(right_crc == int.from_bytes(right_bytes[6:8], "little")),
        "right_standalone_matches_c_golden": bool(right_golden_match),
        "gimbal_request_rad_s": command["gimbal_request_rad_s"],
        "gimbal_wire_codec": "not defined; typed rad/s only",
        "transmitted": command["transmitted"],
    }


def run_replay(checkpoint: str | Path, *, output: str | Path,
               live_url: str | None = None, live_timeout_s: float = 2.0,
               legacy_sb3: bool = False) -> tuple[dict[str, Any], int]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint.suffix.lower() == ".zip" and not legacy_sb3:
        raise ValueError("SB3 .zip replay requires the explicit --legacy-sb3 option")
    if legacy_sb3 and checkpoint.suffix.lower() != ".zip":
        raise ValueError("--legacy-sb3 accepts only an SB3 .zip checkpoint")

    adapter_module, wire = load_depth_modules()
    adapter = make_qa_adapter(adapter_module)
    # Reuse the adapter's registered centered target grid; do not substitute
    # the simulator-independent FieldTask reference K here.
    target_k = np.asarray(adapter.target_k, dtype=np.float64)
    if not np.allclose(target_k, adapter_module.TARGET_K, rtol=0.0, atol=1e-9):
        raise ValueError("depth adapter target K differs from its registered target camera K")

    env = _policy_env(checkpoint, legacy_sb3=legacy_sb3)
    if legacy_sb3:
        cfg = Config()
        # Keep SB3 import isolated to the explicit compatibility path. The
        # legacy replay retains the v6 packer and its historical defaults.
        from isaac_field.policy import PolicyRunner

        runner = PolicyRunner(
            env, checkpoint, device="cpu", input_schema=INPUT_SCHEMA,
        )
        backend = "sb3_legacy"
    else:
        cfg = _rsl_config_from_contract(env.field_policy_contract)
        from isaac_field.rsl_rl_inference import RslPolicyRunner

        runner = RslPolicyRunner(
            env, checkpoint, device="cpu",
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            expected_observation_config=env.observation_config,
        )
        backend = "rsl_rl"
    if str(runner.model.device) != "cpu":
        raise ValueError(f"checkpoint runner did not remain on CPU: {runner.model.device}")
    consumer = LASSCConsumer(adapter, cfg=cfg, max_depth_age_s=0.5,
                             max_other_age_s=0.5, runner=runner)

    source_estimator = make_hardware_feedback_estimator(seed=0)
    current_estimator = make_hardware_feedback_estimator(seed=0)
    history = PoseHistory(capacity=8)
    previous_action = np.zeros(4, dtype=np.float32)
    replay = []
    first_observation = None
    first_action = None
    first_context = None
    last_metadata = None
    last_payload = None
    last_context = None
    last_source_pose = None
    all_finite = True
    observation_contract_checks = []
    recurrent_states = []
    absolute_goal_semantic_checks = []

    # Separate source/current telemetry samples are deliberate. At 250 ms depth
    # age, fresh current CAN/IMU feedback must not be fabricated from command output.
    for index, kind in enumerate(("constant", "checker", "constant"), start=0):
        source_episode = 0.75 + index * 0.05
        consume_episode = source_episode + 0.25
        source_ns = int(round((100.0 + source_episode) * 1e9))
        consume_ns = int(round((100.0 + consume_episode) * 1e9))
        source_xy = (2.0 + index * 0.02, 2.0)
        current_xy = (2.02 + index * 0.02, 2.0)

        source_feedback, source_telemetry = build_telemetry_snapshot(
            source_episode, 100.0 + source_episode, estimator=source_estimator,
            gimbal_q=(0.10 + index * 0.005, 0.20 - index * 0.004),
            attitude_yaw_rad=0.6,
        )
        metadata = make_depth_metadata(seq=index + 1, pipeline_epoch=0,
                                       source_mono_ns=source_ns)
        history.append(session_id=SYNTHETIC_SESSION, pipeline_epoch=0,
                       clock_id=SYNTHETIC_CLOCK, source_mono_ns=source_ns,
                       estimated_xy=source_xy, feedback=source_feedback)
        source_pose = history.lookup(metadata)
        if source_pose is None:
            raise ValueError("exact source-time PoseHistory lookup failed")

        current_feedback, current_telemetry = build_telemetry_snapshot(
            consume_episode, 100.0 + consume_episode, estimator=current_estimator,
            gimbal_q=(0.11 + index * 0.005, 0.19 - index * 0.004),
            attitude_yaw_rad=0.6,
        )
        context = _policy_context(cfg, consume_episode, current_feedback,
                                  current_xy, previous_action)
        meta, payload, _raster, _adapter_module, _wire = build_depth_fixture(
            kind, seq=index + 1, pipeline_epoch=0, source_mono_ns=source_ns)
        result = consumer.consume(
            meta, payload, context=context, source_pose=source_pose,
            consume_mono_ns=consume_ns, consume_clock_id=SYNTHETIC_CLOCK,
        )
        if not result.ready or result.action is None or result.observation is None:
            raise ValueError(f"offline {kind} frame {index + 1} rejected: {result.reasons}")
        action = np.asarray(result.action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
            raise ValueError("real checkpoint produced an invalid normalized action")
        policy_commands = action_commands(
            action, session_id=0x4242, sequence=index + 1, ttl_10ms=10,
            cfg=cfg, drive_radius_m=0.03, reduction=4.0,
        )
        for packet, side in zip(policy_commands["drive_packets"], ("left", "right")):
            if (packet["can_id"] != (0x501 if side == "left" else 0x502)
                    or drive_crc16_ccitt_false(packet["can_id"], packet["dlc"],
                                               packet["data"][:6])
                    != int.from_bytes(packet["data"][6:8], "little")):
                raise ValueError("policy action did not round-trip through the existing CAN encoder")
        if (not np.isfinite(policy_commands["gimbal_request_rad_s"]).all()
                or policy_commands["transmitted"] is not False):
            raise ValueError("policy action produced invalid typed gimbal request")
        if not _state_finite(runner):
            raise ValueError("real checkpoint recurrent state is missing or non-finite")
        observation_fields = {}
        for name, space in env.observation_space.spaces.items():
            value = result.observation.get(name)
            if value is None:
                observation_fields[name] = {"shape": False, "dtype": False,
                                            "finite": False, "contains": False}
                continue
            array = np.asarray(value)
            observation_fields[name] = {
                "shape": tuple(array.shape) == tuple(space.shape),
                "dtype": array.dtype == np.dtype(space.dtype),
                "finite": bool(np.isfinite(array).all()),
                "contains": bool(space.contains(array)),
            }
        observation_contract = {
            "keys_match_checkpoint_space": set(result.observation) == set(env.observation_space.spaces),
            "fields": observation_fields,
        }
        state_goal_features = np.asarray(result.observation["state"][1:3], dtype=np.float64)
        world_goal_normalized = np.clip(
            np.asarray(context.goal_xy, dtype=np.float64) / cfg.size * 2.0 - 1.0,
            -1.0, 1.0,
        )
        delta_world = np.asarray(context.goal_xy, dtype=np.float64) - np.asarray(
            context.estimated_xy, dtype=np.float64,
        )
        heading = float(context.feedback.estimated_heading_rad)
        relative_goal_normalized = np.asarray((
            math.cos(heading) * delta_world[0] + math.sin(heading) * delta_world[1],
            -math.sin(heading) * delta_world[0] + math.cos(heading) * delta_world[1],
        )) / cfg.size
        expected_goal_features = (
            world_goal_normalized
            if cfg.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
            else np.clip(relative_goal_normalized, -1.0, 1.0)
        )
        absolute_goal_semantic_checks.append(bool(
            cfg.input_schema != ABSOLUTE_GOAL_INPUT_SCHEMA
            or (np.allclose(state_goal_features, world_goal_normalized, rtol=0.0, atol=1e-6)
                and abs(heading) > 0.1
                and np.linalg.norm(delta_world) > 0.1
                and not np.allclose(world_goal_normalized, relative_goal_normalized,
                                    rtol=0.0, atol=1e-3))
        ))
        action_space_contains = bool(env.action_space.contains(action))
        all_finite &= all(np.isfinite(value).all() for value in result.observation.values())
        observation_contract_checks.append(observation_contract)
        recurrent_states.append(_state_copy(runner))
        previous_action = action.copy()
        replay.append({
            "seq": index + 1,
            "fixture": kind,
            "age_s": result.age_s,
            "new_frame": result.new_frame,
            "action": action.tolist(),
            "action_space_contains": action_space_contains,
            "observation_contract": observation_contract,
            "goal_state_features": state_goal_features.tolist(),
            "expected_goal_features": expected_goal_features.tolist(),
            "goal_coordinate_semantics": (
                "field_world_xy_normalized" if cfg.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
                else "heading_relative_body_xy_normalized"
            ),
            "estimated_heading_rad": heading,
            "estimated_xy": list(context.estimated_xy),
            "goal_xy": list(context.goal_xy),
            "action_commands": {
                "motor_request_rpm": policy_commands["motor_request_rpm"],
                "drive_packets": [{k: v for k, v in packet.items() if k != "data"}
                                  for packet in policy_commands["drive_packets"]],
                "gimbal_request_rad_s": policy_commands["gimbal_request_rad_s"],
                "transmitted": policy_commands["transmitted"],
            },
            "source_feedback": source_telemetry,
            "current_feedback": current_telemetry,
        })
        if index == 0:
            first_observation = copy.deepcopy(result.observation)
            first_action = action.copy()
            first_context = context
        last_metadata, last_payload = meta, payload
        last_context, last_source_pose = context, source_pose

    states_changed = any(not _state_equal(recurrent_states[i - 1], recurrent_states[i])
                         for i in range(1, len(recurrent_states)))
    # Replaying the same observation after PolicyRunner.reset must reproduce the
    # original deterministic first action using the real checkpoint.
    runner.reset()
    reset_action = np.asarray(runner.action(first_observation), dtype=np.float32).reshape(-1)
    reset_reproduces = bool(np.array_equal(reset_action, first_action))

    # Missing metadata and malformed bytes must be rejected transactionally:
    # accepted counters and the real LSTM state both stay unchanged.
    before_reject_state = _state_copy(runner)
    before_reject_starts = runner.episode_start.copy()
    before_reject_count = consumer.accepted
    missing_meta = dict(last_metadata)
    missing_meta["seq"] = int(last_metadata["seq"] + 1)
    missing_meta.pop("K")
    missing_result = consumer.consume(
        missing_meta, last_payload, context=last_context, source_pose=last_source_pose,
        consume_mono_ns=int(last_metadata["source_mono_ns"] + 250_000_000),
        consume_clock_id=SYNTHETIC_CLOCK,
    )
    missing_rejected_transactionally = bool(
        not missing_result.ready and consumer.accepted == before_reject_count
        and _state_equal(_state_copy(runner), before_reject_state)
        and np.array_equal(runner.episode_start, before_reject_starts)
    )
    bad_payload_meta = dict(last_metadata, seq=99)
    invalid_payload = last_payload
    before_bad_payload_state = _state_copy(runner)
    before_bad_payload_count = consumer.accepted
    bad_payload_result = consumer.consume(
        bad_payload_meta, invalid_payload[:-2], context=last_context,
        source_pose=last_source_pose,
        consume_mono_ns=int(last_metadata["source_mono_ns"] + 250_000_000),
        consume_clock_id=SYNTHETIC_CLOCK,
    )
    truncated_rejected_transactionally = bool(
        not bad_payload_result.ready and consumer.accepted == before_bad_payload_count
        and _state_equal(_state_copy(runner), before_bad_payload_state)
    )

    # A fully invalid sensor raster is a parseable sentinel frame. The shared
    # adapter must preserve an all-false validity mask; it is never fed to policy.
    invalid_source_episode = 0.90
    invalid_source_ns = int(round((100.0 + invalid_source_episode) * 1e9))
    invalid_feedback, _ = build_telemetry_snapshot(
        invalid_source_episode, 100.0 + invalid_source_episode,
        estimator=make_hardware_feedback_estimator(seed=1),
    )
    invalid_pose = SourcePose(SYNTHETIC_SESSION, 0, SYNTHETIC_CLOCK, invalid_source_ns,
        (2.1, 2.0, invalid_feedback.estimated_heading_rad,
         *invalid_feedback.estimated_gimbal_q_rad), True)
    invalid_meta, invalid_wire_payload, _, _, wire = build_depth_fixture(
        "invalid", seq=4, pipeline_epoch=0, source_mono_ns=invalid_source_ns)
    invalid_consume_ns = invalid_source_ns + 250_000_000
    invalid_adapted = _decode_invalid_fixture(
        adapter_module, invalid_meta, invalid_wire_payload,
        invalid_pose, invalid_consume_ns,
    )
    invalid_mask_preserved = bool(
        invalid_adapted.payload_parseable and invalid_adapted.legacy_ready
        and not np.asarray(invalid_adapted.target_valid).any()
        and np.isfinite(invalid_adapted.target_z_m).all()
    )

    # Advance the stream epoch with an exact-time new pose. Consumer must reset
    # the recurrent state before predicting, yielding the same action as frame 1.
    epoch_source_episode = 0.75
    epoch_source_ns = int(round((100.0 + epoch_source_episode) * 1e9))
    epoch_source_estimator = make_hardware_feedback_estimator(seed=2)
    epoch_source_feedback, _ = build_telemetry_snapshot(
        epoch_source_episode, 100.0 + epoch_source_episode,
        estimator=epoch_source_estimator,
        attitude_yaw_rad=0.6,
    )
    epoch_meta = make_depth_metadata(seq=1, pipeline_epoch=1,
                                     source_mono_ns=epoch_source_ns)
    history.append(session_id=SYNTHETIC_SESSION, pipeline_epoch=1,
        clock_id=SYNTHETIC_CLOCK, source_mono_ns=epoch_source_ns,
        estimated_xy=(2.0, 2.0), feedback=epoch_source_feedback)
    epoch_pose = history.lookup(epoch_meta)
    if epoch_pose is None:
        raise ValueError("epoch source-time pose lookup failed")
    epoch_depth = build_depth_fixture("constant", seq=1, pipeline_epoch=1,
                                      source_mono_ns=epoch_source_ns)
    epoch_result = consumer.consume(
        epoch_depth[0], epoch_depth[1], context=first_context,
        source_pose=epoch_pose, consume_mono_ns=epoch_source_ns + 250_000_000,
        consume_clock_id=SYNTHETIC_CLOCK,
    )
    epoch_reset = bool(epoch_result.ready and epoch_result.reset_recurrent
                       and epoch_result.action is not None
                       and np.array_equal(np.asarray(epoch_result.action), first_action))

    codec = _action_codec_check(cfg)
    source_paths = [
        Path(__file__), ROOT / "scripts/lassc_consumer.py",
        ROOT / "isaac_field/task.py",
        ROOT / "isaac_field/feedback.py", ROOT / "isaac_field/telemetry.py",
        ROOT / "isaac_field/action_contract.py", CAN_GOLDEN_PATH,
        CAN_GOLDEN_PROVENANCE_PATH,
    ]
    if legacy_sb3:
        source_paths.append(ROOT / "isaac_field/policy.py")
    else:
        source_paths.extend((
            ROOT / "isaac_field/policy_contract.py",
            ROOT / "isaac_field/sensor_encoder.py",
            ROOT / "isaac_field/rsl_rl_policy.py",
            ROOT / "isaac_field/rsl_rl_inference.py",
        ))
    if DEPTH_CONTRACT_PATH.is_file():
        source_paths.append(DEPTH_CONTRACT_PATH)
    source_hashes = {str(path.resolve()): sha256_file(path) for path in source_paths if path.is_file()}
    source_hashes.update(depth_source_hashes())
    model_checkpoint_sha = getattr(runner.model, "checkpoint_sha256", None)
    if model_checkpoint_sha is not None and model_checkpoint_sha != checkpoint_sha:
        raise ValueError("runner checkpoint SHA-256 differs from the selected checkpoint file")

    checks: dict[str, bool] = {
        "checkpoint_loaded_on_cpu": str(runner.model.device) == "cpu",
        "three_consecutive_frames_ready": len(replay) >= 3 and all(item["new_frame"] for item in replay),
        "finite_observations_actions_and_lstm": bool(all_finite and all(
            np.isfinite(item["action"]).all() for item in replay)
            and all(state is not None and all(np.isfinite(value).all() for value in state)
                    for state in recurrent_states)),
        "observations_match_checkpoint_spaces": bool(all(
            frame["keys_match_checkpoint_space"]
            and all(all(fields.values()) for fields in frame["fields"].values())
            for frame in observation_contract_checks)),
        "actions_match_checkpoint_space": bool(all(
            item["action_space_contains"] for item in replay)),
        "lstm_state_changes_across_frames": bool(states_changed),
        "reset_reproduces_deterministic_action": bool(reset_reproduces),
        "missing_metadata_rejected_without_state_advance": missing_rejected_transactionally,
        "truncated_payload_rejected_without_state_advance": truncated_rejected_transactionally,
        "invalid_depth_pixels_remain_masked": invalid_mask_preserved,
        "epoch_change_resets_recurrent_state": epoch_reset,
        "telemetry_assembler_produces_current_measured_pose": bool(
            all(item["current_feedback"]["pose_valid"] for item in replay)
            and all(item["source_feedback"]["pose_valid"] for item in replay)),
        "current_motor_feedback_is_fresh_measured_can": bool(all(
            item["current_feedback"]["valid_groups"].get("motor_left", False)
            for item in replay)),
        "action_codec_rpm_crc_and_gimbal_units": bool(codec["ok"]),
        "policy_actions_replay_through_command_codec": bool(all(
            len(item["action_commands"]["drive_packets"]) == 2
            and item["action_commands"]["transmitted"] is False
            and np.isfinite(item["action_commands"]["gimbal_request_rad_s"]).all()
            for item in replay)),
        "centered_depth_target_intrinsics": bool(np.allclose(
            target_k, adapter_module.TARGET_K, rtol=0.0, atol=1e-9)),
    }
    if backend == "rsl_rl":
        checks["absolute_goal_state_uses_world_goal_coordinates"] = bool(
            absolute_goal_semantic_checks and all(absolute_goal_semantic_checks))

    live: dict[str, Any] = {"status": "not_tested", "reason": "live_url_not_requested"}
    if live_url:
        try:
            live_metadata, live_payload = fetch_depth(live_url, timeout_s=live_timeout_s)
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            live = {"status": "not_tested", "reason": type(exc).__name__, "detail": str(exc)}
        except Exception as exc:
            live = {"status": "failed", "reason": type(exc).__name__, "detail": str(exc)}
            checks["live_depth_endpoint_returns_valid_wire_record"] = False
        else:
            before_state = _state_copy(runner)
            before_starts = runner.episode_start.copy()
            before_count = consumer.accepted
            live_rejected = consumer.consume(
                # Preserve the producer's actual null/unbound fields. With no
                # exact-time measured pose history, a parseable live response
                # must stay unready and must not be converted into an action.
                live_metadata, live_payload,
                context=last_context, source_pose=None,
                consume_mono_ns=int(live_metadata.get("publish_mono_ns") or 0),
                consume_clock_id=str(live_metadata.get("monotonic_clock_id") or "live-unknown"),
            )
            rejected_without_advance = bool(
                not live_rejected.ready and consumer.accepted == before_count
                and _state_equal(_state_copy(runner), before_state)
                and np.array_equal(runner.episode_start, before_starts))
            readiness_fields = (
                "calibration_id", "extrinsics_id", "preprocessing_id", "source_mono_ns",
                "clock_mapping_id", "clock_quality", "monotonic_clock_id",
            )
            live = {
                "status": "tested" if rejected_without_advance else "failed",
                "url": live_url,
                "wire_record_valid": True,
                "seq": live_metadata.get("seq"),
                "pipeline_epoch": live_metadata.get("pipeline_epoch"),
                "null_or_unbound_readiness_fields": {
                    key: live_metadata.get(key) for key in readiness_fields
                    if live_metadata.get(key) is None
                },
                "payload_sha256": hashlib.sha256(live_payload).hexdigest(),
                "legacy_ready": live_rejected.ready,
                "rejected_without_lstm_or_episode_start_advance": rejected_without_advance,
                "rejection_reasons": list(live_rejected.reasons),
            }
            checks["live_record_unready_without_state_advance"] = rejected_without_advance

    report = {
        "schema": "lassc-cpu-replay-qa-v1",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha,
                       "device": str(runner.model.device),
                       "backend": backend,
                       "label": getattr(runner, "label", f"{backend} checkpoint loaded")},
        "source_sha256": dict(sorted(source_hashes.items())),
        "depth_target_intrinsics": {
            "source": "RK DepthAdapter.TARGET_K; IsaacCamera centered 88x160 renderer target",
            "matrix": target_k.tolist(),
            "synthetic_source_intrinsics": SYNTHETIC_SOURCE_K.tolist(),
            "registrations_are_qa_only": True,
        },
        "telemetry": {
            "source": "checked-in C CAN golden vectors + ATK packet fixture + typed measured gimbal axes",
            "path": "TelemetryAssembler -> FeedbackSample -> hardware estimator -> FeedbackSnapshot",
            "motor_current_semantics": "CAN decoded Iq; no policy action reused as feedback",
        },
        "replay": replay,
        "lstm_state_changed": bool(states_changed),
        "reset_reproduced_action": reset_action.tolist(),
        "epoch_change_action": None if epoch_result.action is None else np.asarray(epoch_result.action).tolist(),
        "action_codec": codec,
        "negative_tests": {
            "missing_metadata": {"rejected": not missing_result.ready,
                "reasons": list(missing_result.reasons),
                "lstm_unchanged": bool(missing_rejected_transactionally)},
            "truncated_payload": {"rejected": not bad_payload_result.ready,
                "reasons": list(bad_payload_result.reasons),
                "lstm_unchanged": bool(truncated_rejected_transactionally)},
        },
        "invalid_pixel_fixture": {
            "valid_source_pixels": int(np.count_nonzero(invalid_adapted.source_valid)),
            "valid_target_pixels": int(np.count_nonzero(invalid_adapted.target_valid)),
            "wire_zero_mm_pixels": int(np.count_nonzero(wire.decode_depth_payload(
                invalid_wire_payload, json.dumps(invalid_meta)) == 0)),
            "adapted_ready_but_not_policy_replayed": bool(invalid_adapted.legacy_ready),
        },
        "live": live,
        "checks": checks,
        "all_required_checks_passed": bool(all(value is True for value in checks.values())),
    }
    if sha256_file(checkpoint) != checkpoint_sha:
        report["all_required_checks_passed"] = False
        report["checks"]["checkpoint_unchanged_during_replay"] = False
    else:
        report["checks"]["checkpoint_unchanged_during_replay"] = True
    report["all_required_checks_passed"] = bool(all(value is True for value in report["checks"].values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    try:
        env.close()
    except Exception:
        pass
    return report, 0 if report["all_required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="native RSL-RL field checkpoint (.pt); SB3 .zip requires --legacy-sb3")
    parser.add_argument("--legacy-sb3", action="store_true",
                        help="explicitly replay an older SB3 RecurrentPPO .zip checkpoint")
    parser.add_argument("--output", required=True, help="write JSON QA report here")
    parser.add_argument("--live-url", help="optional atomic /depth.bin endpoint to probe")
    parser.add_argument("--live-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.live_timeout) or args.live_timeout <= 0:
        parser.error("--live-timeout must be finite and positive")
    try:
        report, code = run_replay(args.checkpoint, output=args.output,
                                  live_url=args.live_url, live_timeout_s=args.live_timeout,
                                  legacy_sb3=args.legacy_sb3)
    except Exception as exc:
        report = {
            "schema": "lassc-cpu-replay-qa-v1",
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "error": f"{type(exc).__name__}: {exc}",
            "checks": {"replay_completed": False},
            "all_required_checks_passed": False,
        }
        Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).expanduser().resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        code = 1
    print(json.dumps({"output": str(Path(args.output).expanduser().resolve()),
                      "all_required_checks_passed": report["all_required_checks_passed"],
                      "live_status": report.get("live", {}).get("status", "not_tested")},
                     ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
