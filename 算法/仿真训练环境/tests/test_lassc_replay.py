"""Small offline tests for the opt-in real-checkpoint LASSC CPU replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.task import Config
from isaac_field.telemetry import make_hardware_feedback_estimator
from scripts.lassc_consumer import PoseHistory
from scripts.lassc_replay import (
    SYNTHETIC_CLOCK,
    SYNTHETIC_SESSION,
    _action_codec_check,
    _policy_env,
    build_depth_fixture,
    build_telemetry_snapshot,
    make_depth_metadata,
)


def test_depth_fixtures_use_rk_wire_codec_and_preserve_invalid_pixels():
    adapter_module = None
    for kind in ("constant", "checker", "invalid"):
        metadata, payload, raster, adapter_module, wire = build_depth_fixture(kind)
        decoded = wire.decode_depth_payload(payload, json.dumps(metadata))
        assert decoded.shape == (240, 131)
        assert metadata["encoding"] == "16UC1"
        assert metadata["convention"] == "camera_z"
        if kind == "constant":
            assert np.all(decoded == 2000)
        elif kind == "checker":
            assert set(np.unique(decoded)).issubset({800, 4200})
            assert len(np.unique(decoded)) == 2
        else:
            assert np.isnan(raster).all()
            assert np.count_nonzero(decoded) == 0
    assert np.array_equal(adapter_module.TARGET_K, np.array(
        [[98.3247958874, 0.0, 44.0],
         [0.0, 98.3247958874, 80.0],
         [0.0, 0.0, 1.0]], dtype=np.float64))


def test_c_can_atk_gimbal_feedback_and_exact_pose_history():
    estimator = make_hardware_feedback_estimator(seed=0)
    snapshot, telemetry = build_telemetry_snapshot(
        0.75, 20.75, estimator=estimator,
        gimbal_q=(0.12, 0.21), gimbal_qd=(0.01, -0.02),
    )
    assert snapshot.pose_valid
    assert telemetry["can_golden_sha256"]
    assert telemetry["can_vectors_fed"] == ["boot_epoch_left", "state_left", "quality_left"]
    assert set(telemetry["atk_groups_decoded"]) == {"accel", "gyro", "attitude", "mag_temperature"}
    assert telemetry["gimbal_units"]["gimbal_yaw"] == {"q_rad": "rad", "qd_rad_s": "rad/s"}
    assert telemetry["feedback_sample_type"] == "FeedbackSample"
    assert telemetry["snapshot_type"] == "FeedbackSnapshot"
    assert telemetry["valid_groups"]["motor_left"] is True
    assert np.isfinite(snapshot.feedback).all()

    history = PoseHistory(capacity=2)
    metadata = make_depth_metadata(seq=1, pipeline_epoch=7, source_mono_ns=20_750_000_000,
                                   session_id=SYNTHETIC_SESSION)
    first = history.append(session_id=SYNTHETIC_SESSION, pipeline_epoch=7,
                           clock_id=SYNTHETIC_CLOCK, source_mono_ns=20_750_000_000,
                           estimated_xy=(2.0, 3.0), feedback=snapshot)
    assert history.lookup(metadata) == first
    off_by_one_ns = dict(metadata, source_mono_ns=20_750_000_001)
    assert history.lookup(off_by_one_ns) is None

    history.append(session_id=SYNTHETIC_SESSION, pipeline_epoch=7,
                   clock_id=SYNTHETIC_CLOCK, source_mono_ns=20_800_000_000,
                   estimated_xy=(2.1, 3.0), feedback=snapshot)
    history.append(session_id=SYNTHETIC_SESSION, pipeline_epoch=7,
                   clock_id=SYNTHETIC_CLOCK, source_mono_ns=20_850_000_000,
                   estimated_xy=(2.2, 3.0), feedback=snapshot)
    assert history.lookup(metadata) is None  # bounded capacity evicted oldest exact sample


def test_action_to_motor_codec_and_typed_gimbal_units():
    result = _action_codec_check(Config())
    assert result["ok"]
    assert result["motor_request_rpm"] == [1200, -1500]
    assert result["left_matches_c_golden"]
    assert result["right_crc_valid"]
    assert result["right_standalone_matches_c_golden"]
    assert result["gimbal_request_rad_s"] == pytest.approx([0.45, -0.6])
    assert result["gimbal_wire_codec"] == "not defined; typed rad/s only"
    assert result["transmitted"] is False


def test_checkpoint_space_adapter_matches_saved_bounds_without_fake_steps():
    checkpoint = os.environ.get("LASSC_REPLAY_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LASSC_REPLAY_CHECKPOINT to inspect trusted checkpoint spaces")
    from stable_baselines3.common.save_util import load_from_zip_file

    checkpoint = Path(checkpoint).expanduser().resolve()
    data, _params, _pytorch_variables = load_from_zip_file(str(checkpoint), device="cpu")
    saved_observation_space = data["observation_space"]
    saved_action_space = data["action_space"]
    assert saved_observation_space is not None and saved_action_space is not None
    expected = {
        "lidar": (0.0, 1.0),
        "depth": (-1.0, 1.0),
        "detections": (-1.0, 1.0),
        "state": (-1.0, 1.0),
        "feedback": (-np.inf, np.inf),
    }
    assert set(saved_observation_space.spaces) == set(expected)
    for name, (low, high) in expected.items():
        space = saved_observation_space.spaces[name]
        assert space.dtype == np.float32
        assert np.all(space.low == low)
        assert np.all(space.high == high)
    assert saved_action_space.shape == (4,)
    assert saved_action_space.dtype == np.float32
    assert np.all(saved_action_space.low == -1.0)
    assert np.all(saved_action_space.high == 1.0)

    env = _policy_env(checkpoint)
    assert env.observation_space == saved_observation_space
    assert env.action_space == saved_action_space
    with pytest.raises(NotImplementedError):
        env.reset()
    with pytest.raises(NotImplementedError):
        env.step(np.zeros(4, dtype=np.float32))
    from isaac_field.policy import validate_policy_spaces
    validate_policy_spaces(env.observation_space, env.action_space)
    env.close()


def test_real_checkpoint_cpu_replay_subprocess_opt_in(tmp_path):
    """Set LASSC_REPLAY_CHECKPOINT and select this node for the larger CPU test."""
    checkpoint = os.environ.get("LASSC_REPLAY_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LASSC_REPLAY_CHECKPOINT to run real recurrent checkpoint QA")
    output = tmp_path / "lassc_replay.json"
    command = [sys.executable, str(ROOT / "scripts/lassc_replay.py"),
               "--checkpoint", checkpoint, "--output", str(output)]
    live_url = os.environ.get("LASSC_REPLAY_LIVE_URL")
    if live_url:
        command.extend(("--live-url", live_url))
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(command, cwd=ROOT, env=child_env,
                               capture_output=True, text=True, timeout=240)
    assert output.is_file(), completed.stdout + "\n" + completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, report
    assert report["all_required_checks_passed"] is True
    assert all(value is True for value in report["checks"].values())
    assert report["checkpoint"]["device"] == "cpu"
    assert report["checkpoint"]["sha256"]
    assert len(report["replay"]) >= 3
    assert report["live"]["status"] in ("not_tested", "tested")
