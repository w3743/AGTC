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
from isaac_field.action_contract import action_config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from isaac_field.policy_contract import OBSERVATION_SHAPES, validate_policy_spaces
from scripts.lassc_consumer import PoseHistory
from scripts.lassc_replay import (
    SYNTHETIC_CLOCK,
    SYNTHETIC_SESSION,
    _action_codec_check,
    _policy_env,
    _rsl_config_from_contract,
    _rsl_spaces_from_contract,
    build_depth_fixture,
    build_telemetry_snapshot,
    make_depth_metadata,
    run_replay,
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
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.suffix.lower() != ".zip":
        pytest.skip("saved-space pickle inspection is only meaningful for explicit SB3 .zip fixtures")
    from stable_baselines3.common.save_util import load_from_zip_file

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

    env = _policy_env(checkpoint, legacy_sb3=True)
    assert env.observation_space == saved_observation_space
    assert env.action_space == saved_action_space
    assert env.input_schema
    with pytest.raises(NotImplementedError):
        env.reset()
    with pytest.raises(NotImplementedError):
        env.step(np.zeros(4, dtype=np.float32))
    validate_policy_spaces(env.observation_space, env.action_space, input_schema=env.input_schema)
    env.close()


def test_rsl_checkpoint_contract_rebuilds_read_only_spaces():
    from test_absolute_goal_policy import _observation_config

    observation_config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    contract = {
        "backend": "rsl_rl",
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "observation_shapes": {key: list(shape) for key, shape in OBSERVATION_SHAPES.items()},
        "observation_config": observation_config,
        "action_config": action_config(),
    }
    env = _rsl_spaces_from_contract(contract)
    assert env.num_envs == 1
    assert env.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert env.observation_config == observation_config
    assert env.action_config == action_config()
    expected = {
        "lidar": (0.0, 1.0),
        "depth": (-1.0, 1.0),
        "detections": (-1.0, 1.0),
        "state": (-1.0, 1.0),
        "feedback": (-np.inf, np.inf),
    }
    assert set(env.observation_space.spaces) == set(expected)
    for name, (low, high) in expected.items():
        space = env.observation_space.spaces[name]
        assert tuple(space.shape) == OBSERVATION_SHAPES[name]
        assert space.dtype == np.float32
        assert np.all(space.low == low)
        assert np.all(space.high == high)
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    with pytest.raises(NotImplementedError):
        env.reset()
    with pytest.raises(NotImplementedError):
        env.step(np.zeros(4, dtype=np.float32))
    env.close()


def test_rsl_replay_requires_complete_task_config():
    with pytest.raises(ValueError, match="task_config"):
        _rsl_config_from_contract({
            "backend": "rsl_rl",
            "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
            "observation_config": {},
        })


def test_sb3_zip_requires_explicit_legacy_selection_before_loading(tmp_path):
    checkpoint = tmp_path / "legacy.zip"
    checkpoint.write_bytes(b"not loaded without opt-in")
    with pytest.raises(ValueError, match="--legacy-sb3"):
        _policy_env(checkpoint)
    with pytest.raises(ValueError, match="only an SB3 .zip"):
        _policy_env(tmp_path / "field.pt", legacy_sb3=True)


def test_native_rsl_checkpoint_runs_cpu_lassc_replay(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict
    from types import SimpleNamespace

    from isaac_field.assets import FrontGuideSpec
    from isaac_field.body_stability import BodyStabilityConfig
    from isaac_field.rsl_rl_policy import RslFieldRNNModel, build_field_policy_contract
    from test_absolute_goal_policy import _observation_config

    torch.set_num_threads(1)
    task_cfg = Config(size=20.0, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
    observation_config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    observation_config["absolute_goal_coordinates"]["field_size_m"] = task_cfg.size
    observations = TensorDict({
        key: torch.zeros((1, *tuple(shape)), dtype=torch.float32)
        for key, shape in OBSERVATION_SHAPES.items()
    }, batch_size=[1])
    groups = {name: list(observations.keys()) for name in ("actor", "critic")}
    actor = RslFieldRNNModel(
        observations, groups, "actor", 4,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": float(np.exp(-2)), "std_type": "log",
        },
    )
    critic = RslFieldRNNModel(
        observations, groups, "critic", 1, cnns=actor.cnns,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    env_contract = SimpleNamespace(
        num_envs=1,
        observation_config=observation_config,
        action_config=action_config(),
        field_cfg=task_cfg,
        front_guide_spec=FrontGuideSpec(),
        body_stability_config=BodyStabilityConfig(),
    )
    contract = build_field_policy_contract(
        actor, critic, obs_groups=groups, env=env_contract,
        num_steps_per_env=128,
    )
    checkpoint = tmp_path / "native_field.pt"
    torch.save({
        "field_policy_contract": contract,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "iter": 0,
    }, checkpoint)

    report, code = run_replay(checkpoint, output=tmp_path / "replay.json")
    assert code == 0, report
    assert report["all_required_checks_passed"] is True
    assert report["checkpoint"]["backend"] == "rsl_rl"
    assert report["checkpoint"]["device"] == "cpu"
    assert report["checkpoint"]["sha256"]
    assert len(report["replay"]) >= 3
    assert report["checks"]["epoch_change_resets_recurrent_state"] is True
    first = report["replay"][0]
    assert abs(first["estimated_heading_rad"]) > 0.1
    assert not np.allclose(first["estimated_xy"], first["goal_xy"])
    np.testing.assert_allclose(first["goal_state_features"], [-0.7, -0.8], atol=1e-6)
    np.testing.assert_allclose(first["goal_state_features"],
                               first["expected_goal_features"], atol=1e-6)
    assert report["checks"]["absolute_goal_state_uses_world_goal_coordinates"] is True


def test_real_checkpoint_cpu_replay_subprocess_opt_in(tmp_path):
    """Set LASSC_REPLAY_CHECKPOINT and select this node for the larger CPU test."""
    checkpoint = os.environ.get("LASSC_REPLAY_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LASSC_REPLAY_CHECKPOINT to run real recurrent checkpoint QA")
    output = tmp_path / "lassc_replay.json"
    command = [sys.executable, str(ROOT / "scripts/lassc_replay.py"),
               "--checkpoint", checkpoint, "--output", str(output)]
    if Path(checkpoint).suffix.lower() == ".zip":
        command.append("--legacy-sb3")
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
    assert report["checkpoint"]["backend"] == (
        "sb3_legacy" if Path(checkpoint).suffix.lower() == ".zip" else "rsl_rl"
    )
    assert len(report["replay"]) >= 3
    assert report["live"]["status"] in ("not_tested", "tested")
