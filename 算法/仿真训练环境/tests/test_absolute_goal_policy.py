"""CPU-only checks for the separately versioned absolute-goal policy schema."""
from __future__ import annotations

import json
import zipfile

import gymnasium as gym
import numpy as np
import pytest

from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA
from isaac_field.policy import (
    ABSOLUTE_GOAL_ENCODER_SCHEMA,
    ENCODER_SCHEMA,
    PolicyRunner,
    _field_encoder_config,
    _validate_checkpoint_encoder_metadata,
    convert_legacy_checkpoint,
    make_model,
)
from policy_contract_helpers import observation_config as _observation_config


class _PolicyEnv(gym.Env):
    def __init__(self, input_schema: str):
        self.input_schema = input_schema
        self.observation_config = _observation_config(input_schema)
        self.observation_space = gym.spaces.Dict(
            {
                "lidar": gym.spaces.Box(0.0, 1.0, shape=(1, 400), dtype=np.float32),
                "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 160, 88), dtype=np.float32),
                "detections": gym.spaces.Box(-1.0, 1.0, shape=(16, 8), dtype=np.float32),
                "state": gym.spaces.Box(-1.0, 1.0, shape=(33,), dtype=np.float32),
                "feedback": gym.spaces.Box(-np.inf, np.inf, shape=(37,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _observation(self):
        return {
            "lidar": np.zeros((1, 400), dtype=np.float32),
            "depth": np.zeros((1, 160, 88), dtype=np.float32),
            "detections": np.zeros((16, 8), dtype=np.float32),
            "state": np.zeros((33,), dtype=np.float32),
            "feedback": np.zeros((37,), dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, False, False, {}


def _vector_env(input_schema: str):
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([lambda: _PolicyEnv(input_schema)])


def _save_model(input_schema: str, path):
    env = _vector_env(input_schema)
    model = make_model(
        env,
        input_schema=input_schema,
        observation_config=_observation_config(input_schema),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        lstm_hidden_size=8,
        device="cpu",
    )
    model.save(path)
    env.close()
    return path.with_suffix(".zip")


def _rewrite_checkpoint(source, destination, mutate):
    with zipfile.ZipFile(source, "r") as source_zip, zipfile.ZipFile(destination, "w") as target_zip:
        for member in source_zip.infolist():
            content = source_zip.read(member.filename)
            if member.filename == "data":
                metadata = json.loads(content)
                mutate(metadata)
                content = json.dumps(metadata).encode("utf-8")
            target_zip.writestr(member, content)
    return destination


def test_absolute_goal_encoder_and_observation_metadata_have_independent_schema():
    config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)

    assert len(config["state_fields"]) == 33
    assert config["state_fields"][1:3] == ["goal_world_x_normalized", "goal_world_y_normalized"]
    assert config["state_fields"][3:7] == [
        "estimated_x_normalized",
        "estimated_y_normalized",
        "heading_sin",
        "heading_cos",
    ]
    assert _field_encoder_config(ABSOLUTE_GOAL_INPUT_SCHEMA)["schema"] == ABSOLUTE_GOAL_ENCODER_SCHEMA
    assert _field_encoder_config(ABSOLUTE_GOAL_INPUT_SCHEMA)["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert _field_encoder_config(INPUT_SCHEMA)["schema"] == ENCODER_SCHEMA


def test_absolute_goal_fresh_model_checkpoint_round_trips_in_runner(tmp_path):
    source = _vector_env(ABSOLUTE_GOAL_INPUT_SCHEMA)
    target = _vector_env(ABSOLUTE_GOAL_INPUT_SCHEMA)
    config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    try:
        model = make_model(
            source,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            observation_config=config,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            lstm_hidden_size=8,
            device="cpu",
        )
        path = tmp_path / "absolute-fresh0.zip"
        model.save(path)

        runner = PolicyRunner(
            target,
            path,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            expected_observation_config=config,
        )
        assert runner.model.field_encoder_config == _field_encoder_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
        assert runner.model.field_observation_config == config
        assert runner.model.policy_kwargs["features_extractor_kwargs"]["encoder_schema"] == ABSOLUTE_GOAL_ENCODER_SCHEMA
        assert runner.action(target.reset()).shape == (1, 4)
    finally:
        source.close()
        target.close()


def test_absolute_goal_model_requires_its_explicit_state_contract():
    env = _vector_env(ABSOLUTE_GOAL_INPUT_SCHEMA)
    try:
        with pytest.raises(ValueError, match="explicit observation_config"):
            make_model(env, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA, n_steps=2, batch_size=2)

        wrong = _observation_config(INPUT_SCHEMA)
        with pytest.raises(ValueError, match="does not match input_schema"):
            make_model(
                env,
                input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                observation_config=wrong,
                n_steps=2,
                batch_size=2,
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "metadata_change, message",
    [
        ("legacy-v6", "encoder schema is missing or incompatible"),
        ("missing-observation-config", "field_observation_config is missing or incompatible"),
        ("wrong-state-fields", "state fields do not match"),
        ("missing-coordinate-contract", "must declare absolute_goal_coordinates"),
        ("wrong-coordinate-unit", "absolute_goal_coordinates contract is missing or incompatible"),
        ("wrong-coordinate-normalization", "absolute_goal_coordinates contract is missing or incompatible"),
        ("invalid-field-size", "field_size_m must be finite and at least 6 m"),
    ],
)
def test_absolute_runner_rejects_wrong_or_missing_schema_before_sb3_load(
    tmp_path, monkeypatch, metadata_change, message
):
    from sb3_contrib import RecurrentPPO

    source_schema = INPUT_SCHEMA if metadata_change == "legacy-v6" else ABSOLUTE_GOAL_INPUT_SCHEMA
    checkpoint = _save_model(source_schema, tmp_path / "source.zip")

    def mutate(metadata):
        if metadata_change == "missing-observation-config":
            metadata.pop("field_observation_config", None)
        elif metadata_change == "wrong-state-fields":
            metadata["field_observation_config"]["state_fields"][1] = "goal_local_x_over_field_size"
        elif metadata_change == "missing-coordinate-contract":
            metadata["field_observation_config"].pop("absolute_goal_coordinates", None)
        elif metadata_change == "wrong-coordinate-unit":
            metadata["field_observation_config"]["absolute_goal_coordinates"]["unit"] = "field_fraction"
        elif metadata_change == "wrong-coordinate-normalization":
            metadata["field_observation_config"]["absolute_goal_coordinates"]["normalization"] = "world_xy / field_size_m"
        elif metadata_change == "invalid-field-size":
            metadata["field_observation_config"]["absolute_goal_coordinates"]["field_size_m"] = 5.0

    if metadata_change != "legacy-v6":
        checkpoint = _rewrite_checkpoint(checkpoint, tmp_path / "mutated.zip", mutate)
    target = _vector_env(ABSOLUTE_GOAL_INPUT_SCHEMA)

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("SB3 load must not run before input-schema validation")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(unexpected_load))
    try:
        with pytest.raises(ValueError, match=message):
            PolicyRunner(
                target,
                checkpoint,
                input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                expected_observation_config=_observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA),
            )
    finally:
        target.close()


def test_legacy_runner_rejects_absolute_goal_checkpoint_before_sb3_load(tmp_path, monkeypatch):
    from sb3_contrib import RecurrentPPO

    checkpoint = _save_model(ABSOLUTE_GOAL_INPUT_SCHEMA, tmp_path / "absolute.zip")
    target = _vector_env(INPUT_SCHEMA)

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("legacy schema check must run before SB3 load")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(unexpected_load))
    try:
        with pytest.raises(ValueError, match="encoder schema is missing or incompatible"):
            PolicyRunner(target, checkpoint)
        with pytest.raises(ValueError, match="encoder schema is missing or incompatible"):
            _validate_checkpoint_encoder_metadata(checkpoint)
    finally:
        target.close()


def test_legacy_conversion_rejects_absolute_environment_before_sb3_load(tmp_path, monkeypatch):
    from sb3_contrib import RecurrentPPO

    checkpoint = _save_model(INPUT_SCHEMA, tmp_path / "legacy-v6.zip")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("legacy conversion must validate the environment before SB3 load")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(unexpected_load))
    with pytest.raises(ValueError, match="environment input_schema .* does not match requested"):
        convert_legacy_checkpoint(
            checkpoint,
            tmp_path / "converted.zip",
            env=_PolicyEnv(ABSOLUTE_GOAL_INPUT_SCHEMA),
        )


def test_legacy_conversion_rejects_marked_absolute_checkpoint_without_env_before_load(
    tmp_path, monkeypatch
):
    from sb3_contrib import RecurrentPPO

    checkpoint = _save_model(ABSOLUTE_GOAL_INPUT_SCHEMA, tmp_path / "absolute.zip")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("absolute-goal metadata must be rejected before SB3 load")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(unexpected_load))
    with pytest.raises(ValueError, match="absolute-goal checkpoints cannot be converted"):
        convert_legacy_checkpoint(checkpoint, tmp_path / "converted.zip")


def test_legacy_conversion_keeps_v6_checkpoints_on_its_existing_load_path(tmp_path, monkeypatch):
    from sb3_contrib import RecurrentPPO

    checkpoint = _save_model(INPUT_SCHEMA, tmp_path / "legacy-v6.zip")

    def reached_legacy_load(*_args, **_kwargs):
        raise RuntimeError("legacy v6 conversion reached SB3 load")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(reached_legacy_load))
    with pytest.raises(RuntimeError, match="legacy v6 conversion reached SB3 load"):
        convert_legacy_checkpoint(checkpoint, tmp_path / "converted.zip")
