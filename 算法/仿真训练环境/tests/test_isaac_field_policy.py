"""Focused tests for the migrated encoder and legacy-checkpoint mapping."""
from __future__ import annotations

import json
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaac_field.policy import (
    BRANCH_DIMS,
    ENCODER_SCHEMA,
    FEATURES_DIM,
    SensorEncoder,
    _legacy_policy_kwargs,
    capture_actor_latent_pi,
    make_exploration_variant,
    make_model,
    PolicyRunner,
    validate_policy_spaces,
)
from isaac_field.action_contract import ACTION_COMPONENTS, ACTION_LABELS, ACTION_SCHEMA, action_config


def _observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "lidar": gym.spaces.Box(0.0, 1.0, shape=(1, 400), dtype=np.float32),
            "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 160, 88), dtype=np.float32),
            "detections": gym.spaces.Box(-1.0, 1.0, shape=(16, 8), dtype=np.float32),
            "state": gym.spaces.Box(-1.0, 1.0, shape=(33,), dtype=np.float32),
            "feedback": gym.spaces.Box(-np.inf, np.inf, shape=(37,), dtype=np.float32),
        }
    )


def _set_candidate_boxes(observation) -> None:
    """Give a handful of rows positive box sizes; padding remains all zero."""
    observation["detections"][:, [0, 3, 7, 11, 15], 2:4] = 0.1


def test_encoder_passes_state_unchanged_without_extra_goal_scaling():
    space = _observation_space()
    encoder = SensorEncoder(space, cache_inference=False)
    observation = {
        key: torch.zeros((1, *sensor_space.shape), dtype=torch.float32)
        for key, sensor_space in space.spaces.items()
    }
    _set_candidate_boxes(observation)
    observation["state"][:, 1:3] = torch.tensor([[0.01, -0.02]])
    observation["state"][:, 12] = 0.003
    original_state = observation["state"].clone()
    received_state: list[torch.Tensor] = []
    handle = encoder.proprioception[0].register_forward_pre_hook(
        lambda _module, inputs: received_state.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            features = encoder(observation)
    finally:
        handle.remove()

    assert features.shape == (1, 192)
    assert torch.isfinite(features).all()
    assert received_state[0][0, 1].item() == pytest.approx(0.01)
    assert received_state[0][0, 2].item() == pytest.approx(-0.02)
    assert received_state[0][0, 12].item() == pytest.approx(0.003)
    assert torch.equal(received_state[0], original_state)
    assert torch.equal(observation["state"], original_state)


def test_encoder_has_requested_depth_and_circular_lidar_branches_and_gradients():
    space = _observation_space()
    encoder = SensorEncoder(space, cache_inference=False)
    assert encoder.encoder_schema == ENCODER_SCHEMA
    assert encoder.branch_dims == BRANCH_DIMS
    assert FEATURES_DIM == 192
    assert encoder.features_dim == 192
    assert encoder.feedback[0].in_features == 37
    assert encoder.feedback[0].out_features == 32
    assert isinstance(encoder.laser[0], torch.nn.Conv1d)
    assert encoder.laser[0].in_channels == 1
    assert encoder.laser[0].out_channels == 16
    assert encoder.laser[0].kernel_size == (5,)
    assert encoder.laser[0].stride == (2,)
    assert encoder.laser[0].padding_mode == "circular"
    assert isinstance(encoder.laser[1], torch.nn.ReLU)
    assert encoder.laser[2].in_channels == 16
    assert encoder.laser[2].out_channels == 32
    assert encoder.laser[4].output_size == 50
    assert encoder.laser[6].in_features == 1600
    assert encoder.laser[6].out_features == 32
    assert encoder.depth[4].output_size == (8, 6)
    assert encoder.depth[6].in_features == 768
    assert encoder.depth[6].out_features == 64

    # The last LiDAR ray is adjacent to the first ray. A first-convolution
    # response at output index zero verifies the circular padding at the seam.
    first_lidar_conv = encoder.laser[0]
    with torch.no_grad():
        first_lidar_conv.weight.zero_()
        first_lidar_conv.bias.zero_()
        first_lidar_conv.weight[0, 0, 1] = 1.0
    lidar = torch.zeros((1, 1, 400), dtype=torch.float32)
    lidar[0, 0, -1] = 1.0
    torch.testing.assert_close(first_lidar_conv(lidar)[0, 0, 0], torch.tensor(1.0))

    observation = {
        key: torch.rand((2, *sensor_space.shape), dtype=torch.float32)
        for key, sensor_space in space.spaces.items()
    }
    _set_candidate_boxes(observation)
    features = encoder(observation)
    assert features.shape == (2, FEATURES_DIM)
    features.square().sum().backward()
    for parameter in (encoder.laser[0].weight, encoder.laser[2].weight, encoder.depth[0].weight, encoder.feedback[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_encoder_rejects_unknown_schema():
    with pytest.raises(ValueError, match="unsupported encoder_schema"):
        SensorEncoder(_observation_space(), encoder_schema="old-flat-lidar-v0")


def test_policy_space_validation_rejects_sensor_dimension_changes():
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    validate_policy_spaces(_observation_space(), action_space)
    invalid = _observation_space()
    invalid.spaces["lidar"] = gym.spaces.Box(0.0, 1.0, shape=(2, 400), dtype=np.float32)

    with pytest.raises(ValueError, match=r"lidar=.*expected \(1, 400\)"):
        validate_policy_spaces(invalid, action_space)

    # The previous policy contract's explicit LiDAR mask channel and larger
    # state/box/feedback tensors must not be accepted as a v6 observation.
    legacy = gym.spaces.Dict({
        "lidar": gym.spaces.Box(0.0, 1.0, shape=(2, 400), dtype=np.float32),
        "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 160, 88), dtype=np.float32),
        "detections": gym.spaces.Box(-1.0, 1.0, shape=(16, 10), dtype=np.float32),
        "state": gym.spaces.Box(-1.0, 1.0, shape=(39,), dtype=np.float32),
        "feedback": gym.spaces.Box(-np.inf, np.inf, shape=(41,), dtype=np.float32),
    })
    with pytest.raises(ValueError):
        validate_policy_spaces(legacy, action_space)


def test_policy_space_validation_requires_finite_normalized_action_bounds():
    nonstandard = gym.spaces.Box(
        low=np.full(4, -1.0, dtype=np.float32),
        high=np.asarray([1.0, 1.0, 1.0, 0.9], dtype=np.float32),
    )
    with pytest.raises(ValueError, match=r"finite and match.*normalized range \[-1, 1\]"):
        validate_policy_spaces(_observation_space(), nonstandard)

    unbounded = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    with pytest.raises(ValueError, match=r"finite and match.*normalized range \[-1, 1\]"):
        validate_policy_spaces(_observation_space(), unbounded)


def test_direct_track_action_contract_is_stable_and_json_safe():
    assert ACTION_SCHEMA == "direct-track-speed-v1"
    assert ACTION_COMPONENTS == (
        "left_track_speed",
        "right_track_speed",
        "gimbal_yaw_rate",
        "gimbal_pitch_rate",
    )
    assert ACTION_LABELS == (
        "左履带速度指令",
        "右履带速度指令",
        "云台偏航角速度指令",
        "云台俯仰角速度指令",
    )
    assert action_config() == {
        "schema": ACTION_SCHEMA,
        "components": list(ACTION_COMPONENTS),
        "normalized_range": [-1.0, 1.0],
    }


def test_removed_goal_gain_and_old_input_contract_are_rejected():
    with pytest.raises(ValueError, match="extra goal-input scaling has been removed"):
        SensorEncoder(_observation_space(), goal_gain=100.0)
    legacy = _observation_space()
    legacy.spaces["depth"] = gym.spaces.Box(0.0, 1.0, shape=(2, 160, 88), dtype=np.float32)
    legacy.spaces["state"] = gym.spaces.Box(-1.0, 1.0, shape=(44,), dtype=np.float32)
    with pytest.raises(ValueError, match="older sensor contracts require retraining"):
        SensorEncoder(legacy)


def test_legacy_checkpoint_metadata_maps_extractor_and_preserves_goal_gain(tmp_path):
    from isaac_field.policy import SensorEncoder

    checkpoint = tmp_path / "legacy.zip"
    stored = {
        "policy_kwargs": {
            "features_extractor_class": "<class 'field_sim.policy.SensorEncoder'>",
            "features_extractor_kwargs": {"goal_gain": 100.0},
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "net_arch": {"pi": [64], "vf": [64]},
            "log_std_init": -2.0,
            "squash_output": False,
            "normalize_images": False,
        }
    }
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps(stored))

    mapped = _legacy_policy_kwargs(checkpoint)

    assert mapped["features_extractor_class"] is SensorEncoder
    assert mapped["features_extractor_kwargs"] == {"goal_gain": 100.0}
    assert mapped["lstm_hidden_size"] == 128
    assert mapped["log_std_init"] == -2.0


class _PolicyVariantEnv(gym.Env):
    def __init__(self):
        self.observation_space = _observation_space()
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


def test_runner_refuses_legacy_checkpoint_without_modifying_it(tmp_path):
    import hashlib
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class LegacyStateEncoder(BaseFeaturesExtractor):
        def __init__(self, observation_space):
            super().__init__(observation_space, features_dim=44)

        def forward(self, observation):
            return observation["state"]

    legacy_env = _PolicyVariantEnv()
    legacy_env.observation_space.spaces["depth"] = gym.spaces.Box(
        0.0, 1.0, shape=(2, 160, 88), dtype=np.float32
    )
    legacy_env.observation_space.spaces["state"] = gym.spaces.Box(
        -1.0, 1.0, shape=(44,), dtype=np.float32
    )
    env = DummyVecEnv([lambda: legacy_env])
    model = RecurrentPPO(
        "MultiInputLstmPolicy", env, n_steps=2, batch_size=2, device="cpu",
        policy_kwargs={"features_extractor_class": LegacyStateEncoder, "lstm_hidden_size": 8},
    )
    checkpoint = tmp_path / "legacy.zip"
    model.save(checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    current_env = DummyVecEnv([_PolicyVariantEnv])
    try:
        with pytest.raises(ValueError, match="requires a newly trained checkpoint"):
            PolicyRunner(current_env, checkpoint)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    finally:
        env.close()
        current_env.close()


def test_runner_rejects_same_shape_legacy_encoder_without_modifying_checkpoint(tmp_path):
    import hashlib
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class LegacySameShapeEncoder(BaseFeaturesExtractor):
        def __init__(self, observation_space):
            super().__init__(observation_space, features_dim=128)
            self.projection = torch.nn.Linear(33, 128)

        def forward(self, observation):
            return self.projection(observation["state"])

    env = DummyVecEnv([_PolicyVariantEnv])
    legacy = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        n_steps=2,
        batch_size=2,
        device="cpu",
        policy_kwargs={"features_extractor_class": LegacySameShapeEncoder, "lstm_hidden_size": 8},
    )
    checkpoint = tmp_path / "same-shape-legacy.zip"
    legacy.save(checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    current_env = DummyVecEnv([_PolicyVariantEnv])
    try:
        with pytest.raises(ValueError, match="requires a newly trained checkpoint"):
            PolicyRunner(current_env, checkpoint)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    finally:
        env.close()
        current_env.close()


def test_runner_translates_checkpoint_tensor_size_mismatch_to_retrain_error(tmp_path, monkeypatch):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([_PolicyVariantEnv])
    source = make_model(env, n_steps=2, batch_size=2, n_epochs=1, lstm_hidden_size=8, device="cpu")
    checkpoint = tmp_path / "current.zip"
    source.save(checkpoint)
    current_env = DummyVecEnv([_PolicyVariantEnv])

    def fail_load(_cls, *_args, **_kwargs):
        raise RuntimeError("Error(s) in loading state_dict: size mismatch for depth.0.weight")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(fail_load))
    try:
        with pytest.raises(ValueError, match="encoder weights are incompatible.*requires a newly trained checkpoint"):
            PolicyRunner(current_env, checkpoint)
    finally:
        env.close()
        current_env.close()


@pytest.mark.parametrize(
    "stored_action_config",
    [
        None,
        {"schema": "forward-differential-track-speed-v2", "components": ["forward", "differential", "gimbal_yaw_rate", "gimbal_pitch_rate"], "normalized_range": [-1.0, 1.0]},
    ],
    ids=["missing-contract", "old-same-shape-semantics"],
)
def test_runner_rejects_same_shape_checkpoint_without_current_action_contract_before_load(
    tmp_path, monkeypatch, stored_action_config
):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    source_env = DummyVecEnv([_PolicyVariantEnv])
    target_env = DummyVecEnv([_PolicyVariantEnv])
    try:
        source = make_model(source_env, n_steps=2, batch_size=2, n_epochs=1, lstm_hidden_size=8, device="cpu")
        source_path = tmp_path / "current.zip"
        source.save(source_path)
        legacy_path = tmp_path / "same-shape-old-action.zip"
        with zipfile.ZipFile(source_path, "r") as source_zip, zipfile.ZipFile(legacy_path, "w") as legacy_zip:
            for member in source_zip.infolist():
                data = source_zip.read(member.filename)
                if member.filename == "data":
                    metadata = json.loads(data)
                    if stored_action_config is None:
                        metadata.pop("field_action_config", None)
                    else:
                        metadata["field_action_config"] = stored_action_config
                    data = json.dumps(metadata).encode("utf-8")
                legacy_zip.writestr(member, data)

        def unexpected_load(_cls, *_args, **_kwargs):
            pytest.fail("SB3 load must not run before action-contract validation")

        monkeypatch.setattr(RecurrentPPO, "load", classmethod(unexpected_load))
        with pytest.raises(ValueError, match="action contract.*requires a newly trained checkpoint"):
            PolicyRunner(target_env, legacy_path)
    finally:
        source_env.close()
        target_env.close()


def test_new_checkpoint_action_contract_round_trips_through_policy_runner(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    source_env = DummyVecEnv([_PolicyVariantEnv])
    target_env = DummyVecEnv([_PolicyVariantEnv])
    try:
        source = make_model(source_env, n_steps=2, batch_size=2, n_epochs=1, lstm_hidden_size=8, device="cpu")
        assert source.field_action_config == action_config()
        checkpoint = tmp_path / "direct-tracks.zip"
        source.save(checkpoint)

        runner = PolicyRunner(target_env, checkpoint)
        assert runner.model.field_action_config == action_config()
        assert runner.action({key: value[None, ...] for key, value in _PolicyVariantEnv()._observation().items()}).shape == (1, 4)
    finally:
        source_env.close()
        target_env.close()


def test_exploration_variants_copy_weights_calibrate_sde_and_keep_fresh_adam():
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([_PolicyVariantEnv])
    source = make_model(
        env,
        seed=37,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.995,
        goal_gain=1.0,
        lstm_hidden_size=8,
        net_arch={"pi": [8], "vf": [8]},
        device="cpu",
    )
    assert source.policy_kwargs["features_extractor_kwargs"]["encoder_schema"] == ENCODER_SCHEMA
    assert source.field_encoder_config == {
        "schema": ENCODER_SCHEMA,
        "branch_dims": BRANCH_DIMS,
        "depth_pool": [8, 6],
        "radar_pool": 50,
        "lidar_channels": ["normalized_range_invalid_zero"],
        "detection_fields": ["cx", "cy", "width", "height", "class_id", "confidence", "range", "range_valid"],
        "detection_presence": "width>0_and_height>0",
        "features_dim": 192,
    }
    assert source.field_action_config == action_config()
    source.num_timesteps = 1234
    # Make the source optimizer non-empty without changing its weights.
    source.policy.log_std.grad = torch.zeros_like(source.policy.log_std)
    source.policy.optimizer.step()
    source.policy.optimizer.zero_grad(set_to_none=True)
    assert source.policy.optimizer.state

    observation = env.reset()
    with capture_actor_latent_pi(source.policy) as latent_outputs:
        source.predict(observation, state=None, episode_start=np.array([True]), deterministic=True)
    assert len(latent_outputs) == 1
    assert latent_outputs[0].shape == (1, 8)
    latent_norm_sq_mean = float(latent_outputs[0].square().sum(dim=-1).mean())
    assert np.isfinite(latent_norm_sq_mean) and latent_norm_sq_mean > 0

    gaussian = make_exploration_variant(source, env, use_sde=False)
    sde = make_exploration_variant(
        source,
        env,
        use_sde=True,
        latent_norm_sq_mean=latent_norm_sq_mean,
    )
    assert gaussian.num_timesteps == sde.num_timesteps == source.num_timesteps
    assert gaussian.seed == sde.seed == source.seed
    assert gaussian.use_sde is False
    assert sde.use_sde is True and sde.sde_sample_freq == 8
    assert sde.policy.action_dist.full_std is True
    assert sde.policy.action_dist.use_expln is False
    assert sde.policy.squash_output is False
    assert gaussian.policy_kwargs["features_extractor_kwargs"]["encoder_schema"] == ENCODER_SCHEMA
    assert sde.policy_kwargs["features_extractor_kwargs"]["encoder_schema"] == ENCODER_SCHEMA
    assert gaussian.field_encoder_config == source.field_encoder_config
    assert sde.field_encoder_config == source.field_encoder_config
    assert gaussian.field_action_config == source.field_action_config == action_config()
    assert sde.field_action_config == source.field_action_config
    assert gaussian.policy.optimizer is not source.policy.optimizer
    assert sde.policy.optimizer is not source.policy.optimizer
    assert gaussian.policy.optimizer is not sde.policy.optimizer
    assert not gaussian.policy.optimizer.state
    assert not sde.policy.optimizer.state

    source_state = source.policy.state_dict()
    gaussian_state = gaussian.policy.state_dict()
    sde_state = sde.policy.state_dict()
    shape_differences = [
        name for name, value in source_state.items() if value.shape != sde_state[name].shape
    ]
    assert shape_differences == ["log_std"]
    for name, value in source_state.items():
        if name != "log_std":
            torch.testing.assert_close(gaussian_state[name], value, rtol=0, atol=0)
            torch.testing.assert_close(sde_state[name], value, rtol=0, atol=0)
    torch.testing.assert_close(gaussian_state["log_std"], source_state["log_std"], rtol=0, atol=0)

    expected_sde_log_std = source.policy.log_std.detach() - 0.5 * np.log(latent_norm_sq_mean)
    torch.testing.assert_close(
        sde.policy.log_std,
        expected_sde_log_std.reshape(1, 4).expand_as(sde.policy.log_std),
    )
    gaussian_variance = source.policy.log_std.detach().exp().square()
    calibrated_variance = latent_norm_sq_mean * sde.policy.log_std.detach().exp().square().mean(dim=0)
    torch.testing.assert_close(calibrated_variance, gaussian_variance)
    with pytest.raises(ValueError, match="source_model must use the supported Gaussian"):
        make_exploration_variant(sde, env, use_sde=False)

    # Deterministic means should remain identical through a short recurrent sequence.
    initial = env.reset()
    states = [None, None, None]
    starts = [np.array([True]), np.array([False]), np.array([False])]
    actions_by_model = [[], [], []]
    for step in range(3):
        obs_t = {key: value.copy() for key, value in initial.items()}
        obs_t["state"][:, 2] = (step + 1) * 0.01
        for index, model in enumerate((source, gaussian, sde)):
            action, states[index] = model.predict(
                obs_t,
                state=states[index],
                episode_start=starts[step],
                deterministic=True,
            )
            actions_by_model[index].append(action)
    for candidate_actions in actions_by_model[1:]:
        for expected, actual in zip(actions_by_model[0], candidate_actions):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    env.close()


def test_exploration_variant_requires_real_positive_sde_calibration():
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([_PolicyVariantEnv])
    source = make_model(
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        lstm_hidden_size=8,
        net_arch={"pi": [8], "vf": [8]},
        device="cpu",
    )
    with pytest.raises(ValueError, match="requires latent_norm_sq_mean"):
        make_exploration_variant(source, env, use_sde=True)
    with pytest.raises(ValueError, match="finite and positive"):
        make_exploration_variant(source, env, use_sde=True, latent_norm_sq_mean=0.0)
    with pytest.raises(ValueError, match="only used for gSDE"):
        make_exploration_variant(source, env, use_sde=False, latent_norm_sq_mean=1.0)
    env.close()


def test_exploration_variant_rejects_source_without_current_action_contract():
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([_PolicyVariantEnv])
    source = make_model(
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        lstm_hidden_size=8,
        net_arch={"pi": [8], "vf": [8]},
        device="cpu",
    )
    source.field_action_config["schema"] = "forward-differential-track-speed-v2"
    try:
        with pytest.raises(ValueError, match="action contract.*requires a newly trained checkpoint"):
            make_exploration_variant(source, env, use_sde=False)
    finally:
        env.close()
