"""Policy feature scaling and checkpoint compatibility tests (no learning)."""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch

from field_sim.policy import PolicyRunner, SensorEncoder, make_model


def _observation_space():
    return spaces.Dict({
        "lidar": spaces.Box(-1.0, 1.0, shape=(2, 8), dtype=np.float32),
        "depth": spaces.Box(-1.0, 1.0, shape=(2, 8, 8), dtype=np.float32),
        "detections": spaces.Box(-1.0, 1.0, shape=(4, 10), dtype=np.float32),
        "state": spaces.Box(-1.0, 1.0, shape=(44,), dtype=np.float32),
    })


def _observation():
    return {
        "lidar": torch.linspace(-1, 1, 16, dtype=torch.float32).reshape(1, 2, 8),
        "depth": torch.linspace(-1, 1, 128, dtype=torch.float32).reshape(1, 2, 8, 8),
        "detections": torch.linspace(-1, 1, 40, dtype=torch.float32).reshape(1, 4, 10),
        "state": torch.linspace(-0.5, 0.5, 44, dtype=torch.float32).reshape(1, 44),
    }


class _PolicyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = _observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _observation(self):
        return {key: np.zeros(space.shape, np.float32)
                for key, space in self.observation_space.spaces.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, False, False, {}


def _legacy_policy_kwargs():
    # Matches the pre-goal_gain checkpoint layout: no extractor kwargs stored.
    return {"features_extractor_class": SensorEncoder,
            "lstm_hidden_size": 128, "n_lstm_layers": 1,
            "net_arch": {"pi": [64], "vf": [64]},
            "normalize_images": False}


def test_default_goal_gain_path_is_bit_identical_to_legacy_extractor():
    torch.manual_seed(4)
    legacy = SensorEncoder(_observation_space(), cache_inference=False)
    current = SensorEncoder(_observation_space(), cache_inference=False, goal_gain=1.0)
    current.load_state_dict(legacy.state_dict())
    observation = _observation()
    original_state = observation["state"].clone()

    with torch.no_grad():
        old_features = legacy(observation)
        current_features = current(observation)

    assert torch.equal(current_features, old_features)
    assert current_features.shape == (1, 128)
    assert torch.equal(observation["state"], original_state)


def test_goal_gain_scales_only_relative_goal_and_tolerance_before_proprioception():
    encoder = SensorEncoder(_observation_space(), cache_inference=False, goal_gain=12.5).eval()
    observation = _observation()
    observation["state"][:, 2:4] = torch.tensor([[0.04, -0.02]])
    observation["state"][:, 19] = 0.004
    original = observation["state"].clone()
    received = []
    handle = encoder.proprioception[0].register_forward_pre_hook(
        lambda _module, inputs: received.append(inputs[0].detach().clone()))
    try:
        with torch.no_grad():
            encoder(observation)
    finally:
        handle.remove()

    expected = original.clone()
    expected[:, 2:4] *= 12.5
    expected[:, 19] *= 12.5
    assert torch.equal(received[0], expected)
    assert torch.equal(observation["state"], original)
    assert received[0][0, 2].item() == pytest.approx(0.5)
    assert received[0][0, 3].item() == pytest.approx(-0.25)
    assert received[0][0, 19].item() == pytest.approx(0.05)


def test_goal_gain_cache_matches_uncached_features_when_goal_changes():
    torch.manual_seed(8)
    cached = SensorEncoder(_observation_space(), goal_gain=12.5).eval()
    uncached = SensorEncoder(_observation_space(), cache_inference=False,
                             goal_gain=12.5).eval()
    uncached.load_state_dict(cached.state_dict())
    first = _observation()
    second = {key: value.clone() for key, value in first.items()}
    second["state"][:, 2:4] += torch.tensor([[0.1, -0.2]])
    original_second_state = second["state"].clone()

    with torch.no_grad():
        cached_first = cached(first)
        cached_second = cached(second)
        expected_first = uncached(first)
        expected_second = uncached(second)

    torch.testing.assert_close(cached_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(cached_second, expected_second, rtol=0, atol=0)
    assert not torch.equal(cached_first, cached_second)
    assert set(cached._cache) == {"lidar", "depth", "detections"}
    assert torch.equal(second["state"], original_second_state)


def test_make_model_options_and_new_and_legacy_checkpoints_roundtrip(tmp_path):
    from sb3_contrib import RecurrentPPO

    configured_env = _PolicyEnv()
    model = make_model(configured_env, seed=11, n_steps=8, batch_size=8,
                       n_epochs=2, learning_rate=2e-4, gamma=0.95,
                       goal_gain=12.5, n_env=1, use_sde=True,
                       sde_sample_freq=32, log_std_init=-1.0,
                       squash_output=True, target_kl=0.03,
                       device=torch.device("cpu"))
    assert model.n_steps == 8
    assert model.batch_size == 8
    assert model.n_epochs == 2
    assert model.learning_rate == pytest.approx(2e-4)
    assert model.gamma == pytest.approx(0.95)
    assert model.use_sde is True
    assert model.sde_sample_freq == 32
    assert model.target_kl == pytest.approx(0.03)
    assert model.policy_kwargs["features_extractor_kwargs"] == {"goal_gain": 12.5}
    assert model.policy_kwargs["log_std_init"] == pytest.approx(-1.0)
    assert model.policy_kwargs["squash_output"] is True
    assert model.device.type == "cpu"
    new_path = tmp_path / "goal_scaled.zip"
    model.save(new_path)
    new_env = _PolicyEnv()
    loaded_new = PolicyRunner(new_env, new_path, device="cpu")
    assert loaded_new.model.device.type == "cpu"
    assert loaded_new.model.policy.features_extractor.goal_gain == pytest.approx(12.5)
    assert loaded_new.model.use_sde is True
    assert loaded_new.model.sde_sample_freq == 32
    assert loaded_new.model.target_kl == pytest.approx(0.03)
    assert loaded_new.model.policy_kwargs["squash_output"] is True
    observation, _ = new_env.reset(seed=10)
    sampled_action, _ = loaded_new.model.predict(observation, deterministic=False)
    assert sampled_action.shape == (4,)
    assert np.all(np.isfinite(sampled_action))
    assert np.all(sampled_action >= new_env.action_space.low)
    assert np.all(sampled_action <= new_env.action_space.high)

    legacy_model = RecurrentPPO(
        "MultiInputLstmPolicy", _PolicyEnv(), seed=11, device="cpu",
        n_steps=8, batch_size=8, n_epochs=1,
        policy_kwargs=_legacy_policy_kwargs())
    legacy_path = tmp_path / "legacy_no_extractor_kwargs.zip"
    legacy_model.save(legacy_path)
    loaded_legacy = PolicyRunner(_PolicyEnv(), legacy_path)
    legacy_extractor = loaded_legacy.model.policy.features_extractor
    assert legacy_extractor.goal_gain == 1.0


def test_make_model_keeps_legacy_default_hyperparameters():
    model = make_model(_PolicyEnv())

    assert model.n_steps == 128
    assert model.batch_size == 64
    assert model.n_epochs == 3
    assert model.learning_rate == pytest.approx(3e-4)
    assert model.gamma == pytest.approx(0.999)
    assert model.policy.features_extractor.goal_gain == 1.0
    assert model.device.type == "cpu"
    assert model.use_sde is False
    assert model.sde_sample_freq == -1
    assert model.target_kl is None
    assert model.policy_kwargs["log_std_init"] == pytest.approx(0.0)
    assert model.policy_kwargs["squash_output"] is False
