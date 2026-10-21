"""CPU regressions for read-only recurrent policy inspection."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from isaac_field.action_contract import ACTION_LABELS, action_config
from isaac_field.policy import PolicyRunner, detection_presence_mask, make_model
from isaac_field.visualization.inspection import PolicyInspector
from scripts.view_policy_netron import (
    INPUT_NAMES,
    PolicyNetronWrapper,
    _compare_runtime,
    _make_observation as make_netron_observation,
    _torch_inputs,
)


class _InspectionEnv(gym.Env):
    def __init__(self):
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return _observation(7), {}

    def step(self, action):
        return _observation(7), 0.0, False, False, {}


def _observation(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = {
        "lidar": rng.uniform(0.0, 1.0, size=(1, 1, 400)).astype(np.float32),
        "depth": rng.uniform(0.0, 1.0, size=(1, 1, 160, 88)).astype(np.float32),
        "detections": np.zeros((1, 16, 8), dtype=np.float32),
        "state": rng.uniform(-1.0, 1.0, size=(1, 33)).astype(np.float32),
        "feedback": rng.normal(0.0, 0.25, size=(1, 37)).astype(np.float32),
    }
    candidates = [1, 4, 9, 14]
    value["detections"][:, candidates, :2] = rng.uniform(0.1, 0.9, size=(1, len(candidates), 2))
    value["detections"][:, candidates, 2:4] = rng.uniform(0.05, 0.25, size=(1, len(candidates), 2))
    value["detections"][:, candidates, 4:6] = rng.uniform(0.2, 0.95, size=(1, len(candidates), 2))
    value["detections"][:, 1, 6] = -0.02
    value["detections"][:, 1, 7] = 0.0
    return value


def _copy_observation(observation):
    return {key: value.copy() for key, value in observation.items()}


def _copy_runner_state(state):
    if state is None:
        return None
    return tuple(
        value.detach().cpu().numpy().copy()
        if isinstance(value, torch.Tensor)
        else np.array(value, copy=True)
        for value in state
    )


def _assert_runner_state_equal(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is not None:
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_allclose(actual_value, expected_value, rtol=0.0, atol=1e-7)


@pytest.fixture
def runtime():
    torch.manual_seed(20261017)
    torch.set_num_threads(1)
    env = DummyVecEnv([_InspectionEnv])
    runner = PolicyRunner(env, device="cpu")
    runner.model.policy.eval()
    inspector = PolicyInspector(runner)
    try:
        yield env, runner, inspector
    finally:
        env.close()


def test_instrumentation_preserves_actions_states_and_captures_cached_branches():
    env = DummyVecEnv([_InspectionEnv])
    reference_env = DummyVecEnv([_InspectionEnv])
    try:
        torch.manual_seed(20261017)
        runner = PolicyRunner(env, device="cpu")
        reference = PolicyRunner(reference_env, device="cpu")
        reference.model.policy.load_state_dict(runner.model.policy.state_dict())
        runner.model.policy.eval()
        reference.model.policy.eval()
        inspector = PolicyInspector(runner)
        encoder = runner.model.policy.features_extractor
        assert encoder.cache_inference is True

        repeated = _observation(19)
        # Populate both inference caches, then clear only recurrent state.
        runner.action(_copy_observation(repeated))
        reference.action(_copy_observation(repeated))
        runner.reset()
        reference.reset()
        assert encoder._cache
        params_before = {
            key: value.detach().clone()
            for key, value in runner.model.policy.state_dict().items()
        }
        observations_before = _copy_observation(repeated)

        reference_lstm = []
        lstm_handle = reference.model.policy.lstm_actor.register_forward_hook(
            lambda _module, _inputs, output: reference_lstm.append(
                tuple(value.detach().clone() for value in output[1])
            )
        )
        module_hook_counts = {
            name: (len(module._forward_hooks), len(module._forward_pre_hooks))
            for name, module in runner.model.policy.named_modules()
        }
        try:
            sequence = [repeated, repeated, _observation(20), repeated]
            inspected_actions = []
            reference_actions = []
            for index, observation in enumerate(sequence):
                reference_action = reference.action(_copy_observation(observation))
                inspected_action = inspector.action(_copy_observation(observation), float(index) * 0.05)
                reference_actions.append(reference_action.copy())
                inspected_actions.append(inspected_action.copy())
                np.testing.assert_allclose(inspected_action, reference_action, rtol=0.0, atol=1e-7)
                assert inspector.snapshot is not None
                assert inspector.snapshot["errors"] == ()
                np.testing.assert_array_equal(
                    inspector.layers["action.log_std"]["values"],
                    runner.model.policy.log_std.detach().cpu().numpy(),
                )
                np.testing.assert_allclose(
                    inspector.layers["action.std"]["values"],
                    runner.model.policy.action_dist.distribution.stddev.detach().cpu().numpy()[0],
                    rtol=0.0,
                    atol=1e-7,
                )
                for key, reference_value in zip(("lstm_actor.h", "lstm_actor.c"), reference_lstm[-1]):
                    expected = reference_value.detach().cpu().numpy()
                    if expected.ndim == 3 and expected.shape[1] == 1:
                        expected = expected[:, 0, :]
                    np.testing.assert_allclose(
                        inspector.layers[key]["values"],
                        expected,
                        rtol=0.0,
                        atol=1e-7,
                    )

            assert len(inspected_actions) == len(reference_actions) == len(sequence)
            assert encoder.cache_inference is True
            # Repeated input traversed the cache-bypassed branches, so each leaf
            # activation should be available in the inspection snapshot.
            layer_values = {layer["key"]: layer["values"] for layer in inspector.layers.values()}
            for key in (
                "features_extractor.laser.0",
                "features_extractor.laser.2",
                "features_extractor.laser.7",
                "features_extractor.depth.7",
                "features_extractor.depth.0",
                "features_extractor.boxes.0",
                "features_extractor.proprioception.0",
            ):
                assert layer_values[key] is not None
            assert layer_values["features_extractor.laser.0"].shape == (16, 200)
            assert layer_values["features_extractor.laser.2"].shape == (32, 100)
            assert layer_values["features_extractor.laser.7"].shape == (32,)
            assert layer_values["features_extractor.depth.7"].shape == (64,)
            assert inspector.layers["features_extractor.laser.0"]["title"].startswith("一维卷积")
            assert "环形填充" in inspector.layers["features_extractor.laser.0"]["title"]
            assert "环形填充" in inspector.layers["features_extractor.laser.0"]["description"]
            assert inspector.layers["features_extractor.laser.2"]["title"].startswith("一维卷积")
            assert inspector.layers["features_extractor.depth.0"]["title"].startswith("二维卷积")
            assert inspector.layers["features_extractor.depth.4"]["title"] == "二维自适应平均池化 · 8×6"
            assert inspector.layers["features_extractor.depth.6"]["title"] == "全连接 768→64"
            assert inspector.layers["features_extractor.laser.7"]["title"].endswith("32维")
            assert inspector.layers["features_extractor.depth.7"]["title"].endswith("64维")
            assert inspector.layers["detections.pooled"]["title"].endswith("32维")
            assert inspector.layers["features_extractor"]["title"] == "五路特征拼接 · 192维"
            assert "雷达32、深度64、检测32、状态32、反馈32拼成192维" in inspector.layers["features_extractor"]["description"]
            assert not any("融合 · 128维" in layer["title"] for layer in inspector.layers.values())
            assert not any("深度图编码输出 · 32维" == layer["title"] for layer in inspector.layers.values())
            assert inspector.snapshot["errors"] == ()
            assert layer_values["input.depth"].shape == (1, 160, 88)
            assert layer_values["input.state"].shape == (33,)
            assert "state.scaled" not in layer_values
            state_encoder_input = inspector.layers["features_extractor.proprioception.0"]["parents"]
            assert state_encoder_input == ("input.state",)
        finally:
            lstm_handle.remove()

        assert module_hook_counts == {
            name: (len(module._forward_hooks), len(module._forward_pre_hooks))
            for name, module in runner.model.policy.named_modules()
        }
        for key, value in runner.model.policy.state_dict().items():
            torch.testing.assert_close(value, params_before[key], rtol=0.0, atol=0.0)
        for key, value in repeated.items():
            np.testing.assert_array_equal(value, observations_before[key])

        # A subsequent ordinary call sees the original cache setting and behavior.
        next_reference = reference.action(_copy_observation(repeated))
        next_inspected = inspector.action(_copy_observation(repeated), 0.25)
        np.testing.assert_allclose(next_inspected, next_reference, rtol=0.0, atol=1e-7)
        assert encoder.cache_inference is True
    finally:
        env.close()
        reference_env.close()


def test_preview_does_not_advance_actor_or_critic_state(runtime):
    _env, runner, inspector = runtime
    inspector.action(_observation(31), 0.0)
    actor_before = _copy_runner_state(runner.state)
    starts_before = runner.episode_start.copy()
    critic_before = _copy_runner_state(inspector.critic_state)

    snapshot = inspector.preview(_observation(32), 0.5)

    assert snapshot["mode"] == "暂停预览（未施加车辆）"
    _assert_runner_state_equal(runner.state, actor_before)
    np.testing.assert_array_equal(runner.episode_start, starts_before)
    _assert_runner_state_equal(inspector.critic_state, critic_before)


def test_actor_and_critic_lstm_gates_reconstruct_observed_states(runtime):
    _env, _runner, inspector = runtime
    inspector.action(_observation(41), 0.0)
    values = {layer["key"]: layer["values"] for layer in inspector.layers.values()}
    layers = inspector.layers

    for branch in ("lstm_actor", "lstm_critic"):
        for gate in ("i", "f", "g", "o"):
            assert layers[f"{branch}.{gate}"]["parents"] == ("features_extractor", f"{branch}.h0")
            assert branch not in layers[f"{branch}.{gate}"]["parents"]
        assert layers[f"{branch}.c"]["parents"] == tuple(
            f"{branch}.{suffix}" for suffix in ("f", "i", "g", "c0")
        )
        assert layers[f"{branch}.h"]["parents"] == (f"{branch}.o", f"{branch}.c")
        assert layers[branch]["parents"] == (f"{branch}.h",)
        for state in ("h0", "c0"):
            assert layers[f"{branch}.{state}"]["parents"] == ()
            assert "跨步反馈折叠" in layers[f"{branch}.{state}"]["description"]
        for suffix in ("h0", "c0", "h", "c", "i", "f", "g", "o"):
            assert values[f"{branch}.{suffix}"] is not None
        for gate in ("i", "f", "o"):
            assert np.all((values[f"{branch}.{gate}"] >= 0.0) & (values[f"{branch}.{gate}"] <= 1.0))
        assert np.all((values[f"{branch}.g"] >= -1.0) & (values[f"{branch}.g"] <= 1.0))

        def layer_state(suffix):
            value = values[f"{branch}.{suffix}"]
            if value.ndim == 3 and value.shape[1] == 1:
                value = value[:, 0, :]
            return value

        h0 = layer_state("h0")
        c0 = layer_state("c0")
        actual_h = layer_state("h")
        actual_c = layer_state("c")
        reconstructed_c = values[f"{branch}.f"] * c0 + values[f"{branch}.i"] * values[f"{branch}.g"]
        reconstructed_h = values[f"{branch}.o"] * np.tanh(reconstructed_c)
        np.testing.assert_allclose(reconstructed_c, actual_c, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(reconstructed_h, actual_h, rtol=2e-5, atol=2e-6)


def test_detection_mask_pool_winners_and_fusion_match_real_forward(runtime):
    _env, runner, inspector = runtime
    observation = _observation(53)
    inspector.action(_copy_observation(observation), 0.0)
    values = {layer["key"]: layer["values"] for layer in inspector.layers.values()}

    np.testing.assert_array_equal(values["input.detections"], observation["detections"][0])
    with torch.no_grad():
        boxes = (
            runner.model.policy.features_extractor.boxes(
                torch.as_tensor(observation["detections"], dtype=torch.float32)
            )[0]
            .cpu()
            .numpy()
        )
    presence = detection_presence_mask(
        torch.as_tensor(observation["detections"], dtype=torch.float32)
    )[0].cpu().numpy()
    assert presence.shape == (16, 1)
    np.testing.assert_array_equal(
        presence[:, 0],
        np.asarray([index in (1, 4, 9, 14) for index in range(16)]),
    )
    # Candidate presence follows its positive box geometry; a negative range
    # and false range-valid bit do not turn a real detection into padding.
    assert observation["detections"][0, 1, 6] < 0.0
    assert observation["detections"][0, 1, 7] == 0.0
    expected_masked = boxes * presence
    expected_pool = expected_masked.max(axis=0)
    expected_winner = expected_masked.argmax(axis=0)
    np.testing.assert_allclose(values["detections.masked"], expected_masked, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(values["detections.pooled"], expected_pool, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(values["detections.winner"], expected_winner)

    expected_fusion = np.concatenate(
        (
            values["features_extractor.laser.7"],
            values["features_extractor.depth.7"],
            expected_pool,
            values["features_extractor.proprioception.1"],
            values["features_extractor.feedback.1"],
        )
    )
    assert expected_fusion.shape == (192,)
    np.testing.assert_allclose(values["features_extractor"], expected_fusion, rtol=1e-6, atol=1e-7)


def test_netron_example_observation_uses_single_depth_channel_and_sensor_feedback():
    observation = make_netron_observation(np.random.default_rng(20261018))

    assert observation["depth"].shape == (1, 1, 160, 88)
    assert observation["lidar"].shape == (1, 1, 400)
    assert observation["detections"].shape == (1, 16, 8)
    assert observation["state"].shape == (1, 33)
    assert observation["feedback"].shape == (1, 37)


def test_netron_wrapper_runtime_matches_policy_with_all_five_sensor_inputs():
    env = DummyVecEnv([_InspectionEnv])
    try:
        model = make_model(
            env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            lstm_hidden_size=8,
            net_arch={"pi": [8], "vf": [8]},
            device="cpu",
        )
        policy = model.policy
        wrapper = PolicyNetronWrapper(policy).eval()
        observation = make_netron_observation(np.random.default_rng(20261020))
        assert INPUT_NAMES[:5] == ("lidar", "depth", "detections", "state", "feedback")
        inputs = _torch_inputs(observation, tuple(
            np.zeros((policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size), dtype=np.float32)
            for _ in range(4)
        ))
        assert len(inputs) == len(INPUT_NAMES) == 9
        np.testing.assert_array_equal(inputs[4].cpu().numpy(), observation["feedback"])
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, inputs, check_trace=False)
            maxima, cases = _compare_runtime(model, policy, wrapper, traced, seed=20261020)
        assert len(cases) == 9
        assert maxima["mean_vs_policy_distribution"] < 2e-5
        assert maxima["clip_vs_predict"] < 2e-5
        assert maxima["value_vs_predict_values"] < 2e-5
    finally:
        env.close()


def test_reset_clears_snapshot_critic_and_all_layer_values(runtime):
    _env, _runner, inspector = runtime
    inspector.action(_observation(61), 0.0)
    assert inspector.snapshot is not None
    assert inspector.critic_state is not None
    assert any(layer["values"] is not None for layer in inspector.layers.values())

    inspector.reset()

    assert inspector.snapshot is None
    assert inspector.critic_state is None
    assert all(layer["values"] is None for layer in inspector.layers.values())


def test_selected_layer_parameters_are_actual_detached_module_tensors(runtime):
    _env, runner, inspector = runtime
    key = "features_extractor.laser.0"
    module = dict(runner.model.policy.named_modules())[key]

    actual = inspector.parameters(key)

    assert set(actual) == {"weight", "bias"}
    assert actual["weight"].shape == (16, 1, 5)
    for name, parameter in module.named_parameters(recurse=False):
        np.testing.assert_array_equal(actual[name], parameter.detach().cpu().numpy())
    actual["weight"].fill(0.0)
    assert torch.count_nonzero(module.weight).item() > 0


def test_gaussian_log_std_node_is_independent_and_exposes_actual_parameter(runtime):
    _env, runner, inspector = runtime
    layers = inspector.layers

    assert layers["action.log_std"]["parents"] == ()
    assert layers["action.log_std"]["module"] is None
    assert layers["action.std"]["parents"] == ("action.log_std",)
    assert layers["action.std"]["module"] is None
    assert layers["action.applied"]["parents"] == ("action_net",)
    assert "、".join(ACTION_LABELS) in layers["action.applied"]["description"]
    assert "左右独立履带速度指令" in layers["action.applied"]["description"]
    assert "云台偏航 / 俯仰角速度语义" in layers["action.applied"]["description"]

    inspector.action(_observation(79), 0.0)
    assert inspector.snapshot["action_config"] == action_config()
    expected = runner.model.policy.log_std.detach().cpu().numpy()
    np.testing.assert_array_equal(layers["action.log_std"]["values"], expected)
    np.testing.assert_allclose(layers["action.std"]["values"], np.exp(expected), rtol=1e-6, atol=1e-7)
    parameters = inspector.parameters("action.log_std")
    assert set(parameters) == {"log_std"}
    np.testing.assert_array_equal(parameters["log_std"], expected)
    parameters["log_std"].fill(0.0)
    np.testing.assert_array_equal(runner.model.policy.log_std.detach().cpu().numpy(), expected)
