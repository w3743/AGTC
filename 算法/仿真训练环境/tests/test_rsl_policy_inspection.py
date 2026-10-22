"""CPU regressions for inspection of native RSL-RL field checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

torch.set_num_threads(1)

from isaac_field.action_contract import action_config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from isaac_field.policy import OBSERVATION_SHAPES
from isaac_field.rsl_rl_inference import RslPolicyRunner
from isaac_field.rsl_rl_policy import RslFieldRNNModel, build_field_policy_contract
from isaac_field.visualization.inspection import PolicyInspector
from isaac_field.visualization.network_panel import PolicyNetworkPanel
from test_absolute_goal_policy import _observation_config


def _make_checkpoint(path: Path):
    torch.manual_seed(20261022)
    config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    observations = TensorDict(
        {key: torch.zeros((1, *shape), dtype=torch.float32) for key, shape in OBSERVATION_SHAPES.items()},
        batch_size=[1],
    )
    groups = {key: list(observations.keys()) for key in ("actor", "critic")}
    actor = RslFieldRNNModel(
        observations,
        groups,
        "actor",
        4,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": float(np.exp(-2.0)),
            "std_type": "log",
            "std_range": [1e-6, 1e6],
        },
    )
    critic = RslFieldRNNModel(
        observations,
        groups,
        "critic",
        1,
        cnns=actor.cnns,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    env = SimpleNamespace(
        num_envs=1,
        observation_config=config,
        action_config=action_config(),
        drive_config={"cpu_fixture": True},
        damage_config={"cpu_fixture": True},
    )
    contract = build_field_policy_contract(actor, critic, obs_groups=groups, env=env, num_steps_per_env=128)
    torch.save(
        {
            "field_policy_contract": contract,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "num_timesteps": 0,
        },
        path,
    )
    return config


def _observation(seed=22):
    rng = np.random.default_rng(seed)
    observation = {
        "lidar": rng.uniform(0.05, 0.95, OBSERVATION_SHAPES["lidar"]).astype(np.float32),
        "depth": rng.uniform(0.05, 0.95, OBSERVATION_SHAPES["depth"]).astype(np.float32),
        "detections": np.zeros(OBSERVATION_SHAPES["detections"], dtype=np.float32),
        "state": rng.uniform(-1.0, 1.0, OBSERVATION_SHAPES["state"]).astype(np.float32),
        "feedback": rng.normal(0.0, 0.25, OBSERVATION_SHAPES["feedback"]).astype(np.float32),
    }
    indices = np.asarray([1, 5, 12])
    observation["detections"][indices, :2] = rng.uniform(0.1, 0.8, (len(indices), 2))
    observation["detections"][indices, 2:4] = rng.uniform(0.05, 0.3, (len(indices), 2))
    observation["detections"][indices, 4:6] = rng.uniform(0.2, 0.95, (len(indices), 2))
    return observation


@pytest.fixture
def runtime(tmp_path):
    config = _make_checkpoint(tmp_path / "field.pt")
    env = SimpleNamespace(num_envs=1, observation_config=config)
    runner = RslPolicyRunner(env, tmp_path / "field.pt", device="cpu", expected_observation_config=config)
    reference = RslPolicyRunner(env, tmp_path / "field.pt", device="cpu", expected_observation_config=config)
    inspector = PolicyInspector(runner)
    return runner, reference, inspector


def _copy_state(state):
    if state is None:
        return None
    return tuple(np.array(value, copy=True) for value in state)


def _assert_state(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is not None:
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)


def test_rsl_graph_points_to_real_modules_and_captures_all_five_inputs(runtime):
    runner, _reference, inspector = runtime
    model = runner.model
    assert not hasattr(model, "policy")
    assert not hasattr(inspector, "policy")

    observation = _observation()
    actor = model.actor
    captured_lstm = []
    captured_mean = []
    hook_counts = {
        name: (len(module._forward_hooks), len(module._forward_pre_hooks))
        for name, module in actor.named_modules()
    }
    handles = [
        actor.rnn.rnn.register_forward_hook(
            lambda _module, _inputs, output: captured_lstm.append(tuple(value.detach().clone() for value in output[1]))
        ),
        actor.mlp[2].register_forward_hook(
            lambda _module, _inputs, output: captured_mean.append(output.detach().clone())
        ),
    ]
    cache = actor.sensor_encoder._cache.copy()
    try:
        action = inspector.action(observation, 0.0)
    finally:
        for handle in handles:
            handle.remove()

    layers = inspector.layers
    values = {key: layer["values"] for key, layer in layers.items()}
    for key in ("lidar", "depth", "detections", "state", "feedback"):
        np.testing.assert_array_equal(values[f"input.{key}"], observation[key])
    assert values["rsl.encoder.fusion"].shape == (192,)
    assert values["rsl.encoder.detections.pooled"].shape == (32,)
    assert values["rsl.actor.lstm.h"].shape == (1, 512)
    assert values["rsl.actor.lstm.c"].shape == (1, 512)
    for gate in ("i", "f", "g", "o"):
        assert values[f"rsl.actor.lstm.{gate}"].shape == (1, 512)
    assert inspector.snapshot["errors"] == ()
    assert len(captured_lstm) == len(captured_mean) == 1
    for suffix, actual in zip(("h", "c"), captured_lstm[0]):
        np.testing.assert_allclose(values[f"rsl.actor.lstm.{suffix}"], actual[:, 0].numpy(), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(values["rsl.action.mean"], captured_mean[0][0].numpy(), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(action, np.clip(values["rsl.action.mean"], -1.0, 1.0), rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(values["action.log_std"], actor.distribution.log_std_param.detach().numpy())
    np.testing.assert_allclose(values["action.std"], np.exp(values["action.log_std"]), rtol=1e-6, atol=1e-7)

    encoder_modules = dict(actor.named_modules())
    assert layers["rsl.encoder.lidar.0"]["module"] == "cnns.field.laser.0"
    assert encoder_modules[layers["rsl.encoder.lidar.0"]["module"]] is actor.sensor_encoder.laser[0]
    assert layers["rsl.action.mean"]["module"] == "mlp.2"
    assert encoder_modules["mlp.2"] is actor.mlp[2]
    assert "actor.cnns.field.laser.0" in layers["rsl.encoder.lidar.0"]["source"]
    assert "RSL-RL" in layers["rsl.action.mean"]["source"]
    assert not any(key.startswith("lstm_actor") or key.startswith("mlp_extractor") for key in layers)
    assert hook_counts == {
        name: (len(module._forward_hooks), len(module._forward_pre_hooks))
        for name, module in actor.named_modules()
    }
    assert actor.sensor_encoder._cache == cache
    assert actor.get_hidden_state() is None
    assert model.critic.get_hidden_state() is None


def test_preview_and_critic_diagnostics_preserve_policy_state(runtime):
    runner, reference, inspector = runtime
    observation = _observation(31)

    actor_state = _copy_state(runner.state)
    critic_state = _copy_state(inspector.critic_state)
    actor_internal = runner.model.actor.get_hidden_state()
    critic_internal = runner.model.critic.get_hidden_state()
    inspector.preview(observation, 0.0)
    _assert_state(runner.state, actor_state)
    _assert_state(inspector.critic_state, critic_state)
    assert runner.episode_start.tolist() == [True]
    assert runner.model.actor.get_hidden_state() is actor_internal
    assert runner.model.critic.get_hidden_state() is critic_internal

    expected = reference.action({key: value.copy() for key, value in observation.items()})
    actual = inspector.action({key: value.copy() for key, value in observation.items()}, 0.05)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    _assert_state(runner.state, reference.state)
    assert runner.episode_start.tolist() == reference.episode_start.tolist() == [False]
    assert inspector.critic_state is not None
    assert all(value.shape == (1, 1, 512) for value in inspector.critic_state)
    assert runner.model.actor.get_hidden_state() is None
    assert runner.model.critic.get_hidden_state() is None

    actor_state = _copy_state(runner.state)
    critic_state = _copy_state(inspector.critic_state)
    inspector.preview(_observation(32), 0.1)
    _assert_state(runner.state, actor_state)
    _assert_state(inspector.critic_state, critic_state)
    assert runner.episode_start.tolist() == [False]

    inspector.reset()
    assert inspector.critic_state is None
    assert inspector.snapshot is None
    assert all(layer["values"] is None for layer in inspector.layers.values())
    _assert_state(runner.state, actor_state)


class _Handle:
    def __init__(self, **values):
        self.__dict__.update(values)

    def on_update(self, _callback):
        pass

    def on_click(self, _callback):
        pass


class _Folder:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Gui:
    def add_folder(self, *_args, **_kwargs):
        return _Folder()

    def add_markdown(self, content):
        return _Handle(content=content)

    def add_button(self, _label):
        return _Handle()

    def add_checkbox(self, _label, *, initial_value):
        return _Handle(value=initial_value)


def test_existing_network_panel_exports_rsl_weights_and_activation(runtime, monkeypatch, tmp_path):
    import isaac_field.visualization.network_panel as network_panel

    runner, _reference, _inspector = runtime
    panel = PolicyNetworkPanel(_Gui(), lambda *_args: None)
    panel.bind(runner)
    observation = _observation(41)
    panel.command("network_preview", None, observation, 1.25)
    panel.publish(force=True)
    monkeypatch.setattr(
        network_panel,
        "__file__",
        str(tmp_path / "isaac_field" / "visualization" / "network_panel.py"),
    )

    panel.export()

    exports = list((tmp_path / "results" / "isaac_field_viewer" / "network_exports").glob("*.npz"))
    assert len(exports) == 1
    with np.load(exports[0], allow_pickle=False) as archive:
        assert archive["activation/rsl.encoder.fusion"].shape == (192,)
        assert archive["activation/rsl.actor.lstm.i"].shape == (1, 512)
        assert archive["weight/cnns.field.laser.0.weight"].shape == (16, 1, 5)
    metadata = json.loads(exports[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "rsl_rl"
    assert metadata["checkpoint_label"] == runner.label
    assert next(layer for layer in metadata["layers"] if layer["key"] == "rsl.action.mean")["source"].startswith("RSL-RL")
    assert not hasattr(runner.model, "policy")
