"""CPU checks for reusing the existing evaluator with native RSL checkpoints."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict

from isaac_field.action_contract import action_config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from isaac_field.learning import _model_or_checkpoint
from isaac_field.rsl_rl_inference import RslPolicyRunner, RslPredictionModel
from isaac_field.rsl_rl_policy import RslFieldRNNModel, build_field_policy_contract
from test_absolute_goal_policy import _observation_config


@pytest.fixture
def rsl_checkpoint(tmp_path):
    torch.set_num_threads(1)
    config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    obs = TensorDict({key: torch.zeros(2, *shape) for key, shape in config["observation_shapes"].items()}, [2])
    groups = {name: list(obs.keys()) for name in ("actor", "critic")}
    actor = RslFieldRNNModel(
        obs, groups, "actor", 4, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        distribution_cfg={"class_name": "rsl_rl.modules.distribution:GaussianDistribution", "init_std": np.exp(-2), "std_type": "log"},
    )
    critic = RslFieldRNNModel(obs, groups, "critic", 1, cnns=actor.cnns, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
    env = SimpleNamespace(num_envs=2, observation_config=config, action_config=action_config())
    contract = build_field_policy_contract(actor, critic, obs_groups=groups, env=env, num_steps_per_env=128)
    path = tmp_path / "model.pt"
    torch.save({"field_policy_contract": contract, "actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(), "iter": 0}, path)
    return path, config


def test_prediction_round_trip_and_independent_episode_state(rsl_checkpoint):
    path, config = rsl_checkpoint
    model = RslPredictionModel(path, expected_observation_config=config)
    obs = {key: np.zeros((2, *shape), np.float32) for key, shape in config["observation_shapes"].items()}
    action, state = model.predict(obs, episode_start=np.ones(2, bool))
    assert action.shape == (2, 4)
    assert all(value.shape == (1, 2, 512) for value in state)
    assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
    repeated, repeated_state = model.predict(obs, episode_start=np.ones(2, bool))
    np.testing.assert_allclose(action, repeated)
    for first, second in zip(state, repeated_state):
        np.testing.assert_allclose(first, second)
    second_action, _ = model.predict(obs, state=state, episode_start=[True, False])
    np.testing.assert_allclose(second_action[0], action[0], atol=1e-6)
    single = {key: value[0] for key, value in obs.items()}
    single_action, _ = model.predict(single, episode_start=[True])
    assert single_action.shape == (4,)
    np.testing.assert_allclose(single_action, action[0], atol=1e-6)
    assert model.actor.get_hidden_state() is None
    assert model.num_timesteps is None  # iteration 0 does not prove a transition count


def test_lazy_critic_uses_the_actor_encoder_without_registering_a_second_copy(rsl_checkpoint):
    path, config = rsl_checkpoint
    model = RslPredictionModel(path, expected_observation_config=config)

    critic = model.critic

    assert critic is model.critic
    assert critic.sensor_encoder is model.actor.sensor_encoder
    assert all(not name.startswith("cnns.") for name, _ in critic.named_parameters())
    assert all(not key.startswith("cnns.") for key in critic.state_dict())
    assert not any(
        parameter is model.actor.sensor_encoder.laser[0].weight
        for parameter in critic.parameters()
    )


def test_predict_values_isolates_explicit_hidden_state_and_resets_only_started_rows(rsl_checkpoint):
    path, config = rsl_checkpoint
    model = RslPredictionModel(path, expected_observation_config=config)
    observation = {
        key: np.zeros((2, *shape), dtype=np.float32)
        for key, shape in config["observation_shapes"].items()
    }
    hidden = np.full((1, 2, 512), 0.75, dtype=np.float32)
    cell = np.full((1, 2, 512), -0.25, dtype=np.float32)
    explicit_state = (hidden.copy(), cell.copy())
    sentinel = (
        torch.full((1, 2, 512), -0.4),
        torch.full((1, 2, 512), 0.6),
    )
    model.critic.reset(hidden_state=sentinel)

    values, next_state = model.predict_values(
        observation, state=explicit_state, episode_start=[True, False]
    )
    reset_values, reset_state = model.predict_values(
        observation, state=explicit_state, episode_start=[True, True]
    )

    assert values.shape == (2, 1)
    assert all(value.shape == (1, 2, 512) for value in next_state)
    np.testing.assert_allclose(values[0], reset_values[0], rtol=0, atol=1e-7)
    assert not np.allclose(values[1], reset_values[1], rtol=0, atol=1e-7)
    assert not np.allclose(next_state[0][:, 1], reset_state[0][:, 1], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(explicit_state[0], hidden)
    np.testing.assert_array_equal(explicit_state[1], cell)
    for actual, expected in zip(model.critic.get_hidden_state(), sentinel):
        torch.testing.assert_close(actual, expected)


def test_policy_runner_carries_partial_resets_as_episode_starts(rsl_checkpoint, monkeypatch):
    path, config = rsl_checkpoint
    env = SimpleNamespace(num_envs=2, observation_config=config)
    runner = RslPolicyRunner(env, path)
    observation = {
        key: np.zeros((2, *shape), dtype=np.float32)
        for key, shape in config["observation_shapes"].items()
    }
    calls = []

    def predict(obs, *, state=None, episode_start=None, deterministic=True):
        calls.append({
            "state": None if state is None else tuple(np.array(value, copy=True) for value in state),
            "episode_start": np.array(episode_start, copy=True),
            "deterministic": deterministic,
        })
        step = len(calls)
        next_state = (
            np.full((1, 2, 512), step, dtype=np.float32),
            np.full((1, 2, 512), -step, dtype=np.float32),
        )
        return np.zeros((2, 4), dtype=np.float32), next_state

    monkeypatch.setattr(runner.model, "predict", predict)
    np.testing.assert_array_equal(runner.action(observation), np.zeros((2, 4), np.float32))
    assert calls[-1]["state"] is None
    np.testing.assert_array_equal(calls[-1]["episode_start"], [True, True])

    runner.action(observation)
    assert calls[-1]["episode_start"].tolist() == [False, False]
    previous = tuple(value.copy() for value in runner.state)
    runner.reset(done=[True, False])
    assert runner.episode_start.tolist() == [True, False]
    for reset_state, before in zip(runner.state, previous):
        np.testing.assert_array_equal(reset_state[:, 0], np.zeros_like(before[:, 0]))
        np.testing.assert_array_equal(reset_state[:, 1], before[:, 1])
    reset_state = tuple(value.copy() for value in runner.state)

    runner.action(observation)
    assert calls[-1]["episode_start"].tolist() == [True, False]
    for passed, expected in zip(calls[-1]["state"], reset_state):
        np.testing.assert_array_equal(passed, expected)
    runner.reset()
    assert runner.state is None
    assert runner.episode_start.tolist() == [True, True]


def test_lazy_critic_rejects_checkpoint_changed_after_actor_load(rsl_checkpoint):
    path, config = rsl_checkpoint
    model = RslPredictionModel(path, expected_observation_config=config)

    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="checkpoint changed since actor loading"):
        _ = model.critic
    assert model._critic is None


def test_existing_evaluator_dispatches_native_checkpoint(rsl_checkpoint):
    path, config = rsl_checkpoint
    model, selected = _model_or_checkpoint(
        path, device="cpu", input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        expected_observation_config=config,
    )
    assert isinstance(model, RslPredictionModel)
    assert selected == path
    same, _ = _model_or_checkpoint(
        model, device="cpu", input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        expected_observation_config=config,
    )
    assert same is model


def test_checkpoint_and_prediction_reject_mismatched_semantics(rsl_checkpoint):
    path, config = rsl_checkpoint
    changed = deepcopy(config)
    changed["absolute_goal_coordinates"]["field_size_m"] = 200.0
    with pytest.raises(ValueError):
        RslPredictionModel(path, expected_observation_config=changed)
    payload = torch.load(path, weights_only=True)
    payload["field_policy_contract"]["action_config"] = {"schema": "unrelated"}
    torch.save(payload, path)
    with pytest.raises(ValueError):
        RslPredictionModel(path, expected_observation_config=config)


def test_rsl_loader_rejects_sb3_checkpoint(rsl_checkpoint):
    path, config = rsl_checkpoint
    wrong = path.with_suffix(".zip")
    wrong.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="SB3"):
        RslPredictionModel(wrong, expected_observation_config=config)


def test_existing_episode_loop_records_rsl_predictions(rsl_checkpoint):
    from isaac_field.learning import evaluate_model
    from test_isaac_field_learning import _FakeAdapter

    path, config = rsl_checkpoint
    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="goto", goal_distance=None, goal_angle=np.pi),
        observation_config=config, action_config=action_config(),
        damage_config={"metric": "stem-contact-v2"}, close=lambda: None,
    )
    zero = {key: np.zeros(shape, np.float32) for key, shape in config["observation_shapes"].items()}

    class Adapter(_FakeAdapter):
        def reset(self, **kwargs):
            _, info = super().reset(**kwargs)
            return zero, info

        def step(self, action):
            _, reward, terminated, truncated, info = super().step(action)
            return zero, reward, terminated, truncated, info

    report = evaluate_model(
        Adapter(engine), path, (12, 13), device="cpu", input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        expected_observation_config=config, record_trajectories=True,
    )
    assert report["model_class"] == "RslPredictionModel"
    assert report["checkpoint"] == str(path)
    assert report["summary"]["episodes"] == 2
    assert [len(row["trajectory"]) for row in report["rows"]] == [2, 2]
