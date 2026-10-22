"""CPU contract checks for the RSL-RL field policy adapter."""
from __future__ import annotations

import copy
import json
import math
import zipfile

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaac_field.action_contract import action_config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, feedback_config
from isaac_field.policy_contract import OBSERVATION_SHAPES
from isaac_field.task import state_fields_for_schema
from isaac_field.rsl_rl_policy import (
    FIELD_OBSERVATION_KEYS,
    RSL_FIELD_CHECKPOINT_FORMAT,
    RslFieldPPO,
    RslFieldRNNModel,
    load_rsl_checkpoint_payload,
    validate_field_policy_contract,
)


torch.set_num_threads(1)


def make_obs(num_envs: int = 2, *, sequence: bool = False, steps: int = 4) -> TensorDict:
    shape = (steps, num_envs) if sequence else (num_envs,)
    data = {
        key: torch.rand(*shape, *dims, dtype=torch.float32)
        for key, dims in OBSERVATION_SHAPES.items()
    }
    return TensorDict(data, batch_size=list(shape), device="cpu")


class SyntheticFieldEnv:
    num_envs = 2
    num_actions = 4
    device = "cpu"
    observation_config = {
        "schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "state_dim": 33,
        "state_fields": list(state_fields_for_schema(ABSOLUTE_GOAL_INPUT_SCHEMA)),
        "observation_shapes": {key: list(value) for key, value in OBSERVATION_SHAPES.items()},
        "feedback": feedback_config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA),
        "absolute_goal_coordinates": {
            "source": "task.goal",
            "coordinate_frame": "field_world_xy",
            "origin_xy_m": [0.0, 0.0],
            "unit": "m",
            "field_size_m": 20.0,
            "normalization": "world_xy / field_size_m * 2 - 1",
            "state_clip": [-1.0, 1.0],
        },
    }
    action_config = action_config()
    drive_config = {"proxy": "cpu-test"}
    damage_config = {"metric": "cpu-test"}

    @property
    def unwrapped(self):
        return self


def make_cfg() -> dict:
    model = "isaac_field.rsl_rl_policy:RslFieldRNNModel"
    policy_kwargs = {
        "hidden_dims": [64],
        "activation": "tanh",
        "obs_normalization": False,
        "rnn_type": "lstm",
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 1,
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "encoder_schema": "depth64-lidar-single1d-absolute-goal-v4",
        "goal_gain": 1.0,
        "cache_inference": True,
        "observation_shapes": {
            key: list(value) for key, value in OBSERVATION_SHAPES.items()
        },
    }
    actor = {
        "class_name": model,
        **policy_kwargs,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": np.exp(-2.0),
            "std_type": "log",
            "std_range": (1e-6, 1e6),
        },
    }
    critic = {
        "class_name": model,
        **policy_kwargs,
        "distribution_cfg": None,
    }
    return {
        "num_steps_per_env": 4,
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "save_interval": 10,
        "torch_compile_mode": None,
        "multi_gpu": None,
        "obs_groups": {
            "actor": list(FIELD_OBSERVATION_KEYS),
            "critic": list(FIELD_OBSERVATION_KEYS),
        },
        "actor": actor,
        "critic": critic,
        "algorithm": {
            "class_name": "isaac_field.rsl_rl_policy:RslFieldPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.995,
            "lam": 0.95,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.0,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": None,
            "normalize_advantage_per_mini_batch": True,
            "use_mixed_precision": False,
            "share_cnn_encoders": True,
        },
    }


def make_algorithm():
    torch.manual_seed(23)
    env = SyntheticFieldEnv()
    cfg = make_cfg()
    algorithm = RslFieldPPO.construct_algorithm(make_obs(), env, cfg, "cpu")
    return env, cfg, algorithm


@pytest.mark.parametrize("init_std", [0.2, 0.5, 1.0, 2.0])
def test_field_gaussian_accepts_finite_positive_exploration_scales(init_std):
    from isaac_field.rsl_rl_policy import _normalized_gaussian_config

    config = {
        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
        "init_std": init_std,
        "std_type": "log",
    }
    normalized = _normalized_gaussian_config(config)
    assert normalized["init_std"] == float(init_std)
    assert normalized["std_type"] == "log"

    observations = make_obs(1)
    groups = {
        "actor": list(FIELD_OBSERVATION_KEYS),
        "critic": list(FIELD_OBSERVATION_KEYS),
    }
    actor = RslFieldRNNModel(
        observations,
        groups,
        "actor",
        4,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        distribution_cfg=config,
    )
    assert actor.distribution.log_std_param.detach().eq(math.log(init_std)).all()
    assert actor.constructor_config()["distribution_cfg"]["init_std"] == float(init_std)


@pytest.mark.parametrize("init_std", [0.0, -0.1, float("nan"), float("inf"), True])
def test_field_gaussian_rejects_nonpositive_or_nonfinite_exploration_scales(init_std):
    from isaac_field.rsl_rl_policy import _normalized_gaussian_config

    with pytest.raises(ValueError, match="finite and positive"):
        _normalized_gaussian_config(
            {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": init_std,
                "std_type": "log",
            }
        )


def test_nondefault_gaussian_init_std_is_preserved_in_checkpoint_contract():
    env = SyntheticFieldEnv()
    cfg = make_cfg()
    cfg["actor"]["distribution_cfg"]["init_std"] = 0.5

    algorithm = RslFieldPPO.construct_algorithm(make_obs(), env, cfg, "cpu")
    contract = validate_field_policy_contract(algorithm.save()["field_policy_contract"])

    assert contract["distribution"]["init_std"] == 0.5
    assert contract["actor_config"]["distribution_cfg"]["init_std"] == 0.5


def test_real_rsl_construct_and_shared_encoder_has_single_optimizer_owner():
    _, cfg, algorithm = make_algorithm()
    actor = algorithm._raw_actor
    critic = algorithm._raw_critic
    assert isinstance(actor, RslFieldRNNModel)
    assert isinstance(critic, RslFieldRNNModel)
    assert actor.sensor_encoder is critic.sensor_encoder
    assert actor.rnn.rnn.hidden_size == critic.rnn.rnn.hidden_size == 512
    assert actor.rnn.rnn.num_layers == critic.rnn.rnn.num_layers == 1
    assert actor.mlp[0].out_features == critic.mlp[0].out_features == 64
    assert actor.mlp[1].__class__.__name__ == critic.mlp[1].__class__.__name__ == "Tanh"
    assert actor.distribution.log_std_param.detach().eq(-2.0).all()

    params = list(algorithm.actor.parameters()) + list(algorithm.critic.parameters())
    assert len({id(param) for param in params}) == len(params)
    assert any(key.startswith("cnns.field.") for key in actor.state_dict())
    assert not any(key.startswith("cnns.") for key in critic.state_dict())
    assert cfg["obs_groups"]["actor"] == list(FIELD_OBSERVATION_KEYS)
    assert algorithm.field_policy_contract["format"] == RSL_FIELD_CHECKPOINT_FORMAT
    json.dumps(algorithm.field_policy_contract, allow_nan=False)


def test_sensor_encoder_gets_gradients_from_actor_and_critic_paths():
    _, _, algorithm = make_algorithm()
    algorithm.train_mode()
    obs = make_obs()
    actor = algorithm._raw_actor
    critic = algorithm._raw_critic
    encoder = actor.sensor_encoder

    actions = actor(obs, stochastic_output=True)
    actor_loss = -actor.get_output_log_prob(actions).mean()
    actor_loss.backward()
    assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in encoder.parameters())
    assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in actor.rnn.parameters())

    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    values = critic(obs)
    critic_loss = values.square().mean()
    critic_loss.backward()
    assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in encoder.parameters())
    assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in critic.rnn.parameters())


def test_recurrent_storage_handles_partial_dones_and_padded_t_by_n_update():
    _, _, algorithm = make_algorithm()
    algorithm.train_mode()
    next_obs = make_obs()
    for step in range(4):
        obs = make_obs()
        algorithm.act(obs)
        dones = torch.tensor([step == 1, step == 2], dtype=torch.long)
        next_obs = make_obs()
        algorithm.process_env_step(
            next_obs,
            rewards=torch.tensor([1.0, 0.5]),
            dones=dones,
            extras={},
        )

    batch = next(algorithm.storage.recurrent_mini_batch_generator(1, 1))
    assert batch.observations["lidar"].shape[:2] == (4, 4)
    assert batch.observations["depth"].shape == (4, 4, 1, 160, 88)
    assert batch.masks.shape == (4, 4)
    assert bool(batch.masks.any()) and bool((~batch.masks).any())
    assert batch.hidden_states[0] is not None and batch.hidden_states[1] is not None

    actor_output = algorithm.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
    )
    critic_output = algorithm.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
    )
    assert actor_output.shape == (4, 2, 4)
    assert critic_output.shape == (4, 2, 1)

    algorithm.compute_returns(next_obs)
    losses = algorithm.update()
    assert algorithm.save()["num_timesteps"] == 8
    _, _, resumed = make_algorithm()
    resumed.load(algorithm.save(), load_cfg=None, strict=True)
    assert resumed.num_timesteps == 8
    resumed.load(algorithm.save(), load_cfg={"actor": True}, strict=True)
    assert resumed.num_timesteps == 0
    assert set(losses) == {"value", "surrogate", "entropy"}
    assert all(math.isfinite(float(value)) for value in losses.values())
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad)
        for param in algorithm._raw_actor.sensor_encoder.parameters()
    )
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad)
        for param in algorithm._raw_actor.rnn.parameters()
    )
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad)
        for param in algorithm._raw_critic.rnn.parameters()
    )


def test_rsl_checkpoint_contract_round_trip_and_rejects_sb3_zip(tmp_path):
    env, cfg, source = make_algorithm()
    payload = source.save()
    assert "field_policy_contract" in payload
    assert type(payload["field_policy_contract"]["actor_config"]["distribution_cfg"]["init_std"]) is float
    assert validate_field_policy_contract(payload["field_policy_contract"]) == source.field_policy_contract

    checkpoint = tmp_path / "field_model.pt"
    torch.save(payload, checkpoint)
    loaded = load_rsl_checkpoint_payload(checkpoint, map_location="cpu")
    assert loaded["field_policy_contract"] == payload["field_policy_contract"]

    restored = RslFieldPPO.construct_algorithm(make_obs(), env, cfg, "cpu")
    restored.load(loaded, load_cfg=None, strict=True)
    assert restored.num_timesteps == payload["num_timesteps"]
    for key, tensor in source._raw_actor.state_dict().items():
        assert torch.equal(tensor, restored._raw_actor.state_dict()[key])
    for key, tensor in source._raw_critic.state_dict().items():
        assert torch.equal(tensor, restored._raw_critic.state_dict()[key])
    assert restored.field_policy_contract == source.field_policy_contract
    legacy_marker = copy.deepcopy(loaded)
    legacy_marker["field_policy_contract"]["encoder_config"]["class_name"] = "isaac_field.policy:SensorEncoder"
    restored.load(legacy_marker, load_cfg=None, strict=True)

    play_env = SyntheticFieldEnv()
    play_env.num_envs = 1
    play_cfg = make_cfg()
    play_cfg["num_steps_per_env"] = 1
    play = RslFieldPPO.construct_algorithm(make_obs(1, steps=1), play_env, play_cfg, "cpu")
    play.load(loaded, load_cfg=None, strict=True)

    changed_contract = copy.deepcopy(loaded)
    changed_contract["field_policy_contract"]["input_schema"] = "wrong-schema"
    with pytest.raises(ValueError, match="encoder schema|input_schema|field_policy_contract"):
        restored.load(changed_contract, load_cfg=None, strict=True)

    changed_goal = copy.deepcopy(loaded)
    changed_goal["field_policy_contract"]["observation_config"]["absolute_goal_coordinates"]["field_size_m"] = 5.0
    with pytest.raises(ValueError, match="observation config"):
        load_rsl_checkpoint_payload(_save_payload(tmp_path, changed_goal, "bad_goal.pt"))

    changed_action = copy.deepcopy(loaded)
    changed_action["field_policy_contract"]["action_config"]["schema"] = "wrong-action-schema"
    with pytest.raises(ValueError, match="action contract"):
        validate_field_policy_contract(changed_action["field_policy_contract"])

    sb3_zip = tmp_path / "legacy_sb3.zip"
    with zipfile.ZipFile(sb3_zip, "w") as archive:
        archive.writestr("data", "{}")
        archive.writestr("policy.pth", b"old sb3 weights")
    with pytest.raises(ValueError, match="SB3 .zip"):
        load_rsl_checkpoint_payload(sb3_zip)
    with pytest.raises(ValueError, match="field_policy_contract"):
        restored.load({"actor_state_dict": {}}, load_cfg=None, strict=True)


def test_legacy_task_contract_missing_failure_cost_matches_only_live_false():
    _env, _cfg, algorithm = make_algorithm()
    live_false = copy.deepcopy(algorithm.save()["field_policy_contract"])
    live_false["task_config"] = {
        "max_time": 60.0,
        "goto_reward_profile": "fast_stop_v1",
        "fast_stop_failure_remaining_time_cost": False,
    }
    old_contract = copy.deepcopy(live_false)
    old_contract["task_config"].pop("fast_stop_failure_remaining_time_cost")

    normalized = validate_field_policy_contract(old_contract, expected=live_false)
    assert normalized["task_config"] == live_false["task_config"]

    explicit_true = copy.deepcopy(live_false)
    explicit_true["task_config"]["fast_stop_failure_remaining_time_cost"] = True
    with pytest.raises(ValueError, match="differs from the live"):
        validate_field_policy_contract(explicit_true, expected=live_false)

    true_live = copy.deepcopy(live_false)
    true_live["task_config"]["fast_stop_failure_remaining_time_cost"] = True
    with pytest.raises(ValueError, match="differs from the live"):
        validate_field_policy_contract(old_contract, expected=true_live)

    changed_other_field = copy.deepcopy(live_false)
    changed_other_field["task_config"]["max_time"] = 61.0
    with pytest.raises(ValueError, match="differs from the live"):
        validate_field_policy_contract(old_contract, expected=changed_other_field)

    missing_other_field = copy.deepcopy(old_contract)
    missing_other_field["task_config"].pop("max_time")
    with pytest.raises(ValueError, match="differs from the live"):
        validate_field_policy_contract(missing_other_field, expected=live_false)


def test_official_runner_learn_zero_saves_inference_loadable_untrained_checkpoint(tmp_path):
    pytest.importorskip("isaaclab")
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from isaac_field.rsl_rl_cfg import FieldFastStopRslRunnerCfg
    from isaac_field.rsl_rl_inference import RslPredictionModel
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.body_stability import BodyStabilityConfig
    from isaac_field.task import Config
    from rsl_rl.runners import OnPolicyRunner
    from scripts.train_isaac_foundation_fresh import FAST_STOP_CONFIG

    class RunnerEnv(SyntheticFieldEnv):
        cfg = {}

        def __init__(self):
            self.field_cfg = Config(
                size=20.0,
                input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                max_time=60.0,
                **FAST_STOP_CONFIG,
            )
            self.front_guide_spec = FrontGuideSpec()
            self.body_stability_config = BodyStabilityConfig()

        def get_observations(self):
            return make_obs(self.num_envs)

    runner_cfg = FieldFastStopRslRunnerCfg()
    runner_cfg.num_steps_per_env = 4
    runner_cfg.algorithm.num_learning_epochs = 1
    runner_cfg.algorithm.num_mini_batches = 1
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, "5.5.1")
    runner = OnPolicyRunner(RunnerEnv(), runner_cfg.to_dict(), log_dir=str(tmp_path), device="cpu")

    assert isinstance(runner.alg, RslFieldPPO)
    assert runner.alg._raw_actor.sensor_encoder is runner.alg._raw_critic.sensor_encoder
    assert runner.alg.field_policy_contract["actor_config"]["distribution_cfg"]["std_range"] == [1e-6, 1e6]
    assert "task_config" in runner.alg.field_policy_contract
    assert "front_guide_spec" in runner.alg.field_policy_contract
    assert "body_stability_config" in runner.alg.field_policy_contract

    actor_before = {
        key: value.detach().clone()
        for key, value in runner.alg._raw_actor.state_dict().items()
    }
    critic_before = {
        key: value.detach().clone()
        for key, value in runner.alg._raw_critic.state_dict().items()
    }
    contract_before = copy.deepcopy(runner.alg.field_policy_contract)
    assert runner.alg.optimizer.state == {}

    # The upstream runner performs its final save even when the learning loop is empty.
    runner.learn(num_learning_iterations=0, init_at_random_ep_len=False)

    checkpoint_path = tmp_path / "model_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["iter"] == 0
    assert payload["num_timesteps"] == 0
    assert payload["infos"] is None
    assert payload["field_policy_contract"] == contract_before
    assert payload["optimizer_state_dict"]["state"] == {}
    for payload_key, before in (
        ("actor_state_dict", actor_before),
        ("critic_state_dict", critic_before),
    ):
        assert payload[payload_key].keys() == before.keys()
        assert all(torch.equal(payload[payload_key][name], before[name]) for name in before)
    assert runner.alg.optimizer.state == {}

    predictor = RslPredictionModel(checkpoint_path, device="cpu")
    observations = {
        key: np.zeros((1, *shape), dtype=np.float32)
        for key, shape in OBSERVATION_SHAPES.items()
    }
    actions, state = predictor.predict(
        observations,
        state=None,
        episode_start=np.ones(1, dtype=bool),
        deterministic=True,
    )
    assert actions.shape == (1, 4)
    assert np.isfinite(actions).all()
    assert len(state) == 2
    assert all(np.isfinite(value).all() for value in state)


def test_checkpoint_records_full_task_reconstruction_config():
    from dataclasses import asdict
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.body_stability import BodyStabilityConfig
    from isaac_field.task import Config
    from scripts.train_isaac_foundation_fresh import FAST_STOP_CONFIG

    env = SyntheticFieldEnv()
    env.field_cfg = Config(size=20.0, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                           max_time=60.0, **FAST_STOP_CONFIG)
    env.front_guide_spec = FrontGuideSpec()
    env.body_stability_config = BodyStabilityConfig()
    algorithm = RslFieldPPO.construct_algorithm(make_obs(), env, make_cfg(), "cpu")
    contract = algorithm.save()["field_policy_contract"]
    assert contract["task_config"] == asdict(env.field_cfg)
    assert contract["front_guide_spec"] == asdict(env.front_guide_spec)
    assert contract["body_stability_config"] == asdict(env.body_stability_config)


def _save_payload(tmp_path, payload, name):
    path = tmp_path / name
    torch.save(payload, path)
    return path
