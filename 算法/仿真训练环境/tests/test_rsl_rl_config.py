"""CPU-only contract checks for the RSL-RL task boundary."""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv

from isaac_field.rsl_rl_cfg import FieldFastStopRslEnvCfg, FieldFastStopRslRunnerCfg
from isaac_field.rsl_rl_env import IsaacFieldRslEnv, expose_policy_observation_space


def test_fast_stop_v2_rsl_configs_keep_the_strict_task_contract():
    env_cfg = FieldFastStopRslEnvCfg()
    runner_cfg = FieldFastStopRslRunnerCfg()

    assert env_cfg.mode == "goto"
    assert env_cfg.task.max_time == 60.0
    assert env_cfg.task.plants == 0
    assert env_cfg.task.terrain_enabled is False
    assert env_cfg.task.rock_density == 0.0
    assert env_cfg.task.damage_metric == "stem-contact-v2"
    assert env_cfg.task.goto_reward_profile == "fast_stop_v2"
    assert env_cfg.task.goal_tolerance == 0.4
    assert env_cfg.task.goal_stop_speed == 0.05
    assert env_cfg.task.goal_stop_angular_speed == 0.1
    assert env_cfg.task.goal_hold == 0.5
    assert env_cfg.task.fast_stop_failure_remaining_time_cost is True
    assert env_cfg.task.input_schema == "sensor-absolute-goal-v1"
    assert env_cfg.is_finite_horizon is True
    assert env_cfg.scene.num_envs == 4
    assert env_cfg.front_guide_spec is not None
    assert env_cfg.body_stability_config["tilt_weight"] == 0.0
    assert env_cfg.body_stability_config["angular_rate_weight"] == 0.0

    leaves = ["lidar", "depth", "detections", "state", "feedback"]
    assert runner_cfg.init_at_random_ep_len is False
    assert runner_cfg.num_steps_per_env == 128
    assert runner_cfg.max_iterations == 384
    assert runner_cfg.save_interval == 16
    assert runner_cfg.obs_groups == {"actor": leaves, "critic": leaves}
    assert runner_cfg.actor.class_name == "isaac_field.rsl_rl_policy:RslFieldRNNModel"
    assert runner_cfg.critic.class_name == "isaac_field.rsl_rl_policy:RslFieldRNNModel"
    assert runner_cfg.actor.input_schema == runner_cfg.critic.input_schema == "sensor-absolute-goal-v1"
    assert runner_cfg.actor.encoder_schema == runner_cfg.critic.encoder_schema == "depth64-lidar-single1d-absolute-goal-v4"
    assert runner_cfg.actor.goal_gain == runner_cfg.critic.goal_gain == 1.0
    assert runner_cfg.actor.cache_inference is runner_cfg.critic.cache_inference is True
    assert runner_cfg.actor.observation_shapes == runner_cfg.critic.observation_shapes == {
        "lidar": [1, 400],
        "depth": [1, 160, 88],
        "detections": [16, 8],
        "state": [33],
        "feedback": [37],
    }
    assert runner_cfg.actor.rnn_type == runner_cfg.critic.rnn_type == "lstm"
    assert runner_cfg.actor.rnn_hidden_dim == runner_cfg.critic.rnn_hidden_dim == 512
    assert runner_cfg.actor.rnn_num_layers == runner_cfg.critic.rnn_num_layers == 1
    assert runner_cfg.actor.hidden_dims == runner_cfg.critic.hidden_dims == [64]
    assert runner_cfg.actor.activation == runner_cfg.critic.activation == "tanh"
    assert runner_cfg.actor.distribution_cfg.init_std == 0.05
    assert runner_cfg.actor.distribution_cfg.std_type == "log"
    assert runner_cfg.algorithm.class_name == "isaac_field.rsl_rl_policy:RslFieldPPO"
    assert runner_cfg.algorithm.share_cnn_encoders is True
    assert runner_cfg.algorithm.max_grad_norm == 0.5
    assert runner_cfg.algorithm.value_loss_coef == 0.5
    assert runner_cfg.algorithm.use_clipped_value_loss is False
    assert runner_cfg.algorithm.entropy_coef == 0.0
    assert runner_cfg.algorithm.gamma == 1.0
    assert runner_cfg.algorithm.lam == 0.99
    assert runner_cfg.algorithm.clip_param == 0.2
    assert runner_cfg.algorithm.learning_rate == 3.0e-4
    assert runner_cfg.algorithm.num_learning_epochs == 5
    assert runner_cfg.algorithm.schedule == "adaptive"
    assert runner_cfg.algorithm.desired_kl == 0.01


def test_hydra_can_override_v2_actor_init_std_to_point_three():
    import isaac_field.tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(
        "Isaac-Field-FastStop-Rsl-Direct",
        "rsl_rl_cfg_entry_point",
        overrides=["agent.actor.distribution_cfg.init_std=0.3"],
    )

    assert env_cfg.task.goto_reward_profile == "fast_stop_v2"
    assert env_cfg.task.fast_stop_failure_remaining_time_cost is True
    assert agent_cfg.actor.distribution_cfg.init_std == 0.3
    assert agent_cfg.algorithm.entropy_coef == 0.0


def test_task_registration_keeps_legacy_and_selects_rsl_for_new_task():
    import isaac_field.tasks  # noqa: F401

    legacy = gym.spec("Isaac-Field-Direct")
    new = gym.spec("Isaac-Field-FastStop-Rsl-Direct")

    assert legacy.entry_point == "isaac_field.env:IsaacFieldEnv"
    assert "default_agent" not in legacy.kwargs
    assert new.entry_point == "isaac_field.rsl_rl_env:IsaacFieldRslEnv"
    assert new.kwargs["default_agent"] == "rsl_rl"
    assert new.kwargs["rsl_rl_cfg_entry_point"] == (
        "isaac_field.rsl_rl_cfg:FieldFastStopRslRunnerCfg"
    )


def test_space_factory_promotes_policy_dict_without_creating_an_environment():
    policy_space = gym.spaces.Dict(
        {
            "lidar": gym.spaces.Box(0.0, 1.0, (1, 400), np.float32),
            "depth": gym.spaces.Box(-1.0, 1.0, (1, 160, 88), np.float32),
            "detections": gym.spaces.Box(-1.0, 1.0, (16, 8), np.float32),
            "state": gym.spaces.Box(-1.0, 1.0, (33,), np.float32),
            "feedback": gym.spaces.Box(-np.inf, np.inf, (37,), np.float32),
        }
    )
    env = SimpleNamespace(
        num_envs=4,
        single_observation_space=gym.spaces.Dict({"policy": policy_space}),
    )

    expose_policy_observation_space(env)

    assert env.single_observation_space is policy_space
    assert set(env.single_observation_space.spaces) == {
        "lidar", "depth", "detections", "state", "feedback"
    }
    assert env.observation_space.spaces["depth"].shape == (4, 1, 160, 88)
    assert env.observation_space.spaces["state"].shape == (4, 33)


def test_observation_boundary_preserves_direct_env_noise_before_promoting_policy():
    assert issubclass(IsaacFieldRslEnv, DirectRLEnv)
    env = object.__new__(IsaacFieldRslEnv)
    source = {key: torch.zeros((4, 1), dtype=torch.float32) for key in (
        "lidar", "depth", "detections", "state", "feedback"
    )}
    env.cfg = SimpleNamespace(observation_noise_model=True)
    env._get_observations = lambda: {"policy": source}
    env._observation_noise_model = lambda obs: {key: value + 1.0 for key, value in obs.items()}

    result = env._compute_observations()

    assert set(result) == set(source)
    assert all(torch.equal(value, torch.ones_like(value)) for value in result.values())
