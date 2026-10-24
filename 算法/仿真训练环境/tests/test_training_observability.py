"""Reward accounting and lightweight pairing must preserve task semantics."""
from copy import deepcopy

import numpy as np
import pytest

from isaac_field.task import Config, FieldTask


def _task(**kwargs):
    cfg = Config(size=12, plants=0, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=4, depth_height=3, **kwargs)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=4, options={"mode": "goto", "start": [3, 3, 0],
                                  "goal": [4, 3], "random_start": False})

    def capture(kind):
        if kind == "lidar":
            return {"ranges": np.full(cfg.lidar_rays, cfg.lidar_range, np.float32),
                    "valid": np.zeros(cfg.lidar_rays, dtype=bool)}
        if kind == "depth":
            return {"values": np.full((cfg.depth_height, cfg.depth_width), cfg.depth_range, np.float32),
                    "valid": np.zeros((cfg.depth_height, cfg.depth_width), dtype=bool)}
        return {"detections": []}

    task.attach_sensors(capture)
    task.initialize_sensors()
    return task


@pytest.mark.parametrize("budget", [None, 5.0])
def test_component_accounting_does_not_change_progress_or_penalties(budget):
    task = _task(goal_progress_budget=budget)
    before = task.robot.copy()
    task.robot[0] += 0.2
    _, reward, terminated, truncated, info = task.finish(
        np.zeros(4, np.float32), before, 1.0, True, True,
    )
    progress = (task.cfg.progress_weight if budget is None else budget) * (1.0 - 0.8)
    expected = -task.cfg.time_penalty * task.cfg.dt - task.cfg.collision_penalty
    expected -= 0.1 * task.cfg.dt
    expected += progress
    assert reward == pytest.approx(expected, abs=1e-12)
    assert sum(info["reward_components"].values()) == pytest.approx(reward, abs=1e-12)
    assert info["reward_components"]["collision"] == -task.cfg.collision_penalty
    assert not terminated and not truncated
    # Returned infos are snapshots, not aliases to the next transition's terms.
    saved = deepcopy(info["reward_components"])
    task.reward_components["time"] = 999
    assert info["reward_components"] == saved


def test_success_hold_and_timeout_remain_distinct_in_reward_accounting():
    task = _task(max_time=1.0, goal_progress_budget=5.0)
    task.robot[:2] = task.goal
    rewards = []
    for _ in range(6):
        _, reward, terminated, truncated, info = task.finish(
            np.zeros(4, np.float32), task.robot.copy(), 0.0,
        )
        rewards.append(reward)
        assert sum(info["reward_components"].values()) == pytest.approx(reward)
    assert terminated and not truncated and info["reason"] == "goal_reached"
    assert info["reward_components"]["goal_success"] == 5.0
    assert sum(rewards) == pytest.approx(5.0 - task.cfg.time_penalty * 0.3)

    timeout_task = _task(max_time=0.05)
    _, reward, terminated, truncated, info = timeout_task.finish(
        np.zeros(4, np.float32), timeout_task.robot.copy(), 1.0,
    )
    assert not terminated and truncated
    assert info["reason"] == "time_limit"
    assert info["reward_components"]["goal_success"] == 0.0
    assert sum(info["reward_components"].values()) == pytest.approx(reward)


def test_lightweight_initial_condition_recording_without_trajectory():
    # Reuse the established orchestration fixture, including its real task-like
    # state transitions; this mode must retain pairing evidence without pixels
    # or the full trajectory payload.
    from test_isaac_field_learning import _FakeAdapter, _FakeModel
    from isaac_field import learning
    from isaac_field.action_contract import action_config
    from types import SimpleNamespace

    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="goto", goal_distance=None, goal_angle=3.14),
        action_config=action_config(), close=lambda: None,
    )
    report = learning.evaluate_model(
        _FakeAdapter(engine), _FakeModel(), (12, 13), record_initial_conditions=True,
    )
    for row in report["rows"]:
        assert row["trajectory_start_matches_reset"] is True
        assert len(row["reset_initial_condition"]["plant_xy_sha256"]) == 64
        assert "trajectory" not in row


@pytest.mark.parametrize(
    ("gae_lambda", "n_steps", "num_envs", "timesteps", "batch_size", "parent_init_std"),
    [
        (0.99, 128, 1, 128, 128, None),
        (0.83, 128, 1, 128, 128, None),
        (0.91, 16, 2, 32, 32, None),
        (0.91, 16, 2, 32, 32, 0.05),
    ],
)
def test_train_stage_forwards_rollout_and_parent_distribution_to_official_rsl_runner(
    tmp_path, monkeypatch, gae_lambda, n_steps, num_envs, timesteps, batch_size, parent_init_std,
):
    from types import SimpleNamespace

    from isaac_field import learning
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.policy_contract import BACKEND_IDENTITY, OBSERVATION_SHAPES
    from isaac_field.action_contract import action_config as make_action_config
    from policy_contract_helpers import observation_config as make_observation_config

    class FakeEnv:
        num_actions = 4
        device = "cpu"
        backend_identity = dict(BACKEND_IDENTITY)
        field_cfg = SimpleNamespace(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
        observation_config = make_observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
        action_config = make_action_config()
        drive_config = {"interface": "cpu-rsl-test"}
        damage_config = {"metric": "cpu-rsl-test"}
        single_observation_space = SimpleNamespace(
            spaces={key: object() for key in OBSERVATION_SHAPES}
        )
        obs_buf = None

        def __init__(self, num_envs):
            self.num_envs = num_envs
            self.reset_count = 0

        @property
        def unwrapped(self):
            return self

        def reset(self):
            self.reset_count += 1
            self.obs_buf = {key: object() for key in OBSERVATION_SHAPES}

    env = FakeEnv(num_envs=num_envs)
    runner_configs = []
    runner_loads = []
    checkpoint = None
    if parent_init_std is not None:
        from isaac_field import rsl_rl_policy

        checkpoint = tmp_path / "parent.pt"
        checkpoint.write_bytes(b"parent checkpoint placeholder")
        distribution = {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": parent_init_std,
            "std_type": "log",
            "std_range": [1e-6, 1e6],
        }
        monkeypatch.setattr(
            rsl_rl_policy,
            "load_rsl_checkpoint_payload",
            lambda path, **kwargs: {
                "field_policy_contract": {
                    "lstm": {"hidden_dim": 512},
                    "distribution": distribution,
                }
            },
        )

    class FakeVecEnvWrapper:
        def __init__(self, wrapped, *, clip_actions):
            assert clip_actions == 1.0
            self.env = wrapped
            self.num_envs = wrapped.num_envs
            self.device = wrapped.device
            # Mirror the upstream wrapper's reset during construction. Fresh
            # DirectRLEnv instances have a valid observation space before obs_buf.
            self.env.reset()

    class FakeOnPolicyRunner:
        def __init__(self, wrapped, cfg, *, log_dir, device):
            self.env = wrapped
            self.cfg = cfg
            runner_configs.append(cfg)
            self.alg = SimpleNamespace(num_timesteps=0)

        def learn(self, *, num_learning_iterations, init_at_random_ep_len):
            assert init_at_random_ep_len is False
            self.alg.num_timesteps += (
                int(num_learning_iterations)
                * self.env.num_envs
                * int(self.cfg["num_steps_per_env"])
            )

        def save(self, path):
            from pathlib import Path
            Path(path).write_bytes(b"fake official RSL checkpoint")

        def load(self, path, *, load_cfg, map_location):
            runner_loads.append((path, load_cfg, map_location))

    import isaaclab_rl.rsl_rl as isaaclab_rsl
    import rsl_rl.runners as rsl_runners

    monkeypatch.setattr(isaaclab_rsl, "RslRlVecEnvWrapper", FakeVecEnvWrapper)
    monkeypatch.setattr(isaaclab_rsl, "handle_deprecated_rsl_rl_cfg", lambda cfg, version: cfg)
    monkeypatch.setattr(rsl_runners, "OnPolicyRunner", FakeOnPolicyRunner)

    output = tmp_path / "trained.pt"
    result = learning.train_stage(
        env, checkpoint, timesteps, output, device="cpu", n_steps=n_steps, n_epochs=1,
        batch_size=batch_size, gae_lambda=gae_lambda,
    )

    assert runner_configs[0]["algorithm"]["lam"] == gae_lambda
    assert runner_configs[0]["num_steps_per_env"] == n_steps
    assert runner_configs[0]["algorithm"]["num_mini_batches"] == 1
    assert result.trained_timesteps == result.total_timesteps == timesteps
    assert result.n_steps == n_steps
    assert result.batch_size == batch_size
    if parent_init_std is not None:
        assert runner_configs[0]["actor"]["distribution_cfg"]["init_std"] == parent_init_std
        assert len(runner_loads) == 1
    else:
        assert runner_configs[0]["actor"]["distribution_cfg"]["init_std"] == pytest.approx(
            0.1353352832366127
        )
    assert result.checkpoint == output
    assert output.read_bytes() == b"fake official RSL checkpoint"
    assert env.reset_count == 1


@pytest.mark.parametrize("shape_mismatch", [False, True])
def test_weights_transfer_records_legacy_parent_and_rejects_shape_mismatch(
    tmp_path, monkeypatch, shape_mismatch,
):
    from types import SimpleNamespace

    import torch

    from isaac_field import learning, rsl_rl_policy
    from isaac_field.action_contract import action_config as make_action_config
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.policy_contract import BACKEND_IDENTITY, OBSERVATION_SHAPES
    from policy_contract_helpers import observation_config as make_observation_config

    live_observation_config = make_observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    compatible_contract = {
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "encoder_config": {"schema": "test-encoder"},
        "observation_shapes": live_observation_config["observation_shapes"],
        "observation_config": live_observation_config,
        "action_config": make_action_config(),
        "actor_config": {"class_name": "test-actor"},
        "critic_config": {"class_name": "test-critic"},
        "obs_groups": {"actor": list(OBSERVATION_SHAPES), "critic": list(OBSERVATION_SHAPES)},
        "lstm": {"hidden_dim": 512, "num_layers": 1},
        "heads": {"action_dim": 4},
        "distribution": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.05,
            "std_type": "log",
            "std_range": [1e-6, 1e6],
        },
        "action_dim": 4,
    }
    source_contract = {
        **compatible_contract,
        "task_config": {"physics": "newton"},
        # Historical Newton checkpoints without identity metadata are retained
        # as legacy_unrecorded lineage during an explicit weights transfer.
    }
    if shape_mismatch:
        source_contract["action_dim"] = 3
    source_payload = {
        "field_policy_contract": source_contract,
        "actor_state_dict": {"weight": torch.tensor([1.0])},
        "critic_state_dict": {"weight": torch.tensor([2.0])},
    }
    monkeypatch.setattr(
        rsl_rl_policy, "load_rsl_checkpoint_payload",
        lambda path, **kwargs: source_payload,
    )

    class FakeEnv:
        num_envs = 1
        num_actions = 4
        device = "cpu"
        backend_identity = dict(BACKEND_IDENTITY)
        field_cfg = SimpleNamespace(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
        observation_config = live_observation_config
        action_config = make_action_config()
        drive_config = {"interface": "cpu-transfer-test"}
        single_observation_space = SimpleNamespace(
            spaces={key: object() for key in OBSERVATION_SHAPES}
        )
        obs_buf = None

        @property
        def unwrapped(self):
            return self

        def reset(self):
            self.obs_buf = {key: object() for key in OBSERVATION_SHAPES}

    env = FakeEnv()

    class FakeVecEnvWrapper:
        def __init__(self, wrapped, *, clip_actions):
            self.env = wrapped
            self.num_envs = wrapped.num_envs
            self.device = wrapped.device
            wrapped.reset()

    actor_loads, critic_loads = [], []

    class FakeAlgorithm:
        field_policy_contract = {
            **compatible_contract,
            **BACKEND_IDENTITY,
            "task_config": {"physics": "physx"},
        }

        def __init__(self):
            self._raw_actor = SimpleNamespace(
                load_state_dict=lambda state, strict: actor_loads.append((state, strict))
            )
            self._raw_critic = SimpleNamespace(
                load_state_dict=lambda state, strict: critic_loads.append((state, strict))
            )
            self.optimizer = SimpleNamespace(state={"old": "adam"})
            self.num_timesteps = 999

    class FakeOnPolicyRunner:
        def __init__(self, wrapped, cfg, *, log_dir, device):
            self.env = wrapped
            self.cfg = cfg
            self.alg = FakeAlgorithm()
            self.learned = False

        def learn(self, *, num_learning_iterations, init_at_random_ep_len):
            self.learned = True
            self.alg.num_timesteps += (
                int(num_learning_iterations)
                * self.env.num_envs
                * int(self.cfg["num_steps_per_env"])
            )

        def save(self, path):
            torch.save({"num_timesteps": self.alg.num_timesteps}, path)

    import isaaclab_rl.rsl_rl as isaaclab_rsl
    import rsl_rl.runners as rsl_runners

    monkeypatch.setattr(isaaclab_rsl, "RslRlVecEnvWrapper", FakeVecEnvWrapper)
    monkeypatch.setattr(isaaclab_rsl, "handle_deprecated_rsl_rl_cfg", lambda cfg, version: cfg)
    monkeypatch.setattr(rsl_runners, "OnPolicyRunner", FakeOnPolicyRunner)

    parent = tmp_path / "historical_newton.pt"
    parent.write_bytes(b"historical RSL source checkpoint")
    output = tmp_path / "physx_transfer.pt"
    if shape_mismatch:
        with pytest.raises(ValueError, match="model structure differs"):
            learning.train_stage(
                env, parent, 128, output, device="cpu", batch_size=128,
                n_epochs=1, weights_transfer=True,
            )
        assert not output.exists()
        assert actor_loads == critic_loads == []
        return

    result = learning.train_stage(
        env, parent, 128, output, device="cpu", batch_size=128,
        n_epochs=1, weights_transfer=True,
    )

    saved = torch.load(output, map_location="cpu", weights_only=True)
    transfer = saved["training_initialization"]
    assert actor_loads[0][1] is critic_loads[0][1] is True
    assert result.trained_timesteps == result.total_timesteps == 128
    assert transfer["mode"] == "weights_transfer"
    assert transfer["parent_backend_identity"] == {
        "simulator": "legacy_unrecorded",
        "physics_backend": "legacy_unrecorded",
        "learning_backend": "legacy_unrecorded",
    }
    assert transfer["target_backend_identity"] == BACKEND_IDENTITY
    assert transfer["parent_task_config"] == {"physics": "newton"}
    assert transfer["optimizer"] == "reinitialized"
    assert transfer["timestep_counter"] == "reset_to_zero"
