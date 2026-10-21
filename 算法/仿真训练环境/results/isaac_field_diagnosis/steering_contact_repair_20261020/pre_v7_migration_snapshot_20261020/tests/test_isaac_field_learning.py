"""Learning orchestration tests use a fake adapter, never Isaac physics."""

import hashlib
import json
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from isaac_field import learning
from isaac_field.action_contract import action_config


class _FakeModel:
    def __init__(self):
        self.starts = []
        self.field_action_config = action_config()

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        self.starts.append((state, np.asarray(episode_start).copy(), deterministic))
        next_state = 1 if state is None else state + 1
        return np.zeros(4, dtype=np.float32), next_state


class _FakeAdapter:
    def __init__(self, engine):
        self.engine = engine
        self.cfg = SimpleNamespace(dt=0.1, max_time=1.0)
        self.task = SimpleNamespace(
            robot=np.zeros(5, dtype=np.float64),
            goal=np.asarray([1.0, 2.0], dtype=np.float64),
            scene=SimpleNamespace(xy=np.asarray([[0.25, 0.75], [1.25, 1.75]], dtype=np.float32)),
            damage_config={"metric": "stem-contact-v2", "track_geometry": "seven-rollers-per-side"},
            time=0.0,
            damaged=set(),
            collisions=0,
            _hold=0.0,
        )
        self.seeds = []
        self.options = []
        self._step = 0
        self._seed = 0

    def reset(self, *, seed=None, options=None):
        self._seed = int(seed)
        self.seeds.append(self._seed)
        self.options.append(options)
        self._step = 0
        self.task.robot[:] = 0.0
        self.task.time = 0.0
        self.task.damaged.clear()
        self.task.collisions = 0
        self.task._hold = 0.0
        return {"state": np.zeros(1, np.float32)}, {"mode": "goto", "terrain_seed": 4}

    def step(self, action):
        assert np.asarray(action).shape == (4,)
        self._step += 1
        self.task.robot[:] = [0.1 * self._step, 0.05 * self._step, 0.1 * self._step,
                              0.2 * self._step, 0.03 * self._step]
        self.task.time = self._step * 0.1
        self.task._hold = self._step * 0.05
        self.task.damaged = set(range(self._seed % 3))
        self.task.collisions = self._seed % 4
        if hasattr(self.engine, "command"):
            self.engine.command = np.asarray([[0.2 * self._step, 0.03 * self._step]], dtype=np.float32)
            self.engine.track_command = np.asarray([[0.2 * self._step, -0.2 * self._step]], dtype=np.float32)
        done = self._step == 2
        info = {
            "mode": "goto",
            "is_success": self._seed % 2 == 0,
            "damage": self._seed % 3,
            "damage_metric": "stem-contact-v2",
            "collisions": self._seed % 4 + 2,
            "distance": self._seed / 10.0,
            "reason": "goal_reached" if self._seed % 2 == 0 else "time_limit",
            "goal": [1.0, 2.0],
        }
        return {"state": np.zeros(1, np.float32)}, 0.25, done, False, info

    def close(self):
        self.engine.close()


def test_evaluate_model_records_raw_episode_metrics_and_resets_recurrent_state():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="mixed", goal_distance=None, goal_angle=3.14),
        action_config=action_config(),
        drive_config={"interface": "forward_and_differential_track_speed_v2"},
        observation_config={"schema": "state-last-command-v2", "state_dim": 44},
        damage_config={"metric": "stem-contact-v2", "track_geometry": "seven-rollers-per-side"},
        closed=False,
        close=lambda: None,
    )
    adapter = _FakeAdapter(engine)
    adapter.task.damage_config = engine.damage_config
    model = _FakeModel()
    model.field_drive_config = {"interface": "legacy"}
    model.field_observation_config = {"schema": "state-last-measured-v1", "state_dim": 44}
    report = learning.evaluate_model(
        adapter,
        model,
        (12, 13),
        goal_distance=None,
        goal_angle=np.pi,
        mode="mixed",
        options_factory=lambda seed: {"mode": "goto", "seed_note": seed},
    )

    assert report["reward"] == "raw_task_reward_distance_cost_zero"
    assert report["action_config"] == action_config()
    assert report["checkpoint_action_config"] == action_config()
    assert report["drive_config"] == engine.drive_config
    assert report["checkpoint_drive_config"] == model.field_drive_config
    assert report["observation_config"] == engine.observation_config
    assert report["checkpoint_observation_config"] == model.field_observation_config
    assert report["damage_config"] == adapter.task.damage_config
    assert report["damage_metric"] == "stem-contact-v2"
    assert report["checkpoint_damage_config"] is None
    assert report["checkpoint_damage_metric"] == "legacy/unrecorded"
    assert [row["damage_metric"] for row in report["rows"]] == ["stem-contact-v2"] * 2
    assert report["seeds"] == [12, 13]
    assert report["summary"] == {
        "episodes": 2,
        "successes": 1,
        "success_rate": 0.5,
        "mean_raw_return": 0.5,
        "mean_damage": 0.5,
        "mean_collisions": 2.5,
        "mean_final_distance": 1.25,
        "timed_out_episodes": 1,
    }
    assert [row["steps"] for row in report["rows"]] == [2, 2]
    assert [row["timed_out"] for row in report["rows"]] == [False, True]
    assert [row["raw_return"] for row in report["rows"]] == [0.5, 0.5]
    assert "trajectory_schema" not in report
    assert all("trajectory" not in row for row in report["rows"])
    assert all("reset_initial_condition" not in row for row in report["rows"])
    assert adapter.options == [{"mode": "goto", "seed_note": 12}, {"mode": "goto", "seed_note": 13}]
    assert model.starts[0][1].tolist() == [True]
    assert model.starts[2][1].tolist() == [True]
    assert model.starts[2][0] is None
    assert json.loads(json.dumps(report))["rows"][0]["success"] is True


def test_evaluate_model_records_checkpoint_damage_config_without_changing_eval_rule():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="goto", goal_distance=None, goal_angle=np.pi),
        action_config=action_config(),
        damage_config={"metric": "stem-contact-v2", "track_geometry": "current"},
        close=lambda: None,
    )
    adapter = _FakeAdapter(engine)
    adapter.task.damage_config = engine.damage_config
    model = _FakeModel()
    model.field_damage_config = {"metric": "root-protection-v1", "plant_geometry": "legacy"}

    report = learning.evaluate_model(adapter, model, (20,), goal_distance=None, goal_angle=np.pi)

    assert report["damage_config"] == engine.damage_config
    assert report["action_config"] == action_config()
    assert report["checkpoint_action_config"] == action_config()
    assert report["checkpoint_damage_config"] == model.field_damage_config
    assert report["checkpoint_damage_metric"] == "root-protection-v1"
    assert report["rows"][0]["damage_metric"] == "stem-contact-v2"
    # The old checkpoint is evaluated under the live task rule; its metadata is descriptive.
    assert model.field_damage_config["metric"] == "root-protection-v1"


def test_safe_goto_assessment_rejects_mixed_damage_metrics_but_keeps_legacy_reports_comparable():
    from scripts.improve_isaac_policy import _safe_goto_assessment

    summary = {
        "clean_successes": 0,
        "damaged_episodes": 0,
        "mean_damage": 0.0,
        "total_collisions": 0,
    }
    legacy_before = {"summary": dict(summary)}
    legacy_after = {"summary": dict(summary)}
    stem_v2 = {"damage_config": {"metric": "stem-contact-v2"}, "summary": dict(summary)}

    assert _safe_goto_assessment(legacy_before, legacy_after) == "no_improvement"
    assert _safe_goto_assessment(legacy_before, stem_v2) == "incomparable_damage_metric"
    mixed_rows = {
        "summary": dict(summary),
        "rows": [{"damage_metric": "root-protection-v1"}, {"damage_metric": "stem-contact-v2"}],
    }
    assert _safe_goto_assessment(legacy_before, mixed_rows) == "incomparable_damage_metric"
    assert _safe_goto_assessment(mixed_rows, mixed_rows) == "incomparable_damage_metric"
    conflicting_rows = {
        "damage_config": {"metric": "stem-contact-v2"},
        "summary": dict(summary),
        "rows": [{"damage_metric": "root-protection-v1"}],
    }
    assert _safe_goto_assessment(conflicting_rows, conflicting_rows) == "incomparable_damage_metric"
    unknown_metric = {"damage_metric": "unrecorded", "summary": dict(summary)}
    assert _safe_goto_assessment(unknown_metric, unknown_metric) == "incomparable_damage_metric"

    def v2_damage_config(**overrides):
        from isaac_field.task import Config, FieldTask

        config = FieldTask(Config(damage_penalty=20.0)).damage_config
        config.update(overrides)
        return config

    current_contract = {
        "drive_config": {"interface": "independent_left_right_track_speed_explicit_pi_v6", "velocity_gain": 0.5},
        "observation_config": {"schema": "sensor-minimal-v6", "state_dim": 33},
    }

    def v2_report(**overrides):
        report = {
            **current_contract,
            "damage_config": v2_damage_config(),
            "summary": dict(summary),
        }
        report.update(overrides)
        return report

    same_v2_before = v2_report(checkpoint_drive_config={"interface": "legacy"})
    same_v2_after = v2_report(checkpoint_drive_config={"interface": "another-old-contract"})
    assert _safe_goto_assessment(same_v2_before, same_v2_after) == "no_improvement"

    different_track_separation = v2_report(
        damage_config=v2_damage_config(track_center_separation_m=0.40)
    )
    assert _safe_goto_assessment(same_v2_before, different_track_separation) == "incomparable_damage_geometry"
    different_damage_penalty = v2_report(
        damage_config=v2_damage_config(penalty_per_plant=4.0)
    )
    assert _safe_goto_assessment(same_v2_before, different_damage_penalty) == "incomparable_damage_geometry"
    incomplete_v2 = v2_report(damage_config={"metric": "stem-contact-v2"})
    assert _safe_goto_assessment(incomplete_v2, incomplete_v2) == "incomparable_damage_geometry"

    different_current_drive = v2_report(
        drive_config={"interface": "track-v2", "velocity_gain": 0.6}
    )
    assert _safe_goto_assessment(same_v2_before, different_current_drive) == "incomparable_environment_contract"
    different_current_observation = v2_report(
        observation_config={"schema": "state-last-command-v3", "state_dim": 44}
    )
    assert _safe_goto_assessment(same_v2_before, different_current_observation) == "incomparable_environment_contract"


def test_evaluate_model_can_record_compact_pose_action_and_goal_trajectory():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="goto", goal_distance=None, goal_angle=np.pi),
        action_config=action_config(),
        drive_config={"interface": "forward_and_differential_track_speed_v2"},
        observation_config={"schema": "state-last-command-v2", "state_dim": 44},
        command=np.zeros((1, 2), dtype=np.float32),
        track_command=np.zeros((1, 2), dtype=np.float32),
        close=lambda: None,
    )
    report = learning.evaluate_model(
        _FakeAdapter(engine),
        _FakeModel(),
        (14,),
        goal_distance=None,
        goal_angle=np.pi,
        mode="goto",
        record_trajectories=True,
    )

    assert report["trajectory_schema"] == "isaac-field-compact-trajectory-v1"
    row = report["rows"][0]
    trajectory = row["trajectory"]
    assert len(trajectory) == row["steps"] == 2
    first = trajectory[0]
    assert first["step"] == 1 and first["time_s"] == pytest.approx(0.1)
    assert first["action_4"] == [0.0, 0.0, 0.0, 0.0]
    assert first["command_v_w"] == pytest.approx([0.2, 0.03])
    assert first["track_command_lr_mps"] == pytest.approx([0.2, -0.2])
    assert first["before"]["pose_xy_yaw"] == [0.0, 0.0, 0.0]
    assert first["after"]["pose_xy_yaw"] == pytest.approx([0.1, 0.05, 0.1])
    assert first["before"]["actual_v_w"] == [0.0, 0.0]
    assert first["after"]["actual_v_w"] == pytest.approx([0.2, 0.03])
    assert first["before"]["goal_robot_frame_m"] == [1.0, 2.0]
    assert first["before"]["goal_distance_m"] == pytest.approx(np.sqrt(5.0))
    assert first["after"]["goal_distance_m"] < first["before"]["goal_distance_m"]
    assert first["after"]["damage"] == 2
    assert first["after"]["hold_s"] == pytest.approx(0.05)
    initial = row["reset_initial_condition"]
    plant_xy = np.ascontiguousarray(_FakeAdapter(engine).task.scene.xy, dtype="<f4")
    expected_hash = hashlib.sha256(
        np.asarray(plant_xy.shape, dtype="<i8").tobytes() + plant_xy.tobytes()
    ).hexdigest()
    assert initial == {
        "seed": 14,
        "start_pose_xy_yaw": [0.0, 0.0, 0.0],
        "goal_xy": [1.0, 2.0],
        "plant_count": 2,
        "plant_xy_sha256": expected_hash,
    }
    assert row["trajectory_start_matches_reset"] is True
    final = trajectory[-1]
    assert final["terminated"] is True
    assert final["truncated"] is False
    assert final["reason"] == "goal_reached"
    assert "observation" not in first and "state_44_before" not in first
    assert json.loads(json.dumps(report))["rows"][0]["trajectory"][0]["action_4"] == [0, 0, 0, 0]


def test_heldout_eval_rejects_shaped_reward():
    engine = SimpleNamespace(
        action_config=action_config(),
        cfg=SimpleNamespace(distance_cost=0.1, mode="mixed"),
    )
    with pytest.raises(ValueError, match="distance_cost=0.0"):
        learning.evaluate_model(_FakeAdapter(engine), _FakeModel(), (5,))


def test_heldout_eval_rejects_checkpoint_with_old_action_semantics():
    engine = SimpleNamespace(
        action_config=action_config(),
        cfg=SimpleNamespace(distance_cost=0.0, mode="mixed"),
    )
    model = _FakeModel()
    model.field_action_config = {"schema": "forward-differential-v2"}

    with pytest.raises(ValueError, match="action contract is missing or incompatible"):
        learning.evaluate_model(_FakeAdapter(engine), model, (5,))


def test_heldout_eval_rejects_old_action_metadata_before_sb3_load(tmp_path):
    import zipfile

    engine = SimpleNamespace(
        action_config=action_config(),
        cfg=SimpleNamespace(distance_cost=0.0, mode="mixed"),
    )
    checkpoint = tmp_path / "old-action.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"field_action_config": {"schema": "forward-differential-v2"}}))

    with pytest.raises(ValueError, match="action contract is missing or incompatible"):
        learning.evaluate_model(_FakeAdapter(engine), checkpoint, (5,))


def test_train_curriculum_uses_disjoint_seeds_and_closes_contexts_sequentially(tmp_path, monkeypatch):
    active = {"engine": None}
    created = []
    stage_calls = []
    evaluations = []

    class FakeEngine:
        def __init__(self, kwargs):
            if active["engine"] is not None:
                raise AssertionError("overlapping physics contexts")
            self.kwargs = kwargs
            self.cfg = SimpleNamespace(distance_cost=kwargs["distance_cost"])
            self.closed = False
            active["engine"] = self
            created.append(self)

        def close(self):
            assert active["engine"] is self
            self.closed = True
            active["engine"] = None

    def env_factory(**kwargs):
        return FakeEngine(kwargs)

    def adapter_factory(engine):
        return _FakeAdapter(engine)

    def fake_evaluate(adapter, model, seeds, **kwargs):
        evaluations.append((model, tuple(seeds), kwargs))
        return {
            "seeds": list(seeds),
            "rows": [{"seed": int(seed)} for seed in seeds],
            "summary": {"success_rate": 0.25},
        }

    def fake_train_stage(env, checkpoint, timesteps, output, callback=None, **kwargs):
        assert active["engine"] is env
        stage_calls.append((env.kwargs, checkpoint, timesteps, callback, kwargs))
        path = tmp_path / f"stage-{len(stage_calls)}.zip"
        path.write_bytes(b"fake")
        return learning.StageTrainingResult(
            model=_FakeModel(),
            checkpoint=path,
            requested_timesteps=timesteps,
            trained_timesteps=128,
            total_timesteps=128 * len(stage_calls),
            seed=kwargs["seed"],
            goal_gain=kwargs["goal_gain"],
            n_steps=128,
            batch_size=kwargs["batch_size"],
            n_epochs=kwargs["n_epochs"],
        )

    start = tmp_path / "initial.zip"
    start.write_bytes(b"fake")
    monkeypatch.setattr(learning, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(learning, "train_stage", fake_train_stage)
    result = learning.train_curriculum(
        learning.Config(size=12, plants=4, terrain_enabled=False, validate_scene=False),
        tmp_path / "course",
        checkpoint=start,
        timesteps_per_stage=64,
        training_seed=7,
        heldout_seeds=(101, 102),
        env_factory=env_factory,
        adapter_factory=adapter_factory,
    )

    assert len(stage_calls) == 4
    assert len(created) == 6  # before-eval + four sequential stages + after-eval
    assert all(engine.closed for engine in created)
    assert active["engine"] is None
    assert [call[0]["mode"] for call in stage_calls] == ["goto", "goto", "goto", "mixed"]
    assert [call[0]["goal_distance"] for call in stage_calls] == [
        (0.45, 1.05),
        (1.0, 3.0),
        (3.0, 8.0),
        None,
    ]
    assert [call[0]["distance_cost"] for call in stage_calls] == [0.0] * 4
    assert [call[0]["seed"] for call in stage_calls] == [7, 8, 9, 10]
    assert evaluations[0][1] == evaluations[1][1] == (101, 102)
    assert result.summary["stage_seeds"] == [7, 8, 9, 10]
    assert result.summary["heldout_seeds"] == [101, 102]
    assert json.loads(result.summary_path.read_text(encoding="utf-8"))["claim_limit"].startswith(
        "Training completion alone"
    )


def test_train_curriculum_rejects_training_heldout_seed_overlap(tmp_path):
    with pytest.raises(ValueError, match="overlap training seeds"):
        learning.train_curriculum(
            learning.Config(size=12, plants=4, terrain_enabled=False),
            tmp_path,
            checkpoint=None,
            training_seed=7,
            heldout_seeds=(7, 99),
            evaluate_before_after=False,
            env_factory=lambda **kwargs: pytest.fail("env should not be created"),
            adapter_factory=lambda engine: engine,
        )


class _TinyRecurrentPolicyEnv:
    """Small real Gymnasium env with the migrated policy's Dict observation contract."""

    metadata = {"render_modes": []}

    def __new__(cls):
        import gymnasium as gym

        class Env(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Dict(
                    {
                        "lidar": gym.spaces.Box(0.0, 1.0, shape=(1, 400), dtype=np.float32),
                        "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 160, 88), dtype=np.float32),
                        "detections": gym.spaces.Box(0.0, 1.0, shape=(16, 8), dtype=np.float32),
                        "state": gym.spaces.Box(-1.0, 1.0, shape=(33,), dtype=np.float32),
                        "feedback": gym.spaces.Box(-np.inf, np.inf, shape=(37,), dtype=np.float32),
                    }
                )
                self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
                self.t = 0

            def _observation(self):
                phase = np.float32((self.t % 9) / 8.0)
                return {
                    "lidar": np.full((1, 400), 0.2 + 0.6 * phase, np.float32),
                    "depth": np.full((1, 160, 88), 0.1 + 0.7 * phase, np.float32),
                    "detections": np.full((16, 8), 0.1 + 0.5 * phase, np.float32),
                    "state": np.full((33,), -0.5 + phase, np.float32),
                    "feedback": np.zeros((37,), dtype=np.float32),
                }

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.t = 0
                return self._observation(), {}

            def step(self, action):
                self.t += 1
                reward = float(np.asarray(action, dtype=np.float32)[0])
                return self._observation(), reward, self.t % 31 == 0, False, {}

        return Env()


@pytest.mark.parametrize("pass_loaded_model", [False, True], ids=["checkpoint", "existing-model"])
def test_train_stage_overrides_loaded_sb3_hyperparameters_and_trains(
    tmp_path, monkeypatch, pass_loaded_model
):
    """Stage overrides must affect both zip-loaded and already-created models."""
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from isaac_field.policy import make_model

    env = DummyVecEnv([_TinyRecurrentPolicyEnv])
    env.action_config = action_config()
    env.drive_config = {"interface": "independent_left_right_track_speed_explicit_pi_v6", "velocity_gain": 20.0}
    env.observation_config = {
        "schema": "sensor-minimal-v6",
        "state_dim": 33,
        "depth_channels": 1,
        "depth_invalid_value": 0.0,
        "previous_command_semantics": "normalized_direct_track_speed_policy_command",
        "previous_command_indices": [13, 14, 15, 16],
    }
    env.damage_config = {
        "metric": "stem-contact-v2",
        "track_geometry": "seven-rollers-per-side",
    }
    monkeypatch.setattr(learning, "_make_adapter_factory", lambda: None)
    from isaac_field import training

    monkeypatch.setattr(training, "_sb3_wrapper", lambda value: value)
    source = make_model(
        env,
        seed=29,
        n_steps=128,
        batch_size=128,
        n_epochs=1,
        learning_rate=7e-3,
        gamma=0.91,
        goal_gain=1.0,
        log_std_init=-1.25,
        target_kl=0.005,
        lstm_hidden_size=16,
        net_arch={"pi": [16], "vf": [16]},
        device="cpu",
    )
    source_path = tmp_path / "source.zip"
    source.save(str(source_path))
    original_parameters = {
        name: value.detach().clone() for name, value in source.policy.state_dict().items()
    }
    cnn_parameter_names = [
        name
        for name in original_parameters
        if name.startswith("features_extractor.laser.") or name.startswith("features_extractor.depth.")
    ]
    assert cnn_parameter_names
    original_log_std = source.policy.log_std.detach().clone()
    loaded = RecurrentPPO.load(str(source_path), env=env, device="cpu")

    # Capture the distribution parameter at the first update boundary. A stage
    # override must not reinitialize the learned log_std from log_std_init.
    observed_initial_log_std = []
    original_learn = RecurrentPPO.learn

    def capture_log_std(model, *args, **kwargs):
        observed_initial_log_std.append(model.policy.log_std.detach().clone())
        return original_learn(model, *args, **kwargs)

    monkeypatch.setattr(RecurrentPPO, "learn", capture_log_std)
    checkpoint_arg = loaded if pass_loaded_model else source_path
    result = learning.train_stage(
        env,
        checkpoint_arg,
        128,
        tmp_path / f"trained-{pass_loaded_model}",
        seed=41,
        device="cpu",
        goal_gain=1.0,
        n_steps=128,
        batch_size=128,
        n_epochs=1,
        learning_rate=2.5e-4,
        gamma=0.83,
        target_kl=0.025,
        log_std_init=-8.0,
    )

    assert result.trained_timesteps == 128
    assert len(observed_initial_log_std) == 1
    torch.testing.assert_close(observed_initial_log_std[0], original_log_std)
    trained = RecurrentPPO.load(str(result.checkpoint), env=env, device="cpu")
    assert trained.field_drive_config == env.drive_config
    assert trained.field_action_config == action_config()
    assert trained.field_observation_config == env.observation_config
    assert trained.field_damage_config == env.damage_config
    assert trained.learning_rate == pytest.approx(2.5e-4)
    assert trained.lr_schedule(1.0) == pytest.approx(2.5e-4)
    assert trained.gamma == pytest.approx(0.83)
    assert trained.rollout_buffer.gamma == pytest.approx(0.83)
    assert trained.target_kl == pytest.approx(0.025)
    assert trained.policy.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert any(
        not torch.equal(original_parameters[name], value)
        for name, value in trained.policy.state_dict().items()
    )
    trained_parameters = trained.policy.state_dict()
    assert any(
        not torch.equal(original_parameters[name], trained_parameters[name])
        for name in cnn_parameter_names
    )

    # Loading for a stage is copy-on-load: it must leave the old candidate intact.
    unchanged = RecurrentPPO.load(str(source_path), env=env, device="cpu")
    assert unchanged.learning_rate == pytest.approx(7e-3)
    assert unchanged.gamma == pytest.approx(0.91)
    assert unchanged.target_kl == pytest.approx(0.005)
    for name, value in unchanged.policy.state_dict().items():
        torch.testing.assert_close(value, original_parameters[name])
    env.close()


@pytest.mark.parametrize("checkpoint_kind", ["path", "model"])
def test_train_stage_rejects_old_action_semantics_before_training(
    tmp_path, monkeypatch, checkpoint_kind,
):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from isaac_field.policy import make_model
    from isaac_field import training

    env = DummyVecEnv([_TinyRecurrentPolicyEnv])
    env.action_config = action_config()
    monkeypatch.setattr(training, "_sb3_wrapper", lambda value: value)
    model = make_model(
        env, seed=13, n_steps=128, batch_size=128, n_epochs=1, device="cpu",
    )
    model.field_action_config = {"schema": "forward-differential-v2"}
    if checkpoint_kind == "path":
        checkpoint = tmp_path / "old-action.zip"
        model.save(str(checkpoint))
    else:
        checkpoint = model

    with pytest.raises(ValueError, match="action contract is missing or incompatible"):
        learning.train_stage(
            env, checkpoint, 128, tmp_path / f"rejected-{checkpoint_kind}",
            device="cpu", n_steps=128, batch_size=128, n_epochs=1,
        )
    env.close()


def test_public_training_path_saves_damage_metric_metadata(tmp_path, monkeypatch):
    import sys

    fake_policy = ModuleType("isaac_field.policy")
    fake_policy.make_model = lambda *_args, **_kwargs: None
    fake_policy.validate_policy_spaces = lambda *_args: None
    fake_policy._validate_model_action_contract = lambda model: (
        None if getattr(model, "field_action_config", None) == action_config()
        else (_ for _ in ()).throw(ValueError("invalid action contract"))
    )
    monkeypatch.setitem(sys.modules, "isaac_field.policy", fake_policy)
    from isaac_field import training

    damage_config = {"metric": "stem-contact-v2", "track_geometry": "seven-rollers-per-side"}
    env = SimpleNamespace(
        num_envs=1,
        action_config=action_config(),
        drive_config={"interface": "track-v2"},
        observation_config={"schema": "state-last-command-v2"},
        damage_config=damage_config,
        observation_space=object(),
        action_space=object(),
    )

    class Model:
        device = "cpu"
        num_timesteps = 0
        field_action_config = action_config()

        def learn(self, *, total_timesteps, **_):
            self.num_timesteps += int(total_timesteps)

        def save(self, path):
            self.saved_damage_config = dict(self.field_damage_config)
            self.saved_action_config = dict(self.field_action_config)
            from pathlib import Path

            Path(f"{path}.zip").write_bytes(b"checkpoint")

    model = Model()
    monkeypatch.setattr(training, "_sb3_wrapper", lambda value: value)
    monkeypatch.setattr(training, "validate_policy_spaces", lambda *_: None)
    monkeypatch.setattr(training, "make_model", lambda *_args, **_kwargs: model)

    result = training.train(
        env,
        tmp_path / "native-training",
        timesteps=4,
        device="cpu",
        config=training.TrainingConfig(n_steps=2, batch_size=2, n_epochs=1),
    )

    assert result.actual_timesteps == 4
    assert model.field_damage_config == damage_config
    assert model.saved_damage_config == damage_config
    assert model.field_action_config == action_config()
    assert model.saved_action_config == action_config()

