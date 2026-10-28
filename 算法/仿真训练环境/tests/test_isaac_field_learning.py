"""Learning orchestration tests use a fake adapter, never Isaac physics."""

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

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


def test_evaluate_model_copies_goal_stop_metrics_and_summarizes_success_time():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="goto", goal_distance=None, goal_angle=np.pi),
        action_config=action_config(),
        drive_config={"interface": "current"},
        observation_config={"schema": "sensor-minimal-v6"},
        damage_config={"metric": "stem-contact-v2"},
        close=lambda: None,
    )
    adapter = _FakeAdapter(engine)
    adapter.task.damage_config = engine.damage_config
    base_step = adapter.step
    stop_row = {
        "schema": "goal_stop_v1", "profile": "fast_stop_v1",
        "linear_speed_m_s": 0.04, "angular_speed_rad_s": 0.08, "hold_s": 0.5,
        "stop_speed_limit_m_s": 0.05, "stop_angular_speed_limit_rad_s": 0.1,
        "hold_required_s": 0.5, "distance_tolerance_m": 0.4,
        "speed_source": "root_com_vel_w",
    }

    def step(action):
        observation, reward, terminated, truncated, info = base_step(action)
        info["goal_stop"] = dict(stop_row)
        return observation, reward, terminated, truncated, info

    adapter.step = step
    report = learning.evaluate_model(adapter, _FakeModel(), (12,), goal_distance=None, goal_angle=np.pi)

    row = report["rows"][0]
    assert row["goal_stop"] == stop_row
    assert row["episode_time_s"] == pytest.approx(0.2)
    assert report["summary"]["mean_success_time_s"] == pytest.approx(0.2)
    assert report["summary"]["median_success_time_s"] == pytest.approx(0.2)


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
        "drive_config": {"interface": "independent_left_right_track_speed_coupled_pi_v7", "velocity_gain": 0.5},
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


def test_heldout_eval_requires_explicit_legacy_switch_for_sb3_zip(tmp_path):
    import zipfile

    engine = SimpleNamespace(
        action_config=action_config(),
        cfg=SimpleNamespace(distance_cost=0.0, mode="mixed"),
    )
    checkpoint = tmp_path / "old-action.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"field_action_config": {"schema": "forward-differential-v2"}}))

    with pytest.raises(ValueError, match="SB3 policy loading is legacy-only"):
        learning.evaluate_model(_FakeAdapter(engine), checkpoint, (5,))


@pytest.mark.parametrize("gamma", [0.995, 1.0])
def test_train_curriculum_uses_disjoint_seeds_and_closes_contexts_sequentially(tmp_path, monkeypatch, gamma):
    from isaac_field import runtime

    monkeypatch.setattr(runtime, "launch_app", lambda **_kwargs: nullcontext())
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
        gamma=gamma,
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
    assert [call[4]["gamma"] for call in stage_calls] == [gamma] * 4
    assert evaluations[0][1] == evaluations[1][1] == (101, 102)
    assert result.summary["stage_seeds"] == [7, 8, 9, 10]
    assert result.summary["heldout_seeds"] == [101, 102]
    assert json.loads(result.summary_path.read_text(encoding="utf-8"))["claim_limit"].startswith(
        "Training completion alone"
    )


def test_train_curriculum_rejects_training_heldout_seed_overlap(tmp_path, monkeypatch):
    from isaac_field import runtime

    monkeypatch.setattr(runtime, "launch_app", lambda **_kwargs: nullcontext())
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


@pytest.mark.parametrize("checkpoint_kind", ["zip", "model"])
def test_rsl_train_stage_rejects_legacy_checkpoint_or_model_object(tmp_path, checkpoint_kind):
    if checkpoint_kind == "zip":
        checkpoint = tmp_path / "legacy.zip"
        checkpoint.write_bytes(b"legacy SB3 archive")
        expected = ValueError
    else:
        checkpoint = SimpleNamespace()
        expected = TypeError

    with pytest.raises(expected, match=r"legacy-only|explicit \.pt checkpoint path"):
        learning.train_stage(
            object(), checkpoint, 128, tmp_path / "new-stage.pt", device="cpu",
            n_steps=128, batch_size=128, n_epochs=1,
        )


def test_default_training_configuration_keeps_512_recurrent_width():
    from isaac_field.training import TrainingConfig

    assert TrainingConfig().lstm_hidden_size == 512


def test_rsl_train_stage_rejects_legacy_pickle_checkpoint(tmp_path):
    checkpoint = tmp_path / "legacy.pkl"
    checkpoint.write_bytes(b"legacy SB3 pickle")
    with pytest.raises(ValueError, match="legacy-only"):
        learning.train_stage(
            object(), checkpoint, 128, tmp_path / "new-stage.pt", device="cpu",
            n_steps=128, batch_size=128, n_epochs=1,
        )


def test_public_training_path_saves_damage_metric_metadata(tmp_path, monkeypatch):
    from isaac_field import training
    calls = {}

    def fake_train_stage(env, checkpoint, timesteps, output, callback=None, **kwargs):
        calls.update(env=env, checkpoint=checkpoint, timesteps=timesteps, output=output, kwargs=kwargs)
        return SimpleNamespace(model=object(), checkpoint=output, trained_timesteps=timesteps)

    monkeypatch.setattr(learning, "train_stage", fake_train_stage)
    result = training.train(
        object(), tmp_path / "native-training", timesteps=256, device="cpu",
        config=training.TrainingConfig(n_epochs=1),
    )

    assert calls["checkpoint"] is None
    assert calls["timesteps"] == 256
    assert calls["output"] == tmp_path / "native-training.pt"
    assert calls["kwargs"]["lstm_hidden_size"] == 512
    assert result.actual_timesteps == 256


def test_public_training_helper_forwards_configured_rollout_length(tmp_path, monkeypatch):
    from isaac_field import training

    calls = {}

    def fake_train_stage(env, checkpoint, timesteps, output, callback=None, **kwargs):
        calls.update(checkpoint=checkpoint, timesteps=timesteps, output=output, kwargs=kwargs)
        return SimpleNamespace(model=object(), checkpoint=output, trained_timesteps=timesteps)

    monkeypatch.setattr(learning, "train_stage", fake_train_stage)
    result = training.train(
        object(), tmp_path / "short-rollout", timesteps=32, device="cpu",
        config=training.TrainingConfig(n_steps=16, batch_size=32, n_epochs=1),
    )

    assert calls["checkpoint"] is None
    assert calls["timesteps"] == 32
    assert calls["kwargs"]["n_steps"] == 16
    assert calls["kwargs"]["batch_size"] == 32
    assert result.actual_timesteps == 32

