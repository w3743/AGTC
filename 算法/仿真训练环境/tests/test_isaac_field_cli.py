from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import types

import numpy as np

from isaac_field.__main__ import main


@dataclass
class _Config:
    plants: int = 0
    terrain_enabled: bool = True
    max_time: float = 30.0
    damage_penalty: float = 4.0
    goal_progress_budget: float | None = None


def _run_cli_viewer(monkeypatch, tmp_path, args):
    observed = {}

    class FakeEnv:
        def __init__(self):
            visualizer = types.SimpleNamespace(cfg=types.SimpleNamespace(visualizer_type="viser"))
            self.sim = types.SimpleNamespace(visualizers=[visualizer])
            self.drive_config = {"velocity_gain": 0.5}
            self.observation_config = {"schema": "state-last-command-v2"}
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAdapter:
        def __init__(self, env):
            self.engine = env
            self.cfg = types.SimpleNamespace(size=100.0)
            self.task = types.SimpleNamespace(scene=types.SimpleNamespace(xy=np.asarray([[1.0, 2.0]])))

        def reset(self, *, seed=None, options=None):
            observed["reset"] = (seed, options)
            return {}, {}

        def state_payload(self):
            return {
                "robot": {"x": 2.0, "y": 3.0, "z": 0.2, "yaw": 0.25, "v": 0.0, "w": 0.0},
                "goal": [4.0, 5.0],
                "gimbal": {"yaw": -0.1, "pitch": 0.2},
            }

    env = FakeEnv()

    def make_env(cfg, mode, num_envs, seed, auto_reset, **kwargs):
        observed["make_env"] = (cfg, mode, num_envs, seed, auto_reset, kwargs)
        return env

    def run_viewer(adapter, visualizer, checkpoint, *, reset_seed=None):
        observed["run_viewer"] = (adapter, visualizer, checkpoint, reset_seed)

    task_module = types.ModuleType("isaac_field.task")
    task_module.Config = _Config
    env_module = types.ModuleType("isaac_field.env")
    env_module.make_env = make_env
    env_module.SingleFieldAdapter = FakeAdapter
    viewer_module = types.ModuleType("isaac_field.viewer")
    viewer_module.run_viewer = run_viewer
    monkeypatch.setitem(sys.modules, "isaac_field.task", task_module)
    monkeypatch.setitem(sys.modules, "isaac_field.env", env_module)
    monkeypatch.setitem(sys.modules, "isaac_field.viewer", viewer_module)
    monkeypatch.setattr(sys, "argv", ["isaac_field", "viewer", *args])

    main()
    assert env.closed
    return observed


def test_viewer_cli_separates_geometry_and_episode_seeds_and_records_replay_state(monkeypatch, tmp_path):
    checkpoint = tmp_path / "candidate.zip"
    output = tmp_path / "fresh-output"
    observed = _run_cli_viewer(
        monkeypatch,
        tmp_path,
        [
            "--seed", "80003", "--episode-seed", "81001", "--reward-profile", "protect_plants",
            "--plants", "16", "--flat", "--max-time", "20", "--mode", "goto",
            "--goal-distance", "1", "3", "--goal-angle", "60", "--checkpoint", str(checkpoint),
            "--legacy-sb3",
            "--output", str(output),
        ],
    )

    cfg, mode, num_envs, seed, auto_reset, kwargs = observed["make_env"]
    assert (cfg.plants, cfg.terrain_enabled, cfg.max_time) == (16, False, 20.0)
    assert (cfg.damage_penalty, cfg.goal_progress_budget) == (20.0, 5.0)
    assert (mode, num_envs, seed, auto_reset) == ("goto", 1, 80003, False)
    np.testing.assert_allclose(kwargs["goal_distance"], (1.0, 3.0))
    assert kwargs["distance_cost"] == 0.0
    assert observed["reset"] == (81001, None)
    assert observed["run_viewer"][3] == 81001

    record_path = output / "viewer_initial_state.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["env_seed"] == 80003
    assert record["episode_seed"] == 81001
    assert record["repeat_reset_seed"] == 81001
    assert record["reward_profile"] == "protect_plants"
    assert record["goal"] == [4.0, 5.0]
    assert record["robot"]["yaw"] == 0.25
    assert record["plant_xy"] == [[1.0, 2.0]]
    assert record["task_config"]["damage_penalty"] == 20.0


def test_viewer_cli_keeps_legacy_profile_and_randomizes_later_resets_by_default(monkeypatch, tmp_path):
    observed = _run_cli_viewer(monkeypatch, tmp_path, ["--seed", "77"])

    cfg = observed["make_env"][0]
    assert (cfg.damage_penalty, cfg.goal_progress_budget) == (4.0, None)
    assert observed["reset"] == (77, None)  # Preserve the previous deterministic initial reset.
    assert observed["run_viewer"][3] is None  # Later UI Reset remains randomized by default.
    assert not (tmp_path / "results" / "isaac_field" / "run" / "viewer_initial_state.json").exists()
