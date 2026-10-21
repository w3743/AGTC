from __future__ import annotations

from dataclasses import dataclass
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
    rock_density: float = 0.01


class _FrontGuideSpec:
    pass


class _BodyStabilityConfig:
    pass


def _run_viewer(monkeypatch, args):
    observed = {}

    class FakeEnv:
        def __init__(self):
            visualizer = types.SimpleNamespace(cfg=types.SimpleNamespace(visualizer_type="viser"))
            self.sim = types.SimpleNamespace(visualizers=[visualizer])
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAdapter:
        def __init__(self, env):
            self.engine = env
            self.task = types.SimpleNamespace(scene=types.SimpleNamespace(xy=np.empty((0, 2))))

        def reset(self, *, seed=None, options=None):
            observed["reset"] = seed
            return {}, {}

        def state_payload(self):
            return {"robot": {}, "gimbal": {}, "goal": None}

    env = FakeEnv()

    def make_env(cfg, mode, num_envs, seed, auto_reset, **kwargs):
        observed["make_env"] = (cfg, mode, num_envs, seed, auto_reset, kwargs)
        return env

    def run_viewer(adapter, visualizer, checkpoint, *, reset_seed=None):
        observed["viewer"] = (checkpoint, reset_seed)

    task_module = types.ModuleType("isaac_field.task")
    task_module.Config = _Config
    env_module = types.ModuleType("isaac_field.env")
    env_module.make_env = make_env
    env_module.SingleFieldAdapter = FakeAdapter
    viewer_module = types.ModuleType("isaac_field.viewer")
    viewer_module.run_viewer = run_viewer
    assets_module = types.ModuleType("isaac_field.assets")
    assets_module.FrontGuideSpec = _FrontGuideSpec
    stability_module = types.ModuleType("isaac_field.body_stability")
    stability_module.BodyStabilityConfig = _BodyStabilityConfig

    for name, module in (
        ("isaac_field.task", task_module),
        ("isaac_field.env", env_module),
        ("isaac_field.viewer", viewer_module),
        ("isaac_field.assets", assets_module),
        ("isaac_field.body_stability", stability_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["isaac_field", "viewer", *args])

    main()
    assert env.closed
    return observed


def test_cli_passes_optional_front_guides_stability_and_rock_density(monkeypatch):
    observed = _run_viewer(
        monkeypatch,
        [
            "--plants", "0", "--rock-density", "0", "--front-guides", "--body-stability",
            "--max-time", "60", "--seed", "80003", "--episode-seed", "87001",
            "--port", "8769", "--reward-profile", "protect_plants",
        ],
    )

    cfg, mode, num_envs, seed, auto_reset, kwargs = observed["make_env"]
    assert (cfg.plants, cfg.rock_density, cfg.terrain_enabled, cfg.max_time) == (0, 0.0, True, 60.0)
    assert (cfg.damage_penalty, cfg.goal_progress_budget) == (20.0, 5.0)
    assert (mode, num_envs, seed, auto_reset) == ("mixed", 1, 80003, False)
    assert isinstance(kwargs["front_guide_spec"], _FrontGuideSpec)
    assert isinstance(kwargs["body_stability_config"], _BodyStabilityConfig)
    assert kwargs["viewer_port"] == 8769
    assert observed["reset"] == 87001
    assert observed["viewer"] == (None, 87001)


def test_cli_defaults_preserve_legacy_optional_physics_and_rock_density(monkeypatch):
    observed = _run_viewer(monkeypatch, [])

    cfg, _mode, _num_envs, _seed, _auto_reset, kwargs = observed["make_env"]
    assert cfg.rock_density == 0.01
    assert kwargs["front_guide_spec"] is None
    assert kwargs["body_stability_config"] is None
    assert observed["viewer"][0] is None
