"""Evaluation contracts tested with a small fake environment; no Gazebo process."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.evaluate_field_gazebo import evaluate


class FakeScene:
    def __init__(self):
        # A crop blocks the direct eastbound route, forcing the evaluator to
        # choose another deterministic clear corridor.
        self.xy = np.asarray([[3.5, 2.0]], dtype=np.float64)
        self.radius = np.asarray([0.25], dtype=np.float64)
        self.terrain = None

    def navigable(self, point):
        point = np.asarray(point, dtype=np.float64)
        vehicle_radius = np.hypot(0.8, 0.3) / 2 + 0.05
        return bool(np.all(point >= vehicle_radius)
                    and np.all(point <= 20.0 - vehicle_radius)
                    and np.all(np.linalg.norm(self.xy - point, axis=1)
                               > self.radius + vehicle_radius))


class FakeEnv:
    def __init__(self, cfg, mode, *, output_dir):
        self.cfg = cfg
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.scene_id = 1
        self.scene = FakeScene()
        self.robot = np.asarray([2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.goal = np.asarray([18.0, 18.0], dtype=np.float64)
        self.time = 0.0
        self.reason = ""
        self.damaged = set()
        self.collisions = 0
        self.closed = False

    def reset(self, *, seed):
        self.seed = int(seed)
        self.robot[:] = [2.0, 2.0, 0.0, 0.0, 0.0]
        self.goal[:] = [18.0, 18.0]
        self.time = 0.0
        self.reason = ""
        self.damaged.clear()
        self.collisions = 0
        return self._observation(), self._info()

    def scene_payload(self):
        return {"scene_id": self.scene_id, "seed": self.seed,
                "plant_xy": self.scene.xy.tolist(), "plant_radius": self.scene.radius.tolist()}

    def set_goal(self, goal):
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        self.mode = "goto"

    def _observation(self):
        return {"state": np.asarray([self.robot[0], self.robot[1]], dtype=np.float32)}

    def _info(self):
        return {"distance": float(np.linalg.norm(self.goal - self.robot[:2])),
                "time": self.time, "sim_time_s": self.time,
                "reason": self.reason,
                "is_success": self.reason == "goal_reached",
                "damage": len(self.damaged), "collisions": self.collisions}

    def step(self, action):
        self.time += self.cfg.dt
        if self.mode == "goto":
            delta = self.goal - self.robot[:2]
            distance = float(np.linalg.norm(delta))
            move = min(distance, 5.0 * self.cfg.dt)
            if distance > 0:
                self.robot[:2] += delta / distance * move
            if np.linalg.norm(self.goal - self.robot[:2]) <= self.cfg.goal_tolerance:
                self.reason = "goal_reached"
                self.robot[3] = 0.0
        elif self.time >= 0.3:
            self.damaged.update((2, 5))
            self.collisions = 1
        truncated = self.time + 1e-9 >= self.cfg.max_time and not self.reason
        if truncated:
            self.reason = "time_limit"
        return self._observation(), 0.0, self.reason == "goal_reached", truncated, self._info()

    def state_payload(self):
        return {"seed": self.seed, "time": self.time, "goal": self.goal.tolist()}

    def close(self):
        self.closed = True


class FakeRunner:
    def __init__(self, env, checkpoint):
        self.env = env

    def action(self, observation):
        return np.zeros(4, dtype=np.float32)


def test_evaluation_keeps_heldout_search_and_reports_stable_goto_metrics(tmp_path):
    result = evaluate("fake-checkpoint.zip", tmp_path / "eval",
                      seeds=(10007,), seconds=1.0, env_factory=FakeEnv,
                      runner_factory=FakeRunner)

    assert result["heldout_seeds"] == [10007]
    assert [episode["mode"] for episode in result["episodes"]] == ["search", "goto"]
    search, goto = result["episodes"]
    assert search["scene_sha256"] == goto["scene_sha256"]
    assert search["crushed_plant_count"] == 2
    assert search["rock_collision_count"] == 1
    assert np.isclose(goto["goal_route"]["selected_distance_m"], 3.0)
    assert goto["goal_route"]["corridor_clearance_m"] >= 0.05
    assert goto["goal_success"] is True
    assert np.isclose(goto["initial_goal_distance_m"], 3.0)
    assert goto["final_goal_distance_m"] <= result["config"]["goal_tolerance"]
    assert 0.0 < goto["sim_seconds"] <= 1.0
    assert result["goto_summary"]["arrivals"] == 1
    assert result["search_summary"]["episodes"] == 1
    assert result["crushed_plant_count_total"] == 2
    assert result["rock_collision_count_total"] == 1


def test_fieldsim_backend_goto_mode_uses_episode_budget_and_short_rollout(tmp_path):
    result = evaluate("fake-checkpoint.zip", tmp_path / "fieldsim-goto",
                      seeds=(10007,), seconds=0.2, episode_budget=4.0,
                      backend="fieldsim", mode="goto", runner_factory=FakeRunner)

    assert result["backend"] == "fieldsim"
    assert result["mode"] == "goto"
    assert result["requested_seconds_per_episode"] == 0.2
    assert result["episode_budget_seconds"] == 4.0
    assert result["config"]["max_time"] == 4.0
    assert result["rollout_max_steps"] == 4
    assert [episode["mode"] for episode in result["episodes"]] == ["goto"]

    episode = result["episodes"][0]
    assert episode["backend"] == "fieldsim"
    assert episode["steps"] == 4
    assert np.isclose(episode["sim_seconds"], 0.2)
    assert episode["rollout_limit_reached"] is True
    assert episode["episode_ended"] is False
    assert result["goto_summary"]["episodes"] == 1
    assert result["search_summary"]["episodes"] == 0
    observation_path = tmp_path / "fieldsim-goto" / "goto_10007" / "policy_observation.npz"
    with np.load(observation_path) as saved:
        # state[17] is normalized by the episode budget, not the short rollout.
        assert np.isclose(saved["state"][17], 0.2 / 4.0)


def test_single_mode_does_not_require_cross_mode_scene_hash(tmp_path):
    result = evaluate("fake-checkpoint.zip", tmp_path / "search-only",
                      seeds=(10007,), seconds=0.2, episode_budget=3.0,
                      backend="fieldsim", mode="search", env_factory=FakeEnv,
                      runner_factory=FakeRunner)

    assert [episode["mode"] for episode in result["episodes"]] == ["search"]
    assert result["config"]["max_time"] == 3.0
    assert result["search_summary"]["episodes"] == 1
    assert result["goto_summary"]["episodes"] == 0

