"""Temporary CPU-only browser fixture for the official Viser goal picker.

This starts the real FastStopGoalControls and Viser 1.0.30, but uses an in-memory
adapter instead of Isaac Lab, Newton, a checkpoint, or a learned policy. It is
for verifying UI events, exact map coordinates, markers, and owner-thread reset
plumbing only; it is not a physical or training preview.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
import types
from typing import Any

import numpy as np
import viser

from isaac_field.task import Config, Scene, sample_episode_start_goal
from scripts.view_fast_stop_goal import FastStopGoalControls


class _TaskState:
    def __init__(self, cfg: Config, scene: Scene):
        self.cfg = cfg
        self.scene = scene
        self.mode = "goto"
        self.time = 0.0
        self._hold = 0.0
        self.done = False
        self.body_angular_velocity = np.zeros(3, dtype=np.float64)
        self.frames: dict[str, Any] = {}

    def _goal_stop_state(self) -> dict[str, float | bool]:
        return {"linear_speed_m_s": 0.0, "hold_s": float(self._hold),
                "within_stop_limits": False}


class _Adapter:
    def __init__(self):
        self.cfg = Config(
            size=100.0,
            cell=0.5,
            plants=0,
            dry_fraction=0.0,
            terrain_enabled=False,
            rock_density=0.0,
            goal_tolerance=0.4,
            goal_hold=0.5,
            goal_stop_speed=0.05,
            goal_stop_angular_speed=0.1,
            validate_scene=True,
            goto_reward_profile="fast_stop_v1",
            goal_progress_budget=0.4,
        )
        self.start = np.asarray((50.0, 50.0, 0.0), dtype=np.float64)
        self.goal = np.asarray((51.0, 50.0), dtype=np.float64)
        rng = np.random.default_rng(20261021)
        self.task = _TaskState(
            self.cfg,
            Scene(self.cfg, np.empty((0, 2), dtype=np.float32),
                  np.empty((0,), dtype=bool), rng),
        )
        self.mode = "goto"
        self.done = False
        self.robot = {"x": 50.0, "y": 50.0, "z": 0.15, "yaw": 0.0,
                      "v": 0.0, "w": 0.0}
        self.reset_events: list[dict[str, Any]] = []
        self.engine = types.SimpleNamespace(
            sim=types.SimpleNamespace(consume_reset_request=lambda: False,
                                      render=lambda: None),
            drive_config={}, observation_config={}, damage_config={"metric": "stem-contact-v2"},
        )

        start, goal = sample_episode_start_goal(
            self.task.scene, np.random.default_rng(0), start=self.start, goal=self.goal,
            random_start=False, mode="goto",
        )
        if not (np.array_equal(start, self.start) and np.array_equal(goal, self.goal)):
            raise RuntimeError("CPU fixture's real Scene sampler changed its exact endpoints")
        connected = self.task.scene.validate_connectivity(start[:2], goal)
        if not connected.get("connected", False):
            raise RuntimeError(f"CPU fixture's real Scene did not validate: {connected}")

    def state_payload(self) -> dict[str, Any]:
        distance = float(np.linalg.norm(self.goal - np.asarray((self.robot["x"], self.robot["y"]))))
        return {
            "mode": self.mode,
            "time": float(self.task.time),
            "robot": dict(self.robot),
            "goal": self.goal.tolist(),
            "done": self.done,
            "gimbal": {"yaw": 0.0, "pitch": 0.0},
            "metrics": {"distance": distance, "damage": 0, "collisions": 0,
                        "found": 0, "targets": 0, "is_success": False,
                        "sensor_sequences": {}},
            "reward": None,
        }

    def reset(self, seed=None, options=None):
        if not isinstance(options, dict):
            raise ValueError("fixture reset requires the exact-goal options payload")
        scene = options.get("scene")
        start = np.asarray(options.get("start"), dtype=np.float64).reshape(-1)
        goal = np.asarray(options.get("goal"), dtype=np.float64).reshape(-1)
        if scene is not self.task.scene or options.get("random_start") is not False:
            raise ValueError("fixture must retain the same scene and fixed start")
        sampled_start, sampled_goal = sample_episode_start_goal(
            scene, np.random.default_rng(0), start=start, goal=goal,
            random_start=False, mode="goto",
        )
        validation = scene.validate_connectivity(start[:2], goal)
        if (not np.array_equal(sampled_start, start) or not np.array_equal(sampled_goal, goal)
                or not validation.get("connected", False)):
            raise ValueError(f"fixture reset rejected exact endpoints: {validation}")

        self.start = start.copy()
        self.goal = goal.copy()
        self.robot.update(x=float(start[0]), y=float(start[1]), yaw=float(start[2]),
                          v=0.0, w=0.0)
        self.mode = "goto"
        self.done = False
        self.task.mode = "goto"
        self.task.time = 0.0
        self.task._hold = 0.0
        self.task.done = False
        self.task.body_angular_velocity = np.zeros(3, dtype=np.float64)
        event = {"start": self.start.tolist(), "goal": self.goal.tolist(),
                 "scene_identity_retained": True, "time_s": self.task.time,
                 "linear_speed_m_s": self.robot["v"], "angular_speed_rad_s": self.robot["w"],
                 "status": "CPU fixture reset accepted"}
        self.reset_events.append(event)
        print("RESET_EVENT " + json.dumps(event, ensure_ascii=False), flush=True)
        return None, {}


class _Visualizer:
    def __init__(self, server: viser.ViserServer):
        self._viewer = types.SimpleNamespace(_server=server)
        self._running = True

    def set_camera_view(self, *, eye, target):
        # The production controls set the public server camera directly.
        return None

    def is_training_paused(self) -> bool:
        return False

    def is_rendering_paused(self) -> bool:
        return False

    def is_running(self) -> bool:
        return self._running


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8795)
    args = parser.parse_args()
    if isinstance(args.port, bool) or not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")

    adapter = _Adapter()
    server = viser.ViserServer(
        host="0.0.0.0", port=args.port, label="CPU控件验证", verbose=False,
    )
    server.gui.add_markdown(
        "**CPU控件验证：不含物理仿真、Newton、GPU或训练模型。**\n\n"
        "绿色点是固定车辆位置，橙色点与黄色圆表示精确目标和 0.4 m 到点半径。",
        order=-20,
    )
    visualizer = _Visualizer(server)
    controls = FastStopGoalControls(visualizer, adapter, checkpoint=None)
    set_panel_label = getattr(controls, "set_panel_label", None)
    if callable(set_panel_label):
        set_panel_label("CPU控件验证")
    controls.process_pending()
    controls._set_top_view()
    controls.publish()

    print(json.dumps({
        "status": "fixture_running",
        "scope": "CPU controls only; no Isaac Lab, Newton physics, GPU, or trained checkpoint",
        "viser_version": viser.__version__,
        "url": f"http://127.0.0.1:{args.port}/",
        "initial_robot_xy_yaw": adapter.start.tolist(),
        "initial_goal_xy": adapter.goal.tolist(),
        "map_size_m": adapter.cfg.size,
        "goal_radius_m": 0.4,
    }, ensure_ascii=False, indent=2), flush=True)

    try:
        while visualizer.is_running():
            controls.process_pending()
            controls.publish()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("FIXTURE_STOP requested by foreground Ctrl-C", flush=True)
    finally:
        visualizer._running = False
        panel = getattr(controls, "_network_panel", None)
        if panel is not None:
            panel.close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
