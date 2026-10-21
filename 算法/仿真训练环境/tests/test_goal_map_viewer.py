"""CPU-only tests for the non-frozen strict-stop map viewer extension."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from queue import SimpleQueue
import shutil
import types

import numpy as np
import pytest

import scripts.view_fast_stop_goal as goal_viewer
from isaac_field.task import Config, Scene

_REAL_TRIAL = os.environ.get("GOAL_MAP_REAL_TRIAL")


class _Wake:
    def set(self):
        pass


def _scene(size=20.0):
    cfg = Config(size=size, plants=0, terrain_enabled=False, validate_scene=True,
                 goto_reward_profile="fast_stop_v1", goal_tolerance=0.4, goal_hold=0.5,
                 goal_stop_speed=0.05, goal_stop_angular_speed=0.1, goal_progress_budget=5.0)
    return Scene(cfg, np.empty((0, 2)), np.empty(0, dtype=bool), np.random.default_rng(7))


class _Task:
    def __init__(self, scene):
        self.scene = scene
        self.cfg = scene.cfg
        self.robot = np.array([5.0, 5.0, 0.0, 0.2, 0.1], dtype=np.float64)
        self.goal = np.array([2.0, 2.0], dtype=np.float64)
        self.time = 12.5
        self._hold = 0.25
        self.root_com_velocity_xy = np.array([0.3, 0.4])
        self.body_angular_velocity = np.array([0.01, 0.02, 0.03])
        self.done = True
        self.reason = "goal_reached"

    def _goal_stop_state(self):
        return {"linear_speed_m_s": float(np.linalg.norm(self.root_com_velocity_xy)),
                "angular_speed_rad_s": float(np.linalg.norm(self.body_angular_velocity)),
                "hold_s": self._hold}


class _Adapter:
    def __init__(self, task):
        self.task = task
        self.cfg = task.cfg
        self.reset_calls = []
        self.observation_calls = 0

    def state_payload(self):
        return {"goal": self.task.goal.tolist(),
                "robot": {"x": float(self.task.robot[0]), "y": float(self.task.robot[1]),
                          "yaw": float(self.task.robot[2])},
                "metrics": {"is_success": self.task.done and self.task.reason == "goal_reached"}}

    def reset(self, *, options=None):
        self.reset_calls.append(options)
        self.task.goal = np.asarray(options["goal"], dtype=np.float64).copy()
        self.task.robot[:3] = np.asarray(options["start"], dtype=np.float64)
        self.task.time = 0.0
        self.task._hold = 0.0
        self.task.root_com_velocity_xy = np.zeros(2)
        self.task.body_angular_velocity = np.zeros(3)
        self.task.done, self.task.reason = False, ""
        return self.current_observation(), {}

    def current_observation(self):
        self.observation_calls += 1
        return {"state": np.zeros(3, dtype=np.float32)}


class _Runner:
    def __init__(self):
        self.state = ("old-lstm",)
        self.reset_calls = 0

    def reset(self, *_):
        self.reset_calls += 1
        self.state = None


class _Panel:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _MarkerScene:
    def add_line_segments(self, *_args, **_kwargs):
        return types.SimpleNamespace(points=None, visible=True)

    def add_point_cloud(self, *_args, **_kwargs):
        return types.SimpleNamespace(points=None, visible=True)


def _controls():
    task = _Task(_scene())
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    controls.adapter = _Adapter(task)
    controls.visualizer = types.SimpleNamespace()
    controls._commands = SimpleQueue()
    controls._wake_event = _Wake()
    controls._map_pick_enabled = True
    controls._map_size = 20.0
    controls._map_status = ""
    controls._map_goal_label = types.SimpleNamespace(content="")
    controls._goal_map_handles = {}
    controls._run_button = types.SimpleNamespace(label="暂停任务")
    controls._status_handle = types.SimpleNamespace(content="")
    controls._task_mode_handle = types.SimpleNamespace(value="")
    controls._goal_x = types.SimpleNamespace(value=0.0)
    controls._goal_y = types.SimpleNamespace(value=0.0)
    controls._runner = _Runner()
    controls._network_panel = _Panel()
    controls._running = True
    controls._terminal = True
    controls._last_reward = 1.0
    controls._last_info = {"old": True}
    controls._latest_observation = {"old": np.ones(1)}
    controls.visualizer._viewer = types.SimpleNamespace(
        _server=types.SimpleNamespace(scene=_MarkerScene(), initial_camera=None))
    controls._focus_vehicle = lambda: None
    return controls


def test_ray_intersection_maps_xy_and_includes_map_edges():
    assert goal_viewer.intersect_ray_with_map((2, 3, 10), (0.2, -0.1, -1), 20) == pytest.approx((4, 2))
    assert goal_viewer.intersect_ray_with_map((0, 20, 10), (0, 0, -1), 20) == (0, 20)


@pytest.mark.parametrize(
    "origin,direction,size,match",
    [
        ((0, 0, 1), (1, 0, 0), 10, "平行"),
        ((0, 0, -1), (0, 0, -1), 10, "后方"),
        ((0, 0, 1), (11, 0, -1), 10, "超出"),
        ((float("nan"), 0, 1), (0, 0, -1), 10, "有限"),
        ((0, 0, 1), (0, 0, -1), float("inf"), "尺寸"),
    ],
)
def test_ray_rejects_bad_or_outside_intersections(origin, direction, size, match):
    with pytest.raises(ValueError, match=match):
        goal_viewer.intersect_ray_with_map(origin, direction, size)


def test_map_camera_fits_whole_map_and_y_axis_is_up():
    for aspect in (16 / 9, 9 / 16, 1.0):
        pose = goal_viewer.map_camera_pose(100.0, aspect)
        assert pose["target"] == (50.0, 50.0, 0.0)
        assert pose["up"] == (0.0, 1.0, 0.0)
        vertical_half = pose["eye"][2] * math.tan(math.radians(30))
        assert vertical_half >= 50.0
        assert vertical_half * aspect >= 50.0


def test_scene_callback_queues_ray_without_reading_flag_or_environment():
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    controls._map_pick_enabled = False
    controls._commands = SimpleQueue()
    controls._wake_event = _Wake()
    controls.adapter = types.SimpleNamespace(
        state_payload=lambda: pytest.fail("Viser callback must not access environment"))
    event = types.SimpleNamespace(ray_origin=(5.0, 5.0, 10.0), ray_direction=(1.0, 0.0, -1.0))
    controls._on_scene_click(event)
    command, value = controls._commands.get_nowait()
    assert command == "map_ray"
    assert value == ((5.0, 5.0, 10.0), (1.0, 0.0, -1.0))


def test_owner_discards_ray_if_picker_was_disabled_before_processing():
    controls = _controls()
    controls._commands.put(("map_ray", ((5.0, 5.0, 10.0), (1.0, 0.0, -1.0))))
    controls._map_pick_enabled = False
    assert controls.process_pending() == 1
    assert controls.adapter.reset_calls == []
    assert controls._runner.reset_calls == 0


@pytest.mark.parametrize("terminal", [False, True])
def test_owner_click_resets_same_pose_and_lstm_then_pauses(terminal):
    controls = _controls()
    controls._terminal = terminal
    controls.adapter.task.done = terminal
    controls.adapter.task.time = 21.0
    controls.adapter.task._hold = 0.4
    start = controls.adapter.task.robot[:3].copy()
    controls._on_scene_click(types.SimpleNamespace(
        ray_origin=(5.0, 5.0, 10.0), ray_direction=(1.0, 0.0, -1.0)))
    assert controls.adapter.reset_calls == []

    assert controls.process_pending() == 1

    options = controls.adapter.reset_calls[0]
    assert options["mode"] == "goto"
    assert options["scene"] is controls.adapter.task.scene
    assert options["random_start"] is False
    assert options["start"] == pytest.approx(start)
    assert options["goal"] == pytest.approx((15.0, 5.0))
    assert controls.adapter.task.robot[:3] == pytest.approx(start)
    assert controls.adapter.task.goal == pytest.approx((15.0, 5.0))
    assert controls.adapter.task.time == 0.0 and controls.adapter.task._hold == 0.0
    assert controls.adapter.task.root_com_velocity_xy == pytest.approx((0.0, 0.0))
    assert controls.adapter.task.body_angular_velocity == pytest.approx((0.0, 0.0, 0.0))
    assert controls._runner.reset_calls == 1 and controls._runner.state is None
    assert controls._network_panel.reset_calls == 1
    assert controls._running is False and controls._terminal is False
    assert controls._run_button.label == "运行任务"
    assert "距离" in controls._map_goal_label.content
    assert "XYZ=" in controls._map_goal_label.content
    assert "hold 0.00/0.50 s" in controls._map_goal_label.content
    assert "<" not in controls._map_goal_label.content


def test_invalid_target_is_rejected_before_reset_or_lstm_clear():
    controls = _controls()
    goal, state = controls.adapter.task.goal.copy(), controls._runner.state
    label = controls._map_goal_label.content
    with pytest.raises(ValueError):
        controls._apply_map_goal((0.1, 0.1))
    assert controls.adapter.reset_calls == []
    assert controls.adapter.task.goal == pytest.approx(goal)
    assert controls._running is True and controls._terminal is True
    assert controls._runner.state == state and controls._runner.reset_calls == 0
    assert controls._map_goal_label.content == label


def test_top_view_sets_up_and_aspect_for_initial_and_connected_cameras():
    controls = _controls()

    class Camera:
        def __init__(self, width, height):
            self.image_width, self.image_height = width, height
            self.fov = math.radians(60)
            self.position = self.look_at = None
            self.up_direction = None

    initial = types.SimpleNamespace(position=None, look_at=None, up=None)
    camera = Camera(1600, 900)
    server = types.SimpleNamespace(initial_camera=initial,
                                  get_clients=lambda: {"web": types.SimpleNamespace(camera=camera)})
    controls.visualizer._viewer._server = server
    controls._update_map_label = lambda: None
    controls._render_if_connected = lambda: None
    controls._set_top_view()
    assert initial.up == (0.0, 1.0, 0.0)
    assert camera.up_direction == (0.0, 1.0, 0.0)
    assert camera.look_at == (10.0, 10.0, 0.0)
    assert camera.position[:2] == (10.0, 10.0)
    assert camera.position[2] == pytest.approx(goal_viewer.map_camera_pose(20, 16 / 9)["eye"][2])


def test_failed_checkpoint_load_refuses_preview_controls(monkeypatch):
    def failed_base_init(self, *_args, **_kwargs):
        self._runner = None
        self._status = "manual paused"
        self._set_status = lambda: None
        self.queued = []
        self._enqueue = lambda command, value=None: self.queued.append((command, value))

    monkeypatch.setattr(goal_viewer.FieldViewerControls, "__init__", failed_base_init)
    fixture = goal_viewer.FastStopGoalControls(object(), object(), checkpoint=None)
    assert fixture.policy_loaded is False
    assert "没有加载策略模型" in fixture._status
    assert fixture.queued == [("map_top_view", None)]
    with pytest.raises(RuntimeError, match="未能加载检查点"):
        goal_viewer.FastStopGoalControls(object(), object(), "bad-or-incomplete-checkpoint.zip")


def test_map_reference_draws_field_boundary_grid_and_origin():
    controls = _controls()
    scene = _MarkerScene()
    scene.lines = []
    scene.labels = []
    scene.add_line_segments = lambda path, points, *_args, **_kwargs: (
        scene.lines.append((path, np.asarray(points).copy())) or types.SimpleNamespace(points=None))
    scene.add_label = lambda *args, **_kwargs: (scene.labels.append(args) or object())
    controls.visualizer._viewer._server.scene = scene
    controls._add_map_reference()
    assert {path for path, _ in scene.lines} == {
        "/FieldTask/ViewerMapGrid", "/FieldTask/ViewerMapBoundary"}
    grid = next(points for path, points in scene.lines if path.endswith("MapGrid"))
    boundary = next(points for path, points in scene.lines if path.endswith("MapBoundary"))
    assert grid.shape == (2, 2, 3)
    assert np.min(boundary[:, :, :2]) == 0.0
    assert np.max(boundary[:, :, :2]) == 20.0
    assert scene.labels and scene.labels[0][1] == "原点 (0, 0)"


def _completed_trial(tmp_path: Path) -> Path:
    trial = tmp_path / "trial"
    trial.mkdir()
    source = goal_viewer.ROOT / "tests/test_goal_map_viewer.py"
    frozen = trial / "source_snapshot/tests/test_goal_map_viewer.py"
    frozen.parent.mkdir(parents=True)
    shutil.copyfile(source, frozen)
    protocol = {
        "schema": "isaac-field-mastery-experiment-v1", "method": "mastery", "profile": "fast_stop_512",
        "config": {"plants": 0, "terrain_enabled": False, "rock_density": 0.0,
                   "goal_tolerance": 0.4, "goal_hold": 0.5, "goal_stop_speed": 0.05,
                   "goal_stop_angular_speed": 0.1, "goto_reward_profile": "fast_stop_v1",
                   "validate_scene": True},
        "stages": [{"name": "near", "mode": "goto", "goal_distance": [0.6, 1.2],
                    "goal_angle": math.radians(15), "plants": 0}],
        "profile_physics_contract": {},
        "source_snapshot": {"files": [{"path": "tests/test_goal_map_viewer.py",
                                       "snapshot_path": "source_snapshot/tests/test_goal_map_viewer.py",
                                       "sha256": goal_viewer._sha256(source)}]},
    }
    (trial / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = goal_viewer._sha256(trial / "protocol.json")
    (trial / "run_status.json").write_text(json.dumps(
        {"status": "completed", "exit_code": 0, "protocol_sha256": protocol_sha}), encoding="utf-8")
    (trial / "summary.json").write_text(json.dumps(
        {"training_finished": True, "profile": "fast_stop_512", "num_timesteps": 8192}), encoding="utf-8")
    (trial / "python_outcome.json").write_text(json.dumps(
        {"status": "completed", "intended_exit_code": 0}), encoding="utf-8")
    (trial / "process_exit.json").write_text(json.dumps({"actual_exit_code": 0}), encoding="utf-8")
    (trial / "python.exit_code").write_text("0" + chr(10), encoding="utf-8")
    return trial


def test_completion_gate_requires_exit_evidence_and_no_active_training(tmp_path, monkeypatch):
    trial = _completed_trial(tmp_path)
    monkeypatch.setattr(goal_viewer, "_active_trial_processes", lambda _trial: [])
    result = goal_viewer.validate_completed_trial(trial)
    assert result["source_file_count"] == 1
    assert result["summary"]["num_timesteps"] == 8192
    monkeypatch.setattr(goal_viewer, "_active_trial_processes",
                        lambda _trial: [{"pid": 44, "argv": "train_isaac_mastery.py --output trial"}])
    with pytest.raises(ValueError, match="训练进程"):
        goal_viewer.validate_completed_trial(trial)


def test_completion_gate_rejects_nonzero_actual_parent_exit(tmp_path, monkeypatch):
    trial = _completed_trial(tmp_path)
    monkeypatch.setattr(goal_viewer, "_active_trial_processes", lambda _trial: [])
    (trial / "process_exit.json").write_text(json.dumps({"actual_exit_code": 143}), encoding="utf-8")
    with pytest.raises(ValueError, match="真实父shell exit 0"):
        goal_viewer.validate_completed_trial(trial)


@pytest.mark.skipif(not _REAL_TRIAL, reason="set GOAL_MAP_REAL_TRIAL for the live-trial integration check")
def test_real_checkpoint_and_near_best_selection_have_full_contract():
    trial = goal_viewer._resolve(_REAL_TRIAL)
    protocol = goal_viewer._json(trial / "protocol.json")
    checkpoint = trial / "checkpoints/steps_0016384_near.zip"
    record = goal_viewer._checkpoint_record(
        checkpoint, "9d374b52335464ae2873a1654b69f8a437e04d26ce0e541a40de471c1e686de3")
    assert (record["num_timesteps"], record["gamma"], record["lstm_hidden_size"],
            record["n_lstm_layers"]) == (16384, 1.0, 512, 1)
    selected = goal_viewer._resolve_checkpoint(trial, protocol, "near-best")
    assert selected["selection"] == "near-best"
    assert selected["label"] == "近距开发候选"
    assert selected["near_probe_episodes"] == 16
    assert 0 <= selected["near_clean_successes"] <= 16
    assert Path(selected["path"]).is_file()
    explicit = goal_viewer._resolve_checkpoint(trial, protocol, "latest", checkpoint)
    assert explicit["selection"] == "explicit"
    assert explicit["sha256"] == record["sha256"]
    with pytest.raises(ValueError, match="没有checkpoint_manifest"):
        goal_viewer._resolve_checkpoint(trial, protocol, "latest", Path("outside-checkpoint.zip"))
