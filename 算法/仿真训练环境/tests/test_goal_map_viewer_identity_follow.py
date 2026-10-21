"""CPU-only checks for model identity and vehicle-follow camera controls."""
from __future__ import annotations

import math
from pathlib import Path
import types

import pytest

import scripts.view_fast_stop_goal as goal_viewer


def _runner(*, steps=98_304, hidden=512, layers=1):
    lstm = types.SimpleNamespace(hidden_size=hidden, num_layers=layers)
    model = types.SimpleNamespace(
        num_timesteps=steps,
        policy=types.SimpleNamespace(lstm_actor=lstm, lstm_critic=lstm),
    )
    return types.SimpleNamespace(model=model)


def _camera():
    return types.SimpleNamespace(
        position=None, look_at=None, up=None, up_direction=None,
        image_width=1600, image_height=900, fov=math.radians(60),
    )


def test_model_identity_marks_near_best_as_development_candidate():
    path = Path("steps_0098304_near.zip")
    content = goal_viewer.FastStopGoalControls._model_identity_content(
        str(path), _runner(),
        {"path": str(path), "selection": "near-best", "num_timesteps": 98_304},
    )

    assert "steps_0098304_near.zip" in content
    assert "98,304" in content
    assert "actor 512×1" in content and "critic 512×1" in content
    assert "near-best" in content and "不是正式 global best" in content


def test_loading_another_checkpoint_replaces_near_best_identity(monkeypatch, tmp_path):
    near_path = tmp_path / "steps_0098304_near.zip"
    other_path = tmp_path / "other.zip"
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    controls._checkpoint = str(near_path)
    controls._checkpoint_selection = {
        "path": str(near_path), "selection": "near-best", "num_timesteps": 98_304,
    }
    controls._runner = _runner()
    controls._model_identity_handle = types.SimpleNamespace(content="")

    def load_checkpoint(self, checkpoint, *, select_policy):
        assert select_policy is False
        self._checkpoint = checkpoint
        self._runner = _runner(steps=3_000, hidden=256, layers=2)

    monkeypatch.setattr(goal_viewer.FieldViewerControls, "_load_checkpoint", load_checkpoint)
    controls._load_checkpoint(str(other_path), select_policy=False)

    assert "other.zip" in controls._model_identity_handle.content
    assert "3,000" in controls._model_identity_handle.content
    assert "actor 256×2" in controls._model_identity_handle.content
    assert "near-best" not in controls._model_identity_handle.content
    assert controls._checkpoint_selection is None


@pytest.mark.parametrize(
    "yaw,expected_eye",
    [
        (0.0, (6.0, 20.0, 5.5)),
        (math.pi / 2, (10.0, 16.0, 5.5)),
    ],
)
def test_follow_camera_stays_behind_vehicle_yaw(yaw, expected_eye):
    pose = goal_viewer.vehicle_follow_camera_pose(
        {"x": 10.0, "y": 20.0, "z": 3.0, "yaw": yaw})

    assert pose["eye"] == pytest.approx(expected_eye)
    assert pose["target"] == pytest.approx((10.0, 20.0, 3.45))
    assert pose["up"] == (0.0, 0.0, 1.0)


def test_follow_view_updates_initial_and_connected_cameras_without_rendering():
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    initial, first, second = _camera(), _camera(), _camera()
    server = types.SimpleNamespace(
        initial_camera=initial,
        get_clients=lambda: {
            "one": types.SimpleNamespace(camera=first),
            "two": types.SimpleNamespace(camera=second),
        },
    )
    controls.visualizer = types.SimpleNamespace(
        _viewer=types.SimpleNamespace(_server=server))
    controls.adapter = types.SimpleNamespace(state_payload=lambda: {
        "robot": {"x": 10.0, "y": 20.0, "z": 3.0, "yaw": math.pi / 2},
    })
    controls._update_map_label = lambda: None
    controls._render_if_connected = lambda: pytest.fail("follow must not render the simulator")
    controls._render_scene = lambda: pytest.fail("follow must not render the simulator")

    controls._set_follow_view()

    for camera in (initial, first, second):
        assert camera.position == pytest.approx((10.0, 16.0, 5.5))
        assert camera.look_at == pytest.approx((10.0, 20.0, 3.45))
    assert initial.up == (0.0, 0.0, 1.0)
    assert first.up_direction == second.up_direction == (0.0, 0.0, 1.0)


def test_map_view_disables_follow_checkbox():
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    initial, client = _camera(), _camera()
    controls.visualizer = types.SimpleNamespace(_viewer=types.SimpleNamespace(_server=types.SimpleNamespace(
        initial_camera=initial,
        get_clients=lambda: {"one": types.SimpleNamespace(camera=client)},
    )))
    controls._map_size = 20.0
    controls._vehicle_follow_enabled = True
    controls._vehicle_follow_handle = types.SimpleNamespace(value=True)
    controls._update_map_label = lambda: None
    controls._render_if_connected = lambda: None

    controls._set_top_view()

    assert controls._vehicle_follow_enabled is False
    assert controls._vehicle_follow_handle.value is False
    assert client.look_at == pytest.approx((10.0, 10.0, 0.0))


def test_map_view_ignores_reentrant_follow_off_command():
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    initial, client = _camera(), _camera()
    controls.visualizer = types.SimpleNamespace(_viewer=types.SimpleNamespace(_server=types.SimpleNamespace(
        initial_camera=initial,
        get_clients=lambda: {"one": types.SimpleNamespace(camera=client)},
    )))
    controls._map_size = 20.0
    controls._vehicle_follow_enabled = True
    controls._vehicle_follow_handle = types.SimpleNamespace(value=True)
    controls._update_map_label = lambda: None
    controls._render_if_connected = lambda: None

    controls._set_top_view()
    map_status = controls._map_status
    controls._apply_command("vehicle_follow", False)

    assert map_status == "全图俯视已启用；地图+Y朝屏幕上方。"
    assert controls._map_status == map_status
    assert controls._vehicle_follow_enabled is False


def test_publish_updates_follow_camera_even_when_paused(monkeypatch):
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    controls._vehicle_follow_enabled = True
    controls._running = False
    controls.adapter = types.SimpleNamespace(state_payload=lambda: {})
    calls = []
    monkeypatch.setattr(goal_viewer.FieldViewerControls, "publish", lambda self: calls.append("base"))
    controls._update_vehicle_marker = lambda _payload=None: calls.append("marker")
    controls._update_map_label = lambda _payload=None: calls.append("label")
    controls._set_follow_view = lambda _payload=None: calls.append("follow")

    controls.publish()

    assert calls == ["base", "marker", "label", "follow"]


def test_follow_toggle_works_while_paused_without_stepping():
    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    controls._vehicle_follow_enabled = False
    controls._running = False
    controls._map_status = ""
    controls._vehicle_follow_handle = types.SimpleNamespace(value=False)
    controls._update_map_label = lambda: None
    controls.adapter = types.SimpleNamespace(
        step=lambda *_args, **_kwargs: pytest.fail("camera toggle must not step physics"))
    calls = []
    controls._set_follow_view = lambda: calls.append("camera")

    controls._apply_command("vehicle_follow", True)
    assert controls._vehicle_follow_enabled is True
    assert controls._vehicle_follow_handle.value is True
    assert calls == ["camera"]

    controls._apply_command("vehicle_follow", False)
    assert controls._vehicle_follow_enabled is False
    assert controls._vehicle_follow_handle.value is False
    assert calls == ["camera"]


def test_model_and_camera_controls_are_added_as_expanded_top_folders(monkeypatch):
    class Folder:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Handle:
        def __init__(self, value=None, content=""):
            self.value, self.content = value, content

        def on_update(self, callback):
            self.update_callback = callback

        def on_click(self, callback):
            self.click_callback = callback

    class Gui:
        def __init__(self):
            self.folders = []

        def add_folder(self, name, **kwargs):
            self.folders.append((name, kwargs))
            return Folder()

        def add_markdown(self, content):
            return Handle(content=content)

        def add_checkbox(self, _label, *, initial_value):
            return Handle(value=initial_value)

        def add_button(self, _label):
            return Handle()

    controls = goal_viewer.FastStopGoalControls.__new__(goal_viewer.FastStopGoalControls)
    controls.adapter = types.SimpleNamespace(
        cfg=types.SimpleNamespace(size=20.0),
        state_payload=lambda: {"goal": [2.0, 2.0]},
    )
    controls._checkpoint_selection = None
    controls._runner = None
    controls._current_goal = lambda _size: [2.0, 2.0]
    scene = types.SimpleNamespace(on_click=lambda: (lambda _callback: None))
    controls._server_scene = lambda: scene
    controls._add_map_reference = lambda: None
    controls._update_goal_marker = lambda _goal: None
    controls._update_vehicle_marker = lambda: None
    monkeypatch.setattr(goal_viewer.FieldViewerControls, "_build_sidebar", lambda *_args: None)
    gui = Gui()

    controls._build_sidebar(gui, None)

    assert gui.folders[:2] == [
        ("当前模型", {"expand_by_default": True, "order": -20}),
        ("地图点选目标", {"expand_by_default": True, "order": -10}),
    ]
    assert controls._vehicle_follow_handle.value is False
    assert "未加载" in controls._model_identity_handle.content
