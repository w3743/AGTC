"""Kitless focused tests for the official-Viser field controls adapter."""

from __future__ import annotations

import types
import sys

import numpy as np
import pytest

from isaac_field.action_contract import ACTION_SCHEMA, action_config
from isaac_field.feedback import INPUT_SCHEMA, STATE_DIM, FEEDBACK_DIM
import isaac_field.viewer as viewer_module
from isaac_field.viewer import attach_field_controls, depth_to_rgb


class _Event:
    def __init__(self, value=None):
        self.target = types.SimpleNamespace(value=value)


class _Handle:
    def __init__(self, value=None, content=None, image=None, label=None):
        self.value = value
        self.content = content
        self.image = image
        self.label = label
        self._click = None
        self._update = None

    def on_click(self, callback):
        self._click = callback
        return callback

    def on_update(self, callback):
        self._update = callback
        return callback

    def click(self):
        assert self._click is not None
        self._click(_Event())

    def update(self, value):
        self.value = value
        assert self._update is not None
        self._update(_Event(value))


class _Folder(_Handle):
    def __init__(self, label, gui):
        super().__init__(label=label)
        self._gui = gui
        self._children = {}

    def __enter__(self):
        self._gui._parents.append(self)
        return self

    def __exit__(self, *_):
        self._gui._parents.pop()
        return False


class _RootGuiContainer:
    def __init__(self):
        self._children = {}


class _Gui:
    def __init__(self):
        self.handles = {}
        self._root = _RootGuiContainer()
        self._container_handle_from_uuid = {"root": self._root}
        self._parents = []
        self._next_id = 0
        self.panel_label = None

    def set_panel_label(self, label):
        self.panel_label = label

    def add_folder(self, label, **_):
        return self._add(label, _Folder(label, self))

    def add_markdown(self, content, **_):
        return self._add(content, _Handle(content=content))

    def add_dropdown(self, label, options, initial_value=None, **_):
        assert options
        return self._add(label, _Handle(value=initial_value, label=label))

    def add_number(self, label, initial_value, **_):
        return self._add(label, _Handle(value=initial_value, label=label))

    def add_button(self, label, **_):
        return self._add(label, _Handle(label=label))

    def add_checkbox(self, label, initial_value=False, **_):
        return self._add(label, _Handle(value=initial_value, label=label))

    def add_slider(self, label, min, max, step, initial_value, **_):
        assert min < max and step > 0
        return self._add(label, _Handle(value=initial_value, label=label))

    def add_text(self, label, initial_value, **_):
        return self._add(label, _Handle(value=initial_value, label=label))

    def add_image(self, image, *, label, **_):
        assert image.shape == (160, 88, 3)
        return self._add(label, _Handle(image=image, label=label))

    def _add(self, label, handle):
        self.handles[label] = handle
        self._next_id += 1
        parent = self._parents[-1] if self._parents else self._root
        parent._children[str(self._next_id)] = handle
        return handle


class _Scene:
    def __init__(self):
        self.nodes = {}
        self.add_calls = []

    def add_point_cloud(self, name, points, colors, **kwargs):
        handle = _Handle()
        handle.name = name
        handle.points = np.asarray(points).copy()
        handle.colors = np.asarray(colors).copy()
        handle.kwargs = kwargs
        self.nodes[name] = handle
        self.add_calls.append(name)
        return handle

    def add_line_segments(self, name, points, colors, **kwargs):
        handle = _Handle()
        handle.name = name
        handle.points = np.asarray(points).copy()
        handle.colors = np.asarray(colors).copy()
        handle.kwargs = kwargs
        self.nodes[name] = handle
        self.add_calls.append(name)
        return handle

class _Space:
    def __init__(self, shape):
        self.shape = shape
        self.low = np.full(shape, -1.0, dtype=np.float32)
        self.high = np.full(shape, 1.0, dtype=np.float32)


class _DictSpace:
    def __init__(self):
        self.spaces = {
            "lidar": _Space((1, 400)),
            "depth": _Space((1, 160, 88)),
            "detections": _Space((16, 8)),
            "state": _Space((STATE_DIM,)),
            "feedback": _Space((FEEDBACK_DIM,)),
        }


class _Adapter:
    def __init__(self):
        self.cfg = types.SimpleNamespace(size=100.0, depth_height=160, depth_width=88, dt=0.05)
        self.observation_space = _DictSpace()
        self.action_space = _Space((4,))
        self.mode = "search"
        self.reset_calls = []
        self.goal_calls = []
        self.actions = []
        self.done = False
        self.time = 0.0
        self.goal = np.asarray([3.0, 4.0], dtype=np.float32)
        self.robot = {"x": 1.0, "y": 2.0, "z": 0.15, "yaw": 0.1}
        self.engine = types.SimpleNamespace(
            sim=_Sim(),
            drive_config={"interface": "independent_left_right_track_speed_coupled_pi_v7", "velocity_gain": 0.5},
            observation_config={
                "schema": INPUT_SCHEMA,
                "state_dim": STATE_DIM,
                "depth_channels": 1,
                "depth_invalid_value": 0.0,
            },
            damage_config={"metric": "stem-contact-v2", "track_geometry": "current"},
        )

    def current_observation(self):
        depth = np.zeros((1, 160, 88), dtype=np.float32)
        depth[0] = 0.5
        return {
            "lidar": np.zeros((1, 400), dtype=np.float32),
            "depth": depth,
            "detections": np.zeros((16, 8), dtype=np.float32),
            "state": np.zeros(STATE_DIM, dtype=np.float32),
            "feedback": np.zeros((FEEDBACK_DIM,), dtype=np.float32),
        }

    def state_payload(self):
        return {
            "mode": self.mode,
            "time": self.time,
            "robot": dict(self.robot),
            "goal": self.goal.tolist(),
            "done": self.done,
            "metrics": {"distance": 2.8, "damage": 0, "collisions": 0, "sensor_sequences": {"lidar": 2, "depth": 3}},
            "reward": 0.0,
        }

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, options))
        self.mode = options["mode"] if options else "search"
        self.done = False
        self.time = 0.0
        self.goal = np.asarray([6.0, 7.0], dtype=np.float32)
        return self.current_observation(), {}

    def set_goal(self, goal):
        self.goal = np.asarray(goal, dtype=np.float32).copy()
        self.goal_calls.append(self.goal.copy())
        self.mode = "goto"

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.current_observation(), 0.5, False, False, {"distance": 2.8}


class _Visualizer:
    def __init__(self):
        self._training_paused = False
        self._rendering_paused = False
        gui = _Gui()
        gui.add_markdown("Physics: Newton MJWarp")
        pause_rendering = gui.add_button("Pause Rendering")
        pause_rendering.on_click(lambda _: self._toggle_native_pause("rendering", pause_rendering))
        pause_simulation = gui.add_button("Pause Simulation")
        pause_simulation.on_click(lambda _: self._toggle_native_pause("simulation", pause_simulation))
        gui.add_button("Reset Episode")
        gui.add_folder("Live Plots")
        with gui.add_folder("Visualization Markers"):
            for label in ("Joints", "Contacts", "Center of Mass", "Particles", "Visual", "Collision", "Springs", "Cloth", "Inertia Boxes"):
                gui.add_checkbox(label)
        self._viewer = types.SimpleNamespace(_server=types.SimpleNamespace(gui=gui, scene=_Scene()))
        self.camera_views = []

    def _toggle_native_pause(self, kind, handle):
        attribute = "_rendering_paused" if kind == "rendering" else "_training_paused"
        paused = not getattr(self, attribute)
        setattr(self, attribute, paused)
        english = ("Resume Rendering" if paused else "Pause Rendering") if kind == "rendering" else (
            "Resume Simulation" if paused else "Pause Simulation"
        )
        handle.label = english

    def set_camera_view(self, *, eye, target):
        self.camera_views.append((tuple(eye), tuple(target)))

    def is_training_paused(self):
        return self._training_paused

    def is_rendering_paused(self):
        return self._rendering_paused

    def is_running(self):
        return True


class _Sim:
    def __init__(self):
        self.reset_requested = False
        self.render_calls = 0

    def render(self):
        self.render_calls += 1

    def consume_reset_request(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested


class _InspectionSpy:
    def __init__(self):
        self.action_calls = []
        self.reset_calls = 0
        self.critic_state = object()
        self.layers = {
            "features_extractor": {
                "key": "features_extractor",
                "title": "特征编码",
                "description": "策略特征",
                "branch": "fusion",
                "column": 1.0,
                "parents": (),
                "module": object(),
                "summary": True,
                "source": "真实策略前向",
                "values": None,
            }
        }
        self.snapshot = None

    def parameters(self, _key):
        return {}

    def reset(self):
        self.reset_calls += 1
        self.snapshot = None
        self.critic_state = None
        for layer in self.layers.values():
            layer["values"] = None

    def action(self, observation, sim_time):
        self.action_calls.append((observation, sim_time))
        action = np.asarray([0.2, -0.1, 0.0, 0.05], dtype=np.float32)
        self.layers["features_extractor"]["values"] = np.asarray([sim_time, 1.0], dtype=np.float32)
        self.snapshot = {
            "sequence": len(self.action_calls),
            "sim_time": sim_time,
            "mode": "runner.action",
            "errors": [],
            "layers": list(self.layers.values()),
        }
        return action


class _RunnerSpy:
    def __init__(self):
        self.action_calls = []
        self.reset_calls = []
        self.action_value = np.asarray([[0.8, 0.0, 0.0, 0.0]], dtype=np.float32)

    def action(self, observation):
        self.action_calls.append(observation)
        return self.action_value

    def reset(self, done=None):
        self.reset_calls.append(done)


class _Network2DSpy:
    def __init__(self, *, clients):
        self.clients = clients
        self.published = []

    def has_clients(self):
        return self.clients

    def publish(self, snapshot, layers, *, state):
        self.published.append((snapshot, layers, state))


def test_depth_pseudocolor_keeps_policy_resolution_and_masks_invalid_pixels():
    depth = np.zeros((1, 160, 88), dtype=np.float32)
    depth[0, :, 1:] = 1.0
    depth[0, 1, 0] = 0.1

    rgb = depth_to_rgb(depth)

    assert rgb.shape == (160, 88, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[0, 0], np.zeros(3, dtype=np.uint8))
    assert rgb[0, 1, 2] > rgb[0, 1, 0]  # far valid pixels are blue-biased
    assert rgb[1, 0, 0] > rgb[1, 0, 2]  # near valid pixels are red-biased


def test_ui_callbacks_only_enqueue_until_owner_thread_processes_them():
    adapter = _Adapter()
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    gui = visualizer._viewer._server.gui

    gui.handles["任务模式"].update("到点任务")
    gui.handles["目标 X（米）"].value = 12.5
    gui.handles["目标 Y（米）"].value = 18.0
    gui.handles["设置目标 / 应用任务模式"].click()
    gui.handles["左履带速度指令"].update(0.4)

    assert adapter.goal_calls == []
    assert adapter.actions == []
    controls.process_pending()

    assert len(adapter.goal_calls) == 1
    np.testing.assert_allclose(adapter.goal_calls[0], [12.5, 18.0])
    assert controls.task_mode == "goto"
    assert controls.action() is None
    gui.handles["运行任务"].click()
    controls.process_pending()
    np.testing.assert_allclose(controls.action(), [0.4, 0.0, 0.0, 0.0])


def test_reset_applies_selected_task_mode_and_step_uses_normalized_manual_action():
    adapter = _Adapter()
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter, reset_seed=81001)
    gui = visualizer._viewer._server.gui

    np.testing.assert_allclose(gui.handles["目标 X（米）"].value, 3.0)
    np.testing.assert_allclose(gui.handles["目标 Y（米）"].value, 4.0)
    gui.handles["任务模式"].update("到点任务")
    gui.handles["目标 X（米）"].value = 7.0
    gui.handles["目标 Y（米）"].value = 9.0
    gui.handles["重置任务"].click()
    gui.handles["云台俯仰角速度指令"].update(-0.25)
    gui.handles["运行任务"].click()

    transition = controls.step_once()

    assert transition is not None
    assert adapter.reset_calls == [(81001, {"mode": "goto"})]
    assert adapter.goal_calls == []  # Reset keeps the new episode's sampled goal.
    np.testing.assert_allclose(adapter.goal, [6.0, 7.0])
    np.testing.assert_allclose(gui.handles["目标 X（米）"].value, 6.0)
    np.testing.assert_allclose(gui.handles["目标 Y（米）"].value, 7.0)
    np.testing.assert_allclose(adapter.actions[-1], [0.0, 0.0, 0.0, -0.25])
    assert controls._metrics_handle.content.find("目标距离") >= 0
    assert controls._depth_handle.image.shape == (160, 88, 3)
    assert visualizer.camera_views[-1] == ((-2.0, -2.0, 3.15), (1.0, 2.0, 0.15))


def test_camera_focus_uses_current_robot_pose_on_init_reset_and_button():
    adapter = _Adapter()
    visualizer = _Visualizer()
    initial_camera = types.SimpleNamespace()
    visualizer._viewer._server.initial_camera = initial_camera
    controls = attach_field_controls(visualizer, adapter)
    scene = visualizer._viewer._server.scene
    assert not hasattr(scene, "add_image")
    assert not hasattr(scene, "add_label")
    assert controls._network_panel is not None
    assert len(visualizer.camera_views) == 1
    assert visualizer.camera_views[-1] == ((-2.0, -2.0, 3.15), (1.0, 2.0, 0.15))
    assert (initial_camera.position, initial_camera.look_at) == visualizer.camera_views[-1]
    assert adapter.engine.sim.render_calls == 1

    adapter.robot = {"x": 48.0, "y": 72.0, "z": 0.22}
    controls._apply_task("goto", 80.0, 90.0, reset=True)
    assert len(visualizer.camera_views) == 2
    assert visualizer.camera_views[-1] == ((45.0, 68.0, 3.22), (48.0, 72.0, 0.22))
    assert adapter.engine.sim.render_calls == 2

    adapter.robot = {"x": 10.0, "y": 20.0, "z": 0.17}
    visualizer._viewer._server.gui.handles["聚焦车辆"].click()
    assert len(visualizer.camera_views) == 2
    assert visualizer.camera_views[-1] == ((45.0, 68.0, 3.22), (48.0, 72.0, 0.22))
    controls.process_pending()
    assert len(visualizer.camera_views) == 3
    assert visualizer.camera_views[-1] == ((7.0, 16.0, 3.17), (10.0, 20.0, 0.17))
    assert (initial_camera.position, initial_camera.look_at) == visualizer.camera_views[-1]
    assert adapter.engine.sim.render_calls == 3


def test_paused_viewer_renders_current_scene_without_stepping_physics():
    adapter = _Adapter()
    adapter.cfg.dt = 0.001
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    initial_render_calls = adapter.engine.sim.render_calls
    running_states = iter((True, False))
    visualizer.is_running = lambda: next(running_states)

    controls.run()

    assert adapter.engine.sim.render_calls == initial_render_calls + 1
    assert adapter.actions == []
    assert controls._depth_handle.image.shape == (160, 88, 3)


def _set_fake_clients(visualizer, client_ids=()):
    server = visualizer._viewer._server
    server.client_ids = tuple(client_ids)
    server.get_clients = lambda: {client_id: object() for client_id in server.client_ids}
    return server


def _count_publishes(controls):
    calls = []
    publish = controls.publish

    def counted_publish():
        calls.append(None)
        publish()

    controls.publish = counted_publish
    return calls


def _stop_after_iterations(visualizer, iterations):
    checks = 0

    def is_running():
        nonlocal checks
        checks += 1
        return checks <= iterations

    visualizer.is_running = is_running


def _policy_viewer_with_2d_client(monkeypatch, *, two_d_clients):
    adapter = _Adapter()
    visualizer = _Visualizer()
    server = _set_fake_clients(visualizer)
    inspector = _InspectionSpy()
    monkeypatch.setitem(
        sys.modules,
        "isaac_field.visualization.inspection",
        types.SimpleNamespace(PolicyInspector=lambda _runner: inspector),
    )
    controls = attach_field_controls(visualizer, adapter)
    runner = _RunnerSpy()
    panel = controls._network_panel
    panel.bind(runner)
    network2d = _Network2DSpy(clients=two_d_clients)
    panel._network2d = network2d
    controls._runner = runner
    controls._control_mode = "policy"
    controls._running = True
    return adapter, visualizer, server, controls, runner, inspector, network2d


def test_running_viewer_keeps_configured_step_cadence_and_publishes_each_transition(monkeypatch):
    adapter = _Adapter()
    adapter.cfg.dt = 0.05
    visualizer = _Visualizer()
    _set_fake_clients(visualizer, ("browser-1",))
    controls = attach_field_controls(visualizer, adapter)
    publish_calls = _count_publishes(controls)
    sleep_calls = []
    monkeypatch.setattr(viewer_module.time, "perf_counter", lambda: 10.0)
    monkeypatch.setattr(viewer_module.time, "sleep", sleep_calls.append)
    _stop_after_iterations(visualizer, 3)

    visualizer._viewer._server.gui.handles["运行任务"].click()
    controls.run()

    assert len(adapter.actions) == 3
    assert len(publish_calls) == 3
    assert sleep_calls == [0.05, 0.05, 0.05]


def test_paused_idle_publishes_and_renders_only_once_without_changes():
    adapter = _Adapter()
    visualizer = _Visualizer()
    _set_fake_clients(visualizer, ("browser-1",))
    controls = attach_field_controls(visualizer, adapter)
    publish_calls = _count_publishes(controls)
    initial_render_calls = adapter.engine.sim.render_calls
    wait_timeouts = []
    controls._wake_event.wait = wait_timeouts.append
    _stop_after_iterations(visualizer, 4)

    controls.run()

    assert len(publish_calls) == 1
    assert adapter.engine.sim.render_calls == initial_render_calls + 1
    assert wait_timeouts == [0.25, 0.25, 0.25, 0.25]
    assert adapter.actions == []


def test_paused_idle_legacy_visualizer_without_client_api_does_not_repeat_updates():
    adapter = _Adapter()
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    publish_calls = _count_publishes(controls)
    initial_render_calls = adapter.engine.sim.render_calls
    wait_timeouts = []
    controls._wake_event.wait = wait_timeouts.append
    _stop_after_iterations(visualizer, 4)

    controls.run()

    assert len(publish_calls) == 1
    assert adapter.engine.sim.render_calls == initial_render_calls + 1
    assert wait_timeouts == [0.25, 0.25, 0.25, 0.25]
    assert adapter.actions == []


def test_idle_viewer_refreshes_for_connection_same_count_reconnect_and_queued_request():
    adapter = _Adapter()
    visualizer = _Visualizer()
    server = _set_fake_clients(visualizer)
    controls = attach_field_controls(visualizer, adapter)
    publish_calls = _count_publishes(controls)
    initial_render_calls = adapter.engine.sim.render_calls
    gui = server.gui
    wait_count = 0

    def wake_or_change_clients(_timeout):
        nonlocal wait_count
        if wait_count == 0:
            server.client_ids = ("browser-1",)
        elif wait_count == 1:
            # A replacement browser can reconnect while the client count stays at one.
            server.client_ids = ("browser-2",)
        elif wait_count == 2:
            gui.handles["左履带速度指令"].update(0.35)
        wait_count += 1

    controls._wake_event.wait = wake_or_change_clients
    _stop_after_iterations(visualizer, 5)

    controls.run()

    assert len(publish_calls) == 3
    assert adapter.engine.sim.render_calls == initial_render_calls + 3
    np.testing.assert_allclose(controls._manual_action, [0.35, 0.0, 0.0, 0.0])
    assert adapter.actions == []


def test_2d_only_client_keeps_policy_inspection_action_and_transition_publication_active(monkeypatch):
    adapter, _visualizer, server, controls, runner, inspector, network2d = _policy_viewer_with_2d_client(
        monkeypatch, two_d_clients=True
    )
    adapter.time = 2.0
    # Let the panel's 4 Hz display throttle admit this transition frame.
    controls._network_panel._last_publish = -1e9

    assert server.client_ids == ()
    assert controls._client_signature() == (("network2d", "active"),)
    assert controls._client_count() == 1

    transition = controls.step_once()

    assert transition is not None
    assert len(inspector.action_calls) == 1
    assert inspector.action_calls[0][1] == 2.0
    assert runner.action_calls == []
    np.testing.assert_allclose(adapter.actions[-1], [0.2, -0.1, 0.0, 0.05])
    assert len(network2d.published) == 1
    snapshot, _layers, state = network2d.published[0]
    assert snapshot is inspector.snapshot
    assert snapshot["sequence"] == 1
    assert state["time"] == 2.0
    assert state["robot"] == adapter.state_payload()["robot"]


def test_2d_without_clients_restores_zero_client_policy_capture_gate(monkeypatch):
    adapter, _visualizer, server, controls, runner, inspector, network2d = _policy_viewer_with_2d_client(
        monkeypatch, two_d_clients=False
    )

    assert server.client_ids == ()
    assert controls._client_signature() == ()
    assert controls._client_count() == 0

    transition = controls.step_once()

    assert transition is not None
    assert inspector.action_calls == []
    assert len(runner.action_calls) == 1
    np.testing.assert_allclose(adapter.actions[-1], runner.action_value[0])
    assert network2d.published == []


def test_run_viewer_closes_independent_2d_server(monkeypatch):
    instances = []

    class FakeNetwork2DServer:
        def __init__(self, port, _enqueue, viewer_url):
            self.port = port
            self.viewer_url = viewer_url
            self.url = f"http://127.0.0.1:{port}/"
            self.closed = False
            instances.append(self)

        def has_clients(self):
            return False

        def publish(self, *_args, **_kwargs):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "isaac_field.visualization.network_2d",
        types.SimpleNamespace(Network2DServer=FakeNetwork2DServer),
    )
    adapter = _Adapter()
    visualizer = _Visualizer()
    server = _set_fake_clients(visualizer)
    server.get_port = lambda: 8769
    visualizer.is_running = lambda: False

    controls = viewer_module.run_viewer(adapter, visualizer)

    assert len(instances) == 1
    assert instances[0].port == 8770
    assert instances[0].viewer_url == "http://127.0.0.1:8769/"
    assert instances[0].closed is True
    assert controls._network_panel._network2d is None


def test_sensor_overlays_use_delivered_frames_and_reuse_viser_handles():
    adapter = _Adapter()
    adapter.task = types.SimpleNamespace(
        frames={
            "lidar": {
                "origin": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "ranges": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "directions": np.asarray(
                    [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
                ),
                "valid": np.asarray([True, False, True, True]),
                "max_range": 8.0,
            },
            "depth": {
                "origin": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "values": np.full((8, 8), 5.0, dtype=np.float32),
                "valid": np.zeros((8, 8), dtype=bool),
                "intrinsics": np.asarray([[4, 0, 4], [0, 4, 4], [0, 0, 1]], dtype=np.float32),
                "basis": np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
            },
        }
    )
    adapter.task.frames["depth"]["valid"][0, 0] = True
    adapter.task.frames["depth"]["values"][0, 0] = 2.0
    adapter.task.frames["depth"]["valid"][2, 2] = True  # omitted by stride-4 display sampling
    adapter.task.frames["depth"]["valid"][4, 4] = True
    adapter.task.frames["depth"]["values"][4, 4] = 3.0

    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    scene = visualizer._viewer._server.scene
    assert len(scene.nodes) == 3
    lidar = scene.nodes["/FieldTask/Sensors/LidarReturns"]
    ring = scene.nodes["/FieldTask/Sensors/LidarMaxRange"]
    depth = scene.nodes["/FieldTask/Sensors/DepthReturns"]
    np.testing.assert_allclose(lidar.points, [[2, 2, 3], [-2, 2, 3], [1, -2, 3]])
    assert lidar.kwargs["point_size"] == 0.015
    assert ring.points.shape == (128, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(ring.points[:, 0] - [1, 2, 3], axis=1), 8.0, atol=1e-5)
    assert ring.kwargs["line_width"] < 1.0
    np.testing.assert_allclose(depth.points, [[3, 4, 5], [4, 2, 3]])
    assert depth.kwargs["point_size"] == 0.02

    lidar_handle = lidar
    adapter.task.frames["lidar"]["ranges"][0] = 2.0
    controls.publish()
    assert scene.nodes["/FieldTask/Sensors/LidarReturns"] is lidar_handle
    assert len(scene.add_calls) == 3
    np.testing.assert_allclose(scene.nodes["/FieldTask/Sensors/LidarReturns"].points[0], [3, 2, 3])


def test_policy_mode_requires_loaded_checkpoint_and_policy_observation_is_batched(monkeypatch):
    observed = {}

    class FakeRunner:
        def __init__(self, env, checkpoint):
            observed["loaded"] = (env, checkpoint)

        def reset(self, done=None):
            observed["reset"] = done

        def action(self, observation):
            observed["observation"] = observation
            return np.asarray([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "isaac_field.policy", types.SimpleNamespace(PolicyRunner=FakeRunner))
    adapter = _Adapter()
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    gui = visualizer._viewer._server.gui

    gui.handles["控制模式"].update("策略控制")
    controls.process_pending()
    assert controls.control_mode == "manual"
    assert not controls.policy_loaded

    gui.handles["检查点路径"].value = "candidate.zip"
    gui.handles["加载检查点"].click()
    controls.process_pending()
    gui.handles["控制模式"].update("策略控制")
    gui.handles["运行任务"].click()
    controls.process_pending()
    np.testing.assert_allclose(controls.action(), [0.1, -0.2, 0.3, -0.4])
    assert controls.control_mode == "policy"

    assert observed["loaded"] == (adapter, "candidate.zip")
    assert observed["observation"]["depth"].shape == (1, 1, 160, 88)
    assert observed["observation"]["state"].shape == (1, STATE_DIM)


def test_initial_checkpoint_loads_without_selecting_policy_or_advancing(monkeypatch):
    loaded = []

    class FakeRunner:
        def __init__(self, env, checkpoint):
            loaded.append(checkpoint)

        def reset(self, done=None):
            pass

    monkeypatch.setitem(sys.modules, "isaac_field.policy", types.SimpleNamespace(PolicyRunner=FakeRunner))
    adapter = _Adapter()
    controls = attach_field_controls(_Visualizer(), adapter, "ready.zip")

    assert loaded == ["ready.zip"]
    assert controls.policy_loaded
    assert controls.control_mode == "manual"
    assert controls.action() is None
    assert adapter.actions == []


def test_official_viewer_reset_is_routed_through_adapter_and_clears_terminal_latch():
    adapter = _Adapter()
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    adapter.done = True
    controls._terminal = True
    adapter.engine.sim.reset_requested = True

    controls.process_pending()

    assert adapter.reset_calls == [(None, {"mode": "search"})]
    assert adapter.done is False
    assert controls._terminal is False
    assert controls.action() is None  # reset does not unpause the local task loop


def test_queued_2d_reset_without_value_resets_episode_and_clears_policy_frame(monkeypatch):
    adapter, visualizer, _server, controls, runner, inspector, _network2d = _policy_viewer_with_2d_client(
        monkeypatch, two_d_clients=True
    )
    adapter.time = 8.25
    adapter.done = True
    adapter.goal = np.asarray([25.0, 31.0], dtype=np.float32)
    controls._terminal = True
    visualizer._viewer._server.gui.handles["任务模式"].update("到点任务")
    controls.process_pending()

    controls._network_panel.action(adapter.current_observation(), adapter.time)
    old_frame = inspector.snapshot
    controls._network_panel.display_snapshot = old_frame
    controls._enqueue("reset")  # The independent 2D page sends no mode/goal tuple.

    processed = controls.process_pending()

    assert processed == 1
    assert adapter.reset_calls == [(None, {"mode": "goto"})]
    assert adapter.time == 0.0
    assert adapter.done is False
    assert adapter.goal_calls == []
    assert controls.task_mode == "goto"
    assert controls._terminal is False
    assert controls._last_reward is None
    assert controls._last_info == {}
    assert runner.reset_calls == [None]
    assert inspector.reset_calls == 1
    assert inspector.snapshot is None
    assert inspector.critic_state is None
    assert all(layer["values"] is None for layer in inspector.layers.values())
    assert controls._network_panel.display_snapshot is None
    assert controls._network_panel._awaiting_transition is False
    np.testing.assert_allclose(adapter.goal, [6.0, 7.0])
    np.testing.assert_allclose(
        [controls.visualizer._viewer._server.gui.handles["目标 X（米）"].value,
         controls.visualizer._viewer._server.gui.handles["目标 Y（米）"].value],
        [6.0, 7.0],
    )


def test_native_reset_racing_with_step_refreshes_state_without_recording_transition():
    adapter = _Adapter()
    adapter.task = types.SimpleNamespace(done=False, time=0.0, mode="search")
    adapter.engine.num_envs = 1
    adapter.engine.cfg = types.SimpleNamespace(auto_reset=False)
    adapter.engine._last_results = [("stale", 9.0, False, False, {"old": True})]
    adapter.engine.reset_terminated = np.asarray([False], dtype=bool)
    adapter.engine.reset_time_outs = np.asarray([False], dtype=bool)
    adapter.engine.episode_length_buf = np.asarray([4], dtype=np.int64)

    def reset_inside_step(action):
        adapter.actions.append(np.asarray(action).copy())
        adapter.engine._last_results[0] = None
        adapter.engine.reset_terminated[0] = True
        adapter.engine.reset_time_outs[0] = False
        adapter.engine.episode_length_buf[0] = 0
        adapter.done = False
        adapter.time = 0.0
        adapter.task.done = False
        adapter.task.time = 0.0
        return None

    adapter.step = reset_inside_step
    controls = attach_field_controls(_Visualizer(), adapter, reset_seed=81001)

    class _Runner:
        def __init__(self):
            self.reset_calls = []
            self.actions = 0

        def action(self, observation):
            self.actions += 1
            return np.asarray([[0.1, 0.0, 0.0, 0.0]], dtype=np.float32)

        def reset(self, done=None):
            self.reset_calls.append(done)

    runner = _Runner()
    controls._runner = runner
    controls._control_mode = "policy"
    controls._running = True
    controls._run_button.label = "暂停任务"
    controls._last_reward = 9.0
    controls._last_info = {"old": True}

    result = controls.step_once()

    assert result is None
    assert runner.actions == 1
    assert runner.reset_calls == [None]
    assert len(adapter.actions) == 1
    assert adapter.reset_calls == []  # Native reset already created the episode.
    assert controls._last_reward is None
    assert controls._last_info == {}
    assert controls._terminal is False
    assert controls._running is False
    assert controls._run_button.label == "运行任务"
    np.testing.assert_array_equal(controls._latest_observation["state"], adapter.current_observation()["state"])
    assert "官方仿真已重置" in controls._status
    assert "**最近奖励：** 未记录" in controls._metrics_handle.content


def test_none_transition_without_native_reset_evidence_raises_diagnostic():
    adapter = _Adapter()
    adapter.step = lambda action: None
    controls = attach_field_controls(_Visualizer(), adapter)
    controls._running = True

    with pytest.raises(RuntimeError, match=r"adapter.step\(\) returned None without a confirmed native reset"):
        controls.step_once()


def test_explicit_episode_seed_repeats_on_each_viewer_reset():
    adapter = _Adapter()
    controls = attach_field_controls(_Visualizer(), adapter, reset_seed=81001)

    controls._apply_task("goto", -100.0, 999.0, reset=True)
    first_goal = adapter.goal.copy()
    controls._apply_task("goto", -100.0, 999.0, reset=True)

    assert adapter.reset_calls == [(81001, {"mode": "goto"}), (81001, {"mode": "goto"})]
    np.testing.assert_allclose(adapter.goal, first_goal)
    assert adapter.goal_calls == []


def test_apply_goal_after_terminal_requires_reset_and_preserves_terminal_state():
    adapter = _Adapter()
    controls = attach_field_controls(_Visualizer(), adapter)
    gui = controls.visualizer._viewer._server.gui
    adapter.done = True
    adapter.time = 20.0
    controls._terminal = True
    gui.handles["任务模式"].update("到点任务")
    gui.handles["目标 X（米）"].value = 14.0
    gui.handles["目标 Y（米）"].value = 15.0
    gui.handles["设置目标 / 应用任务模式"].click()

    controls.process_pending()

    assert adapter.goal_calls == []
    assert adapter.reset_calls == []
    assert adapter.done is True
    assert adapter.time == 20.0
    assert controls._terminal is True
    assert "请先重置任务，再设置目标" in controls._status


def test_terminal_transition_reports_reason_and_stops_the_local_run_loop():
    adapter = _Adapter()
    visualizer = _Visualizer()
    controls = attach_field_controls(visualizer, adapter)
    gui = visualizer._viewer._server.gui
    gui.handles["运行任务"].click()
    controls.process_pending()
    assert controls._running is True
    assert gui.handles["运行任务"].label == "暂停任务"

    adapter.step = lambda action: (
        adapter.current_observation(),
        1.0,
        True,
        False,
        {"reason": "goal_reached", "is_success": True},
    )
    transition = controls.step_once()

    assert transition is not None
    assert controls._terminal is True
    assert controls._running is False
    assert gui.handles["运行任务"].label == "运行任务"
    assert controls._status == "回合结束：已到达目标；结果：成功。"
    assert controls._status_handle.content == controls._status
    assert "回合结束：已到达目标；结果：成功。" in controls._metrics_handle.content


def test_metrics_show_current_and_checkpoint_drive_observation_metadata():
    adapter = _Adapter()
    controls = attach_field_controls(_Visualizer(), adapter)
    controls._checkpoint = "candidate.zip"
    controls._runner = types.SimpleNamespace(
        model=types.SimpleNamespace(
            num_timesteps=32000,
            field_drive_config=dict(adapter.engine.drive_config),
            field_observation_config=dict(adapter.engine.observation_config),
            field_action_config=action_config(),
        )
    )

    controls.publish()

    content = controls._metrics_handle.content
    assert f"增益=0.5；结构版本={INPUT_SCHEMA}" in content
    assert f"**动作协议：** {ACTION_SCHEMA}" in content
    assert "原因=未结束" in content
    assert "步数=32,000；驱动=一致；观测=一致；动作=一致" in content
    assert "当前=stem-contact-v2" in content
    assert "检查点=旧版/未记录（按规则 root-protection-v1 推定）；不一致" in content

    controls._runner.model.field_damage_config = dict(adapter.engine.damage_config)
    controls.publish()
    assert "检查点=一致" in controls._metrics_handle.content

    controls._runner.model.field_damage_config["track_geometry"] = "old-width"
    controls.publish()
    assert "检查点=不一致" in controls._metrics_handle.content
    controls._runner.model.field_damage_config["track_geometry"] = "current"

    controls._runner.model.field_drive_config["velocity_gain"] = 20.0
    controls._runner.model.field_observation_config["schema"] = "legacy-state-v1"
    controls._runner.model.field_damage_config["metric"] = "root-protection-v1"
    controls.publish()
    assert "驱动=不一致；观测=不一致；动作=一致" in controls._metrics_handle.content
    assert "检查点=不一致" in controls._metrics_handle.content

    # Older/fake runners may not expose a model or checkpoint metadata.
    controls._runner = types.SimpleNamespace()
    controls.publish()
    assert "步数=未记录；驱动=旧版/未记录；观测=旧版/未记录；动作=旧版/未记录" in controls._metrics_handle.content
    assert "检查点=旧版/未记录" in controls._metrics_handle.content


def test_native_isaaclab_sidebar_is_translated_without_replacing_callbacks():
    adapter = _Adapter()
    visualizer = _Visualizer()
    gui = visualizer._viewer._server.gui
    pause_rendering = gui.handles["Pause Rendering"]
    pause_simulation = gui.handles["Pause Simulation"]
    rendering_callback = pause_rendering._click
    simulation_callback = pause_simulation._click

    controls = attach_field_controls(visualizer, adapter)

    assert gui.panel_label == "田间仿真"
    assert gui.handles["Physics: Newton MJWarp"].content == "物理引擎：Newton MJWarp"
    assert pause_rendering.label == "暂停渲染"
    assert pause_simulation.label == "暂停仿真"
    assert gui.handles["Reset Episode"].label == "重置回合"
    assert gui.handles["Live Plots"].label == "实时曲线"
    assert gui.handles["Visualization Markers"].label == "可视化标记"
    assert gui.handles["Joints"].label == "关节"
    assert pause_rendering._click is rendering_callback
    assert pause_simulation._click is simulation_callback

    # Isaac Lab's original callback changes these labels back to English.
    pause_rendering.click()
    pause_simulation.click()
    assert pause_rendering.label == "Resume Rendering"
    assert pause_simulation.label == "Resume Simulation"
    assert visualizer.is_rendering_paused() and visualizer.is_training_paused()

    controls._sync_native_pause_labels()
    assert pause_rendering.label == "继续渲染"
    assert pause_simulation.label == "继续仿真"


def test_native_sidebar_translation_degrades_when_private_registry_is_missing():
    adapter = _Adapter()
    visualizer = _Visualizer()
    gui = visualizer._viewer._server.gui
    del gui._container_handle_from_uuid

    controls = attach_field_controls(visualizer, adapter)

    assert controls._run_button.label == "运行任务"
    assert gui.panel_label == "田间仿真"  # public panel API still works
    assert gui.handles["Pause Rendering"].label == "Pause Rendering"
