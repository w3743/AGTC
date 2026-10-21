"""CPU-only tests for the policy capture bridge and its compact Viser controls."""

from __future__ import annotations

from collections import OrderedDict
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from isaac_field.action_contract import action_config
from isaac_field.feedback import STATE_DIM
from isaac_field.visualization.network_panel import PolicyNetworkPanel


class FakeHandle:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)
        self._callbacks = []
        self._click_callbacks = []

    def on_update(self, callback):
        self._callbacks.append(callback)

    def on_click(self, callback):
        self._click_callbacks.append(callback)

    def update(self, value):
        self.value = value
        event = SimpleNamespace(client_id=1, target=self)
        for callback in tuple(self._callbacks):
            callback(event)

    def click(self):
        for callback in tuple(self._click_callbacks):
            callback(None)


class FakeFolder:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeGui:
    def __init__(self):
        self.controls = {}

    def add_folder(self, _label, **_kwargs):
        return FakeFolder()

    def add_markdown(self, content):
        return self._add("markdown", content=content)

    def add_button(self, label):
        return self._add(label)

    def add_checkbox(self, label, *, initial_value):
        return self._add(label, value=initial_value)

    def _add(self, label, **attrs):
        handle = FakeHandle(**attrs)
        self.controls[label] = handle
        return handle


class FakeInspector:
    def __init__(self):
        self.weight = np.asarray([[1.0, 2.0], [7.0, 4.0]], dtype=np.float32)
        self.layers = OrderedDict(
            (
                ("input.state", self._layer("input.state", "状态输入", "state", 0, summary=True)),
                ("features_extractor", self._layer("features_extractor", "特征编码", "fusion", 1,
                                                   parents=("input.state",), summary=True)),
                ("features_extractor.proj", self._layer("features_extractor.proj", "线性层权重", "fusion", 2,
                                                         parents=("features_extractor",))),
            )
        )
        self.snapshot = None
        self.critic_state = {"h": np.asarray([0.1]), "c": np.asarray([0.2])}
        self.reset_count = 0
        self.preview_count = 0
        self.action_count = 0
        self.policy = SimpleNamespace(state_dict=lambda: {})
        self.runner = SimpleNamespace(label="fake checkpoint")

    @staticmethod
    def _layer(key, title, branch, column, *, parents=(), summary=False):
        return {
            "key": key,
            "title": title,
            "description": f"fake {title}",
            "branch": branch,
            "column": float(column),
            "parents": parents,
            "module": None,
            "summary": summary,
            "source": "测试快照",
            "values": None,
        }

    def parameters(self, key):
        if key == "features_extractor.proj":
            return {"weight": self.weight.copy(), "bias": np.asarray([0.25, -0.5], dtype=np.float32)}
        return {}

    def _make_snapshot(self, *, sequence=5, features=(1.0, -2.0, 0.5), mode="runner.action"):
        self.layers["input.state"]["values"] = np.arange(STATE_DIM, dtype=np.float32)
        self.layers["features_extractor"]["values"] = np.asarray(features, dtype=np.float32)
        self.snapshot = {
            "sequence": sequence,
            "sim_time": 1.25,
            "mode": mode,
            "critic": "critic state captured",
            "errors": (),
            "layers": list(self.layers.values()),
        }
        return self.snapshot

    def reset(self):
        self.reset_count += 1
        self.snapshot = None
        self.critic_state = None
        for layer in self.layers.values():
            layer["values"] = None

    def action(self, _observation, _sim_time):
        self.action_count += 1
        self._make_snapshot(sequence=self.action_count)
        return np.asarray([0.5, -0.25, 0.1, -0.1], dtype=np.float32)

    def preview(self, _observation, _sim_time):
        self.preview_count += 1
        return self._make_snapshot(sequence=100 + self.preview_count, mode="暂停预览（未施加车辆）")


class Fake2DServer:
    def __init__(self, port, enqueue, viewer_url):
        self.port = port
        self.enqueue = enqueue
        self.viewer_url = viewer_url
        self.url = f"http://127.0.0.1:{port}/"
        self.bound = []
        self.published = []
        self.clients = False
        self.closed = False

    def bind(self, inspector):
        self.bound.append(inspector)

    def publish(self, snapshot, layers, *, state):
        self.published.append((snapshot, layers, state))

    def has_clients(self):
        return self.clients

    def close(self):
        self.closed = True


def _panel(monkeypatch, *, server=None, inspector=None):
    gui = FakeGui()
    queue = []
    inspector = inspector or FakeInspector()
    inspection_module = ModuleType("isaac_field.visualization.inspection")
    inspection_module.PolicyInspector = lambda _runner: inspector
    monkeypatch.setitem(sys.modules, "isaac_field.visualization.inspection", inspection_module)
    panel = PolicyNetworkPanel(gui, lambda *item: queue.append(item), server=server)
    panel.bind(object())
    return gui, queue, inspector, panel


def test_gui_contains_only_2d_page_controls_and_programmatic_events_are_guarded(monkeypatch):
    gui, queue, _inspector, panel = _panel(monkeypatch)

    assert set(gui.controls) == {
        "markdown",
        "运行策略 / 继续",
        "暂停车辆",
        "读取当前帧（不推进仿真）",
        "导出当前帧与模型权重",
        "采集实时激活",
    }
    assert not hasattr(panel, "scene")
    assert not hasattr(panel, "frozen")
    panel.enabled_handle.value = False  # programmatic writes do not enqueue
    assert queue == []

    gui.controls["运行策略 / 继续"].click()
    gui.controls["暂停车辆"].click()
    gui.controls["读取当前帧（不推进仿真）"].click()
    gui.controls["导出当前帧与模型权重"].click()
    gui.controls["采集实时激活"].update(False)
    assert queue == [
        ("network_run",),
        ("network_pause",),
        ("network_preview",),
        ("network_export",),
        ("network_enabled", False),
    ]


def test_real_viser_programmatic_checkbox_write_is_ignored_but_client_event_enqueues(monkeypatch):
    viser = pytest.importorskip("viser")
    from viser._gui_handles import GuiEvent

    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    queue = []
    try:
        panel = PolicyNetworkPanel(server.gui, lambda *item: queue.append(item))

        # The installed Viser value setter synchronously calls update handlers
        # with client_id=None. The panel must not feed this back to its queue.
        panel.enabled_handle.value = False
        assert queue == []

        callback = panel.enabled_handle._impl.update_cb[-1]
        callback(GuiEvent(client_id=17, client=None, target=panel.enabled_handle))
        assert queue == [("network_enabled", False)]
    finally:
        server.stop()


def test_start_bind_publish_and_close_keep_latest_frame_and_runtime_state(monkeypatch):
    import isaac_field.visualization.network_2d as network_2d

    monkeypatch.setattr(network_2d, "Network2DServer", Fake2DServer)
    gui, _queue, inspector, panel = _panel(monkeypatch)
    panel.start_2d(9123, "http://127.0.0.1:8769/")
    server = panel._network2d
    assert isinstance(server, Fake2DServer)
    assert server.bound == [inspector]
    assert server.viewer_url == "http://127.0.0.1:8769/"
    assert server.published[-1][0] is None
    assert "http://127.0.0.1:9123/" in panel.page_link.content
    assert panel.has_2d_clients() is False
    server.clients = True
    assert panel.has_2d_clients() is True

    inspector.runner.model = SimpleNamespace(num_timesteps=0)
    panel.runtime_state(
        {"time": 12.5, "robot": {"x": 4.0, "y": 5.0}, "done": False},
        running=True,
        status="继续运行",
    )
    first = inspector._make_snapshot(sequence=5, features=(1.0, 2.0, 3.0))
    panel.publish(force=True)
    second = inspector._make_snapshot(sequence=6, features=(9.0, 8.0, 7.0))
    panel.publish(force=True)

    assert panel.display_snapshot is second
    snapshot, layers, state = server.published[-1]
    assert snapshot is second
    np.testing.assert_array_equal(
        next(layer["values"] for layer in layers if layer["key"] == "features_extractor"),
        [9.0, 8.0, 7.0],
    )
    assert state == {
        "running": True,
        "time": 12.5,
        "robot": {"x": 4.0, "y": 5.0},
        "done": False,
        "status": "继续运行",
        "training_steps": 0,
    }
    assert "帧 6" in panel.status.content
    assert "输入 1.250s" in panel.status.content
    assert "采集开启" in panel.status.content

    panel.close()
    assert server.closed is True
    assert panel._network2d is None
    assert panel.has_2d_clients() is False


def test_publish_limits_ordinary_updates_to_four_hz_but_force_is_immediate(monkeypatch):
    import isaac_field.visualization.network_panel as network_panel

    clock = [10.0]
    monkeypatch.setattr(network_panel.time, "monotonic", lambda: clock[0])
    _gui, _queue, inspector, panel = _panel(monkeypatch)
    server = Fake2DServer(9000, lambda *_args: None, "http://127.0.0.1:8769/")
    panel._network2d = server
    panel.runtime_state({"time": 0.0, "robot": {}, "done": False}, running=True, status="运行中")
    inspector._make_snapshot(sequence=1)
    panel.publish(force=True)
    assert len(server.published) == 1

    clock[0] += 0.1
    inspector._make_snapshot(sequence=2)
    panel.runtime_state({"time": 0.1, "robot": {"x": 1.0}, "done": False}, running=True, status="运行中")
    panel.publish()
    assert len(server.published) == 1
    assert panel.display_snapshot["sequence"] == 1

    # A real run/status transition bypasses the ordinary frame-rate limit.
    panel.runtime_state({"time": 0.2, "robot": {"x": 2.0}, "done": True}, running=False, status="完成")
    panel.publish()
    assert len(server.published) == 2
    assert server.published[-1][0]["sequence"] == 2

    clock[0] += 0.1
    inspector._make_snapshot(sequence=3)
    panel.runtime_state({"time": 0.3, "robot": {"x": 3.0}, "done": True}, running=False, status="完成")
    panel.publish()
    assert len(server.published) == 2
    clock[0] = 10.351
    panel.publish()
    assert len(server.published) == 3
    assert server.published[-1][0]["sequence"] == 3


def test_action_records_transition_time_without_mechanical_or_motion_layers(monkeypatch):
    _gui, _queue, inspector, panel = _panel(monkeypatch)
    inspector.layers["action.applied"] = inspector._layer(
        "action.applied", "四维输出动作", "output", 24, parents=("action_net",), summary=True
    )
    action = panel.action({"state": np.zeros((1, STATE_DIM), dtype=np.float32)}, 2.0)
    np.testing.assert_allclose(action, [0.5, -0.25, 0.1, -0.1])
    assert panel._awaiting_transition is True

    adapter = SimpleNamespace(
        engine=SimpleNamespace(
            track_command=np.asarray([0.4, 0.2]),
            wheel_target_velocity=np.arange(14, dtype=np.float32),
            gimbal_target=np.asarray([0.1, -0.2]),
        ),
        state_payload=lambda: {"time": 2.05, "robot": {"v": 0.3, "w": 0.02}},
    )
    panel.record_action_transition(adapter)
    layers = {layer["key"]: layer for layer in inspector.snapshot["layers"]}
    assert not {"actuator.tracks", "actuator.wheels", "actuator.gimbal", "motion.actual"} & layers.keys()
    assert inspector.snapshot["output_time"] == 2.05
    assert panel._awaiting_transition is False

    panel.publish(force=True)
    assert panel.display_snapshot is inspector.snapshot
    panel.reset()
    assert inspector.reset_count == 1
    assert inspector.snapshot is None
    assert inspector.critic_state is None
    assert panel.display_snapshot is None
    assert "等待真实前向" in panel.status.content


def test_command_handles_only_enabled_preview_and_export(monkeypatch):
    _gui, _queue, inspector, panel = _panel(monkeypatch)
    panel.command("network_preview", None, {"state": np.zeros((1, STATE_DIM))}, 1.25)
    assert inspector.preview_count == 1
    assert panel.display_snapshot["mode"].startswith("暂停预览")

    panel.command("network_enabled", False, None, 1.25)
    assert panel.enabled is False
    assert inspector.reset_count == 1
    assert panel.display_snapshot is None

    with pytest.raises(ValueError, match="Unsupported policy network command"):
        panel.command("network_focus", None, None, 1.25)


def test_export_preserves_results_location_and_full_activation_and_state_dict_arrays(monkeypatch, tmp_path):
    import torch
    import isaac_field.visualization.network_panel as module

    _gui, _queue, inspector, panel = _panel(monkeypatch)
    inspector._make_snapshot(sequence=8)
    inspector.snapshot["action_config"] = action_config()
    inspector.policy = SimpleNamespace(
        state_dict=lambda: {"real.weight": torch.tensor([[1.0, -3.0]])}
    )
    inspector.runner = SimpleNamespace(label="test checkpoint")
    panel.publish(force=True)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "isaac_field/visualization/network_panel.py"))

    panel.export()

    files = list((tmp_path / "results/isaac_field_viewer/network_exports").glob("*.npz"))
    assert len(files) == 1
    assert not (tmp_path / "isaac_field/results").exists()
    with np.load(files[0], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["activation/input.state"], np.arange(STATE_DIM, dtype=np.float32))
        np.testing.assert_array_equal(archive["activation/features_extractor"], [1.0, -2.0, 0.5])
        np.testing.assert_array_equal(archive["weight/real.weight"], [[1.0, -3.0]])
    metadata = json.loads(files[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["sequence"] == 8
    assert metadata["action_config"] == action_config()
    assert metadata["checkpoint_label"] == "test checkpoint"
    assert next(layer for layer in metadata["layers"] if layer["key"] == "input.state")["shape"] == [STATE_DIM]
