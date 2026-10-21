"""Viser controls and the capture bridge for the independent 2D policy view."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np


class PolicyNetworkPanel:
    """Collect real policy tensors and publish them to the standalone 2D page."""

    def __init__(self, gui, enqueue, *, server=None):
        self._server = server
        self._network2d = None
        self._runtime_state = {}
        self.enqueue = enqueue
        self.inspector = None
        self.enabled = True
        self.display_snapshot = None
        self._last_publish = 0.0
        self._dirty = True
        self._export_message = ""
        self._awaiting_transition = False

        with gui.add_folder("模型实时激活 · 二维", expand_by_default=True):
            self.page_link = gui.add_markdown("二维展开页将在查看器启动后提供。")
            run = gui.add_button("运行策略 / 继续")
            run.on_click(lambda _event: self.enqueue("network_run"))
            pause = gui.add_button("暂停车辆")
            pause.on_click(lambda _event: self.enqueue("network_pause"))
            preview = gui.add_button("读取当前帧（不推进仿真）")
            preview.on_click(lambda _event: self.enqueue("network_preview"))
            export = gui.add_button("导出当前帧与模型权重")
            export.on_click(lambda _event: self.enqueue("network_export"))
            self.enabled_handle = gui.add_checkbox("采集实时激活", initial_value=True)
            self._on_client_update(
                self.enabled_handle,
                lambda event: self.enqueue("network_enabled", bool(event.target.value)),
            )
            self.status = gui.add_markdown("尚未加载策略。")

    @staticmethod
    def _on_client_update(handle, callback):
        """Ignore Viser's synthetic callbacks from programmatic value writes."""

        def dispatch(event):
            if getattr(event, "client_id", None) is not None:
                callback(event)

        handle.on_update(dispatch)

    def start_2d(self, port, viewer_url):
        """Start the independent localhost page that renders the captured frame."""
        from .network_2d import Network2DServer

        if self._network2d is not None:
            self.close()
        self._network2d = Network2DServer(port, self.enqueue, viewer_url)
        if self.inspector is not None:
            self._network2d.bind(self.inspector)
        self.page_link.content = f"**[打开独立二维展开页]({self._network2d.url})**"
        self.publish(force=True)

    def close(self):
        if self._network2d is not None:
            self._network2d.close()
            self._network2d = None

    def has_2d_clients(self):
        return self._network2d is not None and self._network2d.has_clients()

    def runtime_state(self, payload, *, running, status):
        robot = payload.get("robot", {})
        model = getattr(getattr(self.inspector, "runner", None), "model", None)
        training_steps = getattr(model, "num_timesteps", None)
        updated_state = {
            "running": bool(running),
            "time": float(payload.get("time", 0.0)),
            "robot": dict(robot) if isinstance(robot, dict) else {},
            "done": bool(payload.get("done", False)),
            "status": str(status),
            "training_steps": int(training_steps) if isinstance(training_steps, (int, np.integer)) else None,
        }
        if any(
            self._runtime_state.get(key) != updated_state[key]
            for key in ("running", "done", "status")
        ):
            self._dirty = True
        self._runtime_state = updated_state

    def bind(self, runner):
        # Lazy import keeps the ordinary scene viewer usable without torch/SB3.
        from .inspection import PolicyInspector

        self.inspector = PolicyInspector(runner)
        if self._network2d is not None:
            self._network2d.bind(self.inspector)
        self.display_snapshot = None
        self._awaiting_transition = False
        self._dirty = True
        self.publish(force=True)

    def reset(self):
        self._awaiting_transition = False
        if self.inspector is not None:
            self.inspector.reset()
        self.display_snapshot = None
        self._dirty = True
        self.publish(force=True)

    def command(self, command, value, observation, sim_time):
        if command == "network_enabled":
            enabled = bool(value)
            if enabled != self.enabled:
                self.enabled = enabled
                self.reset()
            return
        if command == "network_preview":
            if self.inspector is not None and observation is not None:
                self.inspector.preview(observation, sim_time)
                self._dirty = True
                self.publish(force=True)
            return
        if command == "network_export":
            self.export()
            self._dirty = True
            self.publish(force=True)
            return
        raise ValueError(f"Unsupported policy network command: {command}")

    def action(self, observation, sim_time):
        if self.inspector is None:
            raise RuntimeError("No policy inspector is bound")
        action = self.inspector.action(observation, sim_time)
        self._awaiting_transition = True
        return action

    def record_actuators(self, adapter):
        """Append actual simulator outputs to the action frame that caused them."""
        if not self._awaiting_transition or self.inspector is None or self.inspector.snapshot is None:
            return
        self._awaiting_transition = False
        engine = getattr(adapter, "engine", None)
        action_layer = self.inspector.layers.get("action.applied", {})
        action_column = float(action_layer.get("column", 15.0))
        definitions = (
            ("actuator.tracks", "左右履带限速指令", "track_command", action_column + 1,
             "实际引擎回读，单位米/秒；动作前进±差速后同比分配，再经加速度限制；不是实测履带速度",
             ("action.applied",)),
            ("actuator.wheels", "14滚轮目标角速度", "wheel_target_velocity", action_column + 2,
             "实际执行器目标回读，单位弧度/秒；履带指令除以代理滚轮半径45mm；不是实测关节转速",
             ("actuator.tracks",)),
            ("actuator.gimbal", "云台目标角", "gimbal_target", action_column + 1,
             "实际执行器目标回读，偏航/俯仰单位弧度；速率积分后执行角度范围约束",
             ("action.applied",)),
        )
        for key, title, attribute, column, description, parents in definitions:
            value = getattr(engine, attribute, None)
            if value is None:
                continue
            if callable(getattr(value, "detach", None)):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.ndim > 1:
                value = value[0]
            self.inspector.layers[key] = dict(
                key=key,
                title=title,
                description=description,
                branch="output",
                column=float(column),
                parents=parents,
                module=None,
                summary=False,
                source="实际仿真步后回读",
                values=value.copy(),
            )
        payload = adapter.state_payload()
        robot = payload.get("robot", {})
        if "v" in robot and "w" in robot:
            key = "motion.actual"
            self.inspector.layers[key] = dict(
                key=key,
                title="车辆实际运动反馈",
                description="物理引擎测得的车体前向速度（米/秒）和偏航角速度（弧度/秒）；指令不等于实际运动",
                branch="output",
                column=action_column + 3,
                parents=("action.applied",),
                module=None,
                summary=True,
                source="实际仿真步后回读",
                values=np.asarray((robot["v"], robot["w"]), dtype=np.float32),
            )
        self.inspector.snapshot["output_time"] = float(payload.get("time", 0.0))
        self.inspector.snapshot["layers"] = list(self.inspector.layers.values())

    def publish(self, *, force=False):
        if self.inspector is None:
            return
        now = time.monotonic()
        # Limit ordinary status/frame publication to four per second. Explicit
        # binds, resets, previews, and completed transitions publish immediately.
        if not force and not self._dirty and now - self._last_publish < 0.25:
            return

        snapshot = self.inspector.snapshot
        layers = list(self.inspector.layers.values())
        self.display_snapshot = snapshot
        if self._network2d is not None:
            self._network2d.publish(snapshot, layers, state=self._runtime_state)

        if snapshot is None:
            status = "策略结构已加载 · 等待真实前向"
        else:
            frame_status = f"帧 {snapshot['sequence']} · 输入 {float(snapshot['sim_time']):.3f}s"
            output_time = snapshot.get("output_time")
            if output_time is not None:
                frame_status += f" → 执行反馈 {float(output_time):.3f}s"
            if self._runtime_state:
                runtime = "运行中" if self._runtime_state.get("running", False) else "已暂停"
                frame_status += f" · 仿真 {float(self._runtime_state.get('time', 0.0)):.2f}s · {runtime}"
            errors = snapshot.get("errors") or ()
            capture = "正常" if not errors else f"异常 {len(errors)} 项"
            status = f"{frame_status} · 激活采集{'开启' if self.enabled else '关闭'} · {capture}"
        if self._export_message:
            status += f" · {self._export_message}"
        self.status.content = status
        self._last_publish = now
        self._dirty = False

    def export(self):
        if self.inspector is None or self.display_snapshot is None:
            raise ValueError("尚无激活帧可导出，请先读取当前帧或运行策略")
        folder = Path(__file__).resolve().parents[2] / "results/isaac_field_viewer/network_exports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (datetime.now().strftime("frame_%Y%m%d_%H%M%S_%f") + ".npz")
        snapshot = self.display_snapshot
        arrays = {
            "activation/" + layer["key"]: layer["values"]
            for layer in snapshot["layers"]
            if layer["values"] is not None
        }
        arrays.update(
            {
                "weight/" + name: value.detach().cpu().numpy()
                for name, value in self.inspector.policy.state_dict().items()
            }
        )
        np.savez_compressed(path, **arrays)
        metadata = {key: value for key, value in snapshot.items() if key != "layers"}
        metadata["checkpoint_label"] = str(getattr(self.inspector.runner, "label", ""))
        metadata["layers"] = [
            {key: value for key, value in layer.items() if key != "values"}
            | {"shape": list(layer["values"].shape) if layer["values"] is not None else None}
            for layer in snapshot["layers"]
        ]
        path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._export_message = f"已导出 {path.name}"


__all__ = ["PolicyNetworkPanel"]
