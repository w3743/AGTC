"""Task controls and policy monitoring for Isaac Lab's official Viser viewer.

This module extends the Viser sidebar and adds a separate policy-layer display;
Isaac Lab/Newton remains responsible for rendering the scene and robot. GUI callbacks enqueue requests because Viser
callbacks run outside the simulation-owner thread. Call ``process_pending`` and
``step_once`` (or ``run``) from the thread that owns the environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from queue import Empty, SimpleQueue
from threading import Event
import time
from typing import Any

import numpy as np

from .action_contract import ACTION_LABELS, ACTION_SCHEMA, action_config
from .task import DAMAGE_METRIC_ROOT_PROTECTION_V1


_CONTROL_MODE_LABELS = {"manual": "手动控制", "policy": "策略控制"}
_CONTROL_MODE_VALUES = {label: value for value, label in _CONTROL_MODE_LABELS.items()}
_CONTROL_MODE_OPTIONS = tuple(_CONTROL_MODE_LABELS.values())
_TASK_MODE_LABELS = {"search": "搜索任务", "goto": "到点任务"}
_TASK_MODE_VALUES = {label: value for value, label in _TASK_MODE_LABELS.items()}
_TASK_MODE_OPTIONS = tuple(_TASK_MODE_LABELS.values())
_CONTROL_MODES = tuple(_CONTROL_MODE_LABELS)
_TASK_MODES = tuple(_TASK_MODE_LABELS)
_MANUAL_ACTION_LABELS = ACTION_LABELS
_NATIVE_GUI_LABELS = {
    "Pause Rendering": "暂停渲染",
    "Resume Rendering": "继续渲染",
    "Pause Simulation": "暂停仿真",
    "Resume Simulation": "继续仿真",
    "Reset Episode": "重置回合",
    "Live Plots": "实时曲线",
    "Visualization Markers": "可视化标记",
    "Joints": "关节",
    "Contacts": "接触",
    "Center of Mass": "质心",
    "Particles": "粒子",
    "Visual": "视觉几何",
    "Collision": "碰撞几何",
    "Springs": "弹簧",
    "Cloth": "布料",
    "Inertia Boxes": "惯性盒",
}
_NATIVE_PAUSE_BUTTONS = {
    "rendering": {"Pause Rendering", "Resume Rendering"},
    "simulation": {"Pause Simulation", "Resume Simulation"},
}
_METADATA_STATUS_LABELS = {
    "not loaded": "未加载",
    "match": "一致",
    "mismatch": "不一致",
    "legacy/unrecorded": "旧版/未记录",
    "unrecorded": "未记录",
    "unavailable": "不可用",
}
_TERMINAL_REASON_LABELS = {
    "all_targets_found_by_evaluator": "已找到全部目标",
    "goal_reached": "已到达目标",
    "plant_damage": "作物受损",
    "time_limit": "达到时限",
    "outside_or_overturned": "越界或倾覆",
    "terminated": "仿真终止",
    "truncated": "仿真截断",
}
_UNOBSERVED_CLIENTS = object()
_LIDAR_CLOUD_NAME = "/FieldTask/Sensors/LidarReturns"
_DEPTH_CLOUD_NAME = "/FieldTask/Sensors/DepthReturns"
_LIDAR_RING_NAME = "/FieldTask/Sensors/LidarMaxRange"
_LIDAR_COLOR = np.asarray((0.0, 1.0, 1.0), dtype=np.float32)
_DEPTH_COLOR = np.asarray((0.12, 0.45, 1.0), dtype=np.float32)
_LIDAR_RING_COLOR = np.asarray((1.0, 0.58, 0.12), dtype=np.float32)


def _registered_gui_handles(gui) -> tuple[Any, ...] | None:
    """Read Viser 1.0.30's GUI tree when available, without requiring it.

    Viser exposes writable handle properties but no public API to retrieve GUI
    elements created by Isaac Lab. The registry is private, so this traversal
    is deliberately feature-detected and callers must tolerate ``None``.
    """
    containers = getattr(gui, "_container_handle_from_uuid", None)
    if not isinstance(containers, Mapping):
        return None
    root = containers.get("root")
    children = getattr(root, "_children", None)
    if not isinstance(children, Mapping):
        return None

    result = []
    seen: set[int] = set()
    stack = list(children.values())
    while stack:
        handle = stack.pop()
        if id(handle) in seen:
            continue
        seen.add(id(handle))
        result.append(handle)
        nested = getattr(handle, "_children", None)
        if isinstance(nested, Mapping):
            stack.extend(nested.values())
    return tuple(result)


def _metadata_status_label(status: str) -> str:
    """Translate compatibility states while retaining raw metric identifiers."""
    legacy_prefix = "legacy/unrecorded (assume "
    if status.startswith(legacy_prefix) and "); " in status:
        assumed, result = status[len(legacy_prefix) :].split("); ", 1)
        return f"旧版/未记录（按规则 {assumed} 推定）；{_METADATA_STATUS_LABELS.get(result, result)}"
    return _METADATA_STATUS_LABELS.get(status, status)


def _terminal_reason_label(reason: str) -> str:
    """Translate known terminal codes for display; preserve unknown diagnostics."""
    return _TERMINAL_REASON_LABELS.get(reason, reason or "未结束")


def depth_to_rgb(depth_observation: np.ndarray) -> np.ndarray:
    """Colorize the policy's single normalized camera-Z depth channel.

    The policy contract is ``(1, H, W)``; zero denotes invalid measurements.
    Invalid pixels are black, including non-finite diagnostic inputs.
    """
    image = np.asarray(depth_observation, dtype=np.float32)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("深度观测的形状必须为 (1, height, width) 或 (1, 1, height, width)")

    distance = np.nan_to_num(image[0], nan=1.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    valid = np.isfinite(image[0]) & (image[0] > 0.0)
    # A compact red-yellow-green-cyan-blue range ramp: near objects are warm,
    # farther valid pixels are cool. The image stays at policy-native HxW.
    stops = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float32)
    colors = np.asarray(
        ((255, 32, 32), (255, 210, 0), (32, 235, 32), (0, 200, 255), (32, 64, 255)),
        dtype=np.float32,
    )
    rgb = np.stack([np.interp(distance, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    rgb[~valid] = 0
    return np.rint(rgb).astype(np.uint8)


def _observation_batch(observation: dict[str, Any], observation_space) -> dict[str, np.ndarray]:
    """Ensure policy inputs have the single-environment vector batch dimension."""
    spaces = getattr(observation_space, "spaces", {})
    result = {}
    for name, value in observation.items():
        array = np.asarray(value)
        expected = tuple(spaces[name].shape) if name in spaces else None
        if expected is not None and tuple(array.shape) == expected:
            array = array[None, ...]
        result[name] = array
    return result


def _metadata_value(value: Any) -> Any:
    """Normalize checkpoint/config values for a stable, JSON-like comparison."""
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class FieldViewerControls:
    """Small task sidebar layered on an initialized Isaac Lab ViserVisualizer.

    The controller never calls the environment from a Viser callback. The main
    simulation loop calls ``process_pending`` and ``step_once``. The official
    ViserVisualizer's built-in Pause/Resume control is honored by ``step_once``.
    """

    def __init__(self, visualizer, adapter, checkpoint: str | None = None, *, reset_seed: int | None = None):
        self.visualizer = visualizer
        self.adapter = adapter
        self._reset_seed = reset_seed
        self._commands: SimpleQueue[tuple[str, Any]] = SimpleQueue()
        self._wake_event = Event()
        self._last_client_signature: (
            tuple[tuple[str, str], ...] | None | object
        ) = _UNOBSERVED_CLIENTS
        self._control_mode = "manual"
        self._task_mode = getattr(adapter, "mode", "search")
        if self._task_mode not in _TASK_MODES:
            self._task_mode = "search"
        self._manual_action = np.zeros(4, dtype=np.float32)
        self._runner = None
        self._checkpoint = checkpoint
        # Keep the physics/action loop stopped until the operator explicitly
        # starts it.  Loading a checkpoint is never an implicit run command.
        self._running = False
        self._terminal = False
        self._last_reward: float | None = None
        self._last_info: dict[str, Any] = {}
        self._latest_observation: dict[str, np.ndarray] | None = None
        self._overlay_handles: dict[str, Any] = {}
        self._status = "手动控制已暂停。点击“运行任务”以推进仿真。"
        self._native_pause_handles: dict[str, list[Any]] = {"rendering": [], "simulation": []}

        viewer = getattr(visualizer, "_viewer", None)
        server = getattr(viewer, "_server", None)
        gui = getattr(server, "gui", None)
        if gui is None:
            raise RuntimeError("必须先初始化 ViserVisualizer，才能添加田间任务控制面板")
        configure_theme = getattr(gui, "configure_theme", None)
        if callable(configure_theme):
            configure_theme(dark_mode=True, control_layout="collapsible", control_width="medium",
                            brand_color=(87, 125, 204), show_share_button=False)
        set_background = getattr(getattr(server, "scene", None), "set_background_image", None)
        if callable(set_background):
            set_background(np.full((1, 1, 3), (12, 16, 25), dtype=np.uint8), format="png")
        from .visualization.network_panel import PolicyNetworkPanel
        self._network_panel = PolicyNetworkPanel(gui, self._enqueue, server=server)
        self._translate_native_sidebar(gui)
        self._build_sidebar(gui, checkpoint)
        if checkpoint:
            try:
                self._load_checkpoint(checkpoint, select_policy=False)
                self._status += " 当前为手动控制；选择“策略控制”并点击“运行任务”后才会执行策略。"
            except Exception as exc:
                self._status = f"加载检查点失败：{exc}。仿真保持在手动暂停状态。"
        self._sync_native_pause_labels()
        self._focus_vehicle()
        self.publish()

    @property
    def control_mode(self) -> str:
        return self._control_mode

    @property
    def task_mode(self) -> str:
        return self._task_mode

    @property
    def checkpoint(self) -> str | None:
        return self._checkpoint

    def _checkpoint_metadata(self) -> tuple[str, str, str, str, str, str]:
        """Return current drive/schema and optional checkpoint compatibility data."""
        engine = getattr(self.adapter, "engine", None)
        try:
            drive_config = getattr(engine, "drive_config", None)
        except Exception:
            drive_config = None
        try:
            observation_config = getattr(engine, "observation_config", None)
        except Exception:
            observation_config = None

        drive_config = drive_config if isinstance(drive_config, Mapping) else None
        observation_config = observation_config if isinstance(observation_config, Mapping) else None
        gain = drive_config.get("velocity_gain") if drive_config is not None else None
        schema = observation_config.get("schema") if observation_config is not None else None
        if isinstance(gain, (int, float, np.integer, np.floating)) and np.isfinite(gain):
            gain_text = f"{float(gain):g}"
        else:
            gain_text = "unavailable"
        schema_text = str(schema) if schema is not None else "unavailable"

        if self._runner is None:
            return gain_text, schema_text, "not loaded", "not loaded", "not loaded", "not loaded"

        model = getattr(self._runner, "model", None)
        raw_steps = getattr(model, "num_timesteps", None) if model is not None else None
        try:
            steps_text = f"{int(raw_steps):,}" if raw_steps is not None else "unrecorded"
        except (TypeError, ValueError, OverflowError):
            steps_text = "unrecorded"

        def compatibility(field_name: str, current: Mapping | None) -> str:
            if model is None:
                return "legacy/unrecorded"
            recorded = getattr(model, field_name, None)
            if not isinstance(recorded, Mapping):
                return "legacy/unrecorded"
            if current is None:
                return "unavailable"
            return (
                "match"
                if _metadata_value(recorded) == _metadata_value(current)
                else "mismatch"
            )

        return (
            gain_text,
            schema_text,
            steps_text,
            compatibility("field_drive_config", drive_config),
            compatibility("field_observation_config", observation_config),
            compatibility("field_action_config", action_config()),
        )

    def _damage_metadata(self) -> tuple[str, str]:
        """Show the active damage metric and whether loaded weights record it."""
        task = getattr(self.adapter, "task", None)
        current = getattr(task, "damage_config", None) if task is not None else None
        if not isinstance(current, Mapping):
            current = getattr(self.adapter, "damage_config", None)
        if not isinstance(current, Mapping):
            current = getattr(getattr(self.adapter, "engine", None), "damage_config", None)
        current_metric = current.get("metric") if isinstance(current, Mapping) else None
        current_text = str(current_metric) if current_metric is not None else "unrecorded"

        if self._runner is None:
            return current_text, "not loaded"
        model = getattr(self._runner, "model", None)
        recorded = getattr(model, "field_damage_config", None) if model is not None else None
        recorded_metric = recorded.get("metric") if isinstance(recorded, Mapping) else None
        if recorded_metric is not None:
            return current_text, (
                "match"
                if isinstance(current, Mapping)
                and _metadata_value(recorded) == _metadata_value(current)
                else "mismatch"
            )

        # Checkpoints predating this metadata are treated as legacy v1 for
        # comparison only; loading never changes the environment's active rule.
        assumed_status = (
            "match" if current_metric == DAMAGE_METRIC_ROOT_PROTECTION_V1 else "mismatch"
        )
        return current_text, f"legacy/unrecorded (assume {DAMAGE_METRIC_ROOT_PROTECTION_V1}); {assumed_status}"

    @property
    def policy_loaded(self) -> bool:
        return self._runner is not None

    def _build_sidebar(self, gui, checkpoint: str | None) -> None:
        cfg = getattr(self.adapter, "cfg", None)
        size = float(getattr(cfg, "size", 100.0))
        depth_height = int(getattr(cfg, "depth_height", 160))
        depth_width = int(getattr(cfg, "depth_width", 88))
        with gui.add_folder("任务设置与传感器", expand_by_default=False):
            gui.add_markdown("Isaac Lab / Newton 任务控制面板。场景与车辆由官方 Viser 实时渲染。")
            self._status_handle = gui.add_markdown(self._status)
            self._run_button = gui.add_button("运行任务")
            self._run_button.on_click(lambda _: self._enqueue("toggle_run"))
            self._control_mode_handle = gui.add_dropdown(
                "控制模式", _CONTROL_MODE_OPTIONS, initial_value=_CONTROL_MODE_LABELS["manual"]
            )
            control = self._control_mode_handle
            control.on_update(lambda event: self._enqueue("control_mode", event.target.value))

            self._task_mode_handle = gui.add_dropdown(
                "任务模式", _TASK_MODE_OPTIONS, initial_value=_TASK_MODE_LABELS[self._task_mode]
            )
            self._task_mode_handle.on_update(lambda event: self._enqueue("task_mode", event.target.value))
            goal = self._current_goal(size)
            self._goal_x = gui.add_number("目标 X（米）", initial_value=float(goal[0]), min=0.0, max=size, step=0.1)
            self._goal_y = gui.add_number("目标 Y（米）", initial_value=float(goal[1]), min=0.0, max=size, step=0.1)
            apply_goal = gui.add_button("设置目标 / 应用任务模式")
            apply_goal.on_click(
                lambda _: self._enqueue(
                    "apply_task",
                    (
                        _TASK_MODE_VALUES.get(str(self._task_mode_handle.value), str(self._task_mode_handle.value)),
                        float(self._goal_x.value),
                        float(self._goal_y.value),
                    ),
                )
            )
            reset = gui.add_button("重置任务")
            reset.on_click(
                lambda _: self._enqueue(
                    "reset",
                    (
                        _TASK_MODE_VALUES.get(str(self._task_mode_handle.value), str(self._task_mode_handle.value)),
                        float(self._goal_x.value),
                        float(self._goal_y.value),
                    ),
                )
            )
            focus = gui.add_button("聚焦车辆")
            focus.on_click(lambda _: self._enqueue("focus_vehicle"))

            gui.add_markdown(
                f"动作协议 {ACTION_SCHEMA}，各滑块范围为 [-1, 1]。前两维分别乘最大履带速度，"
                "独立控制对应一侧履带；后两维分别是云台偏航、俯仰角速度指令。"
            )
            self._manual_sliders = []
            for index, label in enumerate(_MANUAL_ACTION_LABELS):
                slider = gui.add_slider(label, min=-1.0, max=1.0, step=0.01, initial_value=0.0)
                slider.on_update(lambda event, i=index: self._enqueue("manual", (i, float(event.target.value))))
                self._manual_sliders.append(slider)

            self._checkpoint_input = gui.add_text(
                "检查点路径", initial_value=checkpoint or "", hint="SB3 RecurrentPPO .zip 策略文件"
            )
            load = gui.add_button("加载检查点")
            load.on_click(lambda _: self._enqueue("load_checkpoint", str(self._checkpoint_input.value).strip()))

            self._depth_handle = gui.add_image(
                np.zeros((depth_height, depth_width, 3), dtype=np.uint8),
                label="策略深度图（近处红、远处蓝、无效像素黑）",
                format="jpeg",
                jpeg_quality=85,
            )
            self._metrics_handle = gui.add_markdown("等待首帧仿真数据。")

    def _translate_native_sidebar(self, gui) -> None:
        """Translate existing Isaac Lab GUI handles without replacing callbacks.

        Viser 1.0.30 has no public element lookup. Only discovery reads its
        private handle tree; all text writes use public handle properties.
        Missing internals degrade to the unmodified native controls.
        """
        set_panel_label = getattr(gui, "set_panel_label", None)
        if callable(set_panel_label):
            try:
                set_panel_label("田间仿真")
            except Exception:
                pass

        handles = _registered_gui_handles(gui)
        if handles is None:
            return
        for handle in handles:
            try:
                label = getattr(handle, "label", None)
                if isinstance(label, str):
                    for kind, originals in _NATIVE_PAUSE_BUTTONS.items():
                        if label in originals:
                            self._native_pause_handles[kind].append(handle)
                    translated = _NATIVE_GUI_LABELS.get(label)
                    if translated is not None and translated != label:
                        handle.label = translated
                content = getattr(handle, "content", None)
                if isinstance(content, str) and content.startswith("Physics:"):
                    handle.content = f"物理引擎：{content[len('Physics:') :].lstrip()}"
            except Exception:
                # A version-specific or removed handle must not prevent the
                # task panel from attaching to the live viewer.
                continue
        self._sync_native_pause_labels()

    def _sync_native_pause_labels(self) -> None:
        """Keep native toggle labels Chinese after Isaac Lab updates their state."""
        for kind, handles in self._native_pause_handles.items():
            state_method = getattr(
                self.visualizer,
                "is_rendering_paused" if kind == "rendering" else "is_training_paused",
                None,
            )
            if not callable(state_method):
                continue
            try:
                paused = bool(state_method())
            except Exception:
                continue
            label = (
                ("继续渲染" if paused else "暂停渲染")
                if kind == "rendering"
                else ("继续仿真" if paused else "暂停仿真")
            )
            for handle in handles:
                try:
                    if getattr(handle, "label", None) != label:
                        handle.label = label
                except Exception:
                    continue

    def _enqueue(self, command: str, value: Any = None) -> None:
        self._commands.put((command, value))
        self._wake_event.set()

    def _client_signature(self) -> tuple[tuple[str, str], ...] | None:
        """Return stable Viser client identities, or ``None`` if unavailable.

        Isaac Lab's ViserVisualizer wraps the public ViserServer, so the server
        is reached through its viewer object. ``get_clients`` itself is the
        public Viser API. Keeping identities lets a reconnect trigger a refresh
        even when a replacement client arrives before the previous one leaves.
        """
        viewer = getattr(self.visualizer, "_viewer", None)
        server = getattr(viewer, "_server", None)
        get_clients = getattr(server, "get_clients", None)
        if not callable(get_clients):
            return None
        try:
            clients = get_clients()
            identities = clients.keys() if isinstance(clients, Mapping) else clients
            signature = tuple(
                sorted((type(client).__qualname__, repr(client)) for client in identities)
            )
            panel = getattr(self, "_network_panel", None)
            if panel is not None and panel.has_2d_clients():
                signature += (("network2d", "active"),)
            return signature
        except Exception:
            return None

    def _client_count(self) -> int | None:
        """Return connected Viser clients, or ``None`` for older/fake APIs."""
        signature = self._client_signature()
        return None if signature is None else len(signature)

    def _render_if_connected(self) -> None:
        """Render explicit camera/state changes only when a client can see them."""
        clients = self._client_count()
        if clients is None or clients > 0:
            self._render_scene()

    def _current_goal(self, size: float | None = None) -> np.ndarray:
        if size is None:
            size = float(getattr(getattr(self.adapter, "cfg", None), "size", 100.0))
        try:
            goal = np.asarray(self.adapter.state_payload().get("goal"), dtype=np.float32).reshape(2)
            if np.isfinite(goal).all() and np.all((goal >= 0.0) & (goal <= size)):
                return goal
        except (AttributeError, TypeError, ValueError):
            pass
        return np.asarray((size * 0.5, size * 0.5), dtype=np.float32)

    def _sync_task_controls(self, mode: str) -> None:
        """Reflect the reset episode's sampled mode and goal in the sidebar."""
        self._task_mode = mode
        self._task_mode_handle.value = _TASK_MODE_LABELS.get(mode, mode)
        goal = self._current_goal()
        self._goal_x.value = float(goal[0])
        self._goal_y.value = float(goal[1])

    def _episode_ended(self) -> bool:
        if self._terminal:
            return True
        try:
            return bool(self.adapter.state_payload().get("done", False))
        except Exception:
            return bool(getattr(self.adapter, "done", False))

    def process_pending(self) -> int:
        """Apply queued GUI requests on the environment-owning thread."""
        processed = 0
        # Isaac Lab's native viewer reset button queues a SimulationContext
        # reset. Consume it here and route it through the adapter, so our local
        # terminal/policy state is reset together with the task state.
        sim = getattr(getattr(self.adapter, "engine", None), "sim", None)
        consume_reset = getattr(sim, "consume_reset_request", None)
        if callable(consume_reset) and consume_reset():
            payload = self.adapter.state_payload()
            goal = payload.get("goal")
            if goal is None:
                goal = (float(self._goal_x.value), float(self._goal_y.value))
            self._apply_task(
                str(payload.get("mode", self._task_mode)), float(goal[0]), float(goal[1]), reset=True
            )
            processed += 1
        while True:
            try:
                command, value = self._commands.get_nowait()
            except Empty:
                break
            processed += 1
            try:
                self._apply_command(command, value)
            except Exception as exc:
                self._status = f"请求失败：{exc}"
                self._set_status()
        return processed

    def _apply_command(self, command: str, value: Any) -> None:
        if command == "network_run":
            if self._runner is None:
                raise ValueError("请先在任务设置中加载策略检查点")
            if self._episode_ended():
                raise ValueError("本回合已结束，请先重置任务")
            self._control_mode = "policy"
            self._control_mode_handle.value = _CONTROL_MODE_LABELS["policy"]
            self._running = True
            self._run_button.label = "暂停任务"
            self._status = "策略运行中，激活来自同一步真实前向。"
            self._set_status()
            return
        if command == "network_pause":
            self._running = False
            self._run_button.label = "运行任务"
            self._status = "车辆已暂停；可读取当前帧或展开网络。"
            self._set_status()
            return
        if command.startswith("network_") and self._network_panel is not None:
            observation = self._refresh_observation() if command == "network_preview" else None
            if observation is not None:
                observation = _observation_batch(observation, self.adapter.observation_space)
            self._network_panel.command(command, value, observation, self._simulation_time())
            return
        if command == "control_mode":
            requested = _CONTROL_MODE_VALUES.get(str(value), str(value))
            if requested == "policy" and self._runner is None:
                self._control_mode = "manual"
                self._control_mode_handle.value = _CONTROL_MODE_LABELS["manual"]
                self._status = "请先加载检查点，再选择策略控制。"
            elif requested in _CONTROL_MODES:
                self._control_mode = requested
                self._status = f"控制模式：{_CONTROL_MODE_LABELS[requested]}。"
            else:
                raise ValueError(f"不支持的控制模式：{requested!r}")
        elif command == "task_mode":
            requested = _TASK_MODE_VALUES.get(str(value), str(value))
            if requested not in _TASK_MODES:
                raise ValueError(f"不支持的任务模式：{requested!r}")
            self._task_mode = requested
            self._status = f"已选择{_TASK_MODE_LABELS[requested]}；点击“设置目标 / 应用任务模式”后生效。"
        elif command == "manual":
            index, action_value = value
            self._manual_action[int(index)] = np.clip(float(action_value), -1.0, 1.0)
        elif command == "apply_task":
            mode, x, y = value
            self._apply_task(mode, x, y, reset=False)
        elif command == "reset":
            if value is None:
                mode = self._task_mode
                x, y = self._current_goal()
            else:
                mode, x, y = value
            self._apply_task(mode, x, y, reset=True)
        elif command == "load_checkpoint":
            self._load_checkpoint(value, select_policy=True)
        elif command == "focus_vehicle":
            self._focus_vehicle()
        elif command == "toggle_run":
            if not self._running and self._terminal:
                self._status = "本回合已结束。请先重置任务，再次运行。"
            else:
                self._running = not self._running
                self._run_button.label = "暂停任务" if self._running else "运行任务"
                self._status = "任务运行中。" if self._running else "任务已暂停。"
        else:
            raise ValueError(f"未知的查看器命令：{command!r}")
        self._set_status()

    def _apply_task(self, mode: str, x: float, y: float, *, reset: bool) -> None:
        if mode not in _TASK_MODES:
            raise ValueError(f"不支持的任务模式：{mode!r}")
        if not reset and self._episode_ended():
            self._status = "本回合已结束。请先重置任务，再设置目标或切换任务模式。"
            return
        size = float(getattr(getattr(self.adapter, "cfg", None), "size", 100.0))
        goal = np.asarray((x, y), dtype=np.float32)
        if mode == "goto" and not reset and (
            not np.all(np.isfinite(goal)) or np.any(goal < 0.0) or np.any(goal > size)
        ):
            raise ValueError(f"目标点必须位于 [0, {size:g}] × [0, {size:g}] 米范围内")
        did_reset = reset or mode == "search"
        if did_reset:
            self.adapter.reset(seed=self._reset_seed, options={"mode": mode})
            self._terminal = False
            self._last_reward = None
            self._last_info = {}
            if self._runner is not None:
                self._runner.reset()
            self._status = f"已重置{_TASK_MODE_LABELS[mode]}。"
            self._sync_task_controls(mode)
        elif mode == "goto":
            self.adapter.set_goal(goal)
            if self._runner is not None:
                self._runner.reset()
            self._status = f"目标已设为（{x:.2f}, {y:.2f}）米。"
            self._sync_task_controls(mode)
        self._latest_observation = None
        if self._network_panel is not None:
            self._network_panel.reset()
        self._refresh_observation()
        if did_reset:
            self._focus_vehicle()

    def _focus_vehicle(self) -> None:
        """Aim the official Viser camera at the current vehicle pose."""
        set_camera_view = getattr(self.visualizer, "set_camera_view", None)
        if not callable(set_camera_view):
            return
        payload = self.adapter.state_payload()
        robot = payload.get("robot", {})
        x = float(robot.get("x", 0.0))
        y = float(robot.get("y", 0.0))
        z = float(robot.get("z", getattr(getattr(self.adapter, "cfg", None), "reference_height", 0.15)))
        target = (x, y, z)
        eye = (x - 3.0, y - 4.0, z + 3.0)
        server = getattr(getattr(self.visualizer, "_viewer", None), "_server", None)
        initial_camera = getattr(server, "initial_camera", None)
        if initial_camera is not None:
            initial_camera.position = eye
            initial_camera.look_at = target
        set_camera_view(eye=eye, target=target)
        self._render_if_connected()

    def _render_scene(self) -> None:
        """Refresh the official scene visualizer without advancing physics."""
        sim = getattr(getattr(self.adapter, "engine", None), "sim", None)
        render = getattr(sim, "render", None)
        if callable(render):
            render()

    @staticmethod
    def _single_scalar(value: Any) -> Any | None:
        """Read one scalar from CPU/GPU-backed simulator state without assuming torch."""
        try:
            if callable(getattr(value, "detach", None)):
                value = value.detach()
            if callable(getattr(value, "cpu", None)):
                value = value.cpu()
            if callable(getattr(value, "numpy", None)):
                value = value.numpy()
            array = np.asarray(value)
            if array.size != 1:
                return None
            scalar = array.reshape(-1)[0]
            return scalar.item() if callable(getattr(scalar, "item", None)) else scalar
        except (TypeError, ValueError, RuntimeError):
            return None

    def _is_native_reset_without_transition(self) -> bool:
        """Confirm the Isaac Lab native-reset race, not an arbitrary missing result."""
        engine = getattr(self.adapter, "engine", None)
        if engine is None or getattr(engine, "num_envs", None) != 1:
            return False
        if getattr(getattr(engine, "cfg", None), "auto_reset", None) is not False:
            return False
        results = getattr(engine, "_last_results", None)
        if not isinstance(results, (list, tuple)) or len(results) != 1 or results[0] is not None:
            return False
        task = getattr(self.adapter, "task", None)
        if task is None or getattr(task, "done", None) is not False:
            return False
        try:
            task_time = float(task.time)
        except (AttributeError, TypeError, ValueError):
            return False
        if not np.isfinite(task_time) or not np.isclose(task_time, 0.0, rtol=0.0, atol=1e-9):
            return False
        reset_terminated = self._single_scalar(getattr(engine, "reset_terminated", None))
        reset_time_outs = self._single_scalar(getattr(engine, "reset_time_outs", None))
        episode_length = self._single_scalar(getattr(engine, "episode_length_buf", None))
        return (
            reset_terminated is True
            and reset_time_outs is False
            and episode_length == 0
        )

    def _accept_native_reset_without_transition(self) -> None:
        """Refresh an already-reset episode without inventing a policy transition."""
        getter = getattr(self.adapter, "current_observation", None)
        if not callable(getter):
            raise RuntimeError("native reset cleared the transition, but adapter.current_observation() is unavailable")
        observation = getter()
        if not isinstance(observation, Mapping):
            raise RuntimeError("native reset cleared the transition, but no current observation is available")

        self._latest_observation = {key: np.asarray(value) for key, value in observation.items()}
        self._last_reward = None
        self._last_info = {}
        self._terminal = False
        self._running = False
        self._run_button.label = "运行任务"
        if self._runner is not None:
            self._runner.reset()
        if self._network_panel is not None:
            self._network_panel.reset()
        task_mode = getattr(getattr(self.adapter, "task", None), "mode", None)
        if task_mode in _TASK_MODES:
            self._sync_task_controls(task_mode)
        self._status = "官方仿真已重置；已刷新当前状态，请点击“运行任务”继续。"
        self._set_status()
        self._render_if_connected()
        if self._client_count() != 0:
            self.publish()

    def _load_checkpoint(self, checkpoint: str, *, select_policy: bool) -> None:
        if not checkpoint:
            raise ValueError("请先输入检查点路径")
        from .policy import PolicyRunner

        # Keep the current runner if loading a new file fails.
        runner = PolicyRunner(self.adapter, checkpoint)
        runner.reset()
        self._runner = runner
        self._checkpoint = checkpoint
        if self._network_panel is not None:
            self._network_panel.bind(runner)
            observation = self._refresh_observation()
            if observation is not None:
                self._network_panel.inspector.preview(
                    _observation_batch(observation, self.adapter.observation_space), self._simulation_time()
                )
                self._network_panel.publish(force=True)
        if select_policy:
            self._control_mode = "policy"
            self._control_mode_handle.value = _CONTROL_MODE_LABELS["policy"]
        model_kind = (
            "未训练的随机初始化模型"
            if getattr(runner.model, "num_timesteps", None) == 0 else "检查点"
        )
        self._status = f"已加载{model_kind}：{checkpoint}（加载成功不代表策略已具备任务能力）。"

    def _refresh_observation(self) -> dict[str, np.ndarray] | None:
        getter = getattr(self.adapter, "current_observation", None)
        if not callable(getter):
            return self._latest_observation
        observation = getter()
        if observation is not None:
            self._latest_observation = {key: np.asarray(value) for key, value in observation.items()}
        return self._latest_observation

    def _update_sensor_overlays(self) -> None:
        """Update compact point/range overlays from the frames actually delivered to the task."""
        viewer = getattr(self.visualizer, "_viewer", None)
        server = getattr(viewer, "_server", None)
        scene = getattr(server, "scene", None)
        if scene is None:
            return
        task = getattr(self.adapter, "task", None)
        frames = getattr(task, "frames", None)
        if not isinstance(frames, dict):
            for handle in self._overlay_handles.values():
                handle.visible = False
            return

        lidar = frames.get("lidar")
        if isinstance(lidar, dict):
            self._update_lidar_overlay(scene, lidar)
        depth = frames.get("depth")
        if isinstance(depth, dict):
            self._update_depth_overlay(scene, depth)

    def _set_point_cloud(self, scene, key: str, name: str, points: np.ndarray, color, point_size: float) -> None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        handle = self._overlay_handles.get(key)
        if handle is None:
            if len(points) == 0:
                return
            add_point_cloud = getattr(scene, "add_point_cloud", None)
            if not callable(add_point_cloud):
                return
            handle = add_point_cloud(
                name,
                points,
                color,
                point_size=point_size,
                point_shape="circle",
                point_shading="flat",
            )
            self._overlay_handles[key] = handle
        else:
            handle.points = points
            handle.visible = bool(len(points))

    def _update_lidar_overlay(self, scene, frame: dict[str, Any]) -> None:
        try:
            origin = np.asarray(frame["origin"], dtype=np.float32).reshape(3)
            ranges = np.asarray(frame["ranges"], dtype=np.float32).reshape(-1)
            directions = np.asarray(frame["directions"], dtype=np.float32).reshape(-1, 3)
            valid = np.asarray(frame["valid"], dtype=bool).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return
        if len(ranges) != len(directions) or len(valid) != len(ranges):
            return
        valid &= np.isfinite(ranges) & np.isfinite(directions).all(axis=1)
        points = origin + ranges[valid, None] * directions[valid]
        self._set_point_cloud(scene, "lidar", _LIDAR_CLOUD_NAME, points, _LIDAR_COLOR, 0.015)

        usable = directions[np.isfinite(directions).all(axis=1)]
        add_line_segments = getattr(scene, "add_line_segments", None)
        if len(usable) < 3 or not callable(add_line_segments):
            return
        _, _, vh = np.linalg.svd(usable, full_matrices=False)
        normal = vh[-1]
        first = usable[0] - normal * float(np.dot(usable[0], normal))
        first_norm = float(np.linalg.norm(first))
        if first_norm < 1e-6:
            return
        first /= first_norm
        second = np.cross(normal, first)
        second_norm = float(np.linalg.norm(second))
        if second_norm < 1e-6:
            return
        second /= second_norm
        max_range = float(frame.get("max_range", getattr(getattr(self.adapter, "cfg", None), "lidar_range", 8.0)))
        angles = np.linspace(0.0, 2.0 * np.pi, 129, dtype=np.float32)[:-1]
        ring = origin + max_range * (
            np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
        )
        segments = np.stack((ring, np.roll(ring, -1, axis=0)), axis=1)
        ring_handle = self._overlay_handles.get("lidar_ring")
        if ring_handle is None:
            self._overlay_handles["lidar_ring"] = add_line_segments(
                _LIDAR_RING_NAME, segments, _LIDAR_RING_COLOR, line_width=0.6
            )
        else:
            ring_handle.points = segments
            ring_handle.visible = True

    def _update_depth_overlay(self, scene, frame: dict[str, Any]) -> None:
        try:
            values = np.asarray(frame["values"], dtype=np.float32)
            valid = np.asarray(frame["valid"], dtype=bool)
            intrinsics = np.asarray(frame["intrinsics"], dtype=np.float32).reshape(3, 3)
            basis = np.asarray(frame["basis"], dtype=np.float32).reshape(3, 3)
            origin = np.asarray(frame["origin"], dtype=np.float32).reshape(3)
        except (KeyError, TypeError, ValueError):
            return
        if values.ndim != 2 or valid.shape != values.shape:
            return
        height, width = values.shape
        v, u = np.mgrid[0:height:4, 0:width:4]
        z = values[::4, ::4]
        mask = valid[::4, ::4] & np.isfinite(z) & (z > 0.0)
        if not np.any(mask):
            self._set_point_cloud(scene, "depth", _DEPTH_CLOUD_NAME, np.empty((0, 3)), _DEPTH_COLOR, 0.02)
            return
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        if fx <= 0 or fy <= 0:
            return
        x = (u - cx) / fx
        y = (v - cy) / fy
        forward, right, up = basis
        points = (
            origin
            + z[..., None] * forward
            + (x * z)[..., None] * right
            - (y * z)[..., None] * up
        )
        self._set_point_cloud(scene, "depth", _DEPTH_CLOUD_NAME, points[mask], _DEPTH_COLOR, 0.02)

    def action(self) -> np.ndarray | None:
        """Return the next normalized action, or ``None`` while paused/terminal."""
        self.process_pending()
        # A task may have been reset by an external owner-thread operation.
        # Do not keep a stale terminal latch after the adapter reports a live
        # episode.
        if self._terminal and not bool(getattr(self.adapter, "done", False)):
            self._terminal = False
        if self._terminal:
            return None
        if not self._running:
            return None
        paused = getattr(self.visualizer, "is_training_paused", None)
        if callable(paused) and paused():
            return None
        if self._control_mode == "policy":
            if self._runner is None:
                self._status = "尚未加载检查点，当前无法使用策略控制。"
                self._set_status()
                return None
            observation = self._refresh_observation()
            if observation is None:
                raise RuntimeError("adapter.current_observation() 未返回观测数据")
            batched = _observation_batch(observation, self.adapter.observation_space)
            panel = self._network_panel
            if panel is not None and panel.enabled and panel.inspector is not None and self._client_count() != 0:
                predicted = panel.action(batched, self._simulation_time())
            else:
                predicted = self._runner.action(batched)
                if panel is not None and panel.inspector is not None:
                    # The critic is a diagnostic branch. After skipped capture,
                    # restart its own history instead of pretending continuity.
                    panel.inspector.critic_state = None
            action = np.asarray(predicted, dtype=np.float32).reshape(-1, 4)[0]
        else:
            action = self._manual_action.copy()
        space = getattr(self.adapter, "action_space", None)
        if space is not None:
            action = np.clip(action, np.asarray(space.low), np.asarray(space.high))
        return action.astype(np.float32, copy=False)

    def record_transition(self, observation=None, reward=None, terminated=False, truncated=False, info=None) -> None:
        """Update the viewer after an externally driven ``adapter.step`` call."""
        if observation is not None and isinstance(observation, dict):
            self._latest_observation = {key: np.asarray(value) for key, value in observation.items()}
        self._last_reward = None if reward is None else float(np.asarray(reward).reshape(-1)[0])
        self._last_info = dict(info or {})
        if self._network_panel is not None:
            self._network_panel.record_action_transition(self.adapter)
        done = bool(terminated) or bool(truncated)
        if done:
            self._terminal = True
            self._running = False
            self._run_button.label = "运行任务"
            metrics = self._last_info.get("metrics") or self._last_info
            reason = metrics.get("reason") if isinstance(metrics, Mapping) else None
            success = metrics.get("is_success") if isinstance(metrics, Mapping) else None
            if reason is None or success is None:
                try:
                    payload = self.adapter.state_payload()
                    payload_metrics = payload.get("metrics") or {}
                    if reason is None:
                        reason = payload_metrics.get("reason", payload.get("reason"))
                    if success is None:
                        success = payload_metrics.get("is_success")
                except Exception:
                    pass
            reason = str(reason or ("time_limit" if truncated else "terminated"))
            outcome = "成功" if bool(success) else "未成功"
            self._status = f"回合结束：{_terminal_reason_label(reason)}；结果：{outcome}。"
            self._set_status()
            if self._runner is not None:
                self._runner.reset(np.asarray((True,), dtype=bool))
        # The scene/environment step can continue while nobody is connected,
        # but GUI images and metrics need not be republished until a client
        # returns. A new connection is detected by the owner loop below.
        if self._client_count() != 0:
            self.publish()

    def step_once(self):
        """Process pending controls and take one action on the owner thread."""
        action = self.action()
        if action is None:
            self.publish()
            return None
        transition = self.adapter.step(action)
        if transition is None:
            if self._is_native_reset_without_transition():
                # Isaac Lab can consume a native GUI reset after _get_dones()
                # has filled the adapter's private result slot. The reset has
                # already happened; do not call adapter.reset() again or reuse
                # the explicit viewer seed for a second, unintended episode.
                self._accept_native_reset_without_transition()
                return None
            engine = getattr(self.adapter, "engine", None)
            task = getattr(self.adapter, "task", None)
            result_slot = getattr(engine, "_last_results", None)
            result_value = result_slot[0] if isinstance(result_slot, (list, tuple)) and result_slot else "unavailable"
            reset_terminated = self._single_scalar(getattr(engine, "reset_terminated", None))
            reset_time_outs = self._single_scalar(getattr(engine, "reset_time_outs", None))
            episode_length = self._single_scalar(getattr(engine, "episode_length_buf", None))
            raise RuntimeError(
                "adapter.step() returned None without a confirmed native reset "
                f"(last_result={result_value!r}, reset_terminated={reset_terminated!r}, "
                f"reset_time_outs={reset_time_outs!r}, episode_length={episode_length!r}, "
                f"task_done={getattr(task, 'done', None)!r}, task_time={getattr(task, 'time', None)!r})"
            )
        observation, reward, terminated, truncated, info = transition
        self.record_transition(observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    def publish(self) -> None:
        """Push the current policy depth image and compact task metrics to Viser."""
        try:
            observation = self._refresh_observation()
            if observation is not None and "depth" in observation:
                self._depth_handle.image = depth_to_rgb(observation["depth"])
            payload = self.adapter.state_payload()
            self._update_sensor_overlays()
            robot = payload.get("robot", {})
            metrics = payload.get("metrics") or self._last_info
            goal = payload.get("goal")
            goal_text = (
                f"（{float(goal[0]):.2f}, {float(goal[1]):.2f}）米"
                if goal is not None and len(goal) >= 2
                else "不可用"
            )
            seq = metrics.get("sensor_sequences", {})
            gain, schema, timesteps, drive_status, observation_status, action_status = self._checkpoint_metadata()
            damage_metric, damage_status = self._damage_metadata()
            reason = metrics.get("reason", payload.get("reason", ""))
            fields = [
                f"**控制模式：** {_CONTROL_MODE_LABELS.get(self._control_mode, self._control_mode)}  ",
                f"**任务模式：** {_TASK_MODE_LABELS.get(str(payload.get('mode', self._task_mode)), str(payload.get('mode', self._task_mode)))}  ",
                f"**仿真时间：** {float(payload.get('time', 0.0)):.2f} 秒  ",
                f"**车辆：** x={float(robot.get('x', 0.0)):.2f} 米，y={float(robot.get('y', 0.0)):.2f} 米，偏航={float(robot.get('yaw', 0.0)):.2f} 弧度，线速度={float(robot.get('v', 0.0)):.2f} 米/秒，角速度={float(robot.get('w', 0.0)):.2f} 弧度/秒  ",
                f"**云台：** 偏航={float(payload.get('gimbal', {}).get('yaw', 0.0)):.2f}，俯仰={float(payload.get('gimbal', {}).get('pitch', 0.0)):.2f} 弧度  ",
                f"**目标位置：** {goal_text}  ",
                f"**目标距离：** {float(metrics.get('distance', 0.0)):.2f} 米  ",
                f"**搜索进度：** {int(metrics.get('found', 0))}/{int(metrics.get('targets', 0))}；成功={('是' if bool(metrics.get('is_success', False)) else '否')}；原因={_terminal_reason_label(str(reason or ''))}  ",
                f"**损伤 / 碰撞：** {int(metrics.get('damage', 0))} / {int(metrics.get('collisions', 0))}  ",
                f"**传感器序号：** 激光雷达 {seq.get('lidar', '?')}，深度图 {seq.get('depth', '?')}  ",
                "**三维传感器显示：** 青色为激光雷达回波，橙色为 8 米量程，蓝色为有效深度点（每 4 个像素采样）  ",
                f"**最近奖励：** {self._last_reward if self._last_reward is not None else '未记录'}  ",
                f"**检查点：** {self._checkpoint or '无'}  ",
                f"**驱动 / 观测配置：** 增益={_metadata_status_label(gain)}；结构版本={_metadata_status_label(schema)}  ",
                f"**动作协议：** {ACTION_SCHEMA}  ",
                f"**检查点兼容性：** 步数={_metadata_status_label(timesteps)}；驱动={_metadata_status_label(drive_status)}；观测={_metadata_status_label(observation_status)}；动作={_metadata_status_label(action_status)}  ",
                f"**损伤规则：** 当前={_metadata_status_label(damage_metric)}；检查点={_metadata_status_label(damage_status)}  ",
                f"**状态：** {self._status}",
            ]
            self._metrics_handle.content = "\n\n".join(fields)
            if self._network_panel is not None:
                paused = getattr(self.visualizer, "is_training_paused", None)
                native_paused = bool(paused()) if callable(paused) else False
                self._network_panel.runtime_state(payload, running=self._running and not self._terminal and not native_paused, status=self._status)
                self._network_panel.publish()
        except Exception as exc:
            self._status = f"更新查看器失败：{exc}"
            self._set_status()

    def _set_status(self) -> None:
        self._status_handle.content = self._status

    def _simulation_time(self) -> float:
        task = getattr(self.adapter, "task", None)
        if task is not None and hasattr(task, "time"):
            return float(task.time)
        return float(self.adapter.state_payload().get("time", 0.0))

    def run(self) -> None:
        """Own the single-environment loop until the official Viser viewer closes.

        The task starts locally paused and requires an explicit Run task click.
        The official Viser Pause/Resume control remains an independent gate.
        Use this on the same thread that owns the Isaac Lab environment/SimulationContext.
        """
        cfg = getattr(self.adapter, "cfg", None)
        dt = float(getattr(cfg, "dt", 0.05))
        dt = max(1e-3, dt)
        while self._is_viewer_running():
            start = time.perf_counter()
            # Clear before draining the queue: a callback arriving during or
            # after the drain leaves the event set and wakes the idle wait.
            self._wake_event.clear()
            processed = self.process_pending()
            self._sync_native_pause_labels()
            client_signature = self._client_signature()
            clients = None if client_signature is None else len(client_signature)

            paused = getattr(self.visualizer, "is_training_paused", None)
            is_paused = bool(paused()) if callable(paused) else False
            advancing = self._running and not self._terminal and not is_paused
            if advancing:
                # Preserve the existing simulation/action cadence while the
                # task runs. record_transition skips sidebar publication when
                # there are no connected clients.
                self.step_once()
                self._last_client_signature = client_signature
                remaining = dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
                continue

            # Paused/terminal idle: update once for a request or client identity
            # change (including a same-count reconnect), then wait instead of
            # repeatedly colorizing depth and calling SimulationContext.render().
            changed = client_signature != self._last_client_signature
            # process_pending() may have applied a reset, checkpoint, slider,
            # mode, or goal request. Refresh for such requests only when a
            # connected browser can consume the result. If this Viser version
            # lacks get_clients(), assume one may be connected and refresh on
            # first entry and queued requests, then avoid repeated idle updates.
            # The signature is captured above to keep this decision on the owner thread.
            # Queue state itself is handled by the event in _enqueue().
            if (clients is None or clients > 0) and (
                changed or processed > 0 or self._wake_event.is_set()
            ):
                self.publish()
                self._render_scene()
            self._last_client_signature = client_signature
            # Native SimulationContext reset requests do not pass through our
            # callback queue, so retain a low-rate check even with no browser.
            # GUI commands and new callbacks wake this wait immediately.
            self._wake_event.wait(0.5 if clients == 0 else 0.25)

    def _is_viewer_running(self) -> bool:
        running = getattr(self.visualizer, "is_running", None)
        return bool(running()) if callable(running) else True


def attach_field_controls(
    visualizer, adapter, checkpoint: str | None = None, *, reset_seed: int | None = None
) -> FieldViewerControls:
    """Attach field task controls to an initialized official ``ViserVisualizer``."""
    return FieldViewerControls(visualizer, adapter, checkpoint, reset_seed=reset_seed)


def run_viewer(
    adapter, visualizer, checkpoint: str | None = None, *, reset_seed: int | None = None
) -> FieldViewerControls:
    """Attach controls and run the task from the environment-owning thread."""
    controls = attach_field_controls(visualizer, adapter, checkpoint, reset_seed=reset_seed)
    panel = controls._network_panel
    if panel is not None:
        server = panel._server
        get_port = getattr(server, "get_port", None)
        if callable(get_port):
            port = int(get_port())
            try:
                panel.start_2d(port + 1, f"http://127.0.0.1:{port}/")
            except OSError as exc:
                panel.page_link.content = f"二维页启动失败（端口{port + 1}）：{exc}"
    try:
        controls.run()
    finally:
        if panel is not None:
            panel.close()
    return controls


__all__ = ["FieldViewerControls", "attach_field_controls", "depth_to_rgb", "run_viewer"]
