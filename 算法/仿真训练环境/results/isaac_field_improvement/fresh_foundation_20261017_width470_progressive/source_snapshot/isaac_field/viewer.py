"""Task controls and policy monitoring for Isaac Lab's official Viser viewer.

This module extends the Viser sidebar only; Isaac Lab/Newton remains responsible
for rendering the scene and robot. GUI callbacks enqueue requests because Viser
callbacks run outside the simulation-owner thread. Call ``process_pending`` and
``step_once`` (or ``run``) from the thread that owns the environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from queue import Empty, SimpleQueue
import time
from typing import Any

import numpy as np

from .task import DAMAGE_METRIC_ROOT_PROTECTION_V1


_CONTROL_MODES = ("manual", "policy")
_TASK_MODES = ("search", "goto")
_MANUAL_ACTION_LABELS = (
    "Forward demand",
    "Differential steering",
    "Gimbal yaw rate",
    "Gimbal pitch rate",
)
_LIDAR_CLOUD_NAME = "/FieldTask/Sensors/LidarReturns"
_DEPTH_CLOUD_NAME = "/FieldTask/Sensors/DepthReturns"
_LIDAR_RING_NAME = "/FieldTask/Sensors/LidarMaxRange"
_LIDAR_COLOR = np.asarray((0.0, 1.0, 1.0), dtype=np.float32)
_DEPTH_COLOR = np.asarray((0.12, 0.45, 1.0), dtype=np.float32)
_LIDAR_RING_COLOR = np.asarray((1.0, 0.58, 0.12), dtype=np.float32)


def depth_to_rgb(depth_observation: np.ndarray) -> np.ndarray:
    """Colorize the policy depth observation (normalized depth + validity mask).

    The policy contract is ``(2, H, W)`` with channel 0 holding normalized
    camera-Z depth and channel 1 holding validity. Invalid pixels are black.
    """
    image = np.asarray(depth_observation, dtype=np.float32)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] < 2:
        raise ValueError("depth observation must have shape (2, height, width) or (1, 2, height, width)")

    distance = np.nan_to_num(image[0], nan=1.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    valid = np.isfinite(image[1]) & (image[1] > 0.5)
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
        self._status = "Paused in manual control. Choose Run task to advance the environment."

        viewer = getattr(visualizer, "_viewer", None)
        server = getattr(viewer, "_server", None)
        gui = getattr(server, "gui", None)
        if gui is None:
            raise RuntimeError("ViserVisualizer must be initialized before field controls are attached")
        self._build_sidebar(gui, checkpoint)
        if checkpoint:
            try:
                self._load_checkpoint(checkpoint, select_policy=False)
                self._status = f"Loaded checkpoint: {checkpoint}. Paused in manual control; select Policy and Run task to execute it."
            except Exception as exc:
                self._status = f"Checkpoint load failed: {exc}. Paused in manual control."
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

    def _checkpoint_metadata(self) -> tuple[str, str, str, str, str]:
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
            return gain_text, schema_text, "not loaded", "not loaded", "not loaded"

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
        with gui.add_folder("Field Task", expand_by_default=True):
            gui.add_markdown("Isaac Lab / Newton task controls. The scene and robot are rendered by the official Viser viewer.")
            self._status_handle = gui.add_markdown(self._status)
            self._run_button = gui.add_button("Run task")
            self._run_button.on_click(lambda _: self._enqueue("toggle_run"))
            control = gui.add_dropdown("Control", _CONTROL_MODES, initial_value="manual")
            control.on_update(lambda event: self._enqueue("control_mode", event.target.value))

            self._task_mode_handle = gui.add_dropdown("Task mode", _TASK_MODES, initial_value=self._task_mode)
            self._task_mode_handle.on_update(lambda event: self._enqueue("task_mode", event.target.value))
            goal = self._current_goal(size)
            self._goal_x = gui.add_number("Goal X (m)", initial_value=float(goal[0]), min=0.0, max=size, step=0.1)
            self._goal_y = gui.add_number("Goal Y (m)", initial_value=float(goal[1]), min=0.0, max=size, step=0.1)
            apply_goal = gui.add_button("Set goal / apply task mode")
            apply_goal.on_click(
                lambda _: self._enqueue(
                    "apply_task",
                    (str(self._task_mode_handle.value), float(self._goal_x.value), float(self._goal_y.value)),
                )
            )
            reset = gui.add_button("Reset task")
            reset.on_click(
                lambda _: self._enqueue(
                    "reset",
                    (str(self._task_mode_handle.value), float(self._goal_x.value), float(self._goal_y.value)),
                )
            )
            focus = gui.add_button("Focus vehicle")
            focus.on_click(lambda _: self._enqueue("focus_vehicle"))

            self._manual_sliders = []
            for index, label in enumerate(_MANUAL_ACTION_LABELS):
                slider = gui.add_slider(label, min=-1.0, max=1.0, step=0.01, initial_value=0.0)
                slider.on_update(lambda event, i=index: self._enqueue("manual", (i, float(event.target.value))))
                self._manual_sliders.append(slider)

            self._checkpoint_input = gui.add_text(
                "Checkpoint path", initial_value=checkpoint or "", hint="SB3 RecurrentPPO .zip checkpoint"
            )
            load = gui.add_button("Load checkpoint")
            load.on_click(lambda _: self._enqueue("load_checkpoint", str(self._checkpoint_input.value).strip()))

            self._depth_handle = gui.add_image(
                np.zeros((depth_height, depth_width, 3), dtype=np.uint8),
                label="Policy depth (near red, far blue; invalid black)",
                format="jpeg",
                jpeg_quality=85,
            )
            self._metrics_handle = gui.add_markdown("Waiting for the first environment frame.")

    def _enqueue(self, command: str, value: Any = None) -> None:
        self._commands.put((command, value))

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
        self._task_mode_handle.value = mode
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
                self._status = f"Request failed: {exc}"
                self._set_status()
        return processed

    def _apply_command(self, command: str, value: Any) -> None:
        if command == "control_mode":
            requested = str(value)
            if requested == "policy" and self._runner is None:
                self._control_mode = "manual"
                self._status = "Load a checkpoint before selecting Policy control."
            elif requested in _CONTROL_MODES:
                self._control_mode = requested
                self._status = f"Control mode: {requested}."
            else:
                raise ValueError(f"unsupported control mode {requested!r}")
        elif command == "task_mode":
            requested = str(value)
            if requested not in _TASK_MODES:
                raise ValueError(f"unsupported task mode {requested!r}")
            self._task_mode = requested
            self._status = f"Task mode selected: {requested}; apply it to affect the environment."
        elif command == "manual":
            index, action_value = value
            self._manual_action[int(index)] = np.clip(float(action_value), -1.0, 1.0)
        elif command == "apply_task":
            mode, x, y = value
            self._apply_task(mode, x, y, reset=False)
        elif command == "reset":
            mode, x, y = value
            self._apply_task(mode, x, y, reset=True)
        elif command == "load_checkpoint":
            self._load_checkpoint(value, select_policy=True)
        elif command == "focus_vehicle":
            self._focus_vehicle()
        elif command == "toggle_run":
            if not self._running and self._terminal:
                self._status = "Episode ended. Reset the task before running again."
            else:
                self._running = not self._running
                self._run_button.label = "Pause task" if self._running else "Run task"
                self._status = "Task running." if self._running else "Task paused."
        else:
            raise ValueError(f"unknown viewer command {command!r}")
        self._set_status()

    def _apply_task(self, mode: str, x: float, y: float, *, reset: bool) -> None:
        if mode not in _TASK_MODES:
            raise ValueError(f"unsupported task mode {mode!r}")
        if not reset and self._episode_ended():
            self._status = "Episode ended. Reset the task before setting a goal or applying a task mode."
            return
        size = float(getattr(getattr(self.adapter, "cfg", None), "size", 100.0))
        goal = np.asarray((x, y), dtype=np.float32)
        if mode == "goto" and not reset and (
            not np.all(np.isfinite(goal)) or np.any(goal < 0.0) or np.any(goal > size)
        ):
            raise ValueError(f"goal must lie inside [0, {size:g}] x [0, {size:g}]")
        did_reset = reset or mode == "search"
        if did_reset:
            self.adapter.reset(seed=self._reset_seed, options={"mode": mode})
            self._terminal = False
            self._last_reward = None
            self._last_info = {}
            if self._runner is not None:
                self._runner.reset()
            self._status = f"Reset in {mode} mode."
            self._sync_task_controls(mode)
        elif mode == "goto":
            self.adapter.set_goal(goal)
            if self._runner is not None:
                self._runner.reset()
            self._status = f"Goal set to ({x:.2f}, {y:.2f}) m."
            self._sync_task_controls(mode)
        self._latest_observation = None
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
        set_camera_view(eye=(x - 3.0, y - 4.0, z + 3.0), target=target)
        self._render_scene()

    def _render_scene(self) -> None:
        """Refresh the official scene visualizer without advancing physics."""
        sim = getattr(getattr(self.adapter, "engine", None), "sim", None)
        render = getattr(sim, "render", None)
        if callable(render):
            render()

    def _load_checkpoint(self, checkpoint: str, *, select_policy: bool) -> None:
        if not checkpoint:
            raise ValueError("enter a checkpoint path first")
        from .policy import PolicyRunner

        # Keep the current runner if loading a new file fails.
        runner = PolicyRunner(self.adapter, checkpoint)
        runner.reset()
        self._runner = runner
        self._checkpoint = checkpoint
        if select_policy:
            self._control_mode = "policy"
        self._status = f"Loaded checkpoint: {checkpoint} (loading does not imply task competence)."

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
                self._status = "Policy mode is unavailable until a checkpoint is loaded."
                self._set_status()
                return None
            observation = self._refresh_observation()
            if observation is None:
                raise RuntimeError("adapter.current_observation() returned no observation")
            batched = _observation_batch(observation, self.adapter.observation_space)
            action = np.asarray(self._runner.action(batched), dtype=np.float32).reshape(-1, 4)[0]
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
        done = bool(terminated) or bool(truncated)
        if done:
            self._terminal = True
            self._running = False
            self._run_button.label = "Run task"
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
            outcome = "success" if bool(success) else "failure"
            self._status = f"Episode ended: {reason} / {outcome}."
            self._set_status()
            if self._runner is not None:
                self._runner.reset(np.asarray((True,), dtype=bool))
        self.publish()

    def step_once(self):
        """Process pending controls and take one action on the owner thread."""
        action = self.action()
        if action is None:
            self.publish()
            return None
        observation, reward, terminated, truncated, info = self.adapter.step(action)
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
                f"({float(goal[0]):.2f}, {float(goal[1]):.2f}) m"
                if goal is not None and len(goal) >= 2
                else "unavailable"
            )
            seq = metrics.get("sensor_sequences", {})
            gain, schema, timesteps, drive_status, observation_status = self._checkpoint_metadata()
            damage_metric, damage_status = self._damage_metadata()
            fields = [
                f"**Control:** {self._control_mode}  ",
                f"**Task:** {payload.get('mode', self._task_mode)}  ",
                f"**Time:** {float(payload.get('time', 0.0)):.2f} s  ",
                f"**Robot:** x={float(robot.get('x', 0.0)):.2f}, y={float(robot.get('y', 0.0)):.2f}, yaw={float(robot.get('yaw', 0.0)):.2f} rad, v={float(robot.get('v', 0.0)):.2f} m/s, w={float(robot.get('w', 0.0)):.2f} rad/s  ",
                f"**Gimbal:** yaw={float(payload.get('gimbal', {}).get('yaw', 0.0)):.2f}, pitch={float(payload.get('gimbal', {}).get('pitch', 0.0)):.2f} rad  ",
                f"**Goal:** {goal_text}  ",
                f"**Distance:** {float(metrics.get('distance', 0.0)):.2f} m  ",
                f"**Found:** {int(metrics.get('found', 0))}/{int(metrics.get('targets', 0))}; success={bool(metrics.get('is_success', False))}; reason={metrics.get('reason', payload.get('reason', ''))}  ",
                f"**Damage / collisions:** {int(metrics.get('damage', 0))} / {int(metrics.get('collisions', 0))}  ",
                f"**Sensors seq:** LiDAR {seq.get('lidar', '?')}, depth {seq.get('depth', '?')}  ",
                "**3D sensor overlays:** cyan LiDAR returns, orange 8 m range limit, blue valid depth (stride 4)  ",
                f"**Last reward:** {self._last_reward if self._last_reward is not None else payload.get('reward', '?')}  ",
                f"**Checkpoint:** {self._checkpoint or 'none'}  ",
                f"**Drive / observation:** gain={gain}; schema={schema}  ",
                f"**Checkpoint metadata:** timesteps={timesteps}; drive={drive_status}; observation={observation_status}  ",
                f"**Damage metric:** current={damage_metric}; checkpoint={damage_status}  ",
                f"**Status:** {self._status}",
            ]
            self._metrics_handle.content = "\n\n".join(fields)
        except Exception as exc:
            self._status = f"Viewer update error: {exc}"
            self._set_status()

    def _set_status(self) -> None:
        self._status_handle.content = self._status

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
            self.process_pending()
            transition = self.step_once()
            if transition is None:
                # SimulationContext.render() updates Viser from the current
                # physics state; unlike sim.step(), this never advances physics.
                self._render_scene()
            remaining = dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

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
    controls.run()
    return controls


__all__ = ["FieldViewerControls", "attach_field_controls", "depth_to_rgb", "run_viewer"]
