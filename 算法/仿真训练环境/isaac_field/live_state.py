"""Atomic latest-state export for an explicitly attached live viewer."""
from __future__ import annotations

import json
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np


def create_sensor_sample_exporter_from_environment(
    *, run_id, num_envs, lidar_rays, depth_height, depth_width,
):
    """Load the optional pure-CPU sensor sidecar exporter when configured."""
    if not os.environ.get("ISAAC_FIELD_SENSOR_SAMPLES_FILE"):
        return None

    explicit_module = os.environ.get("ISAAC_FIELD_SENSOR_EXPORT_MODULE")
    candidates = [Path(explicit_module).expanduser().resolve()] if explicit_module else []
    if not candidates:
        for ancestor in Path(__file__).resolve().parents:
            candidates.append(ancestor / "tools" / "live_training_observer" / "sensor_export.py")
    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        raise FileNotFoundError(
            "ISAAC_FIELD_SENSOR_SAMPLES_FILE is set, but tools/live_training_observer/sensor_export.py "
            "could not be located; set ISAAC_FIELD_SENSOR_EXPORT_MODULE to its absolute path"
        )

    module_name = "_isaac_field_sensor_export_v1"
    module = sys.modules.get(module_name)
    if module is None or Path(getattr(module, "__file__", "")).resolve() != module_path.resolve():
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load sensor exporter from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return module.create_sensor_sample_exporter_from_environment(
        run_id=run_id,
        num_envs=num_envs,
        lidar_rays=lidar_rays,
        depth_height=depth_height,
        depth_width=depth_width,
    )


class LiveStateExporter:
    def __init__(self, path, *, robot_usd_path, terrain_usd_path, body_names):
        self.path = Path(path).expanduser().resolve()
        self.robot_usd_path = str(robot_usd_path)
        self.terrain_usd_path = str(terrain_usd_path)
        self.body_names = list(body_names)
        self.source_pid = os.getpid()
        self.source_start_time_unix_ns = time.time_ns()
        self.run_id = os.environ.get("ISAAC_FIELD_LIVE_STATE_RUN_ID") or uuid.uuid4().hex
        self.sequence = 0
        self.failed_publish_count = 0
        self.last_write_error = None
        self._unreported_error = False
        self._last_warning_monotonic = None

    def write(self, *, root_link_pose_w, body_link_pose_w, env_origins_w, goals_local_xy, task_times_s,
              sim_time_s=None):
        self.sequence += 1
        origins = np.asarray(env_origins_w).tolist()
        goals = np.asarray(goals_local_xy).tolist()
        packet = {
            "schema_version": 1,
            "sequence": self.sequence,
            "timestamp_unix_ns": time.time_ns(),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "source_pid": self.source_pid,
            "source_start_time_unix_ns": self.source_start_time_unix_ns,
            "run_id": self.run_id,
            "source": "isaac_field._collect_sim_cpu_samples",
            "sim_time_s": sim_time_s,
            "sample_phase": "state_read_before_task_bookkeeping",
            "task_time_semantics": "FieldTask.time recorded at sample time, before current post-physics bookkeeping",
            "publish_status": "recovered" if self._unreported_error else "ok",
            "failed_publish_count": self.failed_publish_count,
            "last_write_error": self.last_write_error,
            "quaternion_order": "xyzw",
            "robot_usd_path": self.robot_usd_path,
            "terrain_usd_path": self.terrain_usd_path,
            "env_origins_w": origins,
            "goals_local_xy": goals,
            "goals_world_xy": [
                [goal[0] + origin[0], goal[1] + origin[1]]
                for goal, origin in zip(goals, origins, strict=True)
            ],
            "task_times_s": np.asarray(task_times_s).tolist(),
            "body_names": self.body_names,
            "root_link_pose_w": np.asarray(root_link_pose_w).tolist(),
            "body_link_pose_w": np.asarray(body_link_pose_w).tolist(),
        }
        temporary = self.path.with_name(f"{self.path.name}.{self.source_pid}.{self.sequence}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(packet, stream, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.path)
        except OSError as exc:
            self.failed_publish_count += 1
            self.last_write_error = f"{type(exc).__name__}: {exc}"
            self._unreported_error = True
            now = time.monotonic()
            if (self._last_warning_monotonic is None
                    or now - self._last_warning_monotonic >= 60.0):
                logging.getLogger(__name__).warning(
                    "live-state export_error at %s (sequence %d, failures %d); retaining last good packet: %s",
                    self.path, self.sequence, self.failed_publish_count, self.last_write_error,
                )
                self._last_warning_monotonic = now
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        self._unreported_error = False
        return True
