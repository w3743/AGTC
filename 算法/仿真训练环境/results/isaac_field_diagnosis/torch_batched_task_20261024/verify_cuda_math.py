"""Bounded CPU/CUDA numerical check for the batched task and sensor helpers.

This is a four-lane correctness check only. It does not launch Isaac Sim,
training, physics, a policy, or a throughput benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT_NAME = "cuda_math.json"
EXPECTED_PYTHONPATH = Path(r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境")
EXPECTED_ENV = {
    "ISAAC_FIELD_DISABLE_NATIVE_RUNTIME": "1",
    "RUN_ISAAC_PHYSICS": "0",
    "CUDA_VISIBLE_DEVICES": "0",
}
_ATTEMPT_CONTEXT: dict[str, Any] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _preserve_attempt(payload: dict[str, Any], result_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = HERE / f"{result_path.stem}_attempt_{stamp}.json"
    _write_once(path, payload)
    return path


def _source_sha256() -> dict[str, str]:
    sources = {
        "isaac_field/tensor_task.py": EXPECTED_PYTHONPATH / "isaac_field" / "tensor_task.py",
        "isaac_field/tensor_sensors.py": EXPECTED_PYTHONPATH / "isaac_field" / "tensor_sensors.py",
        "isaac_field/env.py": EXPECTED_PYTHONPATH / "isaac_field" / "env.py",
        "verify_cuda_math.py": Path(__file__).resolve(),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}


def _forbidden_modules() -> list[str]:
    prefixes = ("isaacsim", "carb", "omni")
    return sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def _require_environment() -> dict[str, str]:
    actual = {key: os.environ.get(key) for key in EXPECTED_ENV}
    for key, expected in EXPECTED_ENV.items():
        if actual[key] != expected:
            raise RuntimeError(f"{key} must be {expected!r}; found {actual[key]!r}")
    pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [Path(value).resolve() for value in pythonpath.split(os.pathsep) if value]
    if EXPECTED_PYTHONPATH.resolve() not in paths:
        raise RuntimeError("PYTHONPATH must include the Isaac training project root")
    return {**actual, "PYTHONPATH": pythonpath}


def _max_abs(left, right) -> float:
    import numpy as np

    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _cpu_array(tensor):
    return tensor.detach().to(device="cpu").numpy()


def _assert_array_equal(label: str, left, right, *, atol: float = 0.0, rtol: float = 0.0) -> float:
    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise AssertionError(f"{label}: shape {left_array.shape} != {right_array.shape}")
    difference = _max_abs(left_array, right_array)
    if atol == 0.0 and rtol == 0.0:
        if not np.array_equal(left_array, right_array):
            raise AssertionError(f"{label}: values differ (max_abs={difference})")
    elif not np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(
            f"{label}: values differ (max_abs={difference}, atol={atol}, rtol={rtol})"
        )
    return difference


def _make_tasks():
    import numpy as np

    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, FeedbackSample
    from isaac_field.task import Config, FieldTask

    tasks = []
    for lane in range(4):
        cfg = Config(
            size=12.0,
            plants=0,
            rock_density=0.0,
            terrain_enabled=False,
            validate_scene=False,
            lidar_rays=8,
            depth_height=4,
            depth_width=5,
            dt=0.1,
            max_time=0.2,
            time_penalty=0.1,
            collision_penalty=0.0,
            goal_tolerance=0.4,
            goal_hold=0.2,
            goal_progress_budget=5.0,
            goal_stop_speed=0.05,
            goal_stop_angular_speed=0.1,
            goto_reward_profile="fast_stop_v2",
            fast_stop_failure_remaining_time_cost=True,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            position_noise=0.025,
        )
        task = FieldTask(cfg, mode="goto")
        task.prepare(seed=8410 + lane, options={"start": [2.0, 2.0, 0.2], "goal": [4.0, 2.0]})
        heading = 0.7 + 0.08 * lane
        task.update_feedback(FeedbackSample(
            sample_time=0.0,
            attitude_quat_xyzw=(0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0)),
            gimbal_q_rad=(0.1 * lane, -0.05 * lane),
            gimbal_qd_rps=(0.02, -0.03),
            motor_iq_a=(1.0 + lane, -1.0 - lane),
            motor_rpm=(100.0 + lane, -100.0 - lane),
            valid={"attitude": True, "gimbal_yaw": True, "gimbal_pitch": True,
                   "motor_left": True, "motor_right": True},
        ))

        def capture(kind, *, _cfg=cfg, _lane=lane):
            if kind == "lidar":
                ranges = np.linspace(_cfg.lidar_min_range, _cfg.lidar_range, _cfg.lidar_rays,
                                     dtype=np.float32)
                ranges[(_lane + 2) % _cfg.lidar_rays] = _cfg.lidar_range
                valid = np.ones(_cfg.lidar_rays, dtype=bool)
                valid[(_lane + 2) % _cfg.lidar_rays] = False
                return {"ranges": ranges, "valid": valid}
            if kind == "depth":
                values = np.linspace(_cfg.depth_min_range, _cfg.depth_range,
                                     _cfg.depth_height * _cfg.depth_width,
                                     dtype=np.float32).reshape(_cfg.depth_height, _cfg.depth_width)
                valid = np.ones_like(values, dtype=bool)
                valid[_lane % _cfg.depth_height, (_lane + 1) % _cfg.depth_width] = False
                return {"values": values, "valid": valid}
            return {"detections": []}

        task.attach_sensors(capture)
        task.initialize_sensors()
        task.last_action = np.asarray([0.1 * lane, -0.1, 0.05, 0.02], dtype=np.float32)
        if lane == 2:
            task._hold = 0.03
        tasks.append(task)
    return tasks


def _pack_observations(tasks, torch, device):
    import numpy as np

    from isaac_field.tensor_task import pack_emptyfield_goto_observation

    sensor_order = ("lidar", "depth", "camera")

    def tensor(rows, dtype=torch.float64):
        return torch.as_tensor(np.stack(rows), dtype=dtype, device="cpu")

    inputs_cpu = {
        "lidar_ranges": tensor([task.frames["lidar"]["ranges"] for task in tasks], torch.float32),
        "lidar_valid": tensor([task.frames["lidar"]["valid"] for task in tasks], torch.bool),
        "depth_values": tensor([task.frames["depth"]["values"] for task in tasks], torch.float32),
        "depth_valid": tensor([task.frames["depth"]["valid"] for task in tasks], torch.bool),
        "empty_detections": torch.zeros((4, 0, 8), dtype=torch.float32),
        "frame_capture_times": tensor([
            [task.frames[key]["capture"] for key in sensor_order] for task in tasks
        ]),
        "frame_estimated_poses": tensor([
            [task.frames[key]["estimated_pose"] for key in sensor_order] for task in tasks
        ]),
        "frame_new": tensor([
            [task._new[key] for key in sensor_order] for task in tasks
        ], torch.bool),
        "estimated_pose": tensor([task._estimated_pose() for task in tasks], torch.float32),
        "goal_xy": tensor([task.goal for task in tasks]),
        "last_actions": tensor([task.last_action for task in tasks], torch.float32),
        "episode_time": tensor([task.time for task in tasks]),
        "feedback": tensor([task._feedback_snapshot.feedback for task in tasks], torch.float32),
        "size": tasks[0].cfg.size,
        "lidar_range": tasks[0].cfg.lidar_range,
        "depth_range": tasks[0].cfg.depth_range,
        "max_time": tasks[0].cfg.max_time,
        "max_detections": tasks[0].cfg.max_detections,
    }
    inputs_cuda = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs_cpu.items()
    }
    expected = [task._observation() for task in tasks]
    cpu_result = pack_emptyfield_goto_observation(**inputs_cpu)
    cuda_result = pack_emptyfield_goto_observation(**inputs_cuda)
    normalized_lidar_cpu = torch.where(
        inputs_cpu["lidar_valid"],
        inputs_cpu["lidar_ranges"] / tasks[0].cfg.lidar_range,
        torch.zeros_like(inputs_cpu["lidar_ranges"]),
    ).unsqueeze(1)
    normalized_depth_cpu = torch.where(
        inputs_cpu["depth_valid"],
        inputs_cpu["depth_values"] / tasks[0].cfg.depth_range,
        torch.zeros_like(inputs_cpu["depth_values"]),
    ).unsqueeze(1)
    normalized_lidar_cuda = normalized_lidar_cpu.to(device=device)
    normalized_depth_cuda = normalized_depth_cpu.to(device=device)
    normalized_before = (
        normalized_lidar_cpu.clone(), normalized_depth_cpu.clone(),
        normalized_lidar_cuda.clone(), normalized_depth_cuda.clone(),
    )
    cpu_cached_result = pack_emptyfield_goto_observation(
        **inputs_cpu, normalized_lidar=normalized_lidar_cpu, normalized_depth=normalized_depth_cpu
    )
    cuda_cached_result = pack_emptyfield_goto_observation(
        **inputs_cuda, normalized_lidar=normalized_lidar_cuda, normalized_depth=normalized_depth_cuda
    )
    if cpu_cached_result["lidar"] is not normalized_lidar_cpu:
        raise AssertionError("CPU normalized LiDAR cache object was copied or replaced")
    if cpu_cached_result["depth"] is not normalized_depth_cpu:
        raise AssertionError("CPU normalized depth cache object was copied or replaced")
    if cuda_cached_result["lidar"] is not normalized_lidar_cuda:
        raise AssertionError("CUDA normalized LiDAR cache object was copied or replaced")
    if cuda_cached_result["depth"] is not normalized_depth_cuda:
        raise AssertionError("CUDA normalized depth cache object was copied or replaced")
    expected_shapes = {
        "lidar": (4, 1, 8),
        "depth": (4, 1, 4, 5),
        "detections": (4, tasks[0].cfg.max_detections, 8),
        "state": (4, 33),
        "feedback": (4, 37),
    }
    if set(cpu_result) != set(expected_shapes) or set(cuda_result) != set(expected_shapes):
        raise AssertionError("observation packer did not return the exact five-key contract")
    output_diffs = {}
    cache_output_diffs = {}
    for key, shape in expected_shapes.items():
        cpu_array, cuda_array = _cpu_array(cpu_result[key]), _cpu_array(cuda_result[key])
        cpu_cached_array = _cpu_array(cpu_cached_result[key])
        cuda_cached_array = _cpu_array(cuda_cached_result[key])
        if tuple(cpu_result[key].shape) != shape or tuple(cuda_result[key].shape) != shape:
            raise AssertionError(f"{key} output shape does not match {shape}")
        if tuple(cpu_cached_result[key].shape) != shape or tuple(cuda_cached_result[key].shape) != shape:
            raise AssertionError(f"cached {key} output shape does not match {shape}")
        if cpu_result[key].dtype != torch.float32 or cuda_result[key].dtype != torch.float32:
            raise AssertionError(f"{key} output dtype must be float32")
        if cpu_cached_result[key].dtype != torch.float32 or cuda_cached_result[key].dtype != torch.float32:
            raise AssertionError(f"cached {key} output dtype must be float32")
        if key in ("lidar", "depth"):
            atol, rtol = 1e-7, 1e-6
        elif key == "state":
            atol, rtol = 2e-7, 0.0
        else:
            atol, rtol = 0.0, 0.0
        output_diffs[key] = {
            "cpu_vs_cuda_max_abs": _assert_array_equal(f"observation.{key}.cpu_cuda", cpu_array,
                                                         cuda_array, atol=atol, rtol=rtol),
            "cpu_vs_fieldtask_max_abs": max(
                _assert_array_equal(f"observation.{key}.fieldtask[{lane}]", cpu_array[lane],
                                    expected[lane][key], atol=atol, rtol=rtol)
                for lane in range(4)
            ),
            "tolerance": {"atol": atol, "rtol": rtol},
        }
        cache_output_diffs[key] = {
            "cpu_cache_vs_default_max_abs": _assert_array_equal(
                f"observation.cached.{key}.cpu_default", cpu_cached_array, cpu_array,
                atol=atol, rtol=rtol,
            ),
            "cuda_cache_vs_default_max_abs": _assert_array_equal(
                f"observation.cached.{key}.cuda_default", cuda_cached_array, cuda_array,
                atol=atol, rtol=rtol,
            ),
            "cpu_cache_vs_cuda_cache_max_abs": _assert_array_equal(
                f"observation.cached.{key}.cpu_cuda", cpu_cached_array, cuda_cached_array,
                atol=atol, rtol=rtol,
            ),
            "cpu_cache_vs_fieldtask_max_abs": max(
                _assert_array_equal(f"observation.cached.{key}.fieldtask[{lane}]",
                                    cpu_cached_array[lane], expected[lane][key],
                                    atol=atol, rtol=rtol)
                for lane in range(4)
            ),
            "cuda_cache_vs_fieldtask_max_abs": max(
                _assert_array_equal(f"observation.cached.{key}.cuda_fieldtask[{lane}]",
                                    cuda_cached_array[lane], expected[lane][key],
                                    atol=atol, rtol=rtol)
                for lane in range(4)
            ),
            "tolerance": {"atol": atol, "rtol": rtol},
        }
    for actual, before in zip((normalized_lidar_cpu, normalized_depth_cpu,
                               normalized_lidar_cuda, normalized_depth_cuda), normalized_before, strict=True):
        if not torch.equal(actual, before):
            raise AssertionError("pack function mutated a caller-owned normalized sensor cache")
    return {
        "default_path_max_abs": output_diffs,
        "normalized_cache_path": {
            "cpu_returned_same_lidar_object": cpu_cached_result["lidar"] is normalized_lidar_cpu,
            "cpu_returned_same_depth_object": cpu_cached_result["depth"] is normalized_depth_cpu,
            "cuda_returned_same_lidar_object": cuda_cached_result["lidar"] is normalized_lidar_cuda,
            "cuda_returned_same_depth_object": cuda_cached_result["depth"] is normalized_depth_cuda,
            "cache_storage_unmodified": True,
            "cache_shapes": {
                "lidar": list(normalized_lidar_cuda.shape),
                "depth": list(normalized_depth_cuda.shape),
            },
            "max_abs_differences": cache_output_diffs,
            "cache_source": "same torch.where normalization expression used by IsaacFieldEnv._build_tensor_observations; CPU FieldTask._observation used as policy reference",
        },
    }


def _step_inputs(tasks, torch, lane_scenarios, active):
    import numpy as np

    rows = {key: [] for key in (
        "position_xy", "old_distance", "goal_xy", "root_com_velocity_xy",
        "body_angular_velocity", "body_tilt_pitch_roll",
    )}
    boundary, physics_event, previous_failure, episode_time = [], [], [], []
    hold_elapsed, initial_distance = [], []
    for lane, task in enumerate(tasks):
        scenario = lane_scenarios[lane]
        rows["position_xy"].append(scenario["position"])
        rows["old_distance"].append(float(np.linalg.norm(task.goal - task.robot[:2])))
        rows["goal_xy"].append(task.goal)
        rows["root_com_velocity_xy"].append(scenario["velocity"])
        rows["body_angular_velocity"].append(scenario["angular"])
        rows["body_tilt_pitch_roll"].append(scenario["tilt"])
        boundary.append(scenario["boundary"])
        physics_event.append(scenario["physics_event"])
        previous_failure.append(task.physics_failure)
        episode_time.append(task.time)
        hold_elapsed.append(task._hold)
        initial_distance.append(task._initial_goal_distance)

    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(value, dtype=dtype, device="cpu")

    return {
        "position_xy": tensor(rows["position_xy"]),
        "old_distance": tensor(rows["old_distance"]),
        "goal_xy": tensor(rows["goal_xy"]),
        "root_com_velocity_xy": tensor(rows["root_com_velocity_xy"]),
        "body_angular_velocity": tensor(rows["body_angular_velocity"]),
        "body_tilt_pitch_roll": tensor(rows["body_tilt_pitch_roll"]),
        "boundary": tensor(boundary, torch.bool),
        "physics_failure_event": tensor(physics_event, torch.bool),
        "previous_physics_failure": tensor(previous_failure, torch.bool),
        "episode_time": tensor(episode_time),
        "hold_elapsed": tensor(hold_elapsed),
        "initial_goal_distance": tensor(initial_distance),
        "active_mask": tensor(active, torch.bool),
        "dt": tasks[0].cfg.dt,
        "max_time": tasks[0].cfg.max_time,
        "goal_tolerance": tasks[0].cfg.goal_tolerance,
        "goal_hold": tasks[0].cfg.goal_hold,
        "goal_stop_speed": tasks[0].cfg.goal_stop_speed,
        "goal_stop_angular_speed": tasks[0].cfg.goal_stop_angular_speed,
        "time_penalty": tasks[0].cfg.time_penalty,
        "goal_progress_budget": tasks[0].cfg.goal_progress_budget,
    }


def _reason_code(reason: str) -> int:
    return {"": 0, "goal_reached": 1, "outside_or_overturned": 2, "time_limit": 3}[reason]


def _run_reward_steps(tasks, torch, device):
    from isaac_field.tensor_task import fast_stop_v2_step

    scenarios = [
        [
            {"position": [3.6, 2.0], "velocity": [0.0, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [1.2, 0.0], "boundary": False, "physics_event": False},
            {"position": [3.6, 2.0], "velocity": [0.05, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [0.0, 0.0], "boundary": False, "physics_event": False},
            {"position": [3.6, 2.0], "velocity": [0.0, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [0.0, 0.0], "boundary": True, "physics_event": False},
            {"position": [3.6, 2.0], "velocity": [0.0, 0.0], "angular": [0.1, 0.0, 0.0],
             "tilt": [0.0, 0.0], "boundary": False, "physics_event": False},
        ],
        [
            {"position": [3.6, 2.0], "velocity": [0.0, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [1.2, 0.0], "boundary": False, "physics_event": False},
            {"position": [3.6, 2.0], "velocity": [0.05, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [0.0, 0.0], "boundary": False, "physics_event": False},
            {"position": [3.6, 2.0], "velocity": [0.0, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [0.0, 0.0], "boundary": False, "physics_event": False},
            {"position": [3.6, 2.0], "velocity": [0.0, 0.0], "angular": [0.0, 0.0, 0.0],
             "tilt": [1.2001, 0.0], "boundary": False, "physics_event": False},
        ],
    ]
    active_masks = [[True, True, True, True], [True, True, False, True]]
    summary: list[dict[str, Any]] = []
    for step_index, (step_scenarios, active) in enumerate(zip(scenarios, active_masks, strict=True)):
        inputs_cpu = _step_inputs(tasks, torch, step_scenarios, active)
        inputs_cuda = {
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs_cpu.items()
        }
        cpu = fast_stop_v2_step(**inputs_cpu)
        cuda = fast_stop_v2_step(**inputs_cuda)
        diffs: dict[str, float] = {}
        reward_reference_diffs: dict[str, float] = {}
        for key in ("rewards", "done", "terminated", "truncated", "reason_codes", "next_time",
                    "next_hold", "next_physics_failure"):
            left, right = _cpu_array(cpu[key]), _cpu_array(cuda[key])
            atol = 2e-6 if key == "rewards" else (1e-12 if key in ("next_time", "next_hold") else 0.0)
            diffs[key] = _assert_array_equal(f"reward.step{step_index}.{key}", left, right, atol=atol)
            if key in ("rewards", "next_time", "next_hold"):
                reward_reference_diffs[key] = _max_abs(left, left)
        if set(cpu["components"]) != set(cuda["components"]):
            raise AssertionError("reward component key sets differ between CPU and CUDA")
        for name in cpu["components"]:
            diffs[f"component:{name}"] = _assert_array_equal(
                f"reward.step{step_index}.component.{name}", _cpu_array(cpu["components"][name]),
                _cpu_array(cuda["components"][name]), atol=2e-6,
            )
            reward_reference_diffs[f"component:{name}"] = 0.0

        references: list[Any | None] = [None] * 4
        for lane, task in enumerate(tasks):
            if not active[lane]:
                continue
            scenario = step_scenarios[lane]
            before = task.robot.copy()
            old_distance = float(inputs_cpu["old_distance"][lane])
            task.robot[:2] = scenario["position"]
            task.root_com_velocity_xy = scenario["velocity"]
            task.body_angular_velocity = scenario["angular"]
            task.body_pose = (0.0, *scenario["tilt"])
            references[lane] = task.finish(
                [0.0, 0.0, 0.0, 0.0], before, old_distance,
                boundary=scenario["boundary"], physics_failure=scenario["physics_event"],
            )

        for lane, task in enumerate(tasks):
            if not active[lane]:
                if cpu["rewards"][lane].item() != 0.0 or bool(cpu["done"][lane].item()):
                    raise AssertionError(f"inactive reward lane {lane} changed output")
                if int(cpu["reason_codes"][lane].item()) != 0:
                    raise AssertionError(f"inactive reward lane {lane} has a terminal reason")
                if float(cpu["next_time"][lane]) != task.time or float(cpu["next_hold"][lane]) != task._hold:
                    raise AssertionError(f"inactive reward lane {lane} changed persistent clocks")
                if bool(cpu["next_physics_failure"][lane].item()) is not task.physics_failure:
                    raise AssertionError(f"inactive reward lane {lane} changed physics failure state")
                for name, values in cpu["components"].items():
                    if float(values[lane].item()) != 0.0:
                        raise AssertionError(f"inactive reward lane {lane} emitted component {name}")
                continue
            _, expected_reward, terminated, truncated, info = references[lane]
            reward_reference_diffs["rewards"] = max(
                reward_reference_diffs.get("rewards", 0.0),
                abs(float(cpu["rewards"][lane]) - expected_reward),
            )
            if not math.isclose(float(cpu["rewards"][lane]), expected_reward, rel_tol=0.0, abs_tol=2e-6):
                raise AssertionError(f"reward lane {lane} differs from real FieldTask.finish")
            if bool(cpu["done"][lane].item()) is not terminated:
                raise AssertionError(f"done lane {lane} differs from real FieldTask.finish")
            if bool(cpu["terminated"][lane].item()) is not terminated or truncated:
                raise AssertionError(f"termination flags lane {lane} differ from fast_stop_v2 reference")
            if int(cpu["reason_codes"][lane].item()) != _reason_code(info["reason"]):
                raise AssertionError(f"reason code lane {lane} differs from real FieldTask.finish")
            if not math.isclose(float(cpu["next_time"][lane]), task.time, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"next_time lane {lane} differs from real FieldTask.finish")
            reward_reference_diffs["next_time"] = max(
                reward_reference_diffs.get("next_time", 0.0),
                abs(float(cpu["next_time"][lane]) - task.time),
            )
            if not math.isclose(float(cpu["next_hold"][lane]), task._hold, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"next_hold lane {lane} differs from real FieldTask.finish")
            reward_reference_diffs["next_hold"] = max(
                reward_reference_diffs.get("next_hold", 0.0),
                abs(float(cpu["next_hold"][lane]) - task._hold),
            )
            if bool(cpu["next_physics_failure"][lane].item()) is not task.physics_failure:
                raise AssertionError(f"physics failure lane {lane} differs from real FieldTask.finish")
            for name, values in cpu["components"].items():
                expected_component = float(info["reward_components"].get(name, 0.0))
                key = f"component:{name}"
                reward_reference_diffs[key] = max(
                    reward_reference_diffs.get(key, 0.0),
                    abs(float(values[lane]) - expected_component),
                )
                if not math.isclose(float(values[lane]), expected_component,
                                    rel_tol=0.0, abs_tol=2e-6):
                    raise AssertionError(f"reward component {name} lane {lane} differs from reference")
        summary.append({
            "step": step_index,
            "active": active,
            "cpu_fieldtask_reason_codes": [
                None if references[lane] is None else _reason_code(references[lane][4]["reason"])
                for lane in range(4)
            ],
            "cpu_cuda_max_abs": diffs,
            "cpu_fieldtask_max_abs": reward_reference_diffs,
            "tolerance": {
                "reward_and_components_abs": 2e-6,
                "clock_and_hold_abs": 1e-12,
                "flags_reason_codes_physics_state": "exact",
            },
        })
    return summary


def _reference_lidar(raw, rng, *, min_range, max_range, relative_error):
    import numpy as np

    valid = np.isfinite(raw) & (raw >= min_range) & (raw < max_range)
    values = np.full(raw.shape, max_range, dtype=np.float32)
    values[valid] = np.clip(
        raw[valid] * (1 + rng.uniform(-relative_error, relative_error, valid.sum())),
        min_range,
        max_range,
    )
    return values, valid


def _reference_depth(raw, rng, *, min_range, max_range, noise_std):
    import numpy as np

    valid = np.isfinite(raw) & (raw >= min_range) & (raw <= max_range)
    values = np.full_like(raw, max_range)
    values[valid] = np.clip(raw[valid] + rng.normal(0, noise_std, valid.sum()), min_range, max_range)
    return values, valid


def _run_sensor_steps(torch, np, device):
    from isaac_field.tensor_sensors import capture_depth_batch, capture_lidar_batch

    lidar_raw = np.asarray([
        [0.12, 0.5, 8.0, np.nan, 1.1, 0.11, 7.9, np.inf, 0.2],
        [0.2, 0.3, 0.4, 0.5, 8.0, -np.inf, 1.0, 3.0, 7.99],
        [8.0, 8.1, np.nan, np.inf, -np.inf, 0.11, 0.12, 8.0, 0.0],
        [0.13, 0.25, 0.75, 1.5, 2.0, 4.0, 7.9, 8.0, np.nan],
    ], dtype=np.float32)
    depth_raw = np.asarray([
        [[0.3, 0.5, 5.0, np.nan], [np.inf, -np.inf, 1.2, 4.9], [0.29, 0.31, 2.2, 3.3]],
        [[0.4, 1.0, 2.0, 5.0], [0.2, 0.3, 4.99, 5.01], [np.nan, 1.3, 2.7, 3.8]],
        [[np.nan, np.inf, -np.inf, 0.0], [0.1, 0.2, 0.29, 5.1], [8.0, 9.0, 10.0, 0.0]],
        [[0.3, 0.3, 0.3, 0.3], [0.4, 1.5, 3.3, 5.0], [5.01, 0.29, np.nan, 2.2]],
    ], dtype=np.float32)
    due_ids = [3, 1, 0, 2]
    seeds = [7120, 7121, 7122, 7123]
    cpu_rngs = [np.random.default_rng(seed) for seed in seeds]
    cuda_rngs = [np.random.default_rng(seed) for seed in seeds]
    reference_rngs = [np.random.default_rng(seed) for seed in seeds]

    cpu_lidar = capture_lidar_batch(
        torch.as_tensor(lidar_raw), due_ids, cpu_rngs,
        min_range=0.12, max_range=8.0, relative_error=0.05,
    )
    cuda_lidar = capture_lidar_batch(
        torch.as_tensor(lidar_raw, device=device), due_ids, cuda_rngs,
        min_range=0.12, max_range=8.0, relative_error=0.05,
    )
    lidar_diffs = {}
    for lane in due_ids:
        expected_values, expected_valid = _reference_lidar(
            lidar_raw[lane], reference_rngs[lane],
            min_range=0.12, max_range=8.0, relative_error=0.05,
        )
        for field in ("ranges", "valid"):
            cpu_array = _cpu_array(cpu_lidar[lane][field])
            cuda_array = _cpu_array(cuda_lidar[lane][field])
            expected = expected_values if field == "ranges" else expected_valid
            atol, rtol = (1e-7, 1e-6) if field == "ranges" else (0.0, 0.0)
            lidar_diffs[f"lane{lane}.{field}"] = {
                "cpu_vs_cuda_max_abs": _assert_array_equal(
                    f"sensor.lidar.lane{lane}.{field}.cpu_cuda", cpu_array, cuda_array,
                    atol=atol, rtol=rtol,
                ),
                "cpu_vs_reference_max_abs": _assert_array_equal(
                    f"sensor.lidar.lane{lane}.{field}.cpu_reference", cpu_array, expected,
                    atol=atol, rtol=rtol,
                ),
                "cuda_vs_reference_max_abs": _assert_array_equal(
                    f"sensor.lidar.lane{lane}.{field}.cuda_reference", cuda_array, expected,
                    atol=atol, rtol=rtol,
                ),
                "tolerance": {"atol": atol, "rtol": rtol},
            }

    cpu_depth = capture_depth_batch(
        torch.as_tensor(depth_raw), due_ids, cpu_rngs,
        min_range=0.3, max_range=5.0, noise_std=0.01,
    )
    cuda_depth = capture_depth_batch(
        torch.as_tensor(depth_raw, device=device), due_ids, cuda_rngs,
        min_range=0.3, max_range=5.0, noise_std=0.01,
    )
    depth_diffs = {}
    for lane in due_ids:
        expected_values, expected_valid = _reference_depth(
            depth_raw[lane], reference_rngs[lane],
            min_range=0.3, max_range=5.0, noise_std=0.01,
        )
        for field in ("values", "valid"):
            cpu_array = _cpu_array(cpu_depth[lane][field])
            cuda_array = _cpu_array(cuda_depth[lane][field])
            expected = expected_values if field == "values" else expected_valid
            atol, rtol = (1e-7, 1e-6) if field == "values" else (0.0, 0.0)
            depth_diffs[f"lane{lane}.{field}"] = {
                "cpu_vs_cuda_max_abs": _assert_array_equal(
                    f"sensor.depth.lane{lane}.{field}.cpu_cuda", cpu_array, cuda_array,
                    atol=atol, rtol=rtol,
                ),
                "cpu_vs_reference_max_abs": _assert_array_equal(
                    f"sensor.depth.lane{lane}.{field}.cpu_reference", cpu_array, expected,
                    atol=atol, rtol=rtol,
                ),
                "cuda_vs_reference_max_abs": _assert_array_equal(
                    f"sensor.depth.lane{lane}.{field}.cuda_reference", cuda_array, expected,
                    atol=atol, rtol=rtol,
                ),
                "tolerance": {"atol": atol, "rtol": rtol},
            }

    rng_diffs = {}
    for lane in range(4):
        cpu_next = cpu_rngs[lane].random(8)
        cuda_next = cuda_rngs[lane].random(8)
        reference_next = reference_rngs[lane].random(8)
        _assert_array_equal(f"sensor.rng.lane{lane}.cpu_cuda", cpu_next, cuda_next, atol=0.0)
        _assert_array_equal(f"sensor.rng.lane{lane}.cpu_reference", cpu_next, reference_next, atol=0.0)
        _assert_array_equal(f"sensor.rng.lane{lane}.cuda_reference", cuda_next, reference_next, atol=0.0)
        rng_diffs[f"lane{lane}"] = 0.0
    return {
        "due_env_ids": due_ids,
        "lidar_shapes": {str(lane): list(cpu_lidar[lane]["ranges"].shape) for lane in due_ids},
        "depth_shapes": {str(lane): list(cpu_depth[lane]["values"].shape) for lane in due_ids},
        "cpu_cuda_max_abs_lidar": lidar_diffs,
        "cpu_cuda_max_abs_depth": depth_diffs,
        "numpy_rng_nextdraw_identical": True,
        "numpy_rng_nextdraw_max_abs": rng_diffs,
    }


def _run() -> dict[str, Any]:
    env = _require_environment()
    import numpy as np
    import torch

    base = {
        "started_utc": _utc_now(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "environment": env,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "numpy_version": np.__version__,
        "batch_size": 4,
        "purpose": "bounded CPU/CUDA numerical correctness only; not training, Isaac Sim, physics, or FPS/throughput",
        "source_sha256": _source_sha256(),
    }
    _ATTEMPT_CONTEXT.update(base)
    _ATTEMPT_CONTEXT["stage"] = "torch_imported"
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; stop without retrying CUDA work")
    _ATTEMPT_CONTEXT["torch_cuda_available"] = True
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}")
    base["cuda_device"] = torch.cuda.get_device_name(device)
    base["cuda_device_count_visible"] = torch.cuda.device_count()
    _ATTEMPT_CONTEXT.update({
        "cuda_device": base["cuda_device"],
        "cuda_device_count_visible": base["cuda_device_count_visible"],
        "stage": "cuda_device_ready",
    })
    if "cuda:0" != str(device):
        raise RuntimeError("visible CUDA device did not resolve to cuda:0")

    tasks = _make_tasks()
    _ATTEMPT_CONTEXT["stage"] = "real_fieldtask_fixtures_ready"
    observation_diffs = _pack_observations(tasks, torch, device)
    _ATTEMPT_CONTEXT["stage"] = "observation_cpu_cuda_and_fieldtask_passed"
    reward_steps = _run_reward_steps(tasks, torch, device)
    _ATTEMPT_CONTEXT["stage"] = "fast_stop_cpu_cuda_and_fieldtask_passed"
    sensor_result = _run_sensor_steps(torch, np, device)
    _ATTEMPT_CONTEXT["stage"] = "tensor_sensor_cpu_cuda_rng_and_numpy_passed"
    torch.cuda.synchronize(device)

    forbidden = _forbidden_modules()
    if forbidden:
        raise RuntimeError(f"forbidden native runtime modules were imported: {forbidden}")
    return {
        **base,
        "status": "passed",
        "finished_utc": _utc_now(),
        "no_simulation": True,
        "training_started": False,
        "physics_sim_started": False,
        "native_runtime_modules_loaded": forbidden,
        "no_isaacsim_carb_omni_modules": not forbidden,
        "torch_loaded": "torch" in sys.modules,
        "torch_cuda_available": torch.cuda.is_available(),
        "observation_contract": {
            "five_keys": ["lidar", "depth", "detections", "state", "feedback"],
            "shapes": {"lidar": [4, 1, 8], "depth": [4, 1, 4, 5],
                       "detections": [4, 16, 8], "state": [4, 33], "feedback": [4, 37]},
            "max_abs_differences": observation_diffs["default_path_max_abs"],
            "normalized_cache_path": observation_diffs["normalized_cache_path"],
            "reference": "real FieldTask._observation",
        },
        "fast_stop_v2": {
            "steps": reward_steps,
            "checks": [
                "goal distance boundary <= 0.4m and tilt exactly 1.2rad are accepted",
                "XY speed exactly 0.05m/s does not accumulate hold",
                "full body angular speed exactly 0.1rad/s does not accumulate hold",
                "0.2s hold succeeds after two 0.1s steps",
                "boundary failure precedes success and preserves prior hold",
                "tilt > 1.2rad precedes timeout",
                "fast_stop_v2 timeout is terminated and not truncated",
                "inactive lane retains its time, hold, and physics-failure state with zero outputs",
            ],
            "reference": "real FieldTask.finish/_finish_step",
        },
        "tensor_sensors": sensor_result,
        "prior_attempts": [{
            "path": "cuda_math_attempt_20261024T094650767641Z.json",
            "status": "failed_before_reward_and_sensor_stages",
            "cause": "initial observation depth CPU/CUDA exact-equality assertion; max_abs=5.960464477539063e-08",
            "classification": "floating-point tolerance assertion, not a CUDA API/device failure",
            "preserved_without_modification": True,
        }],
        "prior_reports": [{
            "path": "cuda_math.json",
            "source_relation": "before normalized observation cache arguments and fixed-shape tensor_sensors gather/H2D optimization",
            "preserved_without_modification": True,
        }],
        "release": "small batch only; no child processes; process exits immediately after writing this record",
    }


def _resolve_result_path(output: str) -> Path:
    requested = Path(output)
    result_path = requested.resolve() if requested.is_absolute() else (HERE / requested).resolve()
    if result_path.parent != HERE.resolve():
        raise ValueError("--output must name a file in the diagnostic result directory")
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_RESULT_NAME,
                        help="write-once result filename in this diagnostic directory")
    args = parser.parse_args(argv)
    result_path = _resolve_result_path(args.output)
    _ATTEMPT_CONTEXT["output_path"] = str(result_path)
    try:
        if result_path.exists():
            raise FileExistsError(f"write-once result already exists: {result_path}")
        result = _run()
        result["result_path"] = str(result_path)
        _write_once(result_path, result)
        return 0
    except BaseException as exc:
        attempt = {
            "status": "failed",
            "started_or_failed_utc": _utc_now(),
            "script": str(Path(__file__).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "attempt_policy": "one bounded invocation; no retry after CUDA errors",
            "requested_result_path": str(result_path),
            "environment": {key: os.environ.get(key) for key in (*EXPECTED_ENV, "PYTHONPATH")},
            "native_runtime_modules_loaded": _forbidden_modules(),
            "context": dict(_ATTEMPT_CONTEXT),
        }
        attempt_path = _preserve_attempt(attempt, result_path)
        print(f"CUDA numerical verification failed; attempt preserved at {attempt_path}", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
