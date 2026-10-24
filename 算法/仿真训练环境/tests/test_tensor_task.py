"""CPU equivalence checks for the empty-field batched Torch task path."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, FeedbackSample, FEEDBACK_DIM
from isaac_field.task import Config, FieldTask
from isaac_field.tensor_task import (
    REASON_GOAL_REACHED,
    REASON_ONGOING,
    REASON_OUTSIDE_OR_OVERTURNED,
    REASON_TIME_LIMIT,
    fast_stop_v2_step,
    pack_emptyfield_goto_observation,
)


def _make_task(*, dt: float = 0.1, max_time: float = 2.0, goal_hold: float = 0.5,
               goal=(4.0, 2.0), start=(2.0, 2.0, 0.2), time_penalty: float = 0.1) -> FieldTask:
    cfg = Config(
        size=12.0,
        plants=0,
        rock_density=0.0,
        terrain_enabled=False,
        validate_scene=False,
        lidar_rays=8,
        depth_height=3,
        depth_width=4,
        dt=dt,
        max_time=max_time,
        time_penalty=time_penalty,
        collision_penalty=0.0,
        goal_tolerance=0.4,
        goal_hold=goal_hold,
        goal_progress_budget=5.0,
        goal_stop_speed=0.05,
        goal_stop_angular_speed=0.1,
        goto_reward_profile="fast_stop_v2",
        fast_stop_failure_remaining_time_cost=True,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=426, options={"start": list(start), "goal": list(goal)})

    def capture(kind):
        if kind == "lidar":
            return {"ranges": np.full(cfg.lidar_rays, cfg.lidar_range, np.float32),
                    "valid": np.zeros(cfg.lidar_rays, dtype=bool)}
        if kind == "depth":
            return {"values": np.full((cfg.depth_height, cfg.depth_width), cfg.depth_range, np.float32),
                    "valid": np.zeros((cfg.depth_height, cfg.depth_width), dtype=bool)}
        return {"detections": []}

    task.attach_sensors(capture)
    task.initialize_sensors()
    return task


def _tensor(value, *, dtype=torch.float64):
    return torch.as_tensor(np.asarray(value), dtype=dtype, device="cpu")


def _batch_call(tasks, *, positions, velocities=None, angular=None, tilts=None,
                boundaries=None, physics_events=None, active=None):
    count = len(tasks)
    velocities = [np.zeros(2) for _ in tasks] if velocities is None else velocities
    angular = [np.zeros(3) for _ in tasks] if angular is None else angular
    tilts = [(task.body_pose[1], task.body_pose[2]) for task in tasks] if tilts is None else tilts
    boundaries = [False] * count if boundaries is None else boundaries
    physics_events = [False] * count if physics_events is None else physics_events
    active = [True] * count if active is None else active
    return fast_stop_v2_step(
        position_xy=_tensor(positions),
        old_distance=_tensor([np.linalg.norm(task.goal - task.robot[:2]) for task in tasks]),
        goal_xy=_tensor([task.goal for task in tasks]),
        root_com_velocity_xy=_tensor(velocities),
        body_angular_velocity=_tensor(angular),
        body_tilt_pitch_roll=_tensor(tilts),
        boundary=torch.tensor(boundaries, dtype=torch.bool),
        physics_failure_event=torch.tensor(physics_events, dtype=torch.bool),
        previous_physics_failure=torch.tensor([task.physics_failure for task in tasks], dtype=torch.bool),
        episode_time=_tensor([task.time for task in tasks]),
        hold_elapsed=_tensor([task._hold for task in tasks]),
        initial_goal_distance=_tensor([task._initial_goal_distance for task in tasks]),
        active_mask=torch.tensor(active, dtype=torch.bool),
        dt=tasks[0].cfg.dt,
        max_time=tasks[0].cfg.max_time,
        goal_tolerance=tasks[0].cfg.goal_tolerance,
        goal_hold=tasks[0].cfg.goal_hold,
        goal_stop_speed=tasks[0].cfg.goal_stop_speed,
        goal_stop_angular_speed=tasks[0].cfg.goal_stop_angular_speed,
        time_penalty=tasks[0].cfg.time_penalty,
        goal_progress_budget=tasks[0].cfg.goal_progress_budget,
    )


def _reference_finish(task, position, velocity, angular, tilt, boundary=False, physics_event=False):
    before = task.robot.copy()
    old_distance = float(np.linalg.norm(task.goal - task.robot[:2]))
    task.robot[:2] = np.asarray(position, dtype=np.float64)
    task.root_com_velocity_xy = np.asarray(velocity, dtype=np.float64)
    task.body_angular_velocity = np.asarray(angular, dtype=np.float64)
    task.body_pose = (0.0, float(tilt[0]), float(tilt[1]))
    return task.finish(
        np.zeros(4, dtype=np.float32), before, old_distance,
        boundary=boundary, physics_failure=physics_event,
    )


def _assert_matches_reference(result, tasks, references, active=None):
    active = [True] * len(tasks) if active is None else active
    reason_code = {"": REASON_ONGOING, "goal_reached": REASON_GOAL_REACHED,
                   "outside_or_overturned": REASON_OUTSIDE_OR_OVERTURNED,
                   "time_limit": REASON_TIME_LIMIT}
    for index, task in enumerate(tasks):
        if not active[index]:
            assert result["rewards"][index].item() == 0.0
            assert not result["done"][index].item()
            assert result["reason_codes"][index].item() == REASON_ONGOING
            assert result["next_time"][index].item() == pytest.approx(task.time)
            assert result["next_hold"][index].item() == pytest.approx(task._hold)
            assert result["next_physics_failure"][index].item() is task.physics_failure
            assert all(component[index].item() == 0.0 for component in result["components"].values())
            continue
        _, reward, terminated, truncated, info = references[index]
        assert result["rewards"][index].item() == pytest.approx(reward, abs=2e-6)
        assert result["done"][index].item() is terminated
        assert result["terminated"][index].item() is terminated
        assert result["truncated"][index].item() is False
        assert truncated is False  # fast_stop_v2 maps its horizon to termination
        assert result["reason_codes"][index].item() == reason_code[info["reason"]]
        assert result["next_time"][index].item() == pytest.approx(task.time, abs=1e-12)
        assert result["next_hold"][index].item() == pytest.approx(task._hold, abs=1e-12)
        assert result["next_physics_failure"][index].item() is task.physics_failure
        for name, component in result["components"].items():
            assert component[index].item() == pytest.approx(info["reward_components"].get(name, 0.0), abs=2e-6)
        assert sum(component[index].item() for component in result["components"].values()) == pytest.approx(reward, abs=2e-6)


def test_batched_fast_stop_matches_real_fieldtask_for_ten_step_hold_and_strict_speed_limits():
    tasks = [_make_task(dt=0.05, max_time=2.0, goal_hold=0.5) for _ in range(3)]
    positions = [[3.6, 2.0], [3.6, 2.0], [3.6, 2.0]]  # exactly within the 0.4 m goal disk
    velocities = [[0.049, 0.0], [0.05, 0.0], [0.049, 0.0]]
    angular = [[0.0, 0.0, 0.099], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]

    for step in range(10):
        result = _batch_call(tasks, positions=positions, velocities=velocities, angular=angular)
        references = [
            _reference_finish(task, positions[index], velocities[index], angular[index], (0.0, 0.0))
            for index, task in enumerate(tasks)
        ]
        _assert_matches_reference(result, tasks, references)
        if step < 9:
            assert result["done"].tolist() == [False, False, False]
        else:
            assert result["reason_codes"].tolist() == [REASON_GOAL_REACHED, REASON_ONGOING, REASON_ONGOING]
            assert result["next_hold"].tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_failure_precedence_tilt_boundary_timeout_and_failure_remaining_time_cost_match_reference():
    tasks = [
        _make_task(dt=0.1, max_time=0.2, goal_hold=0.1, time_penalty=0.1),
        _make_task(dt=0.1, max_time=0.2, goal_hold=0.1, time_penalty=0.1),
        _make_task(dt=0.1, max_time=0.2, goal_hold=0.1, time_penalty=0.1),
        _make_task(dt=0.1, max_time=0.2, goal_hold=0.1, time_penalty=0.1),
    ]
    positions = [[3.6, 2.0], [3.6, 2.0], [2.0, 2.0], [3.6, 2.0]]
    tilts = [(0.0, 0.0), (1.21, 0.0), (0.0, 0.0), (1.2, 0.0)]
    boundaries = [True, False, False, False]
    result = _batch_call(tasks, positions=positions, tilts=tilts, boundaries=boundaries)
    references = [
        _reference_finish(task, positions[index], (0.0, 0.0), (0.0, 0.0, 0.0), tilts[index],
                         boundary=boundaries[index])
        for index, task in enumerate(tasks)
    ]
    _assert_matches_reference(result, tasks, references)

    assert result["reason_codes"].tolist() == [
        REASON_OUTSIDE_OR_OVERTURNED,
        REASON_OUTSIDE_OR_OVERTURNED,
        REASON_ONGOING,
        REASON_GOAL_REACHED,
    ]
    assert result["components"]["failure_remaining_time_cost"].tolist() == pytest.approx([-0.01, -0.01, 0.0, 0.0])
    assert result["next_hold"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.1])

    # The far lane reaches the finite horizon on its second transition; the
    # fast-stop task contract reports timeout as terminated, never truncated.
    # Only the previously ongoing far lane is active in the task reference;
    # terminal lanes are represented as inactive, as in auto_reset=False.
    active = [False, False, True, False]
    # Recompute the batched step with the same active mask for that convention.
    result = _batch_call(tasks, positions=positions, tilts=[(0.0, 0.0)] * 4, active=active)
    references = [None, None, _reference_finish(
        tasks[2], positions[2], (0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0)
    ), None]
    _assert_matches_reference(result, tasks, references, active=active)
    assert result["reason_codes"].tolist() == [REASON_ONGOING, REASON_ONGOING, REASON_TIME_LIMIT, REASON_ONGOING]


def test_active_mask_preserves_inactive_terminal_state_and_partial_reset_only_changes_selected_lane():
    tasks = [_make_task(dt=0.1, max_time=2.0, goal_hold=0.2) for _ in range(2)]
    result = _batch_call(tasks, positions=[[2.0, 2.0], [2.0, 2.0]], boundaries=[True, False])
    refs = [
        _reference_finish(tasks[0], [2.0, 2.0], (0, 0), (0, 0, 0), (0, 0), boundary=True),
        _reference_finish(tasks[1], [2.0, 2.0], (0, 0), (0, 0, 0), (0, 0)),
    ]
    _assert_matches_reference(result, tasks, refs)
    assert tasks[0].done and tasks[0].physics_failure

    frozen = (tasks[0].time, tasks[0]._hold, tasks[0].physics_failure)
    result = _batch_call(tasks, positions=[[2.0, 2.0], [2.0, 2.0]], active=[False, True])
    _assert_matches_reference(result, tasks, [None, _reference_finish(
        tasks[1], [2.0, 2.0], (0, 0), (0, 0, 0), (0, 0)
    )], active=[False, True])
    assert (tasks[0].time, tasks[0]._hold, tasks[0].physics_failure) == frozen
    assert result["done"].tolist() == [False, False]

    # Partial reset: replace lane 0's persistent state with a fresh episode,
    # while lane 1 continues from its own existing time and hold.
    reset_task = _make_task(dt=0.1, max_time=2.0, goal_hold=0.2)
    combined = [reset_task, tasks[1]]
    reset_result = _batch_call(combined, positions=[[2.0, 2.0], [2.0, 2.0]])
    reset_refs = [
        _reference_finish(reset_task, [2.0, 2.0], (0, 0), (0, 0, 0), (0, 0)),
        _reference_finish(tasks[1], [2.0, 2.0], (0, 0), (0, 0, 0), (0, 0)),
    ]
    _assert_matches_reference(reset_result, combined, reset_refs)
    assert reset_result["next_time"].tolist() == pytest.approx([0.1, tasks[1].time])


def test_physics_failure_event_is_sticky_and_failure_keeps_previous_hold():
    tasks = [_make_task(dt=0.1, max_time=1.0, goal_hold=0.5) for _ in range(2)]
    tasks[0]._hold = 0.2
    tasks[1]._hold = 0.3
    tasks[1].physics_failure = True
    result = _batch_call(
        tasks,
        positions=[[3.6, 2.0], [3.6, 2.0]],
        physics_events=[True, False],
    )
    references = [
        _reference_finish(
            tasks[0], [3.6, 2.0], (0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0),
            physics_event=True,
        ),
        _reference_finish(
            tasks[1], [3.6, 2.0], (0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0),
        ),
    ]
    _assert_matches_reference(result, tasks, references)
    assert result["reason_codes"].tolist() == [
        REASON_OUTSIDE_OR_OVERTURNED, REASON_OUTSIDE_OR_OVERTURNED
    ]
    assert result["next_physics_failure"].tolist() == [True, True]
    assert result["next_hold"].tolist() == pytest.approx([0.2, 0.3])


def _make_observation_task(seed: int, start, goal) -> FieldTask:
    cfg = Config(
        size=12.0,
        plants=0,
        rock_density=0.0,
        terrain_enabled=False,
        validate_scene=False,
        lidar_rays=8,
        depth_height=160,
        depth_width=88,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        goto_reward_profile="fast_stop_v2",
        fast_stop_failure_remaining_time_cost=True,
        goal_progress_budget=5.0,
    )
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=seed, options={"start": list(start), "goal": list(goal)})
    heading = 1.1
    task.update_feedback(FeedbackSample(
        sample_time=0.0,
        attitude_quat_xyzw=(0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0)),
        gimbal_q_rad=(0.35, -0.2),
        gimbal_qd_rps=(0.08, -0.03),
        motor_iq_a=(1.5, -1.2),
        motor_rpm=(123.0, -234.0),
        valid={"attitude": True, "gimbal_yaw": True, "gimbal_pitch": True,
               "motor_left": True, "motor_right": True},
    ))

    def capture(kind):
        if kind == "lidar":
            ranges = np.linspace(0.12, cfg.lidar_range, cfg.lidar_rays, dtype=np.float32)
            valid = np.asarray([True, False, True, True, False, True, True, False], dtype=bool)
            return {"ranges": ranges, "valid": valid}
        if kind == "depth":
            values = np.linspace(cfg.depth_min_range, cfg.depth_range,
                                 cfg.depth_height * cfg.depth_width, dtype=np.float32).reshape(
                                     cfg.depth_height, cfg.depth_width)
            valid = np.ones_like(values, dtype=bool)
            valid[::3, ::5] = False
            return {"values": values, "valid": valid}
        return {"detections": []}

    task.attach_sensors(capture)
    task.initialize_sensors()
    task.time = 0.15
    task.robot[:2] += np.asarray([0.25, -0.1])
    task.last_action = np.asarray([0.2, -0.3, 0.1, -0.15], dtype=np.float32)
    task._new["camera"] = False
    return task


def test_batched_observation_matches_real_absolute_goal_fieldtask_frames_and_feedback():
    tasks = [
        _make_observation_task(100, (2.0, 2.0, 0.2), (10.0, 8.0)),
        _make_observation_task(101, (3.0, 2.5, -0.4), (9.0, 7.0)),
    ]
    observations = [task._observation() for task in tasks]
    frame_keys = ("lidar", "depth", "camera")
    inputs = {
        "lidar_ranges": _tensor(np.stack([task.frames["lidar"]["ranges"] for task in tasks]), dtype=torch.float32),
        "lidar_valid": torch.tensor(np.stack([task.frames["lidar"]["valid"] for task in tasks]), dtype=torch.bool),
        "depth_values": _tensor(np.stack([task.frames["depth"]["values"] for task in tasks]), dtype=torch.float32),
        "depth_valid": torch.tensor(np.stack([task.frames["depth"]["valid"] for task in tasks]), dtype=torch.bool),
        "empty_detections": torch.zeros((2, 0, 8), dtype=torch.float32),
        "frame_capture_times": _tensor([[task.frames[key]["capture"] for key in frame_keys] for task in tasks]),
        "frame_estimated_poses": _tensor([[task.frames[key]["estimated_pose"] for key in frame_keys] for task in tasks]),
        "frame_new": torch.tensor([[task._new[key] for key in frame_keys] for task in tasks], dtype=torch.bool),
        "estimated_pose": _tensor([task._estimated_pose() for task in tasks], dtype=torch.float32),
        "goal_xy": _tensor([task.goal for task in tasks]),
        "last_actions": _tensor([task.last_action for task in tasks], dtype=torch.float32),
        "episode_time": _tensor([task.time for task in tasks]),
        "feedback": _tensor([task._feedback_snapshot.feedback for task in tasks], dtype=torch.float32),
        "size": tasks[0].cfg.size,
        "lidar_range": tasks[0].cfg.lidar_range,
        "depth_range": tasks[0].cfg.depth_range,
        "max_time": tasks[0].cfg.max_time,
        "max_detections": tasks[0].cfg.max_detections,
    }
    input_storage_before = {
        name: inputs[name].clone()
        for name in ("lidar_ranges", "lidar_valid", "depth_values", "depth_valid")
    }
    result = pack_emptyfield_goto_observation(**inputs)

    # Build cache entries from FieldTask's real CPU observation. The packer must
    # return those same caller-owned objects and keep all source storage intact.
    normalized_lidar = torch.as_tensor(
        np.stack([observation["lidar"] for observation in observations]), dtype=torch.float32
    )
    normalized_depth = torch.as_tensor(
        np.stack([observation["depth"] for observation in observations]), dtype=torch.float32
    )
    normalized_storage_before = (normalized_lidar.clone(), normalized_depth.clone())
    cached_result = pack_emptyfield_goto_observation(
        **inputs, normalized_lidar=normalized_lidar, normalized_depth=normalized_depth
    )
    lidar_only_result = pack_emptyfield_goto_observation(**inputs, normalized_lidar=normalized_lidar)
    depth_only_result = pack_emptyfield_goto_observation(**inputs, normalized_depth=normalized_depth)

    assert set(result) == {"lidar", "depth", "detections", "state", "feedback"}
    assert result["lidar"].shape == (2, 1, 8)
    assert result["depth"].shape == (2, 1, 160, 88)
    assert result["detections"].shape == (2, 16, 8)
    assert result["state"].shape == (2, 33)
    assert result["feedback"].shape == (2, FEEDBACK_DIM)
    assert all(value.dtype == torch.float32 and value.device.type == "cpu" for value in result.values())
    assert cached_result["lidar"] is normalized_lidar
    assert cached_result["depth"] is normalized_depth
    assert lidar_only_result["lidar"] is normalized_lidar
    assert depth_only_result["depth"] is normalized_depth
    assert cached_result["lidar"].shape == result["lidar"].shape
    assert cached_result["depth"].shape == result["depth"].shape
    torch.testing.assert_close(lidar_only_result["depth"], result["depth"], rtol=0, atol=0)
    torch.testing.assert_close(depth_only_result["lidar"], result["lidar"], rtol=0, atol=0)

    for index, reference in enumerate(observations):
        np.testing.assert_allclose(result["lidar"][index].numpy(), reference["lidar"], rtol=0, atol=0)
        np.testing.assert_allclose(result["depth"][index].numpy(), reference["depth"], rtol=0, atol=0)
        np.testing.assert_allclose(result["detections"][index].numpy(), reference["detections"], rtol=0, atol=0)
        np.testing.assert_allclose(result["state"][index].numpy(), reference["state"], rtol=0, atol=2e-7)
        np.testing.assert_allclose(result["feedback"][index].numpy(), reference["feedback"], rtol=0, atol=0)
        np.testing.assert_allclose(cached_result["lidar"][index].numpy(), reference["lidar"], rtol=0, atol=0)
        np.testing.assert_allclose(cached_result["depth"][index].numpy(), reference["depth"], rtol=0, atol=0)
        for key in ("detections", "state", "feedback"):
            np.testing.assert_allclose(cached_result[key][index].numpy(), reference[key],
                                       rtol=0, atol=2e-7 if key == "state" else 0)
        assert abs(float(tasks[index]._estimated_pose()[2]) - float(tasks[index].robot[2])) > 0.1

    for name, before in input_storage_before.items():
        torch.testing.assert_close(inputs[name], before, rtol=0, atol=0)
    torch.testing.assert_close(normalized_lidar, normalized_storage_before[0], rtol=0, atol=0)
    torch.testing.assert_close(normalized_depth, normalized_storage_before[1], rtol=0, atol=0)


def test_pre_normalized_observation_cache_rejects_bad_shape_dtype_and_device():
    inputs = {
        "lidar_ranges": torch.ones((2, 4), dtype=torch.float32),
        "lidar_valid": torch.ones((2, 4), dtype=torch.bool),
        "depth_values": torch.ones((2, 3, 5), dtype=torch.float32),
        "depth_valid": torch.ones((2, 3, 5), dtype=torch.bool),
        "empty_detections": torch.zeros((2, 0, 8), dtype=torch.float32),
        "frame_capture_times": torch.zeros((2, 3), dtype=torch.float64),
        "frame_estimated_poses": torch.zeros((2, 3, 5), dtype=torch.float64),
        "frame_new": torch.zeros((2, 3), dtype=torch.bool),
        "estimated_pose": torch.zeros((2, 5), dtype=torch.float32),
        "goal_xy": torch.ones((2, 2), dtype=torch.float64),
        "last_actions": torch.zeros((2, 4), dtype=torch.float32),
        "episode_time": torch.zeros(2, dtype=torch.float64),
        "feedback": torch.zeros((2, FEEDBACK_DIM), dtype=torch.float32),
        "size": 12.0,
        "lidar_range": 8.0,
        "depth_range": 5.0,
        "max_time": 1.0,
    }
    bad_caches = (
        ({"normalized_lidar": torch.zeros((2, 4), dtype=torch.float32)}, ValueError, "normalized_lidar must have shape"),
        ({"normalized_lidar": torch.zeros((3, 1, 4), dtype=torch.float32)}, ValueError, "normalized_lidar batch dimension"),
        ({"normalized_lidar": torch.zeros((2, 1, 4), dtype=torch.float64)}, TypeError, "normalized_lidar must use dtype"),
        ({"normalized_lidar": torch.empty((2, 1, 4), dtype=torch.float32, device="meta")}, ValueError, "normalized_lidar must be on device"),
        ({"normalized_depth": torch.zeros((2, 3, 5), dtype=torch.float32)}, ValueError, "normalized_depth must have shape"),
        ({"normalized_depth": torch.zeros((3, 1, 3, 5), dtype=torch.float32)}, ValueError, "normalized_depth batch dimension"),
        ({"normalized_depth": torch.zeros((2, 1, 3, 5), dtype=torch.float64)}, TypeError, "normalized_depth must use dtype"),
        ({"normalized_depth": torch.empty((2, 1, 3, 5), dtype=torch.float32, device="meta")}, ValueError, "normalized_depth must be on device"),
    )
    for kwargs, error, match in bad_caches:
        with pytest.raises(error, match=match):
            pack_emptyfield_goto_observation(**inputs, **kwargs)


def test_tensor_hot_path_has_no_host_array_or_scalar_extraction_calls():
    source_path = Path(__file__).parents[1] / "isaac_field" / "tensor_task.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_methods = {"cpu", "numpy", "item"}
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}

    assert not calls.intersection(forbidden_methods)
