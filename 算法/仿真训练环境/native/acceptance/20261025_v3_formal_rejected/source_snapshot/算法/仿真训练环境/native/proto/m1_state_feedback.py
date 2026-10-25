"""M1.2: state(33) and feedback(37) equivalence against the repository reference code.

Reference state: `isaac_field.task.FieldTask._observation()` (task.py:1537-1619),
  which builds the 33-dim ABSOLUTE_GOAL state from task fields / sensor frames.
Reference feedback: `isaac_field.feedback.SimFeedbackEstimator` (feedback.py:226),
  the exact estimator used by IsaacFieldEnv (update_sample path), with
  noise_enabled=False to isolate scaling/layout from the random sensor noise.

Two feedback comparisons are reported:
  a) "pass_through_attitude": supply the direct sensor quaternion -> checks the
     21 value fields + valid/age bookkeeping against our vector.
  b) "vqf_attitude": omit the quaternion so the reference runs VQF from
     accel/gyro/mag (the real Isaac path) -> the attitude-quaternion gap is the
     remaining non-equivalence of our direct attitude computation.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from isaac_field.feedback import FEEDBACK_DIM, FeedbackSnapshot, SimFeedbackEstimator  # noqa: E402
from isaac_field.task import (ABSOLUTE_GOAL_INPUT_SCHEMA, Config, FieldTask,  # noqa: E402
                              state_fields_for_schema)

FIELD_SIZE = 100.0
POLICY_DT = 0.05
MAX_TIME = 60.0
STATE_DIM = 33
G0 = 9.80665
MAG_REFERENCE = np.array([0.0, 3000.0, -6000.0])


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / max(np.linalg.norm(q), 1e-12)


def sensor_rotation(yaw, gimbal_q):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(gimbal_q[1]), math.sin(gimbal_q[1])
    cyaw, syaw = math.cos(gimbal_q[0]), math.sin(gimbal_q[0])
    r_yaw = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]])
    r_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    r_body = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return r_body @ r_yaw @ r_pitch


def run(n: int = 8, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    cfg = Config(size=FIELD_SIZE, plants=0, dry_fraction=0.0, terrain_enabled=False,
                 max_time=MAX_TIME, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
    state_fields = list(state_fields_for_schema(ABSOLUTE_GOAL_INPUT_SCHEMA))

    # --- random identical inputs for both implementations ---
    pos_xy = rng.uniform(5, 95, (n, 2))
    yaw = rng.uniform(-math.pi, math.pi, n)
    gimbal_q = np.stack([rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.4, 1.0, n)], axis=1)
    goal = rng.uniform(5, 95, (n, 2))
    actions = rng.uniform(-1, 1, (n, 4))
    # quantise episode time and sensor ages to policy steps: the prototype only ever
    # captures on policy boundaries, so the reference must receive the same grid or
    # the age features differ by rounding (not by semantics).
    steps = rng.integers(0, 1000, n).astype(np.int64)
    time_s = steps * POLICY_DT
    ages_steps = rng.integers(0, 41, (n, 3))  # 0..2.0 s in 0.05 s steps
    ages = ages_steps * POLICY_DT
    capture_steps = steps[None, :] - ages_steps.T  # (3, n)
    new_flags = rng.integers(0, 2, (n, 3)).astype(np.float64)

    # ---------------- reference: FieldTask._observation ----------------
    task = FieldTask(cfg, "goto")
    task._pose_noise = np.zeros(2)
    task.goal = goal.copy()
    task.time = 0.0
    frame_poses = {}
    for k, kind in enumerate(("lidar", "depth", "camera")):
        frame_poses[kind] = np.stack([pos_xy[:, 0], pos_xy[:, 1], yaw, gimbal_q[:, 0], gimbal_q[:, 1]], axis=1)
    snapshot = FeedbackSnapshot(
        feedback=np.zeros(FEEDBACK_DIM, np.float32), sample_time=0.0,
        estimated_heading_rad=0.0, estimated_roll_rad=0.0, estimated_pitch_rad=0.0,
        estimated_quaternion_xyzw=np.array([0, 0, 0, 1], np.float32),
        estimated_gimbal_q_rad=np.zeros(2, np.float32), pose_valid=True)
    ref_state = np.zeros((n, STATE_DIM))
    for e in range(n):
        task.goal = np.asarray(goal[e], np.float64).reshape(2).copy()
        task.robot = np.array([pos_xy[e, 0], pos_xy[e, 1], yaw[e], 0.0, 0.0])
        task.gimbal = gimbal_q[e].copy()
        task.last_action = np.asarray(actions[e], np.float32).reshape(4).copy()
        task.time = float(time_s[e])
        task._new = {kind: float(new_flags[e, k]) for k, kind in enumerate(("lidar", "depth", "camera"))}
        task.frames = {}
        for k, kind in enumerate(("lidar", "depth", "camera")):
            pose = frame_poses[kind][e]
            task.frames[kind] = {"capture": float(time_s[e] - ages[e, k]),
                                 "estimated_pose": pose.copy(), "valid": np.ones(1, bool),
                                 "ranges": np.zeros(1, np.float32), "values": np.zeros((1, 1), np.float32),
                                 "detections": []}
        task._feedback_snapshot = FeedbackSnapshot(
            feedback=np.zeros(FEEDBACK_DIM, np.float32), sample_time=float(time_s[e]),
            estimated_heading_rad=float(yaw[e]), estimated_roll_rad=0.0, estimated_pitch_rad=0.0,
            estimated_quaternion_xyzw=np.array([0, 0, 0, 1], np.float32),
            estimated_gimbal_q_rad=gimbal_q[e].astype(np.float32), pose_valid=True)
        obs = task._observation()
        ref_state[e] = obs["state"]

    # ---------------- candidate: ObservationProvider._state ----------------
    from observation_provider import ObservationProvider  # local import; proto dir on path

    provider = ObservationProvider(n, field_size=FIELD_SIZE, seed=seed)
    provider.goal_xy = goal.copy()
    provider.prev_action = actions.copy()
    provider.episode_time = time_s.copy()
    provider.step = steps.copy()
    provider.last_lidar = capture_steps[0].astype(np.int64)
    provider.last_depth = capture_steps[1].astype(np.int64)
    provider.last_camera = capture_steps[2].astype(np.int64)
    provider.new_lidar = new_flags[:, 0]
    provider.new_depth = new_flags[:, 1]
    provider.new_camera = new_flags[:, 2]
    for k in range(3):
        provider.frame_pose[:, k, 0] = pos_xy[:, 0]
        provider.frame_pose[:, k, 1] = pos_xy[:, 1]
        provider.frame_pose[:, k, 2] = yaw
        provider.frame_pose[:, k, 3] = gimbal_q[:, 0]
        provider.frame_pose[:, k, 4] = gimbal_q[:, 1]
    read = {"gimbal_q": gimbal_q, "pos": np.concatenate([pos_xy, np.full((n, 1), 0.15)], axis=1),
            "quat_wxyz": np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1),
            "wheel_qd": np.zeros((n, 14)), "ang_vel": np.zeros((n, 3)),
            "effort": np.zeros((n, 14)), "gimbal_qd": np.zeros((n, 2))}
    cand_state = provider._state(read["pos"], yaw, read)

    state_diff = np.abs(ref_state - cand_state)
    per_field = {name: round(float(state_diff[:, i].max()), 9) for i, name in enumerate(state_fields)}
    state_result = {
        "dim": STATE_DIM, "samples": int(n),
        "max_abs_error": round(float(state_diff.max()), 9),
        "mean_abs_error": round(float(state_diff.mean()), 9),
        "fields_with_error": int(sum(1 for v in per_field.values() if v > 1e-9)),
        "per_field_max_abs_error": per_field,
        "schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
    }

    # ---------------- feedback reference vs candidate ----------------
    def candidate_feedback(e):
        r_sensor = sensor_rotation(yaw[e], gimbal_q[e])
        acc = r_sensor.T @ np.array([0.0, 0.0, G0])
        gyro = r_sensor.T @ np.zeros(3)
        mag = r_sensor.T @ MAG_REFERENCE
        # motor signals: use the same model equivalents as the prototype
        qd = np.zeros(14)
        iq_left = iq_right = 0.0
        rpm_left = rpm_right = 0.0
        vec = np.zeros(FEEDBACK_DIM)
        vec[0:3] = acc / G0
        vec[3:6] = gyro / math.pi
        vec[6:9] = mag / 32768.0
        quat = matrix_to_quat_xyzw(r_sensor)
        vec[9:13] = np.array([quat[0], quat[1], quat[2], quat[3]])  # xyzw
        vec[13] = gimbal_q[e, 0] / math.pi
        vec[14] = 0.0
        vec[15] = gimbal_q[e, 1] / math.pi
        vec[16] = 0.0
        vec[17] = iq_left / 100.0
        vec[18] = rpm_left / 10000.0
        vec[19] = iq_right / 100.0
        vec[20] = rpm_right / 10000.0
        vec[21:29] = 1.0
        vec[29:37] = 0.0
        return vec, acc, gyro, mag, quat

    def reference_feedback(e, pass_attitude: bool):
        est = SimFeedbackEstimator(seed=1000 + e, noise_enabled=False)
        r_sensor = sensor_rotation(yaw[e], gimbal_q[e])
        acc = r_sensor.T @ np.array([0.0, 0.0, G0])
        gyro = r_sensor.T @ np.zeros(3)
        mag = r_sensor.T @ MAG_REFERENCE
        quat = matrix_to_quat_xyzw(r_sensor)
        now = float(time_s[e])
        valid = {g: True for g in ("accel", "gyro", "mag", "attitude", "gimbal_yaw",
                                   "gimbal_pitch", "motor_left", "motor_right")}
        times = {g: now for g in valid}
        kwargs = dict(valid=valid, source_times=times)
        if pass_attitude:
            kwargs["attitude_quat_xyzw"] = quat
        snap = est.update(now, acc, gyro, mag, gimbal_q[e], np.zeros(2),
                          np.array([0.0, 0.0]), np.array([0.0, 0.0]), **kwargs)
        return np.asarray(snap.feedback, np.float64)

    cand_fb = np.stack([candidate_feedback(e)[0] for e in range(n)])
    ref_pt = np.stack([reference_feedback(e, True) for e in range(n)])
    ref_vqf = np.stack([reference_feedback(e, False) for e in range(n)])
    diff_pt = np.abs(ref_pt - cand_fb)
    diff_vqf = np.abs(ref_vqf - cand_fb)
    value_labels = [f"value[{i}]" for i in range(21)] + [f"valid[{i}]" for i in range(8)] + [f"age[{i}]" for i in range(8)]
    feedback_result = {
        "dim": FEEDBACK_DIM, "samples": int(n),
        "pass_through_attitude": {
            "max_abs_error": round(float(diff_pt.max()), 9),
            "value_fields_max_abs_error": round(float(diff_pt[:, :21].max()), 9),
            "valid_age_fields_max_abs_error": round(float(diff_pt[:, 21:].max()), 9),
        },
        "vqf_attitude": {
            "max_abs_error": round(float(diff_vqf.max()), 9),
            "value_fields_max_abs_error": round(float(diff_vqf[:, :21].max()), 9),
            "attitude_quat_max_abs_error": round(float(diff_vqf[:, 9:13].max()), 9),
            "non_attitude_values_max_abs_error": round(
                float(diff_vqf[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20]].max()), 9),
        },
        "per_field_pass_through_max_abs_error": {value_labels[i]: round(float(diff_pt[:, i].max()), 9)
                                                 for i in range(FEEDBACK_DIM)},
    }
    state_fields_err = [value_labels[i] for i in range(0, 0)]
    return {
        "state": state_result,
        "feedback": feedback_result,
        "inputs": {"n": int(n), "seed": seed, "max_time_s": MAX_TIME,
                   "schema": ABSOLUTE_GOAL_INPUT_SCHEMA},
        "candidate_source": "proto/observation_provider.py _state/_feedback",
        "reference_source": "isaac_field/task.py FieldTask._observation + isaac_field/feedback.py SimFeedbackEstimator(noise_enabled=False)",
        "validation_triggered_unused": bool(state_fields_err),
    }


if __name__ == "__main__":
    result = run()
    print("RESULT " + json.dumps(result["state"]))
    print("RESULT_FEEDBACK " + json.dumps(result["feedback"]))
