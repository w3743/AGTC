"""M1 follow-up: VQF bypass verification + CPU cost (no GPU, small memory).

Verifies the prototype's `feedback_backend="vqf"` (python vqf 2.1.2 via
isaac_field.feedback.SimFeedbackEstimator) against an independent reference
estimator fed with identical engine-truth signals, with the documented sim
noise disabled so the comparison isolates the attitude-fusion semantics.
Also measures the serial CPU cost per env update (noise off and on) and the
projected 64-env impact.
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from observation_provider import ObservationProvider  # noqa: E402


def peak_rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def sensor_rotation(quat_wxyz, gimbal_q):
    w, x, y, z = quat_wxyz
    r_base = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    cy, sy = math.cos(gimbal_q[0]), math.sin(gimbal_q[0])
    cp, sp = math.cos(gimbal_q[1]), math.sin(gimbal_q[1])
    r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    r_pitch = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    return r_base @ r_yaw @ r_pitch


def main(n: int = 8, seed: int = 21):
    from isaac_field.feedback import SimFeedbackEstimator

    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.uniform(5, 95, (n, 2)), np.full((n, 1), 0.15)], axis=1)
    yaw = rng.uniform(-math.pi, math.pi, n)
    quat = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)
    ang_vel = rng.normal(0.0, 0.05, (n, 3))
    gimbal_q = np.stack([rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.4, 1.0, n)], axis=1)
    read = {"pos": pos, "quat_wxyz": quat, "lin_vel": rng.normal(0, 0.1, (n, 3)),
            "ang_vel": ang_vel, "wheel_qd": rng.uniform(-5, 5, (n, 14)),
            "gimbal_q": gimbal_q, "gimbal_qd": rng.normal(0, 0.1, (n, 2)),
            "effort": rng.uniform(-1, 1, (n, 14)), "track_command": np.zeros((n, 2))}

    # ---------- equivalence: two fresh estimators at t=0, noise disabled ----------
    provider_eq = ObservationProvider(n, seed=seed, feedback_backend="vqf", feedback_noise=False)
    got = provider_eq._feedback_vqf(quat, gimbal_q, read)
    ref = np.zeros((n, 37))
    for e in range(n):
        est = SimFeedbackEstimator(seed=900 + e, noise_enabled=False)
        r_sensor = sensor_rotation(quat[e], gimbal_q[e])
        acc = r_sensor.T @ np.array([0.0, 0.0, 9.80665])
        gyro = r_sensor.T @ ang_vel[e]
        mag = r_sensor.T @ np.array([0.0, 3000.0, -6000.0])
        effort = read["effort"][e]
        iq = np.array([effort[0:7].mean() / 0.035, effort[7:14].mean() / 0.035])
        rpm = np.array([read["wheel_qd"][e, 0:7].mean() * 4 * 60 / (2 * math.pi),
                        read["wheel_qd"][e, 7:14].mean() * 4 * 60 / (2 * math.pi)])
        snap = est.update(0.0, acc, gyro, mag, gimbal_q[e], read["gimbal_qd"][e], iq, rpm)
        ref[e] = np.asarray(snap.feedback, np.float64)
    diff = np.abs(ref - got.astype(np.float64))
    att_diff = float(diff[:, 9:13].max())
    full_diff = float(diff.max())

    # ---------- cost: advancing episode time (a repeated `now` is a cached no-op) ----------
    reps = 40
    provider = ObservationProvider(n, seed=seed, feedback_backend="vqf", feedback_noise=False)
    provider._feedback_vqf(quat, gimbal_q, read)
    t0 = time.perf_counter()
    for _ in range(reps):
        provider.episode_time += 0.05
        provider._feedback_vqf(quat, gimbal_q, read)
    per_env_us = (time.perf_counter() - t0) / reps / n * 1e6

    provider_noise = ObservationProvider(n, seed=seed, feedback_backend="vqf", feedback_noise=True)
    provider_noise._feedback_vqf(quat, gimbal_q, read)
    t1 = time.perf_counter()
    for _ in range(reps):
        provider_noise.episode_time += 0.05
        provider_noise._feedback_vqf(quat, gimbal_q, read)
    per_env_noise_us = (time.perf_counter() - t1) / reps / n * 1e6

    per_step_64 = per_env_noise_us * 64 / 1e6  # seconds added per policy step, all envs
    result = {
        "phase": "M1_feedback_vqf_bypass",
        "samples": n,
        "reference": "isaac_field/feedback.py SimFeedbackEstimator(noise_enabled=False), independent instance per env",
        "candidate": 'proto observation_provider feedback_backend="vqf" (SimFeedbackEstimator + vqf 2.1.2)',
        "attitude_quaternion_max_abs_error": round(att_diff, 9),
        "full_37_max_abs_error": round(full_diff, 9),
        "target_met_1e_6": bool(att_diff <= 1e-6),
        "per_env_update_ms_no_noise": round(per_env_us / 1e3, 4),
        "per_env_update_ms_with_noise": round(per_env_noise_us / 1e3, 4),
        "serial_cpu_ms_per_policy_step_64env": round(per_step_64 * 1e3, 2),
        "serial_cpu_s_per_16step_iteration_64env": round(per_step_64 * 16, 3),
        "note_on_cost": "python VQF is a serial loop; it adds wall time directly, it is not spread over cores",
    }
    with open(f"{HERE}/m1_vqf_verification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("VQF_RESULT " + json.dumps(result))
    print("peak_rss_mb", peak_rss())
    print("VQF_DONE")


if __name__ == "__main__":
    main()
