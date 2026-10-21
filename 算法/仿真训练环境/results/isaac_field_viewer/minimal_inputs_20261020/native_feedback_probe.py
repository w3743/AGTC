"""Read-only live Newton probe for the minimal policy observation contract."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.feedback import (  # noqa: E402
    FEEDBACK_AGE_SLICE,
    FEEDBACK_GROUPS,
    FEEDBACK_MAX_AGE_S,
    FEEDBACK_VALID_SLICE,
)
from isaac_field.env import make_env  # noqa: E402
from isaac_field.task import Config  # noqa: E402


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    cfg = Config(
        size=100.0,
        plants=0,
        terrain_enabled=False,
        validate_scene=False,
        max_time=2.0,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    env = make_env(
        cfg,
        mode="search",
        num_envs=1,
        seed=69_901,
        auto_reset=False,
    )
    try:
        observation, _ = env.reset(
            seed=69_901,
            options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
        )
        policy = observation["policy"]
        expected_shapes = {
            "lidar": (1, 1, 400),
            "depth": (1, 1, 160, 88),
            "detections": (1, 16, 8),
            "state": (1, 33),
            "feedback": (1, 37),
        }
        observed_shapes = {key: tuple(policy[key].shape) for key in expected_shapes}
        if observed_shapes != expected_shapes:
            raise AssertionError(f"native minimal observation shape mismatch: {observed_shapes}")
        if any(not np.isfinite(_to_numpy(policy[key])).all() for key in expected_shapes):
            raise AssertionError("native observation contains a non-finite value")

        mag_i = FEEDBACK_GROUPS.index("mag")
        valid_i = FEEDBACK_VALID_SLICE.start + mag_i
        age_i = FEEDBACK_AGE_SLICE.start + mag_i
        task = env.tasks[0]
        initial_feedback = np.asarray(task._observation()["feedback"], dtype=np.float32)
        initial_valid = float(initial_feedback[valid_i])
        initial_age_s = float(initial_feedback[age_i] * FEEDBACK_MAX_AGE_S)
        if initial_valid != 1.0 or not math.isclose(initial_age_s, 0.0, abs_tol=1e-6):
            raise AssertionError(
                f"reset magnetometer sample should be fresh and valid: valid={initial_valid}, age_s={initial_age_s}"
            )

        action = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
        samples = []
        for step_index in range(1, 7):
            observation, _, terminated, truncated, _ = env.step(action)
            if bool(_to_numpy(terminated).any()) or bool(_to_numpy(truncated).any()):
                raise AssertionError(f"unexpected episode end during probe at step {step_index}")
            feedback = np.asarray(task._observation()["feedback"], dtype=np.float32)
            valid = float(feedback[valid_i])
            age_s = float(feedback[age_i] * FEEDBACK_MAX_AGE_S)
            expected_age_s = 0.05 if step_index % 2 else 0.0
            samples.append({
                "step": step_index,
                "sample_time_s": float(task.time),
                "mag_valid": valid,
                "mag_age_s": age_s,
            })
            if valid != 1.0:
                raise AssertionError(f"mag valid dropped at step {step_index}: {samples[-1]}")
            if not math.isclose(age_s, expected_age_s, abs_tol=1e-5):
                raise AssertionError(
                    f"mag age did not alternate 0/.05 s at step {step_index}: "
                    f"expected={expected_age_s}, actual={age_s}"
                )

        record = {
            "probe": "native-minimal-observation-feedback-v1",
            "purpose": "live Newton shape and alternating magnetometer freshness verification; no training/FPS measurement",
            "seed": 69_901,
            "shapes": observed_shapes,
            "feedback_contract": {
                "valid_offset": FEEDBACK_VALID_SLICE.start,
                "age_offset": FEEDBACK_AGE_SLICE.start,
                "groups": list(FEEDBACK_GROUPS),
                "mag_valid_index": valid_i,
                "mag_age_index": age_i,
                "reset_sample": {"valid": initial_valid, "age_s": initial_age_s},
                "steps": samples,
            },
        }
        print("NATIVE MINIMAL INPUT PROBE " + json.dumps(record, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
