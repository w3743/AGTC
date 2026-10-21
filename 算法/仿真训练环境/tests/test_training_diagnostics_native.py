"""Actual Newton/SB3 terminal reward accounting, opt-in with native tests."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_ISAAC_PHYSICS") != "1",
                                reason="requires Isaac Lab/Newton native physics")


def test_fast_sb3_wrapper_keeps_terminal_diagnostic_snapshot_through_autoreset():
    from isaac_field.env import make_env
    from isaac_field.task import Config
    from isaac_field.training import _sb3_wrapper

    engine = make_env(
        cfg=Config(plants=0, terrain_enabled=False, max_time=0.1),
        mode="goto", num_envs=1, seed=98_001, auto_reset=True,
        goal_distance=(1.0, 1.2), goal_angle=0.2,
    )
    wrapper = _sb3_wrapper(engine)
    try:
        wrapper.reset()
        for step in range(2):
            _, rewards, dones, infos = wrapper.step(np.zeros((1, 4), dtype=np.float32))
            snapshot = engine.diagnostic_step_infos[0]
            assert sum(snapshot["reward_components"].values()) == pytest.approx(float(rewards[0]), abs=1e-5)
        assert bool(dones[0])
        assert infos[0]["TimeLimit.truncated"]
        assert snapshot["reason"] == "time_limit"
        assert snapshot["time"] == pytest.approx(0.1)
        assert engine.tasks[0].time == 0.0  # current task has already reset
        assert "reward_components" not in infos[0]  # upstream fast path
    finally:
        wrapper.close()
