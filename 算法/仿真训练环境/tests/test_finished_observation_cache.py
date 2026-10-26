"""CPU checks for one-shot reuse of observations produced by ``finish``."""
from types import SimpleNamespace

import numpy as np
import torch

from isaac_field.body_stability import BodyStabilityAccumulator
from isaac_field.env import IsaacFieldEnv
from test_sampling_copy_cache import _bare_env


class _CountingTask:
    def __init__(self, index, *, done=False):
        self.index = index
        self.robot = np.asarray([2.0, 2.0, 0.0, 0.0, 0.0])
        self.body_pose = (0.0, 0.0, 0.0)
        self.body_angular_velocity = np.zeros(3)
        self.root_com_velocity_xy = np.zeros(2)
        self.frames = [object()]
        self.done = done
        self.mode = "goto"
        self.reason = None
        self.time = 0.0
        self.reward = 0.0
        self.total_reward = 0.0
        self.reward_components = {}
        self.body_stability_sample = None
        self.body_stability_stats = BodyStabilityAccumulator()
        self.scene = SimpleNamespace(terrain=None)
        self.observation_calls = 0
        self.finish_calls = 0
        self.feedback_updates = 0

    def _observation(self):
        self.observation_calls += 1
        return {"state": np.asarray([self.index, self.observation_calls], dtype=np.float32)}

    def initialize_sensors(self):
        self.frames = [object()]
        return self._observation(), None

    def finish(self, *args, **kwargs):
        self.finish_calls += 1
        self.time += 0.05
        obs = self._observation()
        return obs, 0.0, False, False, {"distance": 1.0}

    def _info(self):
        return {
            "distance": 1.0,
            "reason": self.reason,
            "is_success": False,
            "reward_components": self.reward_components.copy(),
            "goal_stop": {},
        }

    def update_feedback(self, sample):
        self.feedback_updates += 1


def _step_env(*, auto_reset=True):
    # Reuse the existing CPU-only articulation/scene fixture; this test replaces
    # the task set so it can exercise the four-environment RSL step path.
    env = _bare_env(num_envs=2)
    env.tasks = [_CountingTask(i) for i in range(4)]
    env.cfg = SimpleNamespace(
        auto_reset=auto_reset,
        distance_cost=0.0,
        sim=SimpleNamespace(dt=0.001),
        decimation=50,
    )
    env.field_cfg = SimpleNamespace(size=10.0, goto_reward_profile="fast_stop_v2")
    env.body_stability_config = None
    env.actions = torch.zeros((4, 4), dtype=torch.float32)
    env._before = [task.robot.copy() for task in env.tasks]
    env._old_distance = [1.0] * 4
    env._last_results = [None] * 4
    env.episode_results = []
    env._refresh_state = lambda *, include_feedback=False: setattr(
        env, "_state_observation_fresh", True
    )
    env._refresh_feedback = lambda *, after_step: None
    return env


def test_normal_four_env_step_reuses_finish_observations_once():
    env = _step_env()

    env._get_dones()
    assert [task.finish_calls for task in env.tasks] == [1] * 4
    assert [task.observation_calls for task in env.tasks] == [1] * 4
    assert env._finished_observation_cache_mask == (True,) * 4

    obs = env._get_observations()["policy"]["state"]
    assert [task.observation_calls for task in env.tasks] == [1] * 4
    assert obs.tolist() == [[i, 1.0] for i in range(4)]
    assert env._finished_observation_cache_mask is None

    # A second explicit observation is outside the one-shot post-finish window.
    env._get_observations()
    assert [task.observation_calls for task in env.tasks] == [2] * 4


def test_partial_reset_invalidates_cache_for_every_environment():
    env = _step_env()
    env._get_dones()
    env.tasks[1].frames = []
    sensor_updates = []

    class _Sensor:
        def update(self, dt, *, force_recompute):
            sensor_updates.append((dt, force_recompute))

    class _Scene(dict):
        def write_data_to_sim(self):
            sensor_updates.append("write")

    env.scene = _Scene({name: _Sensor() for name in ("lidar", "rangefinder", "depth", "detector")})
    env.sim.forward = lambda: sensor_updates.append("forward")
    env.sim.render = lambda: sensor_updates.append("render")
    feedback_refreshes = []
    env._refresh_feedback = lambda *, after_step: feedback_refreshes.append(after_step)

    obs = env._get_observations()["policy"]["state"]

    assert [task.observation_calls for task in env.tasks] == [2] * 4
    assert env.tasks[1].frames
    assert obs.tolist() == [[i, 2.0] for i in range(4)]
    assert sensor_updates.count((0.0, True)) == 4
    assert feedback_refreshes == [False]
    assert env._finished_observation_cache_mask is None


def test_done_non_autoreset_environment_does_not_reuse_its_stale_result():
    env = _step_env(auto_reset=False)
    env.tasks[0].done = True
    stale = {"state": np.asarray([-1.0, -1.0], dtype=np.float32)}
    env._last_results[0] = (stale, 99.0, True, False, {})

    env._get_dones()
    assert [task.finish_calls for task in env.tasks] == [0, 1, 1, 1]
    assert env._finished_observation_cache_mask == (False, True, True, True)

    obs = env._get_observations()["policy"]["state"]
    assert [task.observation_calls for task in env.tasks] == [1, 1, 1, 1]
    assert obs[0].tolist() == [0.0, 1.0]


def test_feedback_refresh_clears_pending_observation_cache():
    env = _step_env()
    env._get_dones()
    env._feedback_host_cache = {
        "acceleration": np.zeros((4, 3)),
        "angular_velocity": np.zeros((4, 3)),
        "gimbal_q": np.zeros((4, 2)),
        "gimbal_qd": np.zeros((4, 2)),
        "wheel_qd": np.zeros((4, 14)),
        "wheel_torque": np.zeros((4, 14)),
        "mount_rotation": np.broadcast_to(np.eye(3), (4, 3, 3)).copy(),
    }

    IsaacFieldEnv._refresh_feedback(env, after_step=False)

    assert [task.feedback_updates for task in env.tasks] == [1] * 4
    assert env._finished_observation_cache_mask is None
    env._get_observations()
    assert [task.observation_calls for task in env.tasks] == [2] * 4
