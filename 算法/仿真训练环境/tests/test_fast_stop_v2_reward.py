"""Outcome/ranking regression tests for the explicit goal-set reward profile."""
import numpy as np
import pytest

from isaac_field.task import Config
from test_task import _attach_empty_sensors, _make_fast_stop_task, _step_task_to


def make_task(**updates):
    settings = dict(goto_reward_profile="fast_stop_v2",
                    fast_stop_failure_remaining_time_cost=True, max_time=60.)
    settings.update(updates)
    task = _make_fast_stop_task(**settings)
    task.prepare(seed=426, options={"start": [2., 2., 0.], "goal": [4., 2.]})
    _attach_empty_sensors(task, task.cfg)
    return task


def success_return(wait_steps, terminal_distance):
    task = make_task()
    for _ in range(wait_steps):
        _step_task_to(task, [2., 2.])
    for _ in range(5):
        _step_task_to(task, [4. - terminal_distance, 2.])
    assert task.done and task.reason == "goal_reached"
    return task.total_reward, task.time


def test_success_anywhere_in_goal_set_has_same_progress_and_prefers_less_time():
    edge, edge_time = success_return(0, .4)
    center, center_time = success_return(0, 0.)
    assert edge_time == center_time
    assert edge == pytest.approx(center)
    assert edge == pytest.approx(15. - .1 * edge_time)
    later_center, _ = success_return(3, 0.)
    assert edge > later_center  # Never encourage driving deeper into the success disk.


@pytest.mark.parametrize("terminal_distance", [0., .1, .4])
def test_successful_trajectories_rank_by_time_for_each_fixed_initial_state(terminal_distance):
    fast, fast_time = success_return(1, terminal_distance)
    slow, slow_time = success_return(6, terminal_distance)
    assert fast - slow == pytest.approx(.1 * (slow_time - fast_time))


def failed_return(final_distance, fail_step):
    task = make_task(dt=1., goal_hold=1.)
    # A goal-set arrival with motion must still fail the strict stop contract.
    task.robot[3] = .5
    for step in range(1, fail_step + 1):
        before = task.robot.copy()
        previous_distance = float(np.linalg.norm(task.goal - task.robot[:2]))
        task.robot[:2] = [4. - final_distance, 2.]
        task.finish(np.zeros(4, dtype=np.float32), before, previous_distance,
                    boundary=step == fail_step and fail_step < 60)
        assert sum(task.reward_components.values()) == pytest.approx(task.reward)
    assert task.done and task.reason != "goal_reached"
    return task.total_reward


def test_failed_trajectories_reward_approach_and_do_not_reward_earlier_death():
    close = failed_return(.5, 5)
    far = failed_return(2., 5)
    assert close > far
    assert close == pytest.approx(failed_return(.5, 60))
    assert far == pytest.approx(failed_return(2., 60))


def test_failure_upper_bound_is_below_even_slowest_success():
    failures = [failed_return(distance, when)
                for distance in (0., .2, .5, 2.) for when in (1, 5, 60)]
    assert max(failures) == pytest.approx(-11.)
    assert min(failures) >= -21. - 1e-9
    # An initially-in-range success has zero progress bonus; its 60s return
    # is still at least 4, safely above every failed trajectory's upper bound.
    assert max(failures) < 10. - .1 * 60.


def test_round_trips_cannot_farm_progress_reward():
    task = make_task()
    progress = []
    for position in ([3., 2.], [2., 2.], [3., 2.], [2., 2.]):
        _step_task_to(task, position)
        progress.append(task.reward_components["goal_progress"])
    assert sum(progress) == pytest.approx(0.)
    assert not task.done


@pytest.mark.parametrize("linear,angular", [([0., .05], [0., 0., 0.]),
                                            ([0., 0.], [.1, 0., 0.])])
def test_v2_keeps_strict_full_velocity_and_half_second_hold(linear, angular):
    task = make_task()
    task.root_com_velocity_xy = np.asarray(linear)
    task.body_angular_velocity = np.asarray(angular)
    for _ in range(6):
        _step_task_to(task, [3.8, 2.])
    assert not task.done and task._hold == 0.
    task.root_com_velocity_xy = np.zeros(2)
    task.body_angular_velocity = np.zeros(3)
    for _ in range(5):
        _step_task_to(task, [3.8, 2.])
    assert task.done and task.reason == "goal_reached"
    assert task._info()["goal_stop"]["profile"] == "fast_stop_v2"


def test_v2_requires_remaining_horizon_failure_cost():
    with pytest.raises(ValueError, match="fast_stop_v2.*remaining"):
        Config(goto_reward_profile="fast_stop_v2", goal_progress_budget=5.)
