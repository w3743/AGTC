"""Constraint and reward-wiring tests; the scorer is deliberately synthetic.

These tests do not run a student policy and are not evidence of real
adversarial training. They only exercise the teacher environment contract.
"""

import numpy as np

from isaac_field.adversary import MapDesignerEnv
from isaac_field.task import Config


def _cfg() -> Config:
    return Config(
        size=12.0,
        cell=0.5,
        plants=4,
        dry_fraction=0.5,
        leaves_per_plant=2,
        terrain_enabled=False,
        validate_scene=False,
    )


def _action_for(env: MapDesignerEnv, plant_index: int, xy: np.ndarray) -> np.ndarray:
    count = env._plant_count
    plant_action = -1.0 if count == 1 else 2.0 * plant_index / (count - 1) - 1.0
    slots = env._allowed_grid_indices
    grid_indices = [int(np.argmin(np.abs(env._grid_axis[slots] - value))) for value in xy]
    coordinate_actions = [
        -1.0 if len(slots) == 1 else 2.0 * index / (len(slots) - 1) - 1.0
        for index in grid_indices
    ]
    return np.asarray([plant_action, *coordinate_actions], dtype=np.float32)


def test_teacher_rolls_back_invalid_layout_and_scores_final_map_once():
    calls = []

    def fake_scorer(scene):
        calls.append(scene)
        return 2.5

    env = MapDesignerEnv(_cfg(), fake_scorer, seed=11, edit_budget=8)
    assert env._plant_spacing == 0.28
    np.testing.assert_allclose(np.diff(env._grid_axis), 0.1, atol=1e-12)
    observation, initial_info = env.reset()
    assert observation.shape == env.observation_space.shape
    assert initial_info["fully_connected"] is True
    assert initial_info["target_count"] == 2
    original_positions = np.asarray(env.scene.xy).copy()
    assert env._layout_valid(original_positions, scene=env.scene)[0]

    # Plant 0 is moved onto plant 1, which must be rejected and rolled back.
    _, reward, terminated, truncated, invalid_info = env.step(
        _action_for(env, 0, original_positions[1])
    )
    assert reward == -env.invalid_penalty
    assert not terminated and not truncated
    assert invalid_info["rolled_back"] is True
    assert invalid_info["invalid_reason"] == "plant_spacing"
    np.testing.assert_array_equal(env.scene.xy, original_positions)

    # Keep the remaining edits valid by selecting each plant's current cell.
    final_info = None
    for _ in range(env.edit_budget - 1):
        _, _, terminated, truncated, final_info = env.step(
            _action_for(env, 0, env.scene.xy[0])
        )
        assert not truncated

    assert terminated
    assert final_info is not None
    assert len(calls) == 1
    assert final_info["student_score"] == 2.5
    assert final_info["teacher_objective_reward"] == -2.5
    assert final_info["teacher_objective"] == "naive_minimax"
    assert final_info["paired"] is False
    assert final_info["teacher_reward"] == -2.5
    assert final_info["episode_teacher_return"] == -3.5
    assert final_info["spawn_pose"] is not None
    assert len(final_info["positions"]) == _cfg().plants
    assert len(final_info["checkpoints"]) == env._valid_edits + 1
    assert env.summary()["episodes_recorded"] == 1


def test_layout_edits_keep_scene_geometry_shape_and_targets_fixed():
    env = MapDesignerEnv(_cfg(), lambda scene: 0.0, seed=23, edit_budget=8)
    _, _ = env.reset()
    scene = env.scene
    assert scene is not None
    shape_before = {
        name: np.asarray(getattr(scene, name)).copy()
        for name in ("targets", "radius", "stem_radius", "height", "leaf_plant", "leaf_dry", "normals", "u", "v", "axes")
    }
    old_xy = scene.xy.copy()

    # Find a different location that passes the inexpensive layout checks.
    selected = None
    for x in env._grid_axis[env._allowed_grid_indices]:
        for y in env._grid_axis[env._allowed_grid_indices]:
            candidate = old_xy.copy()
            candidate[0] = (x, y)
            if not np.array_equal(candidate, old_xy) and env._layout_valid(candidate)[0]:
                selected = candidate[0].copy()
                break
        if selected is not None:
            break
    assert selected is not None

    _, reward, terminated, truncated, info = env.step(_action_for(env, 0, selected))
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["rolled_back"] is False
    assert not np.array_equal(env.scene.xy[0], old_xy[0])
    for name, before in shape_before.items():
        np.testing.assert_array_equal(getattr(env.scene, name), before)

