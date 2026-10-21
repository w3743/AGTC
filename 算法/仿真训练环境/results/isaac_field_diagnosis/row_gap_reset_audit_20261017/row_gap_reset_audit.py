"""Small deterministic CPU reproduction of the row-gap reset-anchor rejection."""

from __future__ import annotations

import numpy as np

from isaac_field.navigation import has_grid_anchor
from isaac_field.task import Config, FieldTask, InvalidEpisodePoseError, Scene


def main() -> None:
    cfg = Config(
        size=10.0,
        cell=0.1,
        plants=6,
        terrain_enabled=False,
        validate_scene=True,
        row_spacing=0.8,
        plant_spacing=0.3,
        robot_length=0.8,
        robot_width=0.47,
        lidar_rays=8,
        depth_width=4,
        depth_height=3,
    )
    plant_xy = np.array(
        [[x, y] for y in (4.0, 4.8) for x in (4.7, 5.0, 5.3)], dtype=np.float32
    )
    scene = Scene(cfg, plant_xy, np.zeros(len(plant_xy), dtype=bool), np.random.default_rng(1))
    scene.radius[:] = 0.12
    scene.stem_radius[:] = 0.012
    start = np.array([5.0, 4.4, 0.0])
    goal = np.array([5.5, 4.4])
    adjacent_grid_centers_y = (4.35, 4.45)
    connectivity = scene.validate_connectivity(start[:2], goal)
    legacy_circle_radius = np.hypot(cfg.robot_length, cfg.robot_width) / 2 + 0.05
    side_clearance = (
        cfg.row_spacing / 2
        - cfg.robot_width / 2
        - (float(scene.radius[0]) + 0.025)
    )

    assert scene.pose_clear(start[:2], start[2])
    assert scene.pose_clear(goal, 0.0)
    assert not has_grid_anchor(scene, start[:2])
    assert not has_grid_anchor(scene, goal)
    assert not any(scene.pose_clear([5.0, y], 0.0) for y in adjacent_grid_centers_y)
    assert connectivity["all_free_connected"] is True
    assert connectivity["start_goal_connected"] is False

    try:
        FieldTask(cfg, mode="goto").prepare(
            seed=1,
            options={"scene": scene, "start": start, "goal": goal, "random_start": False},
        )
    except InvalidEpisodePoseError as exc:
        reset_result = f"{type(exc).__name__}: {exc}"
    else:
        raise AssertionError("expected the configured reset validator to reject the start anchor")

    print("cpu_only=True")
    print("row_pitch_m=0.8 vehicle_outer_m=0.8x0.47")
    print("Scene.radius_m=0.12 stem_radius_m=0.012 clearance_margin_m=0.025")
    print(f"centerline_side_clearance_m={side_clearance:.3f}")
    print(f"legacy_Scene.validate_disk_radius_m={legacy_circle_radius:.6f}")
    print(f"start_pose_clear_H={scene.pose_clear(start[:2], start[2])}")
    print(f"goal_pose_clear_H={scene.pose_clear(goal, 0.0)}")
    print(f"start_grid_anchor={has_grid_anchor(scene, start[:2])}")
    print(f"goal_grid_anchor={has_grid_anchor(scene, goal)}")
    print(f"adjacent_grid_centers_y_m={adjacent_grid_centers_y}")
    print(
        "adjacent_grid_center_pose_clear_H="
        f"{tuple(scene.pose_clear([5.0, y], 0.0) for y in adjacent_grid_centers_y)}"
    )
    print(f"all_free_connected={connectivity['all_free_connected']}")
    print(f"start_goal_connected={connectivity['start_goal_connected']}")
    print(f"free_pose_components={connectivity['free_pose_components']}")
    print(f"FieldTask.prepare={reset_result}")


if __name__ == "__main__":
    main()
