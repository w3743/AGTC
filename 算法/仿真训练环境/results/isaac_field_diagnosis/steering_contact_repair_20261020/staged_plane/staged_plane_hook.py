"""Result-only hook for the isolated flat mesh/native plane geometry diagnostic.

Import this helper only from the contact_ground_plane_v6 probe. It deliberately
changes the ground collision representation; it does not implement terrain or a
second simulation path.
"""
from __future__ import annotations

import numpy as np
import newton
import warp as wp


_GROUND_SUFFIX = "/Field/Ground"
_PLANE_SUFFIX = "/Field/GroundPlaneDiagnostic"
_MATERIAL_FIELDS = (
    "shape_material_mu",
    "shape_material_kf",
    "shape_material_ke",
    "shape_material_kd",
    "shape_margin",
    "shape_gap",
)
_V4_GROUND_SOLREF_MODE = 2  # Runtime-dumped v4 ground and all 14 wheel shapes: MJCF_DEFAULT.


def assert_flat_single_env_cfg(cfg) -> None:
    """Keep this diagnostic limited to one plant-free, non-rough environment."""
    if int(cfg.scene.num_envs) != 1:
        raise RuntimeError("the staged plane A/B requires exactly one environment")
    if int(cfg.task.plants) != 0 or bool(cfg.task.terrain_enabled):
        raise RuntimeError("the staged plane A/B requires plants=0 and terrain_enabled=False")


def _current_world_shapes(builder, world_index: int) -> range:
    worlds = getattr(builder, "shape_world", None)
    if worlds is None:
        raise RuntimeError("Newton builder has no shape_world ownership array")
    stop = len(worlds)
    start = stop
    while start > 0 and int(worlds[start - 1]) == world_index:
        start -= 1
    if start == stop:
        raise RuntimeError(f"Newton world {world_index} has no shapes before the hook")
    return range(start, stop)


class FlatGroundPlaneHook:
    """Callable for isaaclab_newton.cloner.newton_builder_world_hook."""

    def __init__(self) -> None:
        self.worlds: list[dict[str, object]] = []

    def __call__(self, builder, world_index: int, position, quaternion) -> None:
        # This probe is intentionally single-world; the source diagnostics use env_0.
        if world_index != 0 or self.worlds:
            raise RuntimeError("the staged plane hook accepts exactly Newton world 0")

        shape_ids = _current_world_shapes(builder, world_index)
        labels = [str(builder.shape_label[index]) for index in shape_ids]
        ground_ids = [
            index for index in shape_ids if str(builder.shape_label[index]).endswith(_GROUND_SUFFIX)
        ]
        if len(ground_ids) != 1:
            raise RuntimeError(f"expected one imported flat ground mesh; found {ground_ids}: {labels}")
        ground_id = ground_ids[0]
        flags = int(builder.shape_flags[ground_id])
        required = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.VISIBLE)
        if flags & required != required:
            raise RuntimeError(f"flat ground must be visible and collidable before substitution: flags={flags}")

        # Preserve the original USD mesh and its appearance; remove only its rigid
        # collision role so it cannot double-contact the new plane.
        builder.shape_flags[ground_id] = flags & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        material = {name: float(getattr(builder, name)[ground_id]) for name in _MATERIAL_FIELDS}

        # Field/Ground is authored at local z=0. Apply the same world transform that
        # Newton replication applied to the USD source. The plane's local normal is +Z.
        world_xform = wp.transform(
            wp.vec3(float(position[0]), float(position[1]), float(position[2])),
            wp.quat(
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ),
        )
        plane_id = builder.add_shape_plane(
            body=-1,
            xform=world_xform,
            width=0.0,
            length=0.0,
            cfg=newton.ModelBuilder.ShapeConfig(),
            color=(0.32, 0.20, 0.10),
            label=f"/World/envs/env_{world_index}{_PLANE_SUFFIX}",
        )
        for name, value in material.items():
            getattr(builder, name)[plane_id] = value
        # RayCaster uses the COLLIDE_SHAPES BVH; keeping this flag makes flat-field
        # rays hit the replacement plane. The plane stays absent from rendered shapes.
        builder.shape_flags[plane_id] = int(newton.ShapeFlags.COLLIDE_SHAPES)

        # Use the actual sparse override if present. Otherwise pin the plane to the
        # v4 runtime-dumped effective mode (2, MJCF_DEFAULT); the v6 final-model
        # audit then requires the retained source ground and all wheels to still be 2.
        solref_mode = builder.custom_attributes.get("mujoco:solref_mode")
        if solref_mode is None:
            raise RuntimeError("MJWarp did not register the per-shape mujoco:solref_mode attribute")
        values = solref_mode.values
        if values is None:
            values = {}
            solref_mode.values = values
        if not isinstance(values, dict):
            raise RuntimeError("MJWarp mujoco:solref_mode storage is not sparse")
        ground_mode = values.get(ground_id, _V4_GROUND_SOLREF_MODE)
        if int(ground_mode) != _V4_GROUND_SOLREF_MODE:
            raise RuntimeError(
                f"v4 source ground solref mode changed: {ground_mode}; expected {_V4_GROUND_SOLREF_MODE}"
            )
        values[plane_id] = int(ground_mode)

        self.worlds.append(
            {
                "world_index": world_index,
                "ground_label": str(builder.shape_label[ground_id]),
                "plane_label": str(builder.shape_label[plane_id]),
                "copied_material": material,
                "ground_solref_mode_source": "explicit_sparse_override" if ground_id in values else "v4_runtime_default",
                "plane_solref_mode_set_explicitly": int(ground_mode),
                "ground_visible": bool(builder.shape_flags[ground_id] & newton.ShapeFlags.VISIBLE),
                "ground_collision_disabled": not bool(
                    builder.shape_flags[ground_id] & newton.ShapeFlags.COLLIDE_SHAPES
                ),
                "plane_collision_enabled": bool(
                    builder.shape_flags[plane_id] & newton.ShapeFlags.COLLIDE_SHAPES
                ),
            }
        )


def _host_array(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _labels_with_basename(labels, name: str) -> list[str]:
    """Match a Newton label by its final path component, independent of nesting."""
    return [str(label) for label in labels if str(label).rsplit("/", 1)[-1] == name]


def validate_final_model(model, wheel_joint_names: list[str], wheel_body_names: list[str]) -> dict[str, object]:
    """Dump a small final-model audit proving one plane and retention of the robot."""
    shape_labels = [str(value) for value in model.shape_label]
    body_labels = [str(value) for value in model.body_label]
    joint_labels = [str(value) for value in model.joint_label]
    flags = _host_array(model.shape_flags).reshape(-1).astype(np.int64)
    label_to_id = {label: index for index, label in enumerate(shape_labels)}
    grounds = [label for label in shape_labels if label.endswith(_GROUND_SUFFIX)]
    planes = [label for label in shape_labels if label.endswith(_PLANE_SUFFIX)]
    robot_bodies = [label for label in body_labels if "/Robot/" in label or label.endswith("/Robot")]
    wheel_joint_hits = {name: _labels_with_basename(joint_labels, name) for name in wheel_joint_names}
    wheel_body_hits = {name: _labels_with_basename(body_labels, name) for name in wheel_body_names}
    invalid_wheel_joints = [name for name, hits in wheel_joint_hits.items() if len(hits) != 1]
    invalid_wheel_bodies = [name for name, hits in wheel_body_hits.items() if len(hits) != 1]
    if len(grounds) != 1 or len(planes) != 1:
        raise RuntimeError(f"expected one retained Ground mesh and one plane; got {grounds}, {planes}")
    if not robot_bodies or invalid_wheel_joints or invalid_wheel_bodies:
        raise RuntimeError(
            "robot retention check failed: "
            f"robot_bodies={len(robot_bodies)}, invalid_wheel_joints={invalid_wheel_joints}, "
            f"wheel_joint_hits={wheel_joint_hits}, invalid_wheel_bodies={invalid_wheel_bodies}, "
            f"wheel_body_hits={wheel_body_hits}"
        )

    ground_flags = int(flags[label_to_id[grounds[0]]])
    plane_flags = int(flags[label_to_id[planes[0]]])
    visible = int(newton.ShapeFlags.VISIBLE)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    if not ground_flags & visible or ground_flags & collide or not plane_flags & collide or plane_flags & visible:
        raise RuntimeError(f"unexpected final ground/plane flags: ground={ground_flags}, plane={plane_flags}")

    return {
        "shape_count": len(shape_labels),
        "body_count": len(body_labels),
        "joint_count": len(joint_labels),
        "retained_ground_mesh_label": grounds[0],
        "replacement_plane_label": planes[0],
        "ground_flags": ground_flags,
        "plane_flags": plane_flags,
        "robot_body_count": len(robot_bodies),
        "wheel_joint_labels": {name: hits[0] for name, hits in wheel_joint_hits.items()},
        "wheel_body_labels": {name: hits[0] for name, hits in wheel_body_hits.items()},
    }
