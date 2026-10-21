"""Proposed result-only fragment for a future drive_config physics child.

This is not imported by production code. Keep the returned values JSON-safe, and
compute/cache this snapshot once after Newton finalizes the model so the viewer
does not synchronize GPU shape arrays on every metadata refresh.
"""
from __future__ import annotations

import numpy as np
from isaac_field.assets import ROBOT_SPEC


_CONTACT_SHAPE_FIELDS = (
    "shape_material_mu",
    "shape_material_kf",
    "shape_material_ke",
    "shape_material_kd",
    "shape_margin",
    "shape_gap",
)


def _canonical_materials_by_role(shape_labels, arrays, wheel_body_names) -> dict:
    """Summarize finite material values without environment paths or multiplicity."""
    roles: dict[str, list[int]] = {"ground": [], "wheel": []}
    for index, label in enumerate(shape_labels):
        label = str(label)
        if label.endswith("/Field/Ground"):
            roles["ground"].append(index)
        elif any(f"/{name}/" in label for name in wheel_body_names):
            roles["wheel"].append(index)

    result = {}
    for role, indices in roles.items():
        if not indices:
            raise RuntimeError(f"runtime contact metadata is missing the {role} shapes")
        fields = {}
        for field, raw in arrays.items():
            values = np.asarray(raw, dtype=np.float64).reshape(-1)[indices]
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite runtime contact material: role={role}, field={field}")
            name = field.removeprefix("shape_material_").removeprefix("shape_")
            fields[name] = sorted({float(value) for value in values})
        result[role] = fields
    return result


def build_physics_contact_metadata(env) -> dict:
    """Build the nested child from actual cfg scalars and finalized shape data."""
    physics = env.cfg.sim.physics
    solver_cfg = physics.solver_cfg
    shape_cfg = physics.default_shape_cfg
    manager = env.sim.physics_manager
    model = manager.get_model()

    wheel_body_names = [f"{wheel.name}_wheel_link" for wheel in ROBOT_SPEC.wheel_specs]
    shape_labels = [str(label) for label in model.shape_label]
    arrays = {}
    for field in _CONTACT_SHAPE_FIELDS:
        value = getattr(model, field)
        arrays[field] = np.asarray(value.numpy() if hasattr(value, "numpy") else value).reshape(-1)
    material_by_role = _canonical_materials_by_role(shape_labels, arrays, wheel_body_names)

    use_mujoco_contacts = bool(solver_cfg.use_mujoco_contacts)
    return {
        "backend": type(physics).__name__,
        "physics_manager": type(manager).__name__,
        "solver_cfg": type(solver_cfg).__name__,
        "solver_type": str(solver_cfg.solver_type),
        "solver": str(solver_cfg.solver),
        "physics_dt_s": float(env.cfg.sim.dt),
        "decimation": int(env.cfg.decimation),
        "policy_dt_s": float(env.step_dt),
        "newton_num_substeps": int(physics.num_substeps),
        "collision_decimation": int(physics.collision_decimation),
        "cone": str(solver_cfg.cone),
        "impratio": float(solver_cfg.impratio),
        "njmax": int(solver_cfg.njmax),
        "nconmax": None if solver_cfg.nconmax is None else int(solver_cfg.nconmax),
        "iterations": int(solver_cfg.iterations),
        "ls_iterations": int(solver_cfg.ls_iterations),
        "tolerance": float(solver_cfg.tolerance),
        "integrator": str(solver_cfg.integrator),
        "use_mujoco_contacts": use_mujoco_contacts,
        "disable_contacts": bool(solver_cfg.disable_contacts),
        "contact_generation": "mujoco_warp" if use_mujoco_contacts else "newton_collision_pipeline",
        "shape_cfg_defaults": {
            "mu": float(shape_cfg.mu),
            "kf": None if not hasattr(shape_cfg, "kf") else float(shape_cfg.kf),
            "ke": float(shape_cfg.ke),
            "kd": float(shape_cfg.kd),
            "margin": float(shape_cfg.margin),
            "gap": float(shape_cfg.gap),
            "kf_source": "cfg" if hasattr(shape_cfg, "kf") else "not_exposed_by_NewtonShapeCfg",
        },
        "actual_wheel_ground_shape_materials": material_by_role,
    }


# Integration notes for the future, unfrozen source revision:
#   1. import copy;
#   2. after `super().__init__(cfg, ...)` has finalized Newton, set
#      `self._physics_contact_metadata = build_physics_contact_metadata(self)`;
#   3. add `physics_contact=copy.deepcopy(self._physics_contact_metadata)` to the
#      existing `return dict(...)` in `drive_config`.
