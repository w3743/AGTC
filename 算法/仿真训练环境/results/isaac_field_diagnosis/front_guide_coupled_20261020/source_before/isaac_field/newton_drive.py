"""Same-side Newton wheel coupling and canonical physics metadata for drive v7."""
from __future__ import annotations

from typing import Any

import numpy as np

MIMIC_TARGET_KIND = 2  # Newton 1.6.1 registered MuJoCo equality target kind.
EQUALITY_SOLREF = (0.004, 1.0)
EQUALITY_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)
CONTACT_SHAPE_FIELDS = (
    "shape_material_mu",
    "shape_material_kf",
    "shape_material_ke",
    "shape_material_kd",
    "shape_margin",
    "shape_gap",
)


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def same_side_wheel_coupling_metadata(enabled: bool) -> dict[str, Any]:
    """Return environment-count-independent physical mechanism metadata."""
    enabled = bool(enabled)
    metadata = {
        "enabled": enabled,
        "treatment": "same_side_wheel0_master_six_mimic_followers_each_side" if enabled else "independent_wheel_joints",
        "relation": "q_follower = 0 + 1*q_leader" if enabled else None,
        "relations_per_world": 12 if enabled else 0,
        "leader_wheel_index_per_side": 0 if enabled else None,
        "follower_wheel_indices_per_side": [1, 2, 3, 4, 5, 6] if enabled else [],
        "equality_solref": list(EQUALITY_SOLREF) if enabled else None,
        "equality_solimp": list(EQUALITY_SOLIMP) if enabled else None,
        "execution": "retain existing per-wheel actuators, PI state and configured effort limits",
        "source_usd_changed": False,
    }
    return metadata


class SameSideMimicBuilderHook:
    """Install twelve same-side equalities in each Newton builder world."""

    def __init__(self, wheel_joint_names_by_side: dict[str, list[str]], expected_world_count: int):
        if set(wheel_joint_names_by_side) != {"left", "right"}:
            raise ValueError("exactly left and right wheel sides are required")
        if isinstance(expected_world_count, bool) or int(expected_world_count) != expected_world_count or expected_world_count < 1:
            raise ValueError("expected_world_count must be a positive integer")
        self.names = {side: list(names) for side, names in wheel_joint_names_by_side.items()}
        for side, names in self.names.items():
            if len(names) != 7 or len(set(names)) != 7:
                raise ValueError(f"{side} must have seven unique wheel-joint names")
            if names[0] != f"{side}_wheel_0_wheel_joint":
                raise ValueError(f"wheel 0 must lead {side}")
        self.expected_world_count = int(expected_world_count)
        self.worlds: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []

    def __call__(self, builder, world_index: int, position, quaternion) -> None:
        if not hasattr(builder, "joint_world") or not hasattr(builder, "current_world"):
            raise RuntimeError("ModelBuilder must expose joint_world/current_world")
        current_world = int(builder.current_world)
        if current_world != int(world_index):
            raise RuntimeError(f"callback world {world_index} != builder.current_world {current_world}")
        if current_world in {item["world_index"] for item in self.worlds}:
            raise RuntimeError(f"duplicate mimic callback for world {current_world}")
        if current_world < 0 or current_world >= self.expected_world_count:
            raise RuntimeError(f"unexpected Newton world {current_world}")

        labels = [str(label) for label in builder.joint_label]
        worlds = _host_array(builder.joint_world).reshape(-1).astype(np.int64)
        if len(labels) != len(worlds):
            raise RuntimeError("joint_label and joint_world sizes differ")
        active_indices = [index for index, world in enumerate(worlds) if int(world) == current_world]
        matches: dict[str, tuple[int, str]] = {}
        for basename in (name for side in ("left", "right") for name in self.names[side]):
            hits = [
                (index, labels[index]) for index in active_indices
                if labels[index].rsplit("/", 1)[-1] == basename
            ]
            if len(hits) != 1:
                raise RuntimeError(f"world {current_world} expected one {basename!r}; got {hits}")
            matches[basename] = hits[0]

        schema = getattr(builder, "custom_attributes", None)
        add_mimic = getattr(builder, "add_constraint_mimic", None)
        add_custom_values = getattr(builder, "add_custom_values", None)
        if not isinstance(schema, dict) or not callable(add_mimic) or not callable(add_custom_values):
            raise RuntimeError("ModelBuilder lacks public mimic/custom-value APIs")
        try:
            solref_type = schema["mujoco:eq_solref"].dtype
            solimp_type = schema["mujoco:eq_solimp"].dtype
        except (KeyError, AttributeError) as exc:
            raise RuntimeError("MuJoCo equality softness schema is unavailable") from exc

        from newton.solvers import SolverMuJoCo  # noqa: PLC0415
        import mujoco  # noqa: PLC0415

        world_relations = []
        for side in ("left", "right"):
            leader_name = self.names[side][0]
            leader_id, leader_label = matches[leader_name]
            for follower_name in self.names[side][1:]:
                follower_id, follower_label = matches[follower_name]
                label = f"same_side_mimic:{side}:{follower_name}->{leader_name}"
                mimic_id = int(add_mimic(
                    joint0=follower_id,
                    joint1=leader_id,
                    coef0=0.0,
                    coef1=1.0,
                    enabled=True,
                    label=label,
                ))
                rows = add_custom_values(**{
                    "mujoco:equality_constraint_type": int(SolverMuJoCo.EqType.JOINT),
                    "mujoco:equality_constraint_objtype": int(mujoco.mjtObj.mjOBJ_JOINT),
                    "mujoco:equality_constraint_joint1": int(follower_id),
                    "mujoco:equality_constraint_joint2": int(leader_id),
                    "mujoco:equality_constraint_polycoef": [0.0, 1.0, 0.0, 0.0, 0.0],
                    "mujoco:equality_constraint_label": label,
                    "mujoco:equality_constraint_enabled": True,
                    "mujoco:equality_constraint_world": current_world,
                    "mujoco:equality_constraint_target_kind": MIMIC_TARGET_KIND,
                    "mujoco:equality_constraint_target": mimic_id,
                    "mujoco:eq_solref": solref_type(*EQUALITY_SOLREF),
                    "mujoco:eq_solimp": solimp_type(*EQUALITY_SOLIMP),
                })
                row_ids = {int(value) for value in rows.values()}
                if len(row_ids) != 1:
                    raise RuntimeError(f"equality fields were assigned to multiple rows: {rows}")
                relation = {
                    "world_index": current_world,
                    "side": side,
                    "leader_id": int(leader_id),
                    "follower_id": int(follower_id),
                    "leader_label": leader_label,
                    "follower_label": follower_label,
                    "mimic_id": mimic_id,
                    "equality_row": row_ids.pop(),
                }
                self.relations.append(relation)
                world_relations.append(relation)
        if len(world_relations) != 12:
            raise RuntimeError(f"world {current_world} got {len(world_relations)} relations")
        self.worlds.append({"world_index": current_world, "relations": world_relations})


def audit_same_side_mimic_model(hook: SameSideMimicBuilderHook, model) -> dict[str, Any]:
    """One-time final Newton model check; solver host/MJWarp audits belong in tests."""
    if int(model.world_count) != hook.expected_world_count:
        raise RuntimeError(f"model has {model.world_count} worlds, expected {hook.expected_world_count}")
    if sorted(item["world_index"] for item in hook.worlds) != list(range(hook.expected_world_count)):
        raise RuntimeError("mimic hook did not visit every world exactly once")
    expected = 12 * hook.expected_world_count
    if int(model.constraint_mimic_count) != expected or len(hook.relations) != expected:
        raise RuntimeError(f"Newton model has {model.constraint_mimic_count} mimic rows, expected {expected}")
    labels = [str(label) for label in model.joint_label]
    worlds = _host_array(model.joint_world).reshape(-1).astype(np.int64)
    arrays = {
        name: _host_array(getattr(model, name)).reshape(-1)
        for name in ("constraint_mimic_joint0", "constraint_mimic_joint1",
                     "constraint_mimic_coef0", "constraint_mimic_coef1")
    }
    equality = model.mujoco
    count = int(equality.equality_constraint_count)
    fields = {name: _host_array(getattr(equality, name)) for name in (
        "equality_constraint_joint1", "equality_constraint_joint2",
        "equality_constraint_target_kind", "equality_constraint_target",
        "equality_constraint_world", "equality_constraint_polycoef", "eq_solref", "eq_solimp",
    )}
    if count != expected or any(len(value) != expected for value in arrays.values()):
        raise RuntimeError("final mimic/equality arrays do not match the expected row count")
    if any(value.shape[0] != expected for value in fields.values()):
        raise RuntimeError("final equality custom arrays do not match the expected row count")
    seen_pairs: set[tuple[int, int]] = set()
    for row in hook.relations:
        leader, follower, world = row["leader_id"], row["follower_id"], row["world_index"]
        if worlds[leader] != world or worlds[follower] != world:
            raise RuntimeError(f"mimic pair crosses worlds: {row}")
        if labels[leader] != row["leader_label"] or labels[follower] != row["follower_label"]:
            raise RuntimeError(f"joint identity changed during finalization: {row}")
        matches = [i for i in range(expected)
                   if int(arrays["constraint_mimic_joint0"][i]) == follower
                   and int(arrays["constraint_mimic_joint1"][i]) == leader]
        if len(matches) != 1:
            raise RuntimeError(f"final model lacks a unique mimic pair: {row}")
        mimic_row = matches[0]
        if not (np.isclose(arrays["constraint_mimic_coef0"][mimic_row], 0.0)
                and np.isclose(arrays["constraint_mimic_coef1"][mimic_row], 1.0)):
            raise RuntimeError(f"mimic coefficients changed: {row}")
        eq_row = int(row["equality_row"])
        if (int(fields["equality_constraint_joint1"][eq_row]) != follower
                or int(fields["equality_constraint_joint2"][eq_row]) != leader
                or int(fields["equality_constraint_world"][eq_row]) != world
                or int(fields["equality_constraint_target_kind"][eq_row]) != MIMIC_TARGET_KIND
                or int(fields["equality_constraint_target"][eq_row]) != row["mimic_id"]
                or not np.allclose(fields["equality_constraint_polycoef"][eq_row], [0, 1, 0, 0, 0])
                or not np.allclose(fields["eq_solref"][eq_row], EQUALITY_SOLREF)
                or not np.allclose(fields["eq_solimp"][eq_row], EQUALITY_SOLIMP)):
            raise RuntimeError(f"final MuJoCo equality does not match mimic relation: {row}")
        pair = (leader, follower)
        if pair in seen_pairs:
            raise RuntimeError(f"duplicate final mimic pair {pair}")
        seen_pairs.add(pair)
    return {
        "status": "passed",
        "worlds_checked": hook.expected_world_count,
        "relations_per_world": 12,
        "total_mimic_rows": expected,
        "total_equality_rows": count,
        "cross_world_pairs": 0,
    }


def _canonical_materials_by_role(
    shape_labels,
    arrays,
    wheel_body_names,
    passive_front_guide_body_names=(),
) -> dict[str, dict[str, list[float]]]:
    roles: dict[str, list[int]] = {"ground": [], "wheel": []}
    if passive_front_guide_body_names:
        roles["passive_front_guide"] = []
    for index, label in enumerate(shape_labels):
        label = str(label)
        if label.endswith("/Field/Ground"):
            roles["ground"].append(index)
        elif any(f"/{name}/" in label for name in wheel_body_names):
            roles["wheel"].append(index)
        elif any(f"/{name}/" in label for name in passive_front_guide_body_names):
            roles["passive_front_guide"].append(index)
    result = {}
    for role, indices in roles.items():
        if not indices:
            raise RuntimeError(f"runtime contact metadata is missing {role} shapes")
        fields = {}
        for field, raw in arrays.items():
            values = np.asarray(raw, dtype=np.float64).reshape(-1)[indices]
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite material value for {role}.{field}")
            name = field.removeprefix("shape_material_").removeprefix("shape_")
            fields[name] = sorted({float(value) for value in values})
        result[role] = fields
    return result


def build_physics_contact_metadata(env) -> dict[str, Any]:
    """Cache JSON-safe config and finalized ground/contact material values once."""
    from .assets import ROBOT_SPEC  # noqa: PLC0415

    physics, solver_cfg = env.cfg.sim.physics, env.cfg.sim.physics.solver_cfg
    shape_cfg, manager = physics.default_shape_cfg, env.sim.physics_manager
    model = manager.get_model()
    wheel_body_names = [f"{wheel.name}_wheel_link" for wheel in ROBOT_SPEC.wheel_specs]
    front_guide_spec = getattr(env, "front_guide_spec", None)
    front_guide_wheels = (
        () if front_guide_spec is None else front_guide_spec.wheel_specs(ROBOT_SPEC)
    )
    front_guide_body_names = [wheel.link_name for wheel in front_guide_wheels]
    arrays = {
        field: _host_array(getattr(model, field)).reshape(-1)
        for field in CONTACT_SHAPE_FIELDS
    }
    material_by_role = _canonical_materials_by_role(
        [str(label) for label in model.shape_label], arrays, wheel_body_names,
        front_guide_body_names,
    )
    metadata = {
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
        "use_mujoco_contacts": bool(solver_cfg.use_mujoco_contacts),
        "disable_contacts": bool(solver_cfg.disable_contacts),
        "contact_generation": "mujoco_warp" if solver_cfg.use_mujoco_contacts else "newton_collision_pipeline",
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
    if front_guide_spec is not None:
        metadata["front_guide_contact_body_names"] = front_guide_body_names
    return metadata
