"""Staged multi-world same-side Newton mimic hook; result-only, not production."""
from __future__ import annotations

from typing import Any

import numpy as np

MIMIC_TARGET_KIND = 2  # Newton 1.6.1 custom schema value used by same_side_mimic_v2.
EQUALITY_SOLREF = (0.004, 1.0)
EQUALITY_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)


def mechanism_metadata() -> dict[str, Any]:
    """Canonical physical-mechanism description; deliberately world-count agnostic."""
    return {
        "treatment": "same_side_wheel0_master_six_mimic_followers_each_side",
        "relation": "q_follower = 0 + 1*q_leader",
        "relations_per_world": 12,
        "leader_wheel_index_per_side": 0,
        "follower_wheel_indices_per_side": [1, 2, 3, 4, 5, 6],
        "equality_solref": list(EQUALITY_SOLREF),
        "equality_solimp": list(EQUALITY_SOLIMP),
        "execution": "retain the existing per-wheel actuator and drive-controller configuration",
        "source_usd_changed": False,
    }


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


class MultiWorldSameSideMimicHook:
    """Add exactly 12 equalities to each active Newton builder world."""

    def __init__(self, wheel_joint_names_by_side: dict[str, list[str]], expected_world_count: int):
        if set(wheel_joint_names_by_side) != {"left", "right"}:
            raise ValueError("exactly left and right sides are required")
        if expected_world_count < 1:
            raise ValueError("expected_world_count must be positive")
        self.names = {side: list(names) for side, names in wheel_joint_names_by_side.items()}
        for side, names in self.names.items():
            if len(names) != 7 or len(set(names)) != 7:
                raise ValueError(f"{side} must provide seven unique wheel joint basenames")
            if names[0] != f"{side}_wheel_0_wheel_joint":
                raise ValueError(f"wheel 0 must be the {side} leader")
        self.expected_world_count = int(expected_world_count)
        self.worlds: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []

    def __call__(self, builder, world_index: int, position, quaternion) -> None:
        if not hasattr(builder, "joint_world") or not hasattr(builder, "current_world"):
            raise RuntimeError("installed builder must expose public joint_world/current_world")
        current_world = int(builder.current_world)
        if current_world != int(world_index):
            raise RuntimeError(
                f"hook world mismatch: callback={world_index}, builder.current_world={current_world}"
            )
        if current_world in {row["world_index"] for row in self.worlds}:
            raise RuntimeError(f"duplicate mimic hook callback for world {current_world}")
        if current_world < 0 or current_world >= self.expected_world_count:
            raise RuntimeError(f"unexpected world index {current_world}")

        labels = [str(label) for label in getattr(builder, "joint_label", [])]
        joint_world = _host_array(builder.joint_world).reshape(-1).astype(np.int64)
        if len(joint_world) != len(labels):
            raise RuntimeError(
                f"joint_world has {len(joint_world)} entries for {len(labels)} joint labels"
            )
        active = {index for index, world in enumerate(joint_world) if int(world) == current_world}
        if not active:
            raise RuntimeError(f"Newton builder contains no joints for current world {current_world}")

        custom_schema = getattr(builder, "custom_attributes", None)
        add_mimic = getattr(builder, "add_constraint_mimic", None)
        add_custom_values = getattr(builder, "add_custom_values", None)
        if not isinstance(custom_schema, dict) or not callable(add_mimic) or not callable(add_custom_values):
            raise RuntimeError("builder is missing public mimic/custom-value APIs or schema")
        try:
            solref_type = custom_schema["mujoco:eq_solref"].dtype
            solimp_type = custom_schema["mujoco:eq_solimp"].dtype
        except (KeyError, AttributeError) as exc:
            raise RuntimeError("builder lacks the public MuJoCo equality-softness schema") from exc

        matches: dict[str, tuple[int, str]] = {}
        for basename in [name for side in ("left", "right") for name in self.names[side]]:
            hits = [
                (index, labels[index]) for index in active
                if labels[index].rsplit("/", 1)[-1] == basename
            ]
            if len(hits) != 1:
                raise RuntimeError(
                    f"expected one {basename!r} joint in world {current_world}; found {hits}"
                )
            matches[basename] = hits[0]

        from newton.solvers import SolverMuJoCo  # noqa: PLC0415
        import mujoco  # noqa: PLC0415

        relations: list[dict[str, Any]] = []
        for side in ("left", "right"):
            leader_name = self.names[side][0]
            leader_index, leader_label = matches[leader_name]
            for follower_name in self.names[side][1:]:
                follower_index, follower_label = matches[follower_name]
                relation_label = f"same_side_mimic:{side}:{follower_name}->{leader_name}"
                mimic_index = int(add_mimic(
                    joint0=follower_index,
                    joint1=leader_index,
                    coef0=0.0,
                    coef1=1.0,
                    enabled=True,
                    label=relation_label,
                ))
                equality_values = add_custom_values(**{
                    "mujoco:equality_constraint_type": int(SolverMuJoCo.EqType.JOINT),
                    "mujoco:equality_constraint_objtype": int(mujoco.mjtObj.mjOBJ_JOINT),
                    "mujoco:equality_constraint_joint1": int(follower_index),
                    "mujoco:equality_constraint_joint2": int(leader_index),
                    "mujoco:equality_constraint_polycoef": [0.0, 1.0, 0.0, 0.0, 0.0],
                    "mujoco:equality_constraint_label": relation_label,
                    "mujoco:equality_constraint_enabled": True,
                    "mujoco:equality_constraint_world": current_world,
                    "mujoco:equality_constraint_target_kind": MIMIC_TARGET_KIND,
                    "mujoco:equality_constraint_target": mimic_index,
                    "mujoco:eq_solref": solref_type(*EQUALITY_SOLREF),
                    "mujoco:eq_solimp": solimp_type(*EQUALITY_SOLIMP),
                })
                row_ids = {int(value) for value in equality_values.values()}
                if len(row_ids) != 1:
                    raise RuntimeError(f"equality values did not map to one row: {equality_values}")
                relation = {
                    "world_index": current_world,
                    "side": side,
                    "leader_name": leader_name,
                    "leader_label": leader_label,
                    "leader_joint_index": int(leader_index),
                    "follower_name": follower_name,
                    "follower_label": follower_label,
                    "follower_joint_index": int(follower_index),
                    "mimic_index": mimic_index,
                    "equality_row_index": row_ids.pop(),
                    "coef0": 0.0,
                    "coef1": 1.0,
                }
                relations.append(relation)
        if len(relations) != 12:
            raise RuntimeError(f"world {current_world} received {len(relations)} relations, expected 12")
        self.relations.extend(relations)
        self.worlds.append({
            "world_index": current_world,
            "joint_labels": matches,
            "relations": relations,
        })


def audit_builder_and_solver(hook: MultiWorldSameSideMimicHook, model, solver) -> dict[str, Any]:
    """Validate actual arrays; do not assume the MuJoCo `neq` scaling rule."""
    if len(hook.worlds) != hook.expected_world_count:
        raise RuntimeError(f"hook visited {len(hook.worlds)} worlds, expected {hook.expected_world_count}")
    if len(hook.relations) != 12 * hook.expected_world_count:
        raise RuntimeError("wrong total hook relation count")
    if int(model.world_count) != hook.expected_world_count:
        raise RuntimeError(f"Newton world_count={model.world_count}, expected {hook.expected_world_count}")

    labels = [str(value) for value in model.joint_label]
    worlds = _host_array(model.joint_world).reshape(-1).astype(np.int64)
    mimic_fields = {
        name: _host_array(getattr(model, name)).reshape(-1)
        for name in (
            "constraint_mimic_joint0", "constraint_mimic_joint1",
            "constraint_mimic_coef0", "constraint_mimic_coef1",
        )
    }
    mimic_count = int(model.constraint_mimic_count)
    if mimic_count != len(hook.relations):
        raise RuntimeError(f"Newton mimic count {mimic_count}; expected {len(hook.relations)}")
    if any(len(values) != mimic_count for values in mimic_fields.values()):
        raise RuntimeError("Newton mimic arrays are not aligned with mimic count")

    pair_counts: dict[tuple[int, int], int] = {}
    for row in hook.relations:
        leader = int(row["leader_joint_index"])
        follower = int(row["follower_joint_index"])
        if worlds[leader] != row["world_index"] or worlds[follower] != row["world_index"]:
            raise RuntimeError(f"mimic pair crosses world boundary: {row}")
        if labels[leader] != row["leader_label"] or labels[follower] != row["follower_label"]:
            raise RuntimeError(f"joint label changed at model finalization: {row}")
        if not np.isclose(float(row["coef0"]), 0.0) or not np.isclose(float(row["coef1"]), 1.0):
            raise RuntimeError(f"unexpected relation coefficients: {row}")
        matches = [
            index for index in range(mimic_count)
            if int(mimic_fields["constraint_mimic_joint0"][index]) == follower
            and int(mimic_fields["constraint_mimic_joint1"][index]) == leader
        ]
        if len(matches) != 1:
            raise RuntimeError(f"mimic pair is absent or duplicated in final Newton model: {row}")
        idx = matches[0]
        pair_counts[(leader, follower)] = pair_counts.get((leader, follower), 0) + 1
        if not np.isclose(float(mimic_fields["constraint_mimic_coef0"][idx]), 0.0):
            raise RuntimeError(f"final coef0 changed for {row}")
        if not np.isclose(float(mimic_fields["constraint_mimic_coef1"][idx]), 1.0):
            raise RuntimeError(f"final coef1 changed for {row}")
    if any(count != 1 for count in pair_counts.values()) or len(pair_counts) != len(hook.relations):
        raise RuntimeError("final mimic pairs are not unique")

    equality = model.mujoco
    equality_count = int(equality.equality_constraint_count)
    equality_fields = (
        "equality_constraint_joint1", "equality_constraint_joint2",
        "equality_constraint_target_kind", "equality_constraint_target",
        "equality_constraint_world", "eq_solref", "eq_solimp",
    )
    eq = {name: _host_array(getattr(equality, name)) for name in equality_fields}
    if any(values.shape[0] != equality_count for values in eq.values()):
        raise RuntimeError("Newton equality custom arrays are not aligned with equality_constraint_count")
    for row in hook.relations:
        idx = int(row["equality_row_index"])
        if idx < 0 or idx >= equality_count:
            raise RuntimeError(f"equality row index out of bounds: {row}")
        if int(eq["equality_constraint_joint1"][idx]) != row["follower_joint_index"]:
            raise RuntimeError(f"equality follower changed: {row}")
        if int(eq["equality_constraint_joint2"][idx]) != row["leader_joint_index"]:
            raise RuntimeError(f"equality leader changed: {row}")
        if int(eq["equality_constraint_world"][idx]) != row["world_index"]:
            raise RuntimeError(f"equality world changed: {row}")
        if int(eq["equality_constraint_target_kind"][idx]) != MIMIC_TARGET_KIND:
            raise RuntimeError(f"equality target-kind changed: {row}")
        if int(eq["equality_constraint_target"][idx]) != row["mimic_index"]:
            raise RuntimeError(f"equality target changed: {row}")
        if not np.allclose(eq["eq_solref"][idx], EQUALITY_SOLREF, atol=1e-7, rtol=0.0):
            raise RuntimeError(f"equality solref changed: {row}")
        if not np.allclose(eq["eq_solimp"][idx], EQUALITY_SOLIMP, atol=1e-7, rtol=0.0):
            raise RuntimeError(f"equality solimp changed: {row}")

    host, warp = solver.mj_model, solver.mjw_model
    host_neq, warp_neq = int(host.neq), int(warp.neq)
    if host_neq != warp_neq:
        raise RuntimeError(f"host neq={host_neq}, MJWarp neq={warp_neq}")
    host_types = _host_array(host.eq_type).reshape(-1)
    warp_types = _host_array(warp.eq_type).reshape(-1)
    host_obj1 = _host_array(host.eq_obj1id).reshape(-1)
    host_obj2 = _host_array(host.eq_obj2id).reshape(-1)
    warp_obj1 = _host_array(warp.eq_obj1id).reshape(-1)
    warp_obj2 = _host_array(warp.eq_obj2id).reshape(-1)
    if not (len(host_types) == len(warp_types) == host_neq):
        raise RuntimeError("host/MJWarp equality type arrays do not match neq")
    if not (np.array_equal(host_types, warp_types)
            and np.array_equal(host_obj1, warp_obj1)
            and np.array_equal(host_obj2, warp_obj2)):
        raise RuntimeError("host and MJWarp equality row identities differ")

    return {
        "expected_world_count": hook.expected_world_count,
        "newton_world_count": int(model.world_count),
        "hook_world_count": len(hook.worlds),
        "mimic_constraints": mimic_count,
        "mujoco_equality_custom_rows": equality_count,
        "solver_host_neq": host_neq,
        "solver_mjwarp_neq": warp_neq,
        "solver_eqtype_codes": host_types.astype(int).tolist(),
        "solver_eq_obj1_ids": host_obj1.astype(int).tolist(),
        "solver_eq_obj2_ids": host_obj2.astype(int).tolist(),
        "worlds": hook.worlds,
        "canonical_mechanism_metadata": mechanism_metadata(),
        "passed": True,
    }
