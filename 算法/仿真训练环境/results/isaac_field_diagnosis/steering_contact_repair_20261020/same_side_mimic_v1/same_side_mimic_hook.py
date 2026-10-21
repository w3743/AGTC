"""Result-only Newton hook and audit for the 14-wheel same-side mimic probe.

The hook adds six position equalities per track side as Newton mimic constraints.
It does not change the source USD, actuator configuration, contact geometry, or
production files.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _basename_matches(labels: list[str], basename: str) -> list[tuple[int, str]]:
    return [
        (index, str(label))
        for index, label in enumerate(labels)
        if str(label).rsplit("/", 1)[-1] == basename
    ]


class SameSideMimicHook:
    """Add one wheel-0 leader and six followers for each side in Newton world 0."""

    def __init__(self, wheel_joint_names_by_side: dict[str, list[str]]) -> None:
        if set(wheel_joint_names_by_side) != {"left", "right"}:
            raise ValueError("the mimic probe requires exactly left and right sides")
        self.wheel_joint_names_by_side = {
            side: list(names) for side, names in wheel_joint_names_by_side.items()
        }
        for side, names in self.wheel_joint_names_by_side.items():
            if len(names) != 7 or len(set(names)) != 7:
                raise ValueError(f"{side} must provide seven unique wheel joint basenames")
        if any(names[0] != f"{side}_wheel_0_wheel_joint"
               for side, names in self.wheel_joint_names_by_side.items()):
            raise ValueError("wheel 0 must be the leader on each side")
        self.worlds: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []

    def __call__(self, builder, world_index: int, position, quaternion) -> None:
        if world_index != 0 or self.worlds:
            raise RuntimeError("the same-side mimic hook accepts exactly Newton world 0 once")
        labels = [str(label) for label in getattr(builder, "joint_label", [])]
        add_mimic = getattr(builder, "add_constraint_mimic", None)
        if not labels or not callable(add_mimic):
            raise RuntimeError("Newton builder must expose joint_label and add_constraint_mimic")

        matches: dict[str, tuple[int, str]] = {}
        for names in self.wheel_joint_names_by_side.values():
            for name in names:
                hits = _basename_matches(labels, name)
                if len(hits) != 1:
                    raise RuntimeError(f"expected one Newton joint basename {name!r}; found {hits}")
                matches[name] = hits[0]

        relations: list[dict[str, Any]] = []
        for side in ("left", "right"):
            names = self.wheel_joint_names_by_side[side]
            leader_name = names[0]
            leader_index, leader_label = matches[leader_name]
            for follower_name in names[1:]:
                follower_index, follower_label = matches[follower_name]
                constraint_index = add_mimic(
                    joint0=follower_index,
                    joint1=leader_index,
                    coef0=0.0,
                    coef1=1.0,
                    enabled=True,
                    label=f"same_side_mimic:{side}:{follower_name}->{leader_name}",
                )
                relations.append({
                    "side": side,
                    "leader_name": leader_name,
                    "leader_label": leader_label,
                    "builder_leader_joint_index": int(leader_index),
                    "follower_name": follower_name,
                    "follower_label": follower_label,
                    "builder_follower_joint_index": int(follower_index),
                    "coef0": 0.0,
                    "coef1": 1.0,
                    "constraint_index": int(constraint_index),
                })
        if len(relations) != 12:
            raise RuntimeError(f"expected 12 mimic relations, created {len(relations)}")
        self.relations = relations
        self.worlds.append({
            "world_index": int(world_index),
            "joint_labels": matches,
            "relations": relations,
            "actuators_retained": True,
            "coefficient_basis": "same authored local +Y axis and same 0.045 m radius; confirm runtime axes before execution",
        })


def _model_joint_match(labels: list[str], basename: str) -> tuple[int, str]:
    hits = _basename_matches(labels, basename)
    if len(hits) != 1:
        raise RuntimeError(f"expected one final-model joint basename {basename!r}; found {hits}")
    return hits[0]


def _mujoco_equality_rows(mj_model, *, include_joint_names: bool) -> dict[str, Any]:
    import mujoco  # noqa: PLC0415

    neq = int(mj_model.neq)
    types = _host_array(mj_model.eq_type).reshape(-1).astype(np.int64)
    obj1 = _host_array(mj_model.eq_obj1id).reshape(-1).astype(np.int64)
    obj2 = _host_array(mj_model.eq_obj2id).reshape(-1).astype(np.int64)
    if len(types) != neq or len(obj1) != neq or len(obj2) != neq:
        raise RuntimeError("MuJoCo equality arrays do not match mj_model.neq")
    joint_type = int(mujoco.mjtEq.mjEQ_JOINT)
    rows: list[dict[str, Any]] = []
    for index in range(neq):
        row: dict[str, Any] = {"index": index, "type_code": int(types[index])}
        if int(types[index]) == joint_type:
            row["type_name"] = "mjEQ_JOINT"
        else:
            row["type_name"] = f"mjtEq({int(types[index])})"
        if include_joint_names and int(types[index]) == joint_type:
            row["joint1_name"] = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(obj1[index])
            )
            row["joint2_name"] = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(obj2[index])
            )
        rows.append(row)
    return {"neq": neq, "joint_equality_type_code": joint_type, "rows": rows}


def finalized_mimic_audit(hook: SameSideMimicHook, model, solver) -> dict[str, Any]:
    """Fail closed unless all 12 relations survived into Newton and MJWarp."""
    if len(hook.worlds) != 1 or len(hook.relations) != 12:
        raise RuntimeError("builder hook did not produce its one-world/12-relation contract")
    labels = [str(value) for value in getattr(model, "joint_label", [])]

    mimic_count = int(getattr(model, "constraint_mimic_count", -1))
    if mimic_count != 12:
        raise RuntimeError(f"final Newton model has {mimic_count} mimic constraints, expected 12")
    fields = (
        "constraint_mimic_joint0", "constraint_mimic_joint1",
        "constraint_mimic_coef0", "constraint_mimic_coef1",
    )
    arrays = {name: _host_array(getattr(model, name)).reshape(-1) for name in fields}
    if any(len(value) != mimic_count for value in arrays.values()):
        raise RuntimeError("final Newton mimic arrays do not match constraint_mimic_count")

    final_relations: list[dict[str, Any]] = []
    expected_pairs: set[tuple[str, str]] = set()
    for relation in hook.relations:
        master_index, master_label = _model_joint_match(labels, relation["leader_name"])
        follower_index, follower_label = _model_joint_match(labels, relation["follower_name"])
        expected_pairs.add((relation["follower_name"], relation["leader_name"]))
        candidates = [
            index for index in range(mimic_count)
            if int(arrays["constraint_mimic_joint0"][index]) == follower_index
            and int(arrays["constraint_mimic_joint1"][index]) == master_index
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"final Newton model did not retain unique pair {relation}: {candidates}")
        index = candidates[0]
        coef0 = float(arrays["constraint_mimic_coef0"][index])
        coef1 = float(arrays["constraint_mimic_coef1"][index])
        if not np.isclose(coef0, 0.0, atol=1e-8) or not np.isclose(coef1, 1.0, atol=1e-8):
            raise RuntimeError(f"final Newton mimic coefficient changed for {relation}: {coef0}, {coef1}")
        final_relations.append({
            **relation,
            "final_model_leader_joint_index": master_index,
            "final_model_leader_label": master_label,
            "final_model_follower_joint_index": follower_index,
            "final_model_follower_label": follower_label,
            "final_mimic_constraint_index": index,
            "final_coef0": coef0,
            "final_coef1": coef1,
        })

    host = getattr(solver, "mj_model", None)
    warp = getattr(solver, "mjw_model", None)
    if host is None or warp is None:
        raise RuntimeError("MJWarp solver must expose both host mj_model and mjw_model")
    host_rows = _mujoco_equality_rows(host, include_joint_names=True)
    warp_rows = _mujoco_equality_rows(warp, include_joint_names=False)
    if host_rows["neq"] != warp_rows["neq"]:
        raise RuntimeError(f"host/MJWarp equality counts differ: {host_rows['neq']} vs {warp_rows['neq']}")
    host_joint_pairs = [
        (row.get("joint1_name"), row.get("joint2_name"))
        for row in host_rows["rows"] if row["type_name"] == "mjEQ_JOINT"
    ]
    expected_pair_matches: list[dict[str, Any]] = []

    def has_expected_suffix(actual: str | None, expected: str) -> bool:
        return actual == expected or (actual is not None and actual.endswith("_" + expected))

    for pair in expected_pairs:
        candidates = [
            actual_pair for actual_pair in host_joint_pairs
            if has_expected_suffix(actual_pair[0], pair[0])
            and has_expected_suffix(actual_pair[1], pair[1])
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"host MuJoCo model has no unique sanitized mjEQ_JOINT for {pair}: {host_joint_pairs}")
        expected_pair_matches.append({"expected_joint_basenames": pair, "actual_host_joint_names": candidates[0]})
    if [row["type_code"] for row in host_rows["rows"]] != [
        row["type_code"] for row in warp_rows["rows"]
    ]:
        raise RuntimeError("host MuJoCo and MJWarp equality type arrays differ")

    return {
        "treatment": "same_side_mimic_wheel0_master_coef0_0_coef1_plus1",
        "source_usd_changed": False,
        "worlds": hook.worlds,
        "newton_model_mimic_count": mimic_count,
        "newton_model_final_relations": final_relations,
        "solver_host_equality": host_rows,
        "solver_mjwarp_equality": warp_rows,
        "expected_pair_matches": expected_pair_matches,
        "all_expected_mjEQ_JOINT_pairs_present_once": True,
        "note": "MuJoCo equality constraints are solver constraints; finite residuals are measured at runtime and are not assumed identically zero.",
    }


def runtime_mimic_state_differences(hook: SameSideMimicHook, model, state) -> dict[str, Any]:
    """Read post-step native q/qd and report follower-minus-master residuals."""
    if len(hook.relations) != 12:
        raise RuntimeError("runtime mimic sample requested before the 12 builder relations exist")
    labels = [str(value) for value in getattr(model, "joint_label", [])]
    q_starts = _host_array(getattr(model, "joint_q_start")).reshape(-1).astype(np.int64)
    qd_starts = _host_array(getattr(model, "joint_qd_start")).reshape(-1).astype(np.int64)
    q = _host_array(getattr(state, "joint_q")).reshape(-1)
    qd = _host_array(getattr(state, "joint_qd")).reshape(-1)
    if len(q_starts) < len(labels) or len(qd_starts) < len(labels):
        raise RuntimeError("Newton joint coordinate offsets are shorter than the final joint-label list")

    offsets: dict[str, tuple[int, int]] = {}
    for name in {r["leader_name"] for r in hook.relations} | {r["follower_name"] for r in hook.relations}:
        index, _label = _model_joint_match(labels, name)
        offsets[name] = (int(q_starts[index]), int(qd_starts[index]))
        if offsets[name][0] >= len(q) or offsets[name][1] >= len(qd):
            raise RuntimeError(f"Newton q/qd state is shorter than coordinate offsets for {name}: {offsets[name]}")
    rows: list[dict[str, Any]] = []
    for relation in hook.relations:
        master_q, master_qd = offsets[relation["leader_name"]]
        follower_q, follower_qd = offsets[relation["follower_name"]]
        row = {
            "side": relation["side"],
            "leader_name": relation["leader_name"],
            "follower_name": relation["follower_name"],
            "coef0": 0.0,
            "coef1": 1.0,
            "leader_q_rad": float(q[master_q]),
            "follower_q_rad": float(q[follower_q]),
            "q_follower_minus_expected_rad": float(q[follower_q] - q[master_q]),
            "leader_qd_rad_s": float(qd[master_qd]),
            "follower_qd_rad_s": float(qd[follower_qd]),
            "qd_follower_minus_expected_rad_s": float(qd[follower_qd] - qd[master_qd]),
        }
        rows.append(row)
    return {
        "source": "NewtonManager.get_state_0() after env.step; post-integration policy-step sample",
        "constraint_semantics": "q_follower = coef0 + coef1*q_leader; qd residual is its derivative",
        "relation_count": len(rows),
        "relations": rows,
    }
