"""Result-only Newton hook and audit for the 14-wheel same-side mimic probe.

The hook adds six position equalities per track side as Newton mimic constraints.
It does not change the source USD, actuator configuration, contact geometry, or
production files.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


# Newton 1.6.1 exposes the equality target fields through the public custom-value
# schema but does not export MjcEqualityTargetKind. Its MIMIC value is 2 in the
# registered `mujoco:equality_constraint_target_kind` attribute.
MIMIC_TARGET_KIND_1_6_1 = 2
EQUALITY_SOLREF = (0.004, 1.0)
EQUALITY_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)


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

    def __init__(
        self,
        wheel_joint_names_by_side: dict[str, list[str]],
        *,
        solref: tuple[float, float] = EQUALITY_SOLREF,
        solimp: tuple[float, float, float, float, float] = EQUALITY_SOLIMP,
    ) -> None:
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
        if len(solref) != 2 or not np.isfinite(solref).all() or solref[0] <= 0 or solref[1] <= 0:
            raise ValueError("equality solref must be a positive finite (timeconst, dampratio) pair")
        if len(solimp) != 5 or not np.isfinite(solimp).all():
            raise ValueError("equality solimp must be five finite values")
        if not (0.0 < solimp[0] <= solimp[1] <= 1.0) or solimp[2] <= 0 or solimp[3] <= 0 or solimp[4] <= 0:
            raise ValueError("equality solimp requires 0 < dmin <= dmax <= 1 and positive width/midpoint/power")
        self.solref = tuple(float(value) for value in solref)
        self.solimp = tuple(float(value) for value in solimp)
        self.worlds: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []

    def __call__(self, builder, world_index: int, position, quaternion) -> None:
        if world_index != 0 or self.worlds:
            raise RuntimeError("the same-side mimic hook accepts exactly Newton world 0 once")
        labels = [str(label) for label in getattr(builder, "joint_label", [])]
        add_mimic = getattr(builder, "add_constraint_mimic", None)
        add_custom_values = getattr(builder, "add_custom_values", None)
        if not labels or not callable(add_mimic) or not callable(add_custom_values):
            raise RuntimeError("Newton builder must expose joint_label, add_constraint_mimic and add_custom_values")
        from newton.solvers import SolverMuJoCo  # noqa: PLC0415
        import mujoco  # noqa: PLC0415

        eq_type_joint = int(SolverMuJoCo.EqType.JOINT)
        eq_objtype_joint = int(mujoco.mjtObj.mjOBJ_JOINT)
        custom_schema = getattr(builder, "custom_attributes", None)
        if not isinstance(custom_schema, dict):
            raise RuntimeError("Newton builder must expose its registered public custom-attribute schema")
        try:
            solref_type = custom_schema["mujoco:eq_solref"].dtype
            solimp_type = custom_schema["mujoco:eq_solimp"].dtype
        except (KeyError, AttributeError) as exc:
            raise RuntimeError("Newton builder is missing public mujoco:eq_solref/eq_solimp schema fields") from exc

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
                label = f"same_side_mimic:{side}:{follower_name}->{leader_name}"
                constraint_index = add_mimic(
                    joint0=follower_index,
                    joint1=leader_index,
                    coef0=0.0,
                    coef1=1.0,
                    enabled=True,
                    label=label,
                )
                # Newton 1.6.1's public low-level API: add one solver equality row
                # in the same custom frequency as the mimic target. This target map
                # prevents SolverMuJoCo from generating a duplicate row for the mimic.
                equality_values = add_custom_values(**{
                    "mujoco:equality_constraint_type": eq_type_joint,
                    "mujoco:equality_constraint_objtype": eq_objtype_joint,
                    "mujoco:equality_constraint_joint1": int(follower_index),
                    "mujoco:equality_constraint_joint2": int(leader_index),
                    "mujoco:equality_constraint_polycoef": [0.0, 1.0, 0.0, 0.0, 0.0],
                    "mujoco:equality_constraint_label": label,
                    "mujoco:equality_constraint_enabled": True,
                    "mujoco:equality_constraint_world": int(builder.current_world),
                    "mujoco:equality_constraint_target_kind": MIMIC_TARGET_KIND_1_6_1,
                    "mujoco:equality_constraint_target": int(constraint_index),
                    "mujoco:eq_solref": solref_type(*self.solref),
                    "mujoco:eq_solimp": solimp_type(*self.solimp),
                })
                equality_indices = set(int(value) for value in equality_values.values())
                if len(equality_indices) != 1:
                    raise RuntimeError(f"equality attributes were not assigned to one shared row: {equality_values}")
                equality_row_index = equality_indices.pop()
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
                    "equality_row_index": equality_row_index,
                    "equality_label": label,
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
            "equality_target_kind": MIMIC_TARGET_KIND_1_6_1,
            "equality_solref": list(self.solref),
            "equality_solimp": list(self.solimp),
            "construction_api": "public ModelBuilder.add_custom_values; no private Newton helper",
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
    solref = _host_array(getattr(mj_model, "eq_solref")).reshape(-1, 2)
    solimp = _host_array(getattr(mj_model, "eq_solimp")).reshape(-1, 5)
    if len(types) != neq or len(obj1) != neq or len(obj2) != neq:
        raise RuntimeError("MuJoCo equality arrays do not match mj_model.neq")
    if len(solref) != neq or len(solimp) != neq:
        raise RuntimeError("MuJoCo equality solref/solimp arrays do not match mj_model.neq")
    joint_type = int(mujoco.mjtEq.mjEQ_JOINT)
    rows: list[dict[str, Any]] = []
    for index in range(neq):
        row: dict[str, Any] = {
            "index": index,
            "type_code": int(types[index]),
            "solref": solref[index].astype(float).tolist(),
            "solimp": solimp[index].astype(float).tolist(),
        }
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
    """Fail closed unless 12 configured Newton equality rows survive into MJWarp."""
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

    equality_namespace = getattr(model, "mujoco", None)
    if equality_namespace is None:
        raise RuntimeError("final Newton model has no public mujoco equality custom-value namespace")
    equality_count = int(getattr(equality_namespace, "equality_constraint_count", -1))
    if equality_count != 12:
        raise RuntimeError(f"Newton custom equality row count is {equality_count}; expected 12")
    equality_array_names = (
        "equality_constraint_type", "equality_constraint_objtype",
        "equality_constraint_joint1", "equality_constraint_joint2",
        "equality_constraint_polycoef", "equality_constraint_enabled",
        "equality_constraint_world", "equality_constraint_target_kind",
        "equality_constraint_target", "eq_solref", "eq_solimp",
    )
    eq_arrays = {name: _host_array(getattr(equality_namespace, name)) for name in equality_array_names}
    for name, value in eq_arrays.items():
        if value.shape[0] != equality_count:
            raise RuntimeError(f"Newton model.mujoco.{name} has {value.shape[0]} rows; expected {equality_count}")
    if not np.allclose(eq_arrays["eq_solref"], np.tile(np.asarray(hook.solref), (equality_count, 1)), rtol=0.0, atol=1e-7):
        raise RuntimeError(f"Newton equality solref did not retain requested {hook.solref}: {eq_arrays['eq_solref']}")
    if not np.allclose(eq_arrays["eq_solimp"], np.tile(np.asarray(hook.solimp), (equality_count, 1)), rtol=0.0, atol=1e-7):
        raise RuntimeError(f"Newton equality solimp did not retain requested {hook.solimp}: {eq_arrays['eq_solimp']}")

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
        eq_candidates = [
            index for index in range(equality_count)
            if int(eq_arrays["equality_constraint_joint1"].reshape(-1)[index]) == follower_index
            and int(eq_arrays["equality_constraint_joint2"].reshape(-1)[index]) == master_index
            and int(eq_arrays["equality_constraint_target_kind"].reshape(-1)[index]) == MIMIC_TARGET_KIND_1_6_1
            and int(eq_arrays["equality_constraint_target"].reshape(-1)[index]) == int(relation["constraint_index"])
        ]
        if len(eq_candidates) != 1:
            raise RuntimeError(f"Newton equality row did not uniquely target the mimic pair {relation}: {eq_candidates}")
        eq_index = eq_candidates[0]
        if eq_index != int(relation["equality_row_index"]):
            raise RuntimeError(f"Newton equality row index changed for {relation}: {eq_index}")
        expected_polycoef = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0])
        if not np.allclose(eq_arrays["equality_constraint_polycoef"][eq_index], expected_polycoef, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"Newton equality polynomial changed for {relation}: {eq_arrays['equality_constraint_polycoef'][eq_index]}")
        if not bool(eq_arrays["equality_constraint_enabled"].reshape(-1)[eq_index]):
            raise RuntimeError(f"Newton equality row is disabled for {relation}")

    host = getattr(solver, "mj_model", None)
    warp = getattr(solver, "mjw_model", None)
    if host is None or warp is None:
        raise RuntimeError("MJWarp solver must expose both host mj_model and mjw_model")
    host_rows = _mujoco_equality_rows(host, include_joint_names=True)
    warp_rows = _mujoco_equality_rows(warp, include_joint_names=False)
    if host_rows["neq"] != 12 or warp_rows["neq"] != 12:
        raise RuntimeError(f"expected exactly 12 host/MJWarp equality rows, got {host_rows['neq']} vs {warp_rows['neq']}; duplicate mimic rows are forbidden")
    if host_rows["neq"] != warp_rows["neq"]:
        raise RuntimeError(f"host/MJWarp equality counts differ: {host_rows['neq']} vs {warp_rows['neq']}")
    host_joint_pairs = [
        (row.get("joint1_name"), row.get("joint2_name"))
        for row in host_rows["rows"] if row["type_name"] == "mjEQ_JOINT"
    ]
    expected_pair_matches: list[dict[str, Any]] = []

    def has_expected_suffix(actual: str | None, expected: str) -> bool:
        return actual == expected or (actual is not None and actual.endswith("_" + expected))

    pair_to_eq_index: dict[tuple[str, str], int] = {}
    for pair in expected_pairs:
        candidates = [
            actual_pair for actual_pair in host_joint_pairs
            if has_expected_suffix(actual_pair[0], pair[0])
            and has_expected_suffix(actual_pair[1], pair[1])
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"host MuJoCo model has no unique sanitized mjEQ_JOINT for {pair}: {host_joint_pairs}")
        pair_to_eq_index[pair] = next(
            row["index"] for row in host_rows["rows"]
            if row.get("joint1_name") == candidates[0][0] and row.get("joint2_name") == candidates[0][1]
        )
        expected_pair_matches.append({"expected_joint_basenames": pair, "actual_host_joint_names": candidates[0]})
    if [row["type_code"] for row in host_rows["rows"]] != [
        row["type_code"] for row in warp_rows["rows"]
    ]:
        raise RuntimeError("host MuJoCo and MJWarp equality type arrays differ")

    for relation in final_relations:
        pair = (relation["follower_name"], relation["leader_name"])
        host_index = pair_to_eq_index[pair]
        host_solref = np.asarray(host_rows["rows"][host_index].get("solref", []), dtype=float)
        host_solimp = np.asarray(host_rows["rows"][host_index].get("solimp", []), dtype=float)
        warp_solref = np.asarray(warp_rows["rows"][host_index].get("solref", []), dtype=float)
        warp_solimp = np.asarray(warp_rows["rows"][host_index].get("solimp", []), dtype=float)
        for owner, observed, expected in (
            ("host solref", host_solref, np.asarray(hook.solref)),
            ("MJWarp solref", warp_solref, np.asarray(hook.solref)),
            ("host solimp", host_solimp, np.asarray(hook.solimp)),
            ("MJWarp solimp", warp_solimp, np.asarray(hook.solimp)),
        ):
            if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=0.0, atol=1e-7):
                raise RuntimeError(f"{owner} is not the requested value for {pair}: {observed}")

    return {
        "treatment": "same_side_mimic_wheel0_master_coef0_0_coef1_plus1",
        "source_usd_changed": False,
        "worlds": hook.worlds,
        "newton_model_mimic_count": mimic_count,
        "newton_model_equality_constraint_count": equality_count,
        "requested_equality_solref": list(hook.solref),
        "requested_equality_solimp": list(hook.solimp),
        "newton_model_equality_solref": eq_arrays["eq_solref"].tolist(),
        "newton_model_equality_solimp": eq_arrays["eq_solimp"].tolist(),
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
