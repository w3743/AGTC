#!/usr/bin/env python3
"""Fixed-policy-step Newton contact/refsafe probe; no training or FPS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
OUT_ROOT = EXP_ROOT / "contact_probe_runs_v4"
V1_PATH = EXP_ROOT / "contact_source_ab_probe.py"
TRIAL_PROTOCOL = ROOT / "results/isaac_field_improvement/fresh_side_pi_20261020/protocol.json"
DECISION_PROTOCOL = EXP_ROOT / "decision_protocol.json"
SEED = 90231
START, GOAL = [50.0, 50.0, 0.0], [90.0, 50.0]
SETTLE_S, DRIVE_S, POLICY_DT = 1.0, 8.0, 0.05
ACTION = [0.10, -0.10, 0.0, 0.0]
KP, KI, EFFORT_LIMIT = 0.04, 0.1, 2.0
DT_DECIMATION = {0.005: 10, 0.002: 25, 0.001: 50, 0.0005: 100}
INSTALLED_VERSIONS = EXP_ROOT / "installed_versions.json"


def load_v1():
    spec = importlib.util.spec_from_file_location("contact_source_ab_v1_helpers", V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reusable v1 helpers from {V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_v1()
as_numpy = V1.as_numpy
sha256_file = V1.sha256_file
json_sha256 = V1.json_sha256


def json_value(value: Any) -> Any:
    try:
        arr = np.asarray(as_numpy(value))
        if arr.ndim == 0:
            result = arr.item()
        else:
            result = arr.tolist()
        json.dumps(result, allow_nan=False)
        return result
    except Exception:
        try:
            result = float(value)
            return result if math.isfinite(result) else "NA"
        except Exception:
            return "NA"


def selected_newton_shape_properties(model, selected: list[tuple[int, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # Newton Model exposes per-shape material arrays with the shape_material_* names.
    for field in ("shape_material_mu", "shape_material_kf", "shape_material_ke", "shape_material_kd",
                  "shape_margin", "shape_gap"):
        raw = getattr(model, field, None)
        if raw is None:
            result[field] = "NA"
            continue
        try:
            values = np.asarray(as_numpy(raw))
            if values.ndim == 0:
                result[field] = values.item()
                continue
            result[field] = {
                label: (values[index].tolist() if values.ndim > 1 else values[index].item())
                for index, label in selected if index < values.shape[0]
            }
        except Exception:
            result[field] = "NA"
    return result


def optional_mj_geometry(solver) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for owner_name in ("mj_model", "mjw_model"):
        owner = getattr(solver, owner_name, None)
        if owner is None:
            result[owner_name] = "NA"
            continue
        row: dict[str, Any] = {}
        names = getattr(owner, "geom_names", None)
        if names is None:
            row["geom_names"] = "NA"
        else:
            try:
                row["geom_names"] = list(names)
            except Exception:
                row["geom_names"] = "NA"
        for field in ("geom_friction", "geom_condim", "geom_solref", "geom_solimp"):
            value = getattr(owner, field, None)
            row[field] = json_value(value) if value is not None else "NA"
        opt = getattr(owner, "opt", None)
        row["actual_opt_cone"] = json_value(getattr(opt, "cone", None)) if opt is not None else "NA"
        row["actual_opt_impratio"] = json_value(getattr(opt, "impratio", None)) if opt is not None else "NA"
        row["actual_opt_impratio_invsqrt"] = (
            json_value(getattr(opt, "impratio_invsqrt", None)) if opt is not None else "NA"
        )
        result[owner_name] = row
    return result


def native_option_scalar(value: Any, field_name: str) -> float:
    """Read scalar solver options as a scalar or one-element API array only."""
    if value is None:
        raise RuntimeError(f"runtime solver option {field_name} is unavailable")
    try:
        values = np.asarray(as_numpy(value), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"cannot convert runtime solver option {field_name}: {exc}") from exc
    if values.size != 1:
        raise RuntimeError(f"runtime solver option {field_name} must have exactly one value; got {values.size}")
    result = float(values[0])
    if not math.isfinite(result):
        raise RuntimeError(f"runtime solver option {field_name} is non-finite: {result}")
    return result


def actual_solver_option_snapshot(solver, solver_cfg) -> dict[str, Any]:
    host_opt = getattr(getattr(solver, "mj_model", None), "opt", None)
    warp_opt = getattr(getattr(solver, "mjw_model", None), "opt", None)
    host_cone = getattr(host_opt, "cone", None) if host_opt is not None else None
    warp_cone = getattr(warp_opt, "cone", None) if warp_opt is not None else None
    host_impratio = getattr(host_opt, "impratio", None) if host_opt is not None else None
    warp_invsqrt = getattr(warp_opt, "impratio_invsqrt", None) if warp_opt is not None else None
    host_ratio = native_option_scalar(host_impratio, "mj_model.opt.impratio")
    inverse_sqrt = native_option_scalar(warp_invsqrt, "mjw_model.opt.impratio_invsqrt")
    expected_cone_code = {"pyramidal": 0, "elliptic": 1}.get(str(solver_cfg.cone))
    expected_ratio = float(solver_cfg.impratio)
    expected_invsqrt = 1.0 / math.sqrt(expected_ratio)
    host_cone_float = native_option_scalar(host_cone, "mj_model.opt.cone")
    warp_cone_float = native_option_scalar(warp_cone, "mjw_model.opt.cone")
    if not host_cone_float.is_integer() or not warp_cone_float.is_integer():
        raise RuntimeError(f"runtime cone option is not an integer enum: host={host_cone_float}, mjwarp={warp_cone_float}")
    host_cone_value = int(host_cone_float)
    warp_cone_value = int(warp_cone_float)
    checks = {
        "host_mj_model_opt_cone_matches_config": host_cone_value == expected_cone_code,
        "mjw_model_opt_cone_matches_config": warp_cone_value == expected_cone_code,
        "host_mj_model_opt_impratio_matches_config": math.isfinite(host_ratio) and math.isclose(
            host_ratio, expected_ratio, rel_tol=1e-6, abs_tol=1e-8),
        "mjw_model_opt_impratio_invsqrt_matches_config": math.isfinite(inverse_sqrt) and math.isclose(
            inverse_sqrt, expected_invsqrt, rel_tol=1e-6, abs_tol=1e-8),
    }
    return {
        "configured": {"cone": str(solver_cfg.cone), "impratio": expected_ratio},
        "host_mj_model_opt": {"cone": json_value(host_cone), "impratio": json_value(host_impratio)},
        "mjw_model_opt": {"cone": json_value(warp_cone),
                           "impratio_invsqrt": json_value(warp_invsqrt),
                           "equivalent_impratio_derived_as_1_over_invsqrt_squared": (
                               1.0 / (inverse_sqrt * inverse_sqrt)
                               if math.isfinite(inverse_sqrt) and inverse_sqrt > 0.0 else "NA")},
        "expected_mujoco_cone_enum": expected_cone_code,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "note": "MJWarp stores frictional-to-normal impedance as impratio_invsqrt=1/sqrt(impratio); host MuJoCo exposes opt.impratio.",
    }


def runtime_properties(cfg, env, wheel_body_names: list[str]) -> dict[str, Any]:
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    solver = NewtonManager._solver
    labels = list(getattr(model, "shape_label", []) or [])
    selected = [(i, label) for i, label in enumerate(labels)
                if any(name in label for name in wheel_body_names) or "ground" in label.lower()]
    shape_cfg = cfg.sim.physics.default_shape_cfg
    solver_cfg = cfg.sim.physics.solver_cfg
    return {
        "manager_single_state": getattr(NewtonManager, "_use_single_state", "NA"),
        "active_solver_class": type(solver).__name__ if solver is not None else "NA",
        "solver_config_optional": {key: json_value(getattr(solver_cfg, key, "NA"))
                                    for key in ("cone", "condim", "solref", "solimp", "iterations", "tolerance")},
        "configured_solver_treatment": {"cone": json_value(getattr(solver_cfg, "cone", "NA")),
                                         "impratio": json_value(getattr(solver_cfg, "impratio", "NA"))},
        "actual_solver_option_snapshot": actual_solver_option_snapshot(solver, solver_cfg),
        "default_shape_config": {key: json_value(getattr(shape_cfg, key, "NA"))
                                  for key in ("mu", "kf", "ke", "kd", "condim", "solref", "solimp", "margin", "gap")},
        "newton_shape_labels_wheel_and_ground": selected,
        "newton_shape_material_properties_wheel_and_ground": selected_newton_shape_properties(model, selected),
        "solver_mujoco_geometry_fields": optional_mj_geometry(solver),
    }


def mjw_contact_parameter_snapshot(solver, model, *, phase: str, policy_step: int,
                                   simulation_time_s: float, physics_dt_s: float) -> dict[str, Any]:
    """Copy only valid MJWarp contact rows after the policy step's final solver substep."""
    data = getattr(solver, "mjw_data", None)
    contact = getattr(data, "contact", None) if data is not None else None
    raw_count = getattr(data, "nacon", None) if data is not None else None
    if raw_count is None or contact is None:
        return {
            "phase": phase, "policy_step_global": policy_step, "simulation_time_s": simulation_time_s,
            "sample_semantics": "after env.step; final completed solver substep of this policy step",
            "nacon": "NA", "valid_contact_rows": "NA", "contact_fields": "NA",
            "body_invweight0": "NA", "shape_solref_mode_by_newton_shape": "NA", "solver_statistics": "NA",
            "refsafe_runtime": "NA",
        }
    count_values = np.asarray(as_numpy(raw_count)).reshape(-1)
    if count_values.size != 1 or not np.isfinite(count_values).all():
        raise RuntimeError(f"expected scalar MJWarp nacon; received shape={count_values.shape}")
    nacon = int(count_values[0])
    naconmax_raw = getattr(data, "naconmax", None)
    naconmax_values = np.asarray(as_numpy(naconmax_raw)).reshape(-1) if naconmax_raw is not None else np.array([])
    naconmax = int(naconmax_values[0]) if naconmax_values.size else "NA"
    if nacon < 0 or (isinstance(naconmax, int) and nacon > naconmax):
        raise RuntimeError(f"invalid MJWarp contact slice nacon={nacon}, naconmax={naconmax}")

    field_names = ("solreffriction", "solref", "solimp", "geom", "dist", "efc_address", "worldid")
    arrays: dict[str, Any] = {}
    array_status: dict[str, str] = {}
    for name in field_names:
        value = getattr(contact, name, None)
        if value is None:
            arrays[name] = "NA"
            array_status[name] = "NA"
            continue
        try:
            values = np.asarray(as_numpy(value))
            if values.ndim == 0 or values.shape[0] < nacon:
                arrays[name] = "NA"
                array_status[name] = f"invalid-leading-dimension:{values.shape}"
            else:
                valid = values[:nacon]
                if np.issubdtype(valid.dtype, np.floating) and not np.isfinite(valid).all():
                    arrays[name] = "NA"
                    array_status[name] = "non-finite-values"
                else:
                    arrays[name] = valid.tolist()
                    array_status[name] = "available-valid-rows-only"
        except Exception as exc:
            arrays[name] = "NA"
            array_status[name] = f"read-error:{type(exc).__name__}: {exc}"

    def read_model_array(owner, field: str) -> np.ndarray | None:
        value = getattr(owner, field, None) if owner is not None else None
        if value is None:
            return None
        try:
            return np.asarray(as_numpy(value))
        except Exception:
            return None

    mj_model = getattr(solver, "mj_model", None)
    mjw_model = getattr(solver, "mjw_model", None)
    geom_bodyid = read_model_array(mjw_model, "geom_bodyid")
    body_invweight0 = read_model_array(mjw_model, "body_invweight0")
    geom_priority = read_model_array(mjw_model, "geom_priority")
    geom_solmix = read_model_array(mjw_model, "geom_solmix")
    shape_to_geom = read_model_array(solver, "newton_shape_to_mjc_geom")
    shape_kf = read_model_array(model, "shape_material_kf")
    shape_body = read_model_array(model, "shape_body")
    mode_raw = getattr(getattr(model, "mujoco", None), "solref_mode", None)
    shape_mode = None
    if mode_raw is not None:
        try:
            shape_mode = np.asarray(as_numpy(mode_raw)).reshape(-1)
        except Exception:
            shape_mode = None
    shape_labels = list(getattr(model, "shape_label", []) or [])
    body_labels = list(getattr(model, "body_label", []) or [])
    mode_names = {0: "SOLREF_MODE_FORCE_SPACE", 1: "SOLREF_MODE_RAW", 2: "SOLREF_MODE_MJCF_DEFAULT"}
    shape_rows: list[dict[str, Any]] = []
    if shape_to_geom is not None:
        flat_shape_to_geom = shape_to_geom.reshape(-1)
        for shape_idx, geom_idx in enumerate(flat_shape_to_geom.tolist()):
            if int(geom_idx) < 0:
                continue
            mode_value: Any = int(shape_mode[shape_idx]) if shape_mode is not None and shape_idx < len(shape_mode) else "NA"
            shape_body_idx: Any = int(shape_body.reshape(-1)[shape_idx]) if shape_body is not None and shape_idx < shape_body.size else "NA"
            shape_rows.append({
                "newton_shape_index": shape_idx,
                "newton_shape_label": shape_labels[shape_idx] if shape_idx < len(shape_labels) else "NA",
                "mjc_geom_id": int(geom_idx),
                "shape_body_index_newton": shape_body_idx,
                "shape_body_label_newton": body_labels[shape_body_idx] if isinstance(shape_body_idx, int) and 0 <= shape_body_idx < len(body_labels) else "NA",
                "shape_material_kf": float(shape_kf.reshape(-1)[shape_idx]) if shape_kf is not None and shape_idx < shape_kf.size else "NA",
                "shape_solref_mode_code": mode_value,
                "shape_solref_mode_name": mode_names.get(mode_value, "NA") if isinstance(mode_value, int) else "NA",
            })

    try:
        import mujoco

        body_object_type = getattr(getattr(mujoco, "mjtObj", None), "mjOBJ_BODY", None)
        geom_object_type = getattr(getattr(mujoco, "mjtObj", None), "mjOBJ_GEOM", None)
    except Exception:
        mujoco = None
        body_object_type = geom_object_type = None

    def object_name(object_type, index: int) -> str:
        if mujoco is None or object_type is None or mj_model is None:
            return "NA"
        try:
            return mujoco.mj_id2name(mj_model, object_type, int(index)) or "unnamed"
        except Exception:
            return "NA"

    def geom_model_value(array: np.ndarray | None, world: int, geom_id: int) -> Any:
        if array is None:
            return None
        try:
            if array.ndim >= 2:
                return array[world, geom_id].item()
            return array[geom_id].item()
        except Exception:
            return None

    contact_rows: list[dict[str, Any]] = []
    referenced_bodies: dict[tuple[int, int], dict[str, Any]] = {}
    raw_geom_rows = arrays.get("geom", "NA")
    raw_world_rows = arrays.get("worldid", "NA")
    raw_imp_rows = arrays.get("solimp", "NA")
    raw_ref_rows = arrays.get("solref", "NA")
    raw_frict_ref_rows = arrays.get("solreffriction", "NA")
    invweight_finite_rows = []
    opt = getattr(mjw_model, "opt", None)
    invsqrt_raw = getattr(opt, "impratio_invsqrt", None) if opt is not None else None
    try:
        impratio_invsqrt = float(np.asarray(as_numpy(invsqrt_raw), dtype=np.float64).reshape(-1)[0])
    except Exception:
        impratio_invsqrt = None

    def shape_candidates_for_geom(geom_id: int) -> list[dict[str, Any]]:
        return [row for row in shape_rows if row["mjc_geom_id"] == geom_id]

    for contact_idx in range(nacon):
        if not isinstance(raw_geom_rows, list) or contact_idx >= len(raw_geom_rows):
            contact_rows.append({"contact_index": contact_idx, "status": "geom-field-NA"})
            continue
        geom_pair = [int(v) for v in raw_geom_rows[contact_idx]]
        world_id = int(raw_world_rows[contact_idx]) if isinstance(raw_world_rows, list) and contact_idx < len(raw_world_rows) else 0
        geom_details: list[dict[str, Any]] = []
        invweights: list[Any] = []
        kf_values: list[Any] = []
        priority_values: list[Any] = []
        solmix_values: list[Any] = []
        for side, geom_id in enumerate(geom_pair):
            body_id_raw = geom_model_value(geom_bodyid, world_id, geom_id)
            body_id = int(body_id_raw) if body_id_raw is not None else None
            invw_vec: Any = None
            invw_trans: Any = None
            if body_id is not None and body_invweight0 is not None:
                try:
                    invw_vec = body_invweight0[world_id, body_id].tolist() if body_invweight0.ndim >= 3 else body_invweight0[body_id].tolist()
                    invw_trans = float(invw_vec[0])
                except Exception:
                    invw_vec = None
            if invw_trans is not None and math.isfinite(invw_trans):
                invweights.append(invw_trans)
                referenced_bodies[(world_id, body_id)] = {
                    "world_id": world_id,
                    "mujoco_body_id": body_id,
                    "mujoco_body_name": object_name(body_object_type, body_id),
                    "body_invweight0_vec2": invw_vec,
                    "body_invweight0_component0_used_for_kf_map": invw_trans,
                    "roles_seen": [],
                }
            geom_name = object_name(geom_object_type, geom_id)
            candidates = shape_candidates_for_geom(geom_id)
            candidate_labels = [str(row.get("newton_shape_label", "NA")) for row in candidates]
            joined = " ".join(candidate_labels + [geom_name]).lower()
            role = "ground" if "ground" in joined else "wheel" if "wheel" in joined else "other"
            if body_id is not None and (world_id, body_id) in referenced_bodies:
                role_list = referenced_bodies[(world_id, body_id)]["roles_seen"]
                if role not in role_list:
                    role_list.append(role)
            for shape_candidate in candidates:
                shape_idx = shape_candidate["newton_shape_index"]
                if shape_kf is not None and shape_idx < shape_kf.size:
                    kf_values.append(float(shape_kf.reshape(-1)[shape_idx]))
            pr = geom_model_value(geom_priority, world_id, geom_id)
            sm = geom_model_value(geom_solmix, world_id, geom_id)
            priority_values.append(float(pr) if pr is not None else None)
            solmix_values.append(float(sm) if sm is not None else None)
            geom_details.append({
                "side": side,
                "mujoco_geom_id": geom_id,
                "mujoco_geom_name": geom_name,
                "role_inferred_from_shape_or_geom_label": role,
                "mujoco_body_id": body_id if body_id is not None else "NA",
                "mujoco_body_name": object_name(body_object_type, body_id) if body_id is not None else "NA",
                "body_invweight0_vec2": invw_vec if invw_vec is not None else "NA",
                "body_invweight0_component0_used_for_kf_map": invw_trans if invw_trans is not None else "NA",
                "matching_newton_shapes": candidates if candidates else "NA",
            })
        mix: float | None = None
        if len(priority_values) == 2 and all(v is not None for v in priority_values):
            if priority_values[0] > priority_values[1]:
                mix = 1.0
            elif priority_values[1] > priority_values[0]:
                mix = 0.0
            elif len(solmix_values) == 2 and all(v is not None for v in solmix_values):
                s1, s2 = solmix_values
                if s1 < 1e-15 and s2 < 1e-15:
                    mix = 0.5
                elif s1 < 1e-15:
                    mix = 0.0
                elif s2 < 1e-15:
                    mix = 1.0
                elif s1 + s2 != 0.0:
                    mix = s1 / (s1 + s2)
        mixed_kf = None
        if mix is not None and len(kf_values) >= 2:
            # A collision geom normally maps to exactly one Newton shape. Keep ambiguity explicit.
            side_kf = []
            for detail in geom_details:
                shapes = detail["matching_newton_shapes"]
                side_kf.append(float(shapes[0]["shape_material_kf"]) if isinstance(shapes, list) and len(shapes) == 1 and isinstance(shapes[0].get("shape_material_kf"), (int, float)) else None)
            if len(side_kf) == 2 and all(v is not None for v in side_kf):
                mixed_kf = mix * side_kf[0] + (1.0 - mix) * side_kf[1]
        imp_row = raw_imp_rows[contact_idx] if isinstance(raw_imp_rows, list) and contact_idx < len(raw_imp_rows) else None
        dmax = float(imp_row[1]) if isinstance(imp_row, list) and len(imp_row) > 1 else None
        expected_tc = None
        if (mixed_kf is not None and len(invweights) == 2 and dmax is not None
                and impratio_invsqrt is not None and mixed_kf > 0.0):
            denom = mixed_kf * sum(invweights) * ((1.0 - dmax) * impratio_invsqrt**2 + dmax)
            if denom > 0.0 and math.isfinite(denom):
                expected_tc = 2.0 / denom
        friction_ref = raw_frict_ref_rows[contact_idx] if isinstance(raw_frict_ref_rows, list) and contact_idx < len(raw_frict_ref_rows) else None
        normal_ref = raw_ref_rows[contact_idx] if isinstance(raw_ref_rows, list) and contact_idx < len(raw_ref_rows) else None
        actual_tc = float(friction_ref[0]) if isinstance(friction_ref, list) and friction_ref else None
        tc_floor = 2.0 * physics_dt_s
        if actual_tc is not None and math.isfinite(actual_tc) and actual_tc > 0.0:
            invweight_finite_rows.append(actual_tc)
        friction_ref_format = (
            "positive_timeconst_format" if actual_tc is not None and actual_tc > 0.0
            else "zero_inherits_normal_solref" if actual_tc == 0.0
            else "negative_direct_format" if actual_tc is not None and actual_tc < 0.0
            else "NA"
        )
        contact_rows.append({
            "contact_index_valid_slice": contact_idx,
            "world_id": world_id,
            "geom_pair_ids": geom_pair,
            "geom_sides": geom_details,
            "dist_mjwarp_contact_distance": arrays.get("dist", "NA")[contact_idx] if isinstance(arrays.get("dist"), list) else "NA",
            "solref_actual": normal_ref if normal_ref is not None else "NA",
            "solimp_actual": imp_row if imp_row is not None else "NA",
            "solreffriction_actual": friction_ref if friction_ref is not None else "NA",
            "solreffriction_actual_timeconst_s": actual_tc if actual_tc is not None else "NA",
            "solreffriction_first_value_format": friction_ref_format,
            "dmax_from_actual_contact_solimp_index1": dmax if dmax is not None else "NA",
            "geom_mix_from_runtime_priority_solmix": mix if mix is not None else "NA",
            "mixed_shape_material_kf_from_runtime_mapping": mixed_kf if mixed_kf is not None else "NA",
            "solreffriction_timeconst_expected_from_newton_1_6_1_formula_s": expected_tc if expected_tc is not None else "NA",
            "refsafe_floor_2dt_s": tc_floor,
            "actual_solreffriction_tc_at_or_below_2dt": (
                bool(actual_tc <= tc_floor + 1e-9)
                if actual_tc is not None and actual_tc > 0.0 else "NA"
            ),
        })

    refsafe_state: dict[str, Any] = {"status": "NA"}
    try:
        import mujoco

        disablebit = int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
        refsafe_state = {"status": "available", "refsafe_disable_bit": disablebit, "owners": {}}
        for owner_name, owner in (("host_mj_model", mj_model), ("mjw_model", mjw_model)):
            opt = getattr(owner, "opt", None) if owner is not None else None
            flags_raw = getattr(opt, "disableflags", None) if opt is not None else None
            if flags_raw is None:
                refsafe_state["owners"][owner_name] = {"disableflags": "NA", "refsafe_enabled": "NA"}
            else:
                flags = int(np.asarray(as_numpy(flags_raw)).reshape(-1)[0])
                refsafe_state["owners"][owner_name] = {
                    "disableflags": flags,
                    "refsafe_enabled": (flags & disablebit) == 0,
                }
    except Exception as exc:
        refsafe_state = {"status": "NA", "error": f"{type(exc).__name__}: {exc}"}

    solver_statistics: dict[str, Any] = {}
    statistic_fields = ("solver_niter", "solver_gradient", "solver_improvement", "solver_fwdinv",
                        "solver_residual", "solver_grad", "solver_iter")
    for field in statistic_fields:
        value = getattr(data, field, None)
        if value is None:
            solver_statistics[field] = "NA"
            continue
        try:
            values = np.asarray(as_numpy(value))
            if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
                solver_statistics[field] = "NA"
            else:
                solver_statistics[field] = values.tolist()
        except Exception as exc:
            solver_statistics[field] = f"read-error:{type(exc).__name__}: {exc}"

    return {
        "phase": phase,
        "policy_step_global": policy_step,
        "simulation_time_s": simulation_time_s,
        "sample_semantics": "after env.step; fields copied from the final completed solver substep of this policy step; valid contact slots only [0:nacon]",
        "nacon": nacon,
        "naconmax": naconmax,
        "contact_field_status": array_status,
        "mjw_data_contact_valid_rows": arrays,
        "contact_rows_with_geom_body_invweight_and_formula": contact_rows,
        "referenced_mujoco_bodies_body_invweight0": list(referenced_bodies.values()),
        "shape_solref_mode_by_newton_shape": shape_rows if shape_rows else "NA",
        "solver_statistics": solver_statistics,
        "impratio_invsqrt_actual": impratio_invsqrt if impratio_invsqrt is not None else "NA",
        "refsafe_runtime": refsafe_state,
        "refsafe_assessment": {
            "threshold_2dt_s": 2.0 * physics_dt_s,
            "actual_friction_tc_min_s": min(invweight_finite_rows) if invweight_finite_rows else "NA",
            "actual_friction_tc_max_s": max(invweight_finite_rows) if invweight_finite_rows else "NA",
            "contacts_with_tc_at_or_below_2dt": sum(
                row.get("actual_solreffriction_tc_at_or_below_2dt") is True for row in contact_rows
            ),
            "nonpositive_solreffriction_first_parameter_rows_excluded_from_2dt_test": sum(
                row.get("solreffriction_first_value_format") in ("zero_inherits_normal_solref", "negative_direct_format")
                for row in contact_rows
            ),
            "contacts_with_expected_formula_available": sum(row.get("solreffriction_timeconst_expected_from_newton_1_6_1_formula_s") != "NA" for row in contact_rows),
            "note": "2dt comparison flags contacts whose observed solreffriction time constant could be affected by MuJoCo refsafe; no solver option is changed by this probe.",
        },
    }


def find_native_wheel_offsets(model, wheel_names: list[str]) -> tuple[list[int] | None, list[dict[str, Any]]]:
    labels = list(getattr(model, "joint_label", []) or [])
    try:
        starts = np.asarray(as_numpy(model.joint_qd_start)).reshape(-1).astype(int).tolist()
    except Exception:
        return None, []
    offsets: list[int] = []
    rows: list[dict[str, Any]] = []
    for name in wheel_names:
        hits = [(i, int(starts[i]), label) for i, label in enumerate(labels[:len(starts)])
                if label.endswith("/" + name)]
        if len(hits) != 1:
            return None, rows
        joint_index, offset, label = hits[0]
        offsets.append(offset)
        rows.append({"joint_name": name, "joint_label": label,
                     "model_joint_index": joint_index, "native_joint_qd_offset": offset})
    return offsets, rows


try:
    import warp as _wp

    @_wp.kernel
    def record_native_wheel_step(
        state_qd: _wp.array(dtype=_wp.float32),
        qd_offsets: _wp.array(dtype=_wp.int32),
        wheel_ids: _wp.array(dtype=_wp.int32),
        target: _wp.array(dtype=_wp.float32),
        integral: _wp.array(dtype=_wp.float32),
        applied: _wp.array2d(dtype=_wp.float32),
        contact_count: _wp.array(dtype=_wp.int32),
        contact_available: int,
        qd_available: int,
        command_available: int,
        kp: float,
        qd_tape: _wp.array2d(dtype=_wp.float32),
        target_tape: _wp.array2d(dtype=_wp.float32),
        integral_tape: _wp.array2d(dtype=_wp.float32),
        effort_tape: _wp.array2d(dtype=_wp.float32),
        predicted_tape: _wp.array2d(dtype=_wp.float32),
        contact_tape: _wp.array(dtype=_wp.int32),
        counter: _wp.array(dtype=_wp.int32),
        capacity: int,
    ):
        slot = _wp.atomic_add(counter, 0, 1)
        if slot < capacity:
            contact_tape[slot] = contact_count[0] if contact_available == 1 else -1
            for wheel in range(14):
                joint_id = wheel_ids[wheel]
                dof_id = qd_offsets[wheel]
                target_value = target[dof_id] if command_available == 1 else 0.0
                integral_value = integral[dof_id] if command_available == 1 else 0.0
                effort_value = applied[0, joint_id]
                target_tape[slot, wheel] = target_value
                integral_tape[slot, wheel] = integral_value
                effort_tape[slot, wheel] = effort_value
                if qd_available == 1:
                    qd_value = state_qd[dof_id]
                    qd_tape[slot, wheel] = qd_value
                    predicted_tape[slot, wheel] = (integral_value + kp * (target_value - qd_value)
                                                   if command_available == 1 else 0.0)
                else:
                    qd_tape[slot, wheel] = 0.0
                    predicted_tape[slot, wheel] = 0.0

except Exception:  # pragma: no cover - imported in the Isaac runtime
    record_native_wheel_step = None


def positive_finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("RUN_CONTACT_FRICTION_V4") != "1":
        raise SystemExit("Requires RUN_CONTACT_FRICTION_V4=1 after parent authorization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-dt", type=float, choices=tuple(DT_DECIMATION), required=True)
    parser.add_argument("--cone", choices=("pyramidal", "elliptic"), default="pyramidal")
    parser.add_argument("--impratio", type=positive_finite_float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    decimation = DT_DECIMATION[args.physics_dt]
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output directory must be inside {OUT_ROOT}") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (args.physics_dt != 0.002 or args.cone != "elliptic" or not math.isclose(args.impratio, 1.0, abs_tol=1e-12)):
        raise SystemExit("v4 is pinned to the authorized 2 ms / elliptic / impratio 1 baseline")
    summary_path = output_dir / "contact_friction_v4_summary.json"
    samples_path = output_dir / "policy_step_contact_samples.jsonl"
    native_path = output_dir / "physics_step_native_qd_effort.jsonl"
    contact_parameters_path = output_dir / "mjwarp_contact_parameter_snapshots.jsonl"
    summary: dict[str, Any] = {
        "schema": "isaac-field-contact-friction-v4",
        "status": "running",
        "physics_dt_s": args.physics_dt,
        "decimation": decimation,
        "solver_treatment": {"cone": args.cone, "impratio": args.impratio},
        "comparison_note": "Compare only with an external baseline whose full canonical initial-state hash and fixed physics contract match; no automatic dt-only pairing is performed.",
        "policy_dt_s": POLICY_DT,
        "policy_dt_product_s": args.physics_dt * decimation,
        "seed": SEED,
        "start_xy_yaw": START,
        "goal_xy": GOAL,
        "settle_s": SETTLE_S,
        "drive_s": DRIVE_S,
        "action": ACTION,
        "contact_source": "Newton collision pipeline (use_mujoco_contacts=False)",
        "output_paths": {"policy_samples": str(samples_path), "native_physics_samples": str(native_path),
                          "mjwarp_contact_parameter_snapshots": str(contact_parameters_path)},
        "contact_capture_plan": {
            "policy_steps": {"settle_end": round(SETTLE_S / POLICY_DT),
                              "drive_2s": round(SETTLE_S / POLICY_DT) + round(2.0 / POLICY_DT),
                              "drive_4s": round(SETTLE_S / POLICY_DT) + round(4.0 / POLICY_DT),
                              "drive_8s": round(SETTLE_S / POLICY_DT) + round(8.0 / POLICY_DT)},
            "source": "solver.mjw_data.nacon and solver.mjw_data.contact fields read on CPU after env.step completes",
            "valid_slice": "Only contact arrays [:nacon] are serialized; padded MJWarp capacity is excluded.",
            "refsafe": "No model option is modified; host and MJWarp disableflags are recorded when available.",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = None
    try:
        training_hashes, diagnostic_hashes = V1.BASE.source_hashes()
        helper_path = Path(__file__).resolve()
        runner_path = EXP_ROOT / "run_contact_friction_v4.sh"
        version_record = json.loads(INSTALLED_VERSIONS.read_text(encoding="utf-8")) if INSTALLED_VERSIONS.is_file() else {}
        summary.update({
            "training_source_sha256": training_hashes,
            "training_source_file_count": len(training_hashes),
            "training_protocol_sha256": diagnostic_hashes["training_protocol_sha256"],
            "training_protocol": str(TRIAL_PROTOCOL),
            "diagnostic_source_sha256": diagnostic_hashes,
            "v1_helper_sha256": sha256_file(V1_PATH),
            "v4_helper_sha256": sha256_file(helper_path),
            "v4_runner_sha256": sha256_file(runner_path),
            "installed_versions_record": {
                "path": str(INSTALLED_VERSIONS),
                "sha256": sha256_file(INSTALLED_VERSIONS) if INSTALLED_VERSIONS.is_file() else "NA",
                "checked_at": version_record.get("checked_at", "NA"),
                "versions": version_record.get("versions", {}),
                "source_actual_exit_code": version_record.get("actual_exit_code", "NA"),
            },
            "decision_protocol_sha256": sha256_file(DECISION_PROTOCOL) if DECISION_PROTOCOL.is_file() else "NA",
        })
        import torch
        import warp as wp
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_newton.physics import NewtonManager
        from isaac_field.assets import ROBOT_SPEC, ensure_robot_asset
        from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
        from isaac_field.task import Config

        import newton
        newton_root = Path(newton.__file__).resolve().parent
        source_refs = {
            "solver_mujoco.py": {
                "path": str(newton_root / "_src/solvers/mujoco/solver_mujoco.py"),
                "sha256": sha256_file(newton_root / "_src/solvers/mujoco/solver_mujoco.py"),
                "relevant_lines": "4643-4687: reads model.mujoco.solref_mode and converts Newton contacts into mjw_data.contact fields",
            },
            "kernels.py": {
                "path": str(newton_root / "_src/solvers/mujoco/kernels.py"),
                "sha256": sha256_file(newton_root / "_src/solvers/mujoco/kernels.py"),
                "relevant_lines": "637-648: maps mixed shape_material_kf, body_invweight0, solimp dmax, and impratio_invsqrt to per-contact solreffriction time constant",
            },
            "test_mujoco_solver.py": {
                "path": str(newton_root / "tests/test_mujoco_solver.py"),
                "sha256": sha256_file(newton_root / "tests/test_mujoco_solver.py"),
                "relevant_lines": "5073-5097: reads nacon and per-contact fields, independently states expected kf/impratio formula",
            },
        }
        summary["installed_newton_contact_source_evidence"] = source_refs

        cfg_task = Config(size=100.0, plants=0, layout="rows", terrain_enabled=False,
                          validate_scene=False, max_time=20.0)
        cfg = FieldEnvCfg(task=cfg_task, mode="search", auto_reset=False, seed=SEED)
        cfg.scene.num_envs = 1
        cfg.sim.dt = args.physics_dt
        cfg.decimation = decimation
        cfg.sim.physics.solver_cfg.cone = args.cone
        cfg.sim.physics.solver_cfg.impratio = args.impratio
        cfg.scene.wheel_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/.*_wheel_link",
            update_period=0.0, history_length=1, track_friction_forces=True,
        )
        cfg.sim.physics.solver_cfg.use_mujoco_contacts = False
        env = IsaacFieldEnv(cfg)

        physics, solver_cfg, shape = cfg.sim.physics, cfg.sim.physics.solver_cfg, cfg.sim.physics.default_shape_cfg
        physics_config = {
            "physics_dt_s": float(cfg.sim.dt), "policy_decimation": int(cfg.decimation),
            "policy_dt_s": float(cfg.sim.dt * cfg.decimation),
            "use_cuda_graph": bool(physics.use_cuda_graph),
            "use_mujoco_contacts": bool(solver_cfg.use_mujoco_contacts),
            "solver_integrator": str(solver_cfg.integrator),
            "configured_solver_cone": str(solver_cfg.cone),
            "configured_solver_impratio": float(solver_cfg.impratio),
            "solver_njmax": int(solver_cfg.njmax), "solver_nconmax": int(solver_cfg.nconmax),
            "shape_mu": float(shape.mu), "shape_margin_m": float(shape.margin),
            "shape_gap_m": float(shape.gap),
            "collision_cfg_present": getattr(physics, "collision_cfg", None) is not None,
        }
        if (not math.isclose(physics_config["policy_dt_s"], POLICY_DT, abs_tol=1e-12)
                or physics_config["use_mujoco_contacts"] is not False
                or physics_config["collision_cfg_present"]
                or not math.isclose(physics_config["shape_mu"], 0.8, abs_tol=1e-12)
                or physics_config["configured_solver_cone"] != args.cone
                or not math.isclose(physics_config["configured_solver_impratio"], args.impratio, abs_tol=1e-12)):
            raise RuntimeError(f"unexpected simulation/contact config: {physics_config}")
        drive_config = env.drive_config
        actuator = cfg.scene.robot.actuators["drive"]
        drive_checks = {
            "interface": drive_config.get("interface") == "independent_left_right_track_speed_explicit_pi_v6",
            "P_gain_0.04": math.isclose(float(drive_config.get("velocity_gain")), KP, abs_tol=1e-12),
            "Ki_gain_0.1": math.isclose(float(drive_config.get("integral_gain_nm_per_proxy_wheel_per_rad_s_s")), KI, abs_tol=1e-12),
            "policy_step_dt_0.05": math.isclose(float(env.step_dt), POLICY_DT, abs_tol=1e-12),
            "actuator_cap_2Nm": math.isclose(float(actuator.actuator_effort_limit), EFFORT_LIMIT, abs_tol=1e-12),
            "joint_cap_2Nm": math.isclose(float(actuator.joint_effort_limit), EFFORT_LIMIT, abs_tol=1e-12),
        }
        if not all(drive_checks.values()):
            raise RuntimeError(f"current drive preflight failed: {drive_checks}")

        wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
        wheel_body_names = [f"{wheel.name}_wheel_link" for wheel in ROBOT_SPEC.wheel_specs]
        wheel_ids = env._wheel_ids.detach().cpu().numpy().astype(np.int64).tolist()
        actual_names = [env.rover.joint_names[index] for index in wheel_ids]
        if len(wheel_ids) != 14 or actual_names != wheel_names:
            raise RuntimeError(f"expected 14 wheel joints in order: {actual_names}")
        wheel_body_ids = [env.rover.body_names.index(name) for name in wheel_body_names]
        wheel_sides = as_numpy(env._wheel_track_ids).reshape(-1).astype(int).tolist()
        sensor_names = list(env.scene["wheel_contacts"].sensor_names or [])
        if len(sensor_names) != 14 or set(sensor_names) != set(wheel_body_names):
            raise RuntimeError(f"wheel sensor body names mismatch: {sensor_names}")
        sensor_ids = [sensor_names.index(name) for name in wheel_body_names]

        field_fingerprint = hashlib.sha256(json.dumps(
            {"config": V1.asdict(cfg_task), "seed": cfg.seed, "asset_revision": 8},
            sort_keys=True).encode("utf-8")).hexdigest()[:16]
        terrain_path = ROOT / "results/isaac_field/assets" / field_fingerprint / "terrain.usda"
        robot_asset_path = ensure_robot_asset(ROOT / "results/isaac_field/assets")
        if not terrain_path.is_file() or not robot_asset_path.is_file():
            raise FileNotFoundError(f"same geometry assets unavailable: {terrain_path}, {robot_asset_path}")
        invariant = {
            "task_config": V1.asdict(cfg_task), "seed": SEED, "start": START, "goal": GOAL,
            "plants": 0,
            "flat_field_terrain_usd_sha256": sha256_file(terrain_path),
            "robot_usd_sha256": sha256_file(robot_asset_path),
            "robot_spec": V1.asdict(ROBOT_SPEC),
            "contact_source": "Newton; use_mujoco_contacts=False",
            "default_shape_mu": float(shape.mu), "P": KP, "Ki": KI,
            "actuator_effort_limit_nm": float(actuator.actuator_effort_limit),
            "joint_effort_limit_nm": float(actuator.joint_effort_limit),
            "action": ACTION, "settle_s": SETTLE_S, "drive_s": DRIVE_S,
            "variable_physics_dt_s": args.physics_dt,
            "derived_decimation": decimation,
            "solver_treatment": {"cone": args.cone, "impratio": args.impratio},
            "fixed_policy_dt_s": POLICY_DT,
        }
        fixed_invariant = {key: value for key, value in invariant.items()
                           if key not in ("variable_physics_dt_s", "derived_decimation", "solver_treatment")}
        summary.update({
            "comparison_contract": invariant,
            "comparison_contract_sha256": json_sha256(invariant),
            "fixed_physical_comparison_contract": fixed_invariant,
            "fixed_physical_comparison_contract_sha256": json_sha256(fixed_invariant),
            "physics_config": physics_config,
            "drive_config": drive_config,
            "drive_preflight": drive_checks,
            "wheel_order_joints": wheel_names,
            "wheel_order_bodies": wheel_body_names,
            "wheel_sides": wheel_sides,
            "assets": {"field_fingerprint": field_fingerprint,
                       "terrain_usd": str(terrain_path), "terrain_usd_sha256": sha256_file(terrain_path),
                       "robot_usd": str(robot_asset_path), "robot_usd_sha256": sha256_file(robot_asset_path)},
            "body_joint_mass_inertia_armature": V1.body_and_joint_properties(env),
            "runtime_physics_properties": runtime_properties(cfg, env, wheel_body_names),
            "sampling_plan": {"settle_policy_steps": round(SETTLE_S / POLICY_DT),
                              "drive_policy_steps": round(DRIVE_S / POLICY_DT),
                              "physics_substeps_per_policy_step": decimation,
                              "physics_sample_hz": 1.0 / args.physics_dt,
                              "policy_sample_hz": 1.0 / POLICY_DT,
                              "surface": "original generated flat Field mesh; no added plane or replacement collider"},
        })
        if not summary["runtime_physics_properties"]["actual_solver_option_snapshot"]["all_checks_pass"]:
            raise RuntimeError(
                "actual host-MuJoCo/MJWarp cone or impratio fields do not match configured treatment: "
                f"{summary['runtime_physics_properties']['actual_solver_option_snapshot']}"
            )
        env.reset(seed=SEED, options={"mode": "search", "start": START, "goal": GOAL})
        initial_state = V1.initial_state_snapshot(env)
        summary["initial_state_after_reset_before_settle"] = initial_state
        summary["initial_state_canonical_sha256"] = json_sha256(initial_state)
        summary["paired_initial_state_checks"] = {
            "status": "pending_external_match",
            "matched_reference": None,
            "required_match_fields": ["initial_state_canonical_sha256", "fixed_physical_comparison_contract_sha256"],
            "note": "Do not infer a pair from dt alone; cone and impratio are explicit treatment variables.",
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        model = NewtonManager.get_model()
        offsets, native_offset_rows = find_native_wheel_offsets(model, wheel_names)
        qd_available = offsets is not None and len(offsets) == 14
        summary["native_qd_access"] = {
            "status": "available" if qd_available else "NA",
            "source": "NewtonManager.get_state_0().joint_qd read from post-actuator callback before solver substeps; getter is called in the callback each iteration",
            "manager_single_state": getattr(NewtonManager, "_use_single_state", "NA"),
            "native_joint_qd_offsets": native_offset_rows if qd_available else "NA",
        }
        native_control = NewtonManager.get_control()
        native_target = getattr(native_control, "joint_target_qd", None) if native_control is not None else None
        native_integral = getattr(native_control, "joint_act", None) if native_control is not None else None
        command_available = native_target is not None and native_integral is not None
        summary["native_control_command_access"] = {
            "status": "available" if command_available else "NA",
            "target_source": "NewtonManager.get_control().joint_target_qd indexed at native joint_qd_start offsets",
            "integral_feedforward_source": "NewtonManager.get_control().joint_act indexed at native joint_qd_start offsets",
        }
        if record_native_wheel_step is None:
            qd_available = False
            summary["native_qd_access"]["status"] = "NA"
            summary["native_qd_access"]["error"] = "Warp sampling kernel unavailable"

        settle_steps, drive_steps = round(SETTLE_S / POLICY_DT), round(DRIVE_S / POLICY_DT)
        expected = (settle_steps + drive_steps) * decimation
        qd_tape = wp.empty((expected, 14), dtype=wp.float32, device=env.device)
        target_tape = wp.empty((expected, 14), dtype=wp.float32, device=env.device)
        integral_tape = wp.empty((expected, 14), dtype=wp.float32, device=env.device)
        effort_tape = wp.empty((expected, 14), dtype=wp.float32, device=env.device)
        predicted_tape = wp.empty((expected, 14), dtype=wp.float32, device=env.device)
        contact_tape = wp.empty(expected, dtype=wp.int32, device=env.device)
        counter = wp.zeros(1, dtype=wp.int32, device=env.device)
        wheel_ids_warp = wp.from_torch(torch.as_tensor(wheel_ids, device=env.device, dtype=torch.int32).contiguous())
        offsets_warp = wp.from_torch(torch.as_tensor(offsets or [0] * 14, device=env.device, dtype=torch.int32).contiguous())
        dummy_size = max(offsets or [0]) + 1
        target_warp = native_target if command_available else wp.zeros(dummy_size, dtype=wp.float32, device=env.device)
        integral_warp = native_integral if command_available else wp.zeros(dummy_size, dtype=wp.float32, device=env.device)
        applied_warp = env.rover.actuators.applied_effort.warp
        dummy_qd = wp.zeros(dummy_size, dtype=wp.float32, device=env.device)
        dummy_contact = wp.full(1, -1, dtype=wp.int32, device=env.device)

        def capture_after_native_actuator_step() -> None:
            state = NewtonManager.get_state_0()
            state_qd = getattr(state, "joint_qd", None) if qd_available else None
            if state_qd is None:
                state_qd = dummy_qd
            contacts = NewtonManager.get_contacts()
            contact_array = getattr(contacts, "rigid_contact_count", None) if contacts is not None else None
            contact_flag = 1 if contact_array is not None else 0
            if contact_array is None:
                contact_array = dummy_contact
            wp.launch(record_native_wheel_step, dim=1, device=env.device, inputs=[
                state_qd, offsets_warp, wheel_ids_warp, target_warp, integral_warp, applied_warp,
                contact_array, contact_flag, int(qd_available), int(command_available), KP,
                qd_tape, target_tape, integral_tape, effort_tape, predicted_tape,
                contact_tape, counter, expected,
            ])

        if record_native_wheel_step is not None:
            NewtonManager.register_post_actuator_callback(capture_after_native_actuator_step)
        zero_action = torch.zeros((1, 4), device=env.device, dtype=torch.float32)
        drive_action = torch.as_tensor([ACTION], device=env.device, dtype=torch.float32)
        pair_unavailable = sensor_unavailable = 0
        sample_count = 0
        contact_parameter_snapshots: list[dict[str, Any]] = []
        drive_snapshot_steps = {round(2.0 / POLICY_DT), round(4.0 / POLICY_DT), round(8.0 / POLICY_DT)}
        with samples_path.open("x", encoding="utf-8") as stream:
            for phase, action_tensor, steps in (("settle", zero_action, settle_steps), ("drive", drive_action, drive_steps)):
                for phase_step in range(1, steps + 1):
                    env.step(action_tensor)
                    sample_count += 1
                    callback_count = int(as_numpy(counter).reshape(-1)[0]) if record_native_wheel_step else 0
                    if record_native_wheel_step and callback_count != sample_count * decimation:
                        raise RuntimeError(f"native sample count {callback_count}, expected {sample_count * decimation}")
                    row = V1.sample(env, wheel_body_names, sensor_ids, wheel_body_ids, wheel_sides,
                                    phase, phase_step, sample_count, decimation, args.physics_dt, POLICY_DT)
                    state_after = NewtonManager.get_state_0()
                    if qd_available and getattr(state_after, "joint_qd", None) is not None:
                        state_qd_after = np.asarray(as_numpy(state_after.joint_qd)).reshape(-1)
                        final_qd = [float(state_qd_after[offset]) for offset in offsets]
                    else:
                        final_qd = [None] * 14
                    for wheel_row, qd in zip(row["wheel_forces"], final_qd):
                        wheel_row["wheel_joint_velocity_rad_s"] = qd
                    row["native_qd_sample_source"] = "NewtonManager.get_state_0().joint_qd after env.step; not ArticulationData cache"
                    pair_unavailable += row["contact_identity_api"]["status"] != "available"
                    sensor_unavailable += row["wheel_contact_sensor_status"] != "available"
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    should_capture_contacts = (
                        (phase == "settle" and phase_step == settle_steps)
                        or (phase == "drive" and phase_step in drive_snapshot_steps)
                    )
                    if should_capture_contacts:
                        checkpoint = "settle_end" if phase == "settle" else f"drive_{phase_step * POLICY_DT:.0f}s"
                        contact_parameter_snapshots.append(mjw_contact_parameter_snapshot(
                            NewtonManager._solver, NewtonManager.get_model(), phase=checkpoint,
                            policy_step=sample_count, simulation_time_s=sample_count * POLICY_DT,
                            physics_dt_s=args.physics_dt,
                        ))

        with contact_parameters_path.open("x", encoding="utf-8") as stream:
            for snapshot in contact_parameter_snapshots:
                stream.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")

        captured = int(as_numpy(counter).reshape(-1)[0]) if record_native_wheel_step else 0
        if record_native_wheel_step and captured != expected:
            raise RuntimeError(f"captured {captured} native samples, expected {expected}")
        if record_native_wheel_step:
            qd = as_numpy(qd_tape)[:captured]
            targets = as_numpy(target_tape)[:captured]
            integral = as_numpy(integral_tape)[:captured]
            efforts = as_numpy(effort_tape)[:captured]
            predicted = as_numpy(predicted_tape)[:captured]
            contacts = as_numpy(contact_tape)[:captured]
            if qd_available and not np.isfinite(qd).all():
                raise RuntimeError("native wheel qd tape contains non-finite values")
            if not np.isfinite(efforts).all():
                raise RuntimeError("native applied-effort tape contains non-finite values")
            uncapped = np.abs(predicted) < (EFFORT_LIMIT - 1e-4)
            residual = efforts - predicted
            formula_available = qd_available and command_available
            usable = uncapped & np.isfinite(qd) if formula_available else np.zeros_like(uncapped, dtype=bool)
            no_contact_usable = usable & (contacts[:, None] == 0)
            formula = {
                "status": "available" if formula_available else "NA",
                "expression": "integral_feedforward + 0.04 * (wheel_target_rad_s - Newton input joint_qd)",
                "captured_native_qd_substeps": captured if qd_available else "NA",
                "applied_effort_samples": captured,
                "uncapped_wheel_samples": int(usable.sum()),
                "uncapped_no_contact_wheel_samples": int(no_contact_usable.sum()),
                "max_abs_effort_residual_uncapped_nm": float(np.max(np.abs(residual[usable]))) if usable.any() else "NA",
                "mean_abs_effort_residual_uncapped_nm": float(np.mean(np.abs(residual[usable]))) if usable.any() else "NA",
                "uncapped_expression_match_at_1e-4_nm": bool(usable.any() and np.max(np.abs(residual[usable])) <= 1e-4),
                "no_contact_expression_match_at_1e-4_nm": bool(no_contact_usable.any() and np.max(np.abs(residual[no_contact_usable])) <= 1e-4),
                "native_contact_record_count_zero_samples": int((contacts == 0).sum()),
                "native_contact_count_semantics": "global Newton contact records for this one-env world, captured at post-actuator/pre-solver callback; not per-wheel contact count",
            }
            with native_path.open("x", encoding="utf-8") as stream:
                for slot in range(captured):
                    policy_step = slot // decimation + 1
                    phase = "settle" if policy_step <= settle_steps else "drive"
                    item = {
                        "physics_iteration_global": slot + 1,
                        "simulation_time_s": (slot + 1) * args.physics_dt,
                        "policy_step_global": policy_step,
                        "phase": phase,
                        "physics_substep_in_policy": slot % decimation + 1,
                        "active_state_read": "NewtonManager.get_state_0().joint_qd at post-actuator callback before solver substep",
                        "native_joint_qd_rad_s_by_wheel_order": qd[slot].tolist() if qd_available else "NA",
                        "target_rad_s_by_wheel_order": targets[slot].tolist() if command_available else "NA",
                        "integral_feedforward_nm_by_wheel_order": integral[slot].tolist() if command_available else "NA",
                        "predicted_uncapped_effort_nm_by_wheel_order": predicted[slot].tolist() if formula_available else "NA",
                        "native_applied_effort_nm_by_wheel_order": efforts[slot].tolist(),
                        "native_global_contact_record_count": int(contacts[slot]) if contacts[slot] >= 0 else "NA",
                    }
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            formula = {"status": "NA", "reason": "Warp sampler unavailable; motion probe completed with policy-step samples only"}

        max_effort = float(np.max(np.abs(efforts))) if record_native_wheel_step else None
        if max_effort is not None and max_effort > EFFORT_LIMIT + 1e-5:
            raise RuntimeError(f"native actuator effort exceeded {EFFORT_LIMIT} Nm: {max_effort}")
        final_training_hashes, final_diagnostic_hashes = V1.BASE.source_hashes()
        if final_training_hashes != training_hashes or final_diagnostic_hashes != diagnostic_hashes:
            raise RuntimeError("training or diagnostic source hashes changed during probe")
        if sha256_file(helper_path) != summary["v4_helper_sha256"] or sha256_file(runner_path) != summary["v4_runner_sha256"]:
            raise RuntimeError("v4 helper or runner changed during probe")
        policy_rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
        final_policy_rows = policy_rows[-round(1.0 / POLICY_DT):]
        final_yaw_values = [float(row["root_angular_velocity_world_rad_s"][2]) for row in final_policy_rows]
        last_second_yaw_mean = float(np.mean(final_yaw_values)) if final_yaw_values else "NA"
        contact_rows_flat = [
            row for snapshot in contact_parameter_snapshots
            for row in snapshot.get("contact_rows_with_geom_body_invweight_and_formula", [])
            if isinstance(row, dict) and "contact_index_valid_slice" in row
        ]
        observed_tc = [float(row["solreffriction_actual_timeconst_s"]) for row in contact_rows_flat
                       if isinstance(row.get("solreffriction_actual_timeconst_s"), (int, float))
                       and math.isfinite(float(row["solreffriction_actual_timeconst_s"]))]
        formula_pairs = [
            (float(row["solreffriction_actual_timeconst_s"]),
             float(row["solreffriction_timeconst_expected_from_newton_1_6_1_formula_s"]))
            for row in contact_rows_flat
            if isinstance(row.get("solreffriction_actual_timeconst_s"), (int, float))
            and float(row["solreffriction_actual_timeconst_s"]) > 0.0
            and isinstance(row.get("solreffriction_timeconst_expected_from_newton_1_6_1_formula_s"), (int, float))
        ]
        contact_capture_summary = {
            "snapshot_count": len(contact_parameter_snapshots),
            "snapshots": [{"phase": s.get("phase"), "nacon": s.get("nacon"),
                           "solver_statistics": s.get("solver_statistics", "NA"),
                           "refsafe_runtime": s.get("refsafe_runtime"),
                           "refsafe_assessment": s.get("refsafe_assessment")} for s in contact_parameter_snapshots],
            "valid_contact_rows_across_snapshots": len(contact_rows_flat),
            "threshold_2dt_s": 2.0 * args.physics_dt,
            "observed_solreffriction_timeconst_min_s": min(observed_tc) if observed_tc else "NA",
            "observed_solreffriction_timeconst_max_s": max(observed_tc) if observed_tc else "NA",
            "observed_contact_rows_at_or_below_2dt": sum(tc <= 2.0 * args.physics_dt + 1e-9 for tc in observed_tc),
            "newton_formula_available_contact_rows": len(formula_pairs),
            "newton_formula_max_abs_timeconst_residual_s": max((abs(a - e) for a, e in formula_pairs), default="NA"),
            "newton_formula_max_relative_timeconst_residual": max((abs(a - e) / max(abs(e), 1e-12) for a, e in formula_pairs), default="NA"),
        }
        summary.update({
            "status": "complete",
            "policy_samples_captured": sample_count,
            "policy_samples_expected": settle_steps + drive_steps,
            "contact_identity_api_unavailable_samples": int(pair_unavailable),
            "wheel_contact_sensor_unavailable_samples": int(sensor_unavailable),
            "native_physics_substeps_expected": expected,
            "native_physics_substeps_captured": captured,
            "native_qd_formula_check": formula,
            "mjwarp_contact_parameter_capture": contact_capture_summary,
            "native_effort_max_abs_nm": max_effort,
            "native_effort_cap_nm": EFFORT_LIMIT,
            "native_effort_cap_pass": bool(max_effort is not None and max_effort <= EFFORT_LIMIT + 1e-5),
            "end_last_1s_mean_root_yaw_rate_rad_s": last_second_yaw_mean,
            "end_last_1s_policy_sample_count": len(final_yaw_values),
            "source_hashes_unchanged_after_probe": True,
        })
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "physics_dt_s": args.physics_dt,
                          "decimation": decimation, "policy_samples": sample_count,
                          "native_substeps": captured, "max_effort_nm": max_effort,
                          "native_qd_formula_check": formula,
                          "initial_state_sha256": summary["initial_state_canonical_sha256"]}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
