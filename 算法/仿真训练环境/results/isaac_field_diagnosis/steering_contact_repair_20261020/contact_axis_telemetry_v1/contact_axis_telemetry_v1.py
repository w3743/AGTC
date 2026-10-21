#!/usr/bin/env python3
"""Add actual wheel/contact arrays to the frozen v4 snapshots without changing physics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP_ROOT = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
V4_PATH = EXP_ROOT / "contact_friction_v4.py"
OUT_ROOT = EXP_ROOT / "contact_probe_runs_v4"
CASE = "axis-telemetry-v1-attempt3__baseline__dt-2ms__cone-elliptic__impratio-1__mu-0.8__kf-1000"
OUT_DIR = OUT_ROOT / CASE
WRAPPER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MODEL_FIELDS = (
    "geom_quat", "geom_pos", "geom_type", "geom_size", "geom_bodyid", "geom_contype", "geom_conaffinity",
    "jnt_axis", "jnt_bodyid", "jnt_type", "jnt_limited", "jnt_actfrclimited", "jnt_actgravcomp",
    "jnt_group", "body_weldid", "body_mocapid", "body_jntadr", "body_jntnum", "body_geomadr",
    "body_geomnum", "body_invweight0", "dof_damping", "dof_frictionloss", "dof_armature",
    "joint_flags", "joint_enabled", "body_flags",
)


def load_v4():
    spec = importlib.util.spec_from_file_location("frozen_contact_friction_v4", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v4: {V4_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V4 = load_v4()


def copy_array(owner, field: str, rows: int | None = None, *, world_rows: bool = False):
    value = getattr(owner, field, None) if owner is not None else None
    if value is None:
        return "NA"
    try:
        array = np.asarray(V4.as_numpy(value))
        if rows is not None:
            array = array[0, :rows] if world_rows and array.ndim >= 2 and array.shape[0] == 1 else array[:rows]
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            return {"status": "non_finite", "shape": list(array.shape)}
        return {"shape": list(array.shape), "values": array.tolist()}
    except Exception as exc:
        return {"status": f"read_error:{type(exc).__name__}:{exc}"}


def copy_names(owner):
    names = {}
    for field in ("geom_names", "jnt_names", "joint_names", "body_names", "shape_label", "joint_label", "body_label"):
        value = getattr(owner, field, None) if owner is not None else None
        try:
            names[field] = list(value) if value is not None else "NA"
        except Exception:
            names[field] = "NA"
    return names


def telemetry(solver, model, phase, policy_step, simulation_time_s, physics_dt_s):
    data = getattr(solver, "mjw_data", None)
    try:
        nacon = int(np.asarray(V4.as_numpy(data.nacon)).reshape(-1)[0])
    except Exception:
        nacon = 0
    try:
        nefc = int(np.asarray(V4.as_numpy(data.nefc)).reshape(-1)[0])
    except Exception:
        nefc = None
    contacts = getattr(data, "contact", None) if data is not None else None
    efc = getattr(data, "efc", None) if data is not None else None
    return {
        "wrapper_source": str(Path(__file__).resolve()),
        "wrapper_sha256": WRAPPER_SHA256,
        "sample": {"utc": datetime.now(timezone.utc).isoformat(), "wall_time_ns": time.time_ns(),
                   "monotonic_ns": time.monotonic_ns(), "phase": phase,
                   "policy_step_global": int(policy_step), "simulation_time_s": float(simulation_time_s),
                   "physics_dt_s": float(physics_dt_s)},
        "models": {
            name: {"names": copy_names(getattr(solver, name, None)),
                   "arrays": {field: copy_array(getattr(solver, name, None), field) for field in MODEL_FIELDS}}
            for name in ("mj_model", "mjw_model")
        },
        "mjw_data": {field: copy_array(data, field) for field in ("geom_xmat", "geom_xpos", "xpos", "cvel", "qvel")},
        "contacts_valid_rows_only": {
            "nacon": nacon,
            "fields": {field: copy_array(contacts, field, nacon)
                       for field in ("frame", "pos", "friction", "dim", "efc_address", "geom", "dist", "worldid")},
        },
        "efc_active_rows_only": {
            "nefc": nefc,
            "fields": {field: copy_array(efc, field, nefc, world_rows=True) if nefc is not None else "NA"
                       for field in ("vel", "aref", "force", "D", "state", "type", "id")},
            "contact_mapping_field": "contacts_valid_rows_only.fields.efc_address",
        },
        "axis_basis_note": "MuJoCo cylinders use local +Z; geom_quat maps this into the body-local frame. Compare with jnt_axis for the same jnt_bodyid/geom_bodyid; do not infer from USD authoring axis.",
        "time_semantics": "Contact and EFC arrays are from the last solve (pre-integration; efc.vel is J*qvel). mjw_data pose/qvel sampled after env.step may be post-integration and are not simultaneous with those solve rows.",
        "physics_mutation": False,
    }


ORIGINAL_COLLECTOR = V4.mjw_contact_parameter_snapshot


def wrapped_collector(solver, model, *, phase, policy_step, simulation_time_s, physics_dt_s):
    snapshot = ORIGINAL_COLLECTOR(solver, model, phase=phase, policy_step=policy_step,
                                  simulation_time_s=simulation_time_s, physics_dt_s=physics_dt_s)
    snapshot["axis_telemetry_v1"] = telemetry(solver, model, phase, policy_step, simulation_time_s, physics_dt_s)
    return snapshot


def main() -> int:
    if os.environ.get("RUN_CONTACT_AXIS_TELEMETRY_V1") != "1":
        raise SystemExit("Requires RUN_CONTACT_AXIS_TELEMETRY_V1=1 after parent authorization")
    V4.mjw_contact_parameter_snapshot = wrapped_collector
    args = ["--physics-dt", "0.002", "--cone", "elliptic", "--impratio", "1", "--output-dir", str(OUT_DIR)]
    rc = V4.main(args)
    identity = {"schema": "contact-axis-telemetry-v1-run-identity", "wrapper_source": str(Path(__file__).resolve()),
                "wrapper_sha256": WRAPPER_SHA256, "v4_source": str(V4_PATH.resolve()),
                "v4_sha256": V4.sha256_file(V4_PATH), "case": CASE, "arguments": args,
                "contract": {"physics_dt_s": 0.002, "decimation": 25, "policy_dt_s": 0.05,
                             "cone": "elliptic", "impratio": 1.0, "mu": 0.8, "kf": 1000.0,
                             "seed": 90231, "start_xy_yaw": [50, 50, 0], "action": [0.1, -0.1, 0, 0],
                             "settle_s": 1.0, "drive_s": 8.0}, "telemetry_only": True, "physics_mutation": False}
    with (OUT_DIR / "telemetry_run_identity.json").open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
