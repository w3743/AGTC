#!/usr/bin/env python3
"""One-shot Newton actuator effort limit smoke; not training, PI, or FPS."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import traceback

import numpy as np
import torch
import warp as wp


@wp.kernel
def _record_applied_effort(
    applied: wp.array2d(dtype=wp.float32),
    wheel_ids: wp.array(dtype=wp.int32),
    tape: wp.array2d(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    capacity: int,
):
    slot = wp.atomic_add(counter, 0, 1)
    if slot < capacity:
        for j in range(14):
            tape[slot, j] = applied[0, wheel_ids[j]]


def _plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _object_fields(obj, names):
    fields = {}
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str, type(None), np.integer, np.floating)):
            fields[name] = _plain(value)
        elif isinstance(value, (list, tuple)):
            fields[name] = [_plain(item) for item in value]
    try:
        items = vars(obj).items()
    except Exception:
        items = ()
    for name, value in items:
        if name.startswith("_"):
            continue
        if isinstance(value, (bool, int, float, str, type(None), np.integer, np.floating)):
            fields.setdefault(name, _plain(value))
    return fields


def _sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _usd_drive_evidence(env):
    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return {"available": False, "reason": "no active USD stage"}
        wanted = set(env._wheel_names)
        joints = []
        max_effort_api = []
        for prim in stage.Traverse():
            schemas = list(prim.GetAppliedSchemas())
            if "NewtonMaxEffortClampingAPI" in schemas:
                attr = prim.GetAttribute("newton:maxEffort")
                max_effort_api.append({
                    "prim_path": str(prim.GetPath()),
                    "schema": "NewtonMaxEffortClampingAPI",
                    "max_effort": _plain(attr.Get()) if attr.IsValid() else None,
                })
            if prim.GetName() not in wanted:
                continue
            has_drive = bool(prim.HasAPI(UsdPhysics.DriveAPI, "angular"))
            max_force = None
            if has_drive:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                attr = drive.GetMaxForceAttr()
                max_force = _plain(attr.Get()) if attr.IsValid() else None
            joints.append({
                "prim_path": str(prim.GetPath()),
                "prim_type": str(prim.GetTypeName()),
                "applied_schemas": schemas,
                "has_angular_drive_api": has_drive,
                "angular_drive_max_force": max_force,
            })
        return {
            "available": True,
            "wheel_joint_count": len(joints),
            "wheel_joints": joints,
            "newton_max_effort_clamping_apis": max_effort_api,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _model_actuator_evidence(env):
    manager = env.sim.physics_manager
    model = manager.get_model()
    raw = getattr(model, "actuators", None)
    entries = []
    if raw is not None:
        try:
            for index, actuator in enumerate(raw):
                clamp_items = []
                for clamp in (getattr(actuator, "clamping", None) or []):
                    clamp_items.append({
                        "type": type(clamp).__name__,
                        "fields": _object_fields(clamp, ("max_effort", "limit", "max_force", "effort_limit")),
                    })
                entries.append({
                    "index": index,
                    "type": type(actuator).__name__,
                    "name": next((getattr(actuator, key, None) for key in ("name", "label", "actuator_name")
                                   if getattr(actuator, key, None) is not None), None),
                    "controller_type": type(getattr(actuator, "controller", None)).__name__,
                    "fields": _object_fields(actuator, ("max_effort", "effort_limit", "actuator_effort_limit")),
                    "clamping": clamp_items,
                })
        except Exception as exc:
            entries.append({"iteration_error": f"{type(exc).__name__}: {exc}"})
    drive_cfg = env.cfg.scene.robot.actuators.get("drive")
    cfg_fields = _object_fields(
        drive_cfg,
        ("joint_effort_limit", "actuator_effort_limit", "joint_velocity_limit", "stiffness", "damping"),
    ) if drive_cfg is not None else None
    try:
        actual_joint_limits = env.rover.data.joint_effort_limits.torch[:, env._wheel_ids].detach().cpu().tolist()
    except Exception as exc:
        actual_joint_limits = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "model_actuator_count": len(entries),
        "model_actuators": entries,
        "drive_config_class": type(drive_cfg).__name__ if drive_cfg is not None else None,
        "drive_config": cfg_fields,
        "articulation_joint_effort_limits_nm": actual_joint_limits,
    }


def main():
    if os.environ.get("RUN_ISAAC_PHYSICS") != "1":
        raise SystemExit("Refusing Newton smoke without RUN_ISAAC_PHYSICS=1")
    out = Path(__file__).resolve().parent
    summary_path = out / "effort_limit_summary.json"
    samples_path = out / "effort_limit_substeps.jsonl"
    if summary_path.exists() or samples_path.exists():
        raise SystemExit(f"Refusing to overwrite outputs in {out}")

    helper = runpy.run_path("tests/test_env_integration.py")
    make_env, step = helper["_make_env"], helper["_step"]
    from isaaclab_newton.physics import NewtonManager

    source_paths = (
        "isaac_field/env.py",
        "isaac_field/assets.py",
        "tests/test_env_integration.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/newton/kernels.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
        "external/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py",
    )
    env = None
    sample_count = 0
    capacity = 2048
    try:
        env = make_env(plants=0, max_time=20.0, terrain_enabled=False, seed=90231)
        dt = float(env.cfg.sim.dt)
        decimation = int(env.cfg.decimation)
        if not math.isclose(dt, 0.005, abs_tol=1e-12) or decimation != 10:
            raise RuntimeError(f"unexpected physics cadence: dt={dt}, decimation={decimation}")
        env.reset(
            seed=90231,
            options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
        )
        ids_torch = env._wheel_ids.to(dtype=torch.int32).contiguous()
        ids_warp = wp.from_torch(ids_torch)
        tape = wp.empty((capacity, 14), dtype=wp.float32, device=env.device)
        counter = wp.zeros(1, dtype=wp.int32, device=env.device)
        applied_warp = env.rover.actuators.applied_effort.warp

        def capture_applied_effort():
            wp.launch(
                _record_applied_effort,
                dim=1,
                inputs=[applied_warp, ids_warp, tape, counter, capacity],
                device=env.device,
            )

        NewtonManager.register_post_actuator_callback(capture_applied_effort)

        original_apply = env._apply_action
        active_case = {"name": None, "ff": None}
        boundary_reads = []
        wheel_zero = torch.zeros_like(env.wheel_target_velocity)

        def wrapped_apply_action():
            if active_case["name"] is not None:
                last = env.rover.actuators.applied_effort.torch[:, env._wheel_ids].detach().cpu().numpy()[0]
                boundary_reads.append({
                    "case": active_case["name"],
                    "policy_boundary_previous_applied_effort_nm": [float(value) for value in last],
                })
            original_apply()
            env.rover.actuators.target_command.set_velocity_index(
                value=wheel_zero, joint_ids=env._wheel_ids,
            )
            ff = active_case["ff"]
            if ff is None:
                ff = torch.zeros_like(env.wheel_target_velocity)
            env.rover.actuators.target_command.set_effort_index(
                value=ff, joint_ids=env._wheel_ids,
            )

        env._apply_action = wrapped_apply_action
        model_evidence = _model_actuator_evidence(env)
        usd_evidence = _usd_drive_evidence(env)
        side = env._wheel_track_ids
        left = side == 0
        right = side == 1
        if int(left.sum().item()) != 7 or int(right.sum().item()) != 7:
            raise RuntimeError("Expected exactly seven drive joints per side")

        patterns = (
            ("same_positive", 4.0, 4.0),
            ("same_negative", -4.0, -4.0),
            ("opposed_left_positive", 4.0, -4.0),
            ("opposed_left_negative", -4.0, 4.0),
        )
        cases = []
        rows_for_file = []
        zero_action = np.zeros(4, dtype=np.float32)
        for case_name, left_nm, right_nm in patterns:
            env.reset(
                seed=90231,
                options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
            )
            active_case["name"] = None
            active_case["ff"] = None
            settle_before = int(counter.numpy()[0])
            step(env, zero_action, count=round(1.0 / float(env.step_dt)))
            settle_after = int(counter.numpy()[0])
            ff = torch.zeros_like(env.wheel_target_velocity)
            ff[:, left] = left_nm
            ff[:, right] = right_nm
            active_case["name"] = case_name
            active_case["ff"] = ff
            test_before = int(counter.numpy()[0])
            step(env, zero_action, count=2)
            test_after = int(counter.numpy()[0])
            if test_after > capacity:
                raise RuntimeError(f"sample buffer overrun: {test_after}>{capacity}")
            all_rows = np.asarray(tape.numpy()[:test_after], dtype=np.float64)
            test_rows = all_rows[test_before:test_after]
            if test_rows.shape[0] != 2 * decimation:
                raise RuntimeError(
                    f"expected {2 * decimation} captured physics iterations, got {test_rows.shape[0]}"
                )
            left_values = test_rows[:, np.asarray(left.detach().cpu(), dtype=bool)]
            right_values = test_rows[:, np.asarray(right.detach().cpu(), dtype=bool)]
            max_abs = float(np.max(np.abs(test_rows)))
            within_cap = bool(max_abs <= 2.0 + 1e-4)
            expected_left_sign = 1 if left_nm > 0 else -1
            expected_right_sign = 1 if right_nm > 0 else -1
            sign_ok = bool(
                np.all(left_values * expected_left_sign > 0.0)
                and np.all(right_values * expected_right_sign > 0.0)
            )
            case = {
                "name": case_name,
                "requested_feedforward_nm": {"left": left_nm, "right": right_nm},
                "target_wheel_velocity_rad_s": 0.0,
                "independent_reset": True,
                "settle_s": 1.0,
                "settle_policy_steps": round(1.0 / float(env.step_dt)),
                "captured_physics_iterations": int(test_rows.shape[0]),
                "physics_iteration_range_zero_based": [test_before, test_after - 1],
                "left_applied_effort_min_max_nm": [float(left_values.min()), float(left_values.max())],
                "right_applied_effort_min_max_nm": [float(right_values.min()), float(right_values.max())],
                "max_abs_applied_effort_nm": max_abs,
                "all_samples_within_2nm_cap": within_cap,
                "signs_match_requested_pattern": sign_ok,
                "per_substep_applied_effort_nm": test_rows.tolist(),
            }
            cases.append(case)
            for local_index, values in enumerate(test_rows):
                rows_for_file.append({
                    "case": case_name,
                    "substep_in_case": local_index,
                    "physics_iteration_global": test_before + local_index,
                    "dt_s": dt,
                    "left_ff_nm": left_nm,
                    "right_ff_nm": right_nm,
                    "wheel_order": list(env._wheel_names),
                    "applied_effort_nm": [float(value) for value in values],
                })
            print("EFFORT LIMIT CASE " + json.dumps({k: v for k, v in case.items() if k != "per_substep_applied_effort_nm"}, sort_keys=True), flush=True)
            active_case["name"] = None
            active_case["ff"] = None

        sample_count = int(counter.numpy()[0])
        result = {
            "schema": "isaac-field-effort-limit-smoke-v1",
            "purpose": "isolated native actuator command-limit diagnosis; not training, policy action, PI, FPS, or a motor rating",
            "execution_status": "completed",
            "cap_assertion_status": (
                "pass" if all(case["all_samples_within_2nm_cap"] for case in cases)
                else "limit_not_enforced"
            ),
            "cap_assertion_rule": "max absolute official Newton applied_effort <= 2.0001 Nm for every wheel and captured physics iteration",
            "environment": {
                "helper": "tests/test_env_integration.py::_make_env/_step",
                "plants": 0,
                "terrain_enabled": False,
                "seed": 90231,
                "start_xy_yaw": [50.0, 50.0, 0.0],
                "goal_xy": [90.0, 50.0],
                "max_time_s": 20.0,
                "physics_dt_s": dt,
                "policy_dt_s": float(env.step_dt),
                "decimation": decimation,
                "physics_iterations_per_policy_tick": decimation,
                "test_policy_ticks_per_pattern": 2,
                "drive_joint_count": int(len(env._wheel_names)),
                "drive_joint_order": list(env._wheel_names),
                "action_all_zero": True,
            },
            "runtime_newton_actuator_clamping": model_evidence,
            "runtime_usd_joint_drive_and_newton_clamping_apis": usd_evidence,
            "cases": cases,
            "policy_boundary_previous_effort_reads": boundary_reads,
            "captured_physics_iteration_count_including_settle": sample_count,
            "capture_tape_capacity": capacity,
            "source_sha256": {path: _sha(path) for path in source_paths},
        }
        with samples_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows_for_file:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print("EFFORT LIMIT SMOKE " + json.dumps({
            "execution_status": result["execution_status"],
            "cap_assertion_status": result["cap_assertion_status"],
            "case_count": len(cases),
            "captured_physics_iteration_count_including_settle": sample_count,
            "summary": str(summary_path),
            "samples": str(samples_path),
        }, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {
            "schema": "isaac-field-effort-limit-smoke-v1",
            "execution_status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "partial_captured_physics_iteration_count": sample_count,
        }
        try:
            with (Path(__file__).resolve().parent / "effort_limit_failure.json").open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(failure, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except Exception:
            pass
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()