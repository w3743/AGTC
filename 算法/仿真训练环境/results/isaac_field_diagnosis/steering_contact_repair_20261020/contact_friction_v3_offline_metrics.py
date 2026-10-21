#!/usr/bin/env python3
"""Summarize the frozen v2 baseline and completed v3 raw contact tapes; no simulator."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
V2_ROOT = EXP / "contact_probe_runs_v2"
V3_ROOT = EXP / "contact_probe_runs_v3"
OUTPUT = EXP / "contact_friction_v3_comparison.json"
BASELINE = "newton_2ms_v2"
TREATMENTS = (
    "dt-2ms__cone-elliptic__impratio-1__attempt-retry-01",
    "dt-2ms__cone-elliptic__impratio-10",
    "dt-2ms__cone-elliptic__impratio-100",
)
FINAL_SECOND_START_S = 8.0


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def describe(values) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return {"count": int(array.size), "mean": "NA", "median": "NA", "p95": "NA", "max_abs": "NA"}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max_abs": float(np.max(np.abs(array))),
    }


def summarize(case: str, summary_path: Path, raw_dir: Path, is_v3: bool, baseline_hash: str, fixed_hash: str) -> dict:
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise RuntimeError(f"{case} is not complete: {summary.get('status')}")
    state_hash = summary.get("initial_state_canonical_sha256")
    contract_hash = summary.get("fixed_physical_comparison_contract_sha256") or summary.get(
        "fixed_comparison_contract_excluding_dt_sha256"
    )
    if state_hash != baseline_hash or contract_hash != fixed_hash:
        raise RuntimeError(f"{case} does not match frozen baseline initial state/fixed contract")

    policy_rows = read_jsonl(raw_dir / "policy_step_contact_samples.jsonl")
    native_rows = read_jsonl(raw_dir / "physics_step_native_qd_effort.jsonl")
    policy_last = [row for row in policy_rows if float(row["simulation_time_s"]) > FINAL_SECOND_START_S]
    native_last = [row for row in native_rows if float(row["simulation_time_s"]) > FINAL_SECOND_START_S]
    if len(policy_last) != 20 or len(native_last) != 500:
        raise RuntimeError(f"{case} final-second sample counts policy={len(policy_last)} native={len(native_last)}")

    wheel_names = summary["wheel_order_joints"]
    qd = np.asarray([row["native_joint_qd_rad_s_by_wheel_order"] for row in native_last], dtype=np.float64)
    effort = np.asarray([row["native_applied_effort_nm_by_wheel_order"] for row in native_last], dtype=np.float64)
    target = np.asarray([row["target_rad_s_by_wheel_order"] for row in native_last], dtype=np.float64)
    if qd.shape != (500, 14) or effort.shape != (500, 14) or target.shape != (500, 14):
        raise RuntimeError(f"{case} unexpected native final-second shapes qd={qd.shape}, effort={effort.shape}, target={target.shape}")

    side_indices = {
        "left": [i for i, name in enumerate(wheel_names) if "left_" in name],
        "right": [i for i, name in enumerate(wheel_names) if "right_" in name],
    }
    if len(side_indices["left"]) != 7 or len(side_indices["right"]) != 7:
        raise RuntimeError(f"{case} wheel ordering does not contain seven joints per track")

    qd_per_wheel_std = np.std(qd, axis=0)
    qd_side = {
        side: {
            "mean_rad_s": float(np.mean(qd[:, indices])),
            "mean_target_rad_s": float(np.mean(target[:, indices])),
            "mean_temporal_std_per_wheel_rad_s": float(np.mean(qd_per_wheel_std[indices])),
            "max_temporal_std_per_wheel_rad_s": float(np.max(qd_per_wheel_std[indices])),
        }
        for side, indices in side_indices.items()
    }

    normal_total = []
    normal_side = {"left": [], "right": []}
    slip_all = []
    slip_loaded = []
    slip_longitudinal_loaded = []
    slip_lateral_loaded = []
    friction_dot_slip_loaded = []
    loaded_normals = []
    wheel_radius = float(summary["comparison_contract"]["robot_spec"]["wheel_radius_m"])
    up = np.asarray([0.0, 0.0, 1.0])
    for row in policy_last:
        root_position = np.asarray(row["root_pose_world_xyzw"][:3], dtype=np.float64)
        root_velocity = np.asarray(row["root_linear_velocity_world_mps"], dtype=np.float64)
        root_omega = np.asarray(row["root_angular_velocity_world_rad_s"], dtype=np.float64)
        forward = np.asarray(row["root_forward_world_unit"], dtype=np.float64)
        right = np.cross(forward, up)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        wheel_rows = row["wheel_forces"]
        normals = np.asarray([wheel["normal_force_world_up_component_N"] for wheel in wheel_rows], dtype=np.float64)
        normal_total.append(float(np.sum(normals)))
        for side, indices in side_indices.items():
            normal_side[side].append(float(np.mean(normals[indices])))
        for index, wheel in enumerate(wheel_rows):
            center = np.asarray(wheel["wheel_center_world_m"], dtype=np.float64)
            qd_post_step = float(wheel["wheel_joint_velocity_rad_s"])
            contact_arm = center - wheel_radius * up - root_position
            contact_velocity = root_velocity + np.cross(root_omega, contact_arm)
            slip = contact_velocity - qd_post_step * wheel_radius * forward
            speed = float(np.linalg.norm(slip))
            slip_all.append(speed)
            if normals[index] > 0.0:
                slip_loaded.append(speed)
                slip_longitudinal_loaded.append(float(np.dot(slip, forward)))
                slip_lateral_loaded.append(float(np.dot(slip, right)))
                friction = np.asarray(wheel["friction_force_world_aggregate_N"], dtype=np.float64)
                friction_dot_slip_loaded.append(float(np.dot(friction, slip)))
                loaded_normals.append(float(normals[index]))

    result = {
        "case": case,
        "summary_path": str(summary_path),
        "status": summary["status"],
        "actual_exit_code": 0 if is_v3 else "frozen-v2-baseline-exit0",
        "initial_state_canonical_sha256": state_hash,
        "fixed_physical_comparison_contract_sha256": contract_hash,
        "solver_treatment": summary.get("solver_treatment", {"cone": "pyramidal", "impratio": 1.0}),
        "actual_solver_option_snapshot": summary.get("runtime_physics_properties", {}).get("actual_solver_option_snapshot", {
            "status": "v2 did not record host mj_model.opt.impratio or MJWarp impratio_invsqrt"
        }),
        "end_last_1s_mean_root_yaw_rate_rad_s": float(summary["end_last_1s_mean_root_yaw_rate_rad_s"]),
        "delta_yaw_vs_v2_pyramidal_1_rad_s": None,
        "native_final_second_samples": len(native_last),
        "native_qd_mean_and_variability_rad_s": qd_side,
        "native_effort_final_second_nm": {
            "mean_abs": float(np.mean(np.abs(effort))),
            "p95_abs": float(np.percentile(np.abs(effort), 95)),
            "max_abs": float(np.max(np.abs(effort))),
            "full_run_max_abs": summary["native_effort_max_abs_nm"],
            "expression_max_abs_residual_uncapped_nm": summary["native_qd_formula_check"]["max_abs_effort_residual_uncapped_nm"],
            "uncapped_expression_match_at_1e-4_nm": summary["native_qd_formula_check"]["uncapped_expression_match_at_1e-4_nm"],
        },
        "normal_force_final_second_N": {
            "total_upward_mean_per_policy_sample": float(np.mean(normal_total)),
            "total_upward_p05_p95": [float(np.percentile(normal_total, 5)), float(np.percentile(normal_total, 95))],
            "mean_per_wheel_upward_by_side": {side: float(np.mean(values)) for side, values in normal_side.items()},
            "loaded_wheel_sample_count": len(loaded_normals),
            "loaded_wheel_upward_force_N": describe(loaded_normals),
        },
        "approx_flat_contact_slip_final_second_mps": {
            "formula": "v_root + omega_root x (wheel_center - radius*world_up - root_position) - wheel_qd_post_step*radius*root_forward",
            "all_wheel_samples": describe(slip_all),
            "loaded_wheel_samples_normal_up_gt_0": describe(slip_loaded),
            "loaded_forward_signed_mps": describe(slip_longitudinal_loaded),
            "loaded_lateral_signed_mps": describe(slip_lateral_loaded),
            "loaded_aggregate_friction_force_dot_slip_W_proxy": describe(friction_dot_slip_loaded),
            "positive_friction_dot_slip_fraction": (
                float(np.mean(np.asarray(friction_dot_slip_loaded) > 0.0)) if friction_dot_slip_loaded else "NA"
            ),
            "limitation": "kinematic flat-ground contact-point proxy from policy-step aggregated sensor forces; not solver contact-point velocity, per-pair work, or calibrated terrain slip",
        },
    }
    return result


def main() -> int:
    baseline_summary_path = V2_ROOT / BASELINE / "contact_dt_v2_summary.json"
    baseline_hashes = read_json(baseline_summary_path)
    baseline_state = baseline_hashes["initial_state_canonical_sha256"]
    baseline_fixed = baseline_hashes["fixed_comparison_contract_excluding_dt_sha256"]
    baseline = summarize(
        BASELINE, baseline_summary_path, V2_ROOT / BASELINE, False, baseline_state, baseline_fixed
    )
    results = [baseline]
    for case in TREATMENTS:
        summary_path = V3_ROOT / case / "contact_friction_v3_summary.json"
        results.append(summarize(case, summary_path, V3_ROOT / case, True, baseline_state, baseline_fixed))
    for row in results[1:]:
        row["delta_yaw_vs_v2_pyramidal_1_rad_s"] = (
            row["end_last_1s_mean_root_yaw_rate_rad_s"] - baseline["end_last_1s_mean_root_yaw_rate_rad_s"]
        )
    output = {
        "schema": "isaac-field-contact-friction-v3-offline-comparison",
        "created_date": "2026-10-20",
        "source_protocol": str(EXP / "contact_friction_v3_protocol.json"),
        "gpu_or_simulator_used_for_this_summary": False,
        "comparison_checks": {
            "baseline_initial_state_canonical_sha256": baseline_state,
            "baseline_fixed_physical_contract_sha256": baseline_fixed,
            "all_runs_match_both_hashes": True,
            "v3_treatment_runs": len(results) - 1,
            "failed_preflight_attempt_excluded": "dt-2ms__cone-elliptic__impratio-1; retained separately, actual exit 1 before reset/motion",
        },
        "metrics": results,
    }
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing comparison: {OUTPUT}")
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "runs": [
        {"case": row["case"], "yaw": row["end_last_1s_mean_root_yaw_rate_rad_s"],
         "yaw_delta": row["delta_yaw_vs_v2_pyramidal_1_rad_s"],
         "loaded_slip_mean_mps": row["approx_flat_contact_slip_final_second_mps"]["loaded_wheel_samples"]["mean"]}
        for row in results
    ]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
