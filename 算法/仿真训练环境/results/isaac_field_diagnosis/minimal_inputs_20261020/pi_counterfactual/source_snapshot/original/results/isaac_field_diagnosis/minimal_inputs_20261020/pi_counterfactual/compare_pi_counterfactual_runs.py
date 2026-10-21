#!/usr/bin/env python3
"""Verify exact pairing and summarize the capped-P versus per-side-PI probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    p_path = root / "capped_p_reference" / "counterfactual_summary.json"
    pi_path = root / "pi_candidate" / "counterfactual_summary.json"
    out_path = args.output.resolve()
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite {out_path}")
    p, pi = read_json(p_path), read_json(pi_path)
    if p.get("schema") != "isaac-field-capped-drive-counterfactual-v1" or pi.get("schema") != p.get("schema"):
        raise ValueError("counterfactual summary schema mismatch")
    p_ki = p["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"]
    pi_ki = pi["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"]
    if p_ki != 0.0 or pi_ki != 0.1:
        raise ValueError(f"expected capped-P Ki=0 and PI Ki=0.1, got {p_ki}, {pi_ki}")
    if any(
        run["counterfactual_controller"].get("controller_architecture")
        != "side_integral_plus_native_per_wheel_P"
        for run in (p, pi)
    ):
        raise ValueError("controller architecture metadata is missing or inconsistent")
    if p.get("source_sha256") != pi.get("source_sha256"):
        raise ValueError("source hashes differ between runs")
    if p.get("base_drive_config") != pi.get("base_drive_config"):
        raise ValueError("base v5 drive configuration differs between runs")
    if p.get("environment") != pi.get("environment"):
        raise ValueError("environment configuration differs between runs")
    if p.get("case_count") != 10 or pi.get("case_count") != 10:
        raise ValueError("expected ten paired cases")

    p_cases = {row["case"]: row for row in p["cases"]}
    pi_cases = {row["case"]: row for row in pi["cases"]}
    if set(p_cases) != set(pi_cases) or len(p_cases) != 10 or len(pi_cases) != 10:
        raise ValueError("case identities are not a unique 10-case match")

    paired = []
    for case in p_cases:
        p_row, pi_row = p_cases[case], pi_cases[case]
        if p_row["policy_normalized_action"] != pi_row["policy_normalized_action"]:
            raise ValueError(f"{case}: action differs")
        if p_row["initial_state_after_settle"] != pi_row["initial_state_after_settle"]:
            raise ValueError(f"{case}: reset/settle state differs; exact pairing failed")
        p_drive, pi_drive = p_row["drive"], pi_row["drive"]
        item = {
            "case": case,
            "action": p_row["policy_normalized_action"],
            "initial_state_after_settle_exact_match": True,
            "capped_p_drive": {
                "end_actual_v_w": p_drive["end_actual_v_w"],
                "mean_actual_v_w": p_drive["mean_actual_v_w"],
                "end_xy_m": p_drive["end_xy_m"],
                "net_yaw_delta_rad": p_drive["net_yaw_delta_unwrapped_rad"],
                "last_1s_mean_abs_qd_tracking_error_rad_s": p_drive["last_1s_mean_abs_qd_tracking_error_rad_s"],
                "max_abs_applied_effort_nm": p_drive["max_abs_applied_effort_nm"],
            },
            "pi_drive": {
                "end_actual_v_w": pi_drive["end_actual_v_w"],
                "mean_actual_v_w": pi_drive["mean_actual_v_w"],
                "end_xy_m": pi_drive["end_xy_m"],
                "net_yaw_delta_rad": pi_drive["net_yaw_delta_unwrapped_rad"],
                "last_1s_mean_abs_qd_tracking_error_rad_s": pi_drive["last_1s_mean_abs_qd_tracking_error_rad_s"],
                "max_abs_applied_effort_nm": pi_drive["max_abs_applied_effort_nm"],
                "max_abs_side_integral_feedforward_nm_per_proxy_wheel": pi_drive[
                    "max_abs_side_integral_feedforward_nm_per_proxy_wheel"
                ],
            },
            "difference_pi_minus_capped_p_end_actual_v_w": [
                pi_drive["end_actual_v_w"][index] - p_drive["end_actual_v_w"][index]
                for index in range(2)
            ],
        }
        if "brake" in p_row or "brake" in pi_row:
            if "brake" not in p_row or "brake" not in pi_row:
                raise ValueError(f"{case}: brake phase missing from one run")
            p_brake, pi_brake = p_row["brake"], pi_row["brake"]
            item["capped_p_brake"] = {
                "last_0_3s_mean_actual_v_w": p_brake["last_0_3s_mean_actual_v_w"],
                "low_speed_thresholds_met": p_brake["last_0_3s_meets_both_low_speed_thresholds"],
                "end_side_integral_ff_nm_per_proxy_wheel": p_brake[
                    "last_0_3s_side_integral_feedforward_nm_per_proxy_wheel"]["end_per_side"]
                if "last_0_3s_side_integral_feedforward_nm_per_proxy_wheel" in p_brake else None,
            }
            item["pi_brake"] = {
                "last_0_3s_mean_actual_v_w": pi_brake["last_0_3s_mean_actual_v_w"],
                "low_speed_thresholds_met": pi_brake["last_0_3s_meets_both_low_speed_thresholds"],
                "end_side_integral_ff_nm_per_proxy_wheel": pi_brake[
                    "last_0_3s_side_integral_feedforward_nm_per_proxy_wheel"]["end_per_side"]
                if "last_0_3s_side_integral_feedforward_nm_per_proxy_wheel" in pi_brake else None,
            }
        paired.append(item)

    for run in (p, pi):
        monitor = run["physics_substep_applied_effort_monitor"]
        if monitor["checked_physics_substeps"] != run["expected_physics_substeps"]:
            raise ValueError(f"{run['run_label']}: incomplete actuator-iteration capture")
        if monitor["max_abs_applied_effort_nm"] > 2.0001:
            raise ValueError(f"{run['run_label']}: observed effort exceeded 2.0001 Nm")
    if p["counterfactual_controller"].get("ki_status") != "fixed zero for capped-P reference":
        raise ValueError("Ki=0 reference metadata is incorrect")
    p_integral_max = max(
        row["drive"]["max_abs_side_integral_feedforward_nm_per_proxy_wheel"]
        for row in p["cases"]
    )
    if p_integral_max != 0.0:
        raise ValueError(f"Ki=0 reference produced non-zero integral output: {p_integral_max}")

    result = {
        "schema": "isaac-field-capped-drive-counterfactual-pair-v1",
        "source_runs": {
            "capped_p_reference_summary": str(p_path),
            "pi_candidate_summary": str(pi_path),
        },
        "exact_pairing": {
            "same_source_hashes": True,
            "same_base_drive_config": True,
            "same_environment": True,
            "same_actions": True,
            "same_settle_state_per_case": True,
            "case_count": len(paired),
        },
        "effort_cap": {
            "configured_limit_nm_per_proxy_wheel": 2.0,
            "capped_p_max_actual_effort_nm": p["physics_substep_applied_effort_monitor"]["max_abs_applied_effort_nm"],
            "pi_max_actual_effort_nm": pi["physics_substep_applied_effort_monitor"]["max_abs_applied_effort_nm"],
            "capped_p_captured_physics_iterations": p["physics_substep_applied_effort_monitor"]["checked_physics_substeps"],
            "pi_captured_physics_iterations": pi["physics_substep_applied_effort_monitor"]["checked_physics_substeps"],
        },
        "capped_p_integral_max_abs_nm_per_proxy_wheel": p_integral_max,
        "paired_cases": paired,
        "interpretation_guard": [
            "This compares two runtime-only simulations under the v5 base drive configuration.",
            "The nonzero Ki is an uncalibrated assumption and does not represent a validated hardware controller.",
            "Native IdealPD remains per proxy wheel; only the two diagnostic integral states are per track side.",
            "Brake thresholds are diagnostic low-speed checks and do not establish navigation or goal completion.",
            "Simulator applied effort is not measured motor torque and does not establish training causality.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PAIR_COMPARISON_WRITTEN {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
