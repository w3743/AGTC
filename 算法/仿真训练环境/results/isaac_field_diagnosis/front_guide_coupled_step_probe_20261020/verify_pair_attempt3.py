#!/usr/bin/env python3
"""CPU-only pairing audit for the fixed-action front-guide mode runs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ROWS = 200
STEP_REAR_EDGE_X_M = 3.4
# A conservative orientation-independent bound from the proxy geometry: the
# farthest guide-wheel point is sqrt(.42^2 + .205^2 + .055^2) +
# sqrt(.045^2 + .03^2) < .525 m from the chassis origin.
WHOLE_PROXY_ENVELOPE_RADIUS_M = 0.525
STATE_FIELDS = (
    ("x_m", "x_m"),
    ("y_m", "y_m"),
    ("yaw_rad", "yaw_rad"),
    ("z_m", "z_m"),
    ("pitch_rad", "pitch_rad"),
    ("roll_rad", "roll_rad"),
    ("forward_speed_mps", "forward_speed_mps"),
    ("yaw_rate_rad_s", "yaw_rate_rad_s"),
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def no_rotation_mode(spec: dict[str, Any]) -> dict[str, Any]:
    result = dict(spec)
    result.pop("rotation_coupling", None)
    return result


def no_selected_mode(contract: dict[str, Any]) -> dict[str, Any]:
    result = dict(contract)
    result.pop("rotation_coupling", None)
    return result


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def runtime_guide_mode_matches(summary: dict[str, Any], mode: str) -> bool:
    audit = summary["runtime"]["guide_constraint_audit"]
    expected_guide_relations = 2 if mode == "same_side_track_speed" else 0
    expected_total = 12 + expected_guide_relations
    equality = audit["guide_joint_equality_present_by_joint"]
    no_drive = (
        not any(audit["guide_independent_actuator_joint_present_by_joint"].values())
        and not any(audit["guide_usd_driveapi_present_by_joint"].values())
        and audit["guide_actuator_count_from_drive_metadata"] == 0
        and audit["guide_pid_count_added"] == 0
    )
    relation_matches = (
        audit["support_wheel_relations_per_world"] == 12
        and audit["guide_relations_per_world"] == expected_guide_relations
        and audit["total_mimic_relations_per_world"] == expected_total
        and (
            all(equality.values())
            if expected_guide_relations == 2
            else not any(equality.values())
        )
    )
    if mode == "same_side_track_speed":
        relation_matches = relation_matches and all(
            audit["guide_equality_leader_matches_same_side_track_by_joint"].values()
        ) and all(audit["guide_equality_speed_ratio_matches_by_joint"].values())
    return relation_matches and no_drive


def trajectory_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = summary["initial_state_after_reset"]["robot"]
    start_x, start_y = float(start["x_m"]), float(start["y_m"])
    states = [row["body_state"] for row in rows]
    xy_path = sum(
        math.hypot(float(right["x_m"]) - float(left["x_m"]),
                   float(right["y_m"]) - float(left["y_m"]))
        for left, right in zip([{"x_m": start_x, "y_m": start_y}] + states[:-1], states)
    )
    final = states[-1]
    whole_proxy_rear_bound_x = [
        float(state["x_m"]) - WHOLE_PROXY_ENVELOPE_RADIUS_M for state in states
    ]
    clear_steps = [
        index for index, rear_bound_x in enumerate(whole_proxy_rear_bound_x, start=1)
        if rear_bound_x > STEP_REAR_EDGE_X_M
    ]
    return {
        "initial_xy_m": [start_x, start_y],
        "final_xy_m": [float(final["x_m"]), float(final["y_m"])],
        "final_displacement_from_reset_xy_m": math.hypot(
            float(final["x_m"]) - start_x, float(final["y_m"]) - start_y
        ),
        "sampled_xy_path_length_m": xy_path,
        "final_state": {key: float(final[key]) for key, _ in STATE_FIELDS},
        "max_abs_pitch_rad": max(abs(float(state["pitch_rad"])) for state in states),
        "max_abs_roll_rad": max(abs(float(state["roll_rad"])) for state in states),
        "min_body_z_m": min(float(state["z_m"]) for state in states),
        "max_body_z_m": max(float(state["z_m"]) for state in states),
        "whole_vehicle_proxy_rear_x_lower_bound_m_at_final": whole_proxy_rear_bound_x[-1],
        "step_rear_edge_x_m": STEP_REAR_EDGE_X_M,
        "whole_vehicle_proxy_conservative_clear_of_step_by_final": (
            whole_proxy_rear_bound_x[-1] > STEP_REAR_EDGE_X_M
        ),
        "first_policy_step_conservatively_clear_of_step_rear": (
            clear_steps[0] if clear_steps else None
        ),
        "whole_vehicle_clearance_bound_note": (
            "Conservative orientation-independent envelope for chassis, support rollers, and front guides; "
            "requires the complete proxy envelope to lie beyond the step rear edge, not merely the root past the front edge."
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_pair_attempt3.py PAIR_DIRECTORY")
    pair_dir = Path(sys.argv[1]).resolve()
    allowed_root = ROOT / "results/isaac_field_diagnosis/front_guide_coupled_step_probe_20261020"
    if not pair_dir.is_relative_to(allowed_root) or pair_dir == allowed_root:
        raise SystemExit(f"Refusing pair directory outside this experiment: {pair_dir}")

    free_dir = pair_dir / "free_run_003"
    coupled_dir = pair_dir / "coupled_run_003"
    free_summary = read_json(free_dir / "summary.json")
    coupled_summary = read_json(coupled_dir / "summary.json")
    free_rows = read_jsonl(free_dir / "policy_step_samples.jsonl")
    coupled_rows = read_jsonl(coupled_dir / "policy_step_samples.jsonl")

    checks: dict[str, bool] = {}
    checks["both_probe_runs_report_passed"] = (
        free_summary.get("status") == "passed" and coupled_summary.get("status") == "passed"
    )
    checks["same_probe_sources_and_effective_source_hashes"] = (
        free_summary.get("source_sha256") == coupled_summary.get("source_sha256")
    )
    checks["same_front_guide_geometry_and_mass"] = (
        no_rotation_mode(free_summary["runtime"]["front_guide_spec"])
        == no_rotation_mode(coupled_summary["runtime"]["front_guide_spec"])
    )
    guide_geometry = no_rotation_mode(free_summary["runtime"]["front_guide_spec"])
    checks["front_guide_spec_matches_requested_trial_geometry"] = guide_geometry == {
        "x_m": 0.42,
        "z_offset_from_support_m": 0.05,
        "radius_m": 0.045,
        "width_m": 0.06,
        "mass_per_wheel_kg": 0.25,
    }
    checks["selected_rotation_modes_are_free_and_coupled"] = (
        free_summary["runtime"]["front_guide_spec"].get("rotation_coupling") == "free"
        and coupled_summary["runtime"]["front_guide_spec"].get("rotation_coupling")
        == "same_side_track_speed"
    )
    checks["runtime_constraints_match_12_vs_14_variant_topology"] = (
        runtime_guide_mode_matches(free_summary, "free")
        and runtime_guide_mode_matches(coupled_summary, "same_side_track_speed")
    )
    checks["zero_guide_actuator_output_and_2nm_track_cap"] = all(
        run["sampling_result"]["checks"]["guide_qfrc_actuator_zero_all_policy_steps"]
        and run["sampling_result"]["checks"]["guide_applied_effort_zero_all_policy_steps"]
        and run["sampling_result"]["checks"]["wheel_effort_cap_2nm_respected_all_policy_steps"]
        and run["sampling_result"]["drive_effort_limit_nm"] == 2.0
        for run in (free_summary, coupled_summary)
    )
    checks["same_reset_initial_state"] = (
        free_summary.get("initial_state_after_reset") == coupled_summary.get("initial_state_after_reset")
    )
    reset_robot = free_summary.get("initial_state_after_reset", {}).get("robot", {})
    checks["reset_start_xy_yaw_matches_trial_contract"] = all(
        math.isclose(float(reset_robot.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1e-9)
        for key, expected in (("x_m", 2.0), ("y_m", 2.0), ("yaw_rad", 0.0))
    )
    checks["same_field_usd_content_hash"] = (
        free_summary.get("field_asset_sha256") == coupled_summary.get("field_asset_sha256")
    )
    checks["same_task_obstacle_and_sampling_contract"] = all(
        (
            no_selected_mode(free_summary["sampling_contract"])
            == no_selected_mode(coupled_summary["sampling_contract"])
        )
        if key == "sampling_contract"
        else free_summary.get(key) == coupled_summary.get(key)
        for key in ("sampling_contract", "task_contract", "obstacle_contract")
    )
    expected_sampling = {
        "policy_steps": EXPECTED_ROWS,
        "physics_dt_s": 0.001,
        "decimation": 50,
        "policy_dt_s": 0.05,
        "fixed_normalized_action_left_right_gimbal_yaw_pitch": [0.05, 0.05, 0.0, 0.0],
        "trained_policy_comparison": False,
        "free_vs_coupled_fixed_action_physics_pair": True,
        "checkpoint_loaded": False,
        "claim_scope": "paired fixed-action physics diagnostics, not policy learning, navigation competence, or a continuous-track model",
    }
    checks["fixed_action_and_timing_match_trial_contract"] = (
        no_selected_mode(free_summary["sampling_contract"]) == expected_sampling
        and no_selected_mode(coupled_summary["sampling_contract"]) == expected_sampling
    )
    expected_task = {
        "mode": "goto",
        "plants": 0,
        "terrain_enabled": False,
        "terrain_amplitude_m": 0.0,
        "rock_density_per_m2": 0.0,
        "start_xy_yaw": [2.0, 2.0, 0.0],
        "goal_xy": [6.0, 2.0],
        "goal_distance_config": None,
        "max_time_s": 12.0,
        "auto_reset": False,
    }
    expected_obstacle = {
        "primitive": "Isaac Lab USD Cuboid via AssetBaseCfg + sim_utils.CuboidCfg",
        "prim_path": "{ENV_REGEX_NS}/DiagnosticStep",
        "size_xyz_m": [0.8, 2.0, 0.06],
        "center_xyz_m": [3.0, 2.0, 0.03],
        "front_edge_x_m": 2.6,
        "collision_material_override": None,
        "collision_note": "Uses the actual configured Newton default shape material; value is recorded from the imported model.",
    }
    checks["task_and_step_match_trial_contract"] = all(
        run.get("task_contract") == expected_task
        and run.get("obstacle_contract") == expected_obstacle
        for run in (free_summary, coupled_summary)
    )
    checks["same_physics_timing_and_drive_caps"] = all(
        free_summary["runtime"].get(key) == coupled_summary["runtime"].get(key)
        for key in (
            "physics_dt_s",
            "decimation",
            "policy_dt_s",
            "max_episode_steps",
            "body_stability_config",
            "drive_interface",
            "drive_velocity_gain",
            "drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s",
            "drive_effort_limit_nm",
            "newton_contact_backend",
            "configured_default_shape_material",
        )
    )
    checks["each_run_has_exactly_200_finite_rows"] = all(
        len(rows) == EXPECTED_ROWS
        and [row.get("policy_step") for row in rows] == list(range(1, EXPECTED_ROWS + 1))
        and all(row.get("finite") is True for row in rows)
        for rows in (free_rows, coupled_rows)
    )
    checks["fixed_actions_match_per_step"] = (
        len(free_rows) == len(coupled_rows) == EXPECTED_ROWS
        and all(
        left.get("normalized_action_left_right_gimbal_yaw_pitch")
        == right.get("normalized_action_left_right_gimbal_yaw_pitch")
        for left, right in zip(free_rows, coupled_rows)
        )
    )
    checks["trajectory_state_values_are_finite"] = all(
        finite_number(row["body_state"][key])
        for rows in (free_rows, coupled_rows)
        for row in rows
        for key, _ in STATE_FIELDS
    )

    delta_rows = []
    rows_usable = (
        checks["each_run_has_exactly_200_finite_rows"]
        and checks["trajectory_state_values_are_finite"]
        and checks["same_probe_sources_and_effective_source_hashes"]
        and checks["same_reset_initial_state"]
        and checks["reset_start_xy_yaw_matches_trial_contract"]
        and checks["same_field_usd_content_hash"]
        and checks["same_task_obstacle_and_sampling_contract"]
        and checks["same_physics_timing_and_drive_caps"]
    )
    for free_row, coupled_row in (zip(free_rows, coupled_rows) if rows_usable else ()):
        free_state = free_row["body_state"]
        coupled_state = coupled_row["body_state"]
        delta = {
            field: float(coupled_state[key]) - float(free_state[key])
            for key, field in STATE_FIELDS
        }
        free_pairs = free_row["contact_snapshot"].get("guide_step_pairs", [])
        coupled_pairs = coupled_row["contact_snapshot"].get("guide_step_pairs", [])
        delta_rows.append({
            "policy_step": free_row["policy_step"],
            "simulation_time_s": free_row["simulation_time_s"],
            "coupled_minus_free_body_state": delta,
            "free_whole_vehicle_proxy_rear_x_lower_bound_m": (
                float(free_state["x_m"]) - WHOLE_PROXY_ENVELOPE_RADIUS_M
            ),
            "coupled_whole_vehicle_proxy_rear_x_lower_bound_m": (
                float(coupled_state["x_m"]) - WHOLE_PROXY_ENVELOPE_RADIUS_M
            ),
            "free_guide_step_pair_count": len(free_pairs),
            "coupled_guide_step_pair_count": len(coupled_pairs),
            "free_guide_qd_rad_s": {
                name: values["velocity_rad_s"]
                for name, values in free_row["front_guide_joints"].items()
            },
            "coupled_guide_qd_rad_s": {
                name: values["velocity_rad_s"]
                for name, values in coupled_row["front_guide_joints"].items()
            },
            "coupled_guide_qd_relation": coupled_row.get("guide_speed_relation"),
        })

    results = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "pairing_scope": (
            "Same-scene fixed-action physical response for free versus same-side-track-speed guide rotation; "
            "not a trained-policy comparison, continuous-track model, maximum step-height test, or force-causality analysis."
        ),
        "checks": checks,
        "run_paths": {"free": str(free_dir), "same_side_track_speed": str(coupled_dir)},
        "reset_initial_state_equal": checks["same_reset_initial_state"],
        "field_usd_sha256_equal": checks["same_field_usd_content_hash"],
        "field_usd_sha256": {
            "free": free_summary.get("field_asset_sha256"),
            "same_side_track_speed": coupled_summary.get("field_asset_sha256"),
        },
        "robot_usd_sha256": {
            "free": free_summary.get("robot_asset_sha256"),
            "same_side_track_speed": coupled_summary.get("robot_asset_sha256"),
            "note": "Separate rotation-mode asset variants are expected to have distinct USD metadata and cache hashes.",
        },
        "guide_spec_sha256": {
            "free": free_summary["runtime"].get("front_guide_spec_sha256"),
            "same_side_track_speed": coupled_summary["runtime"].get("front_guide_spec_sha256"),
        },
        "runtime_constraint_counts": {
            mode: {
                "support_wheel_relations_per_world": run["runtime"]["guide_constraint_audit"][
                    "support_wheel_relations_per_world"
                ],
                "guide_relations_per_world": run["runtime"]["guide_constraint_audit"][
                    "guide_relations_per_world"
                ],
                "total_mimic_relations_per_world": run["runtime"]["guide_constraint_audit"][
                    "total_mimic_relations_per_world"
                ],
                "guide_equality_present_by_joint": run["runtime"]["guide_constraint_audit"][
                    "guide_joint_equality_present_by_joint"
                ],
                "guide_equality_polycoef_by_joint": run["runtime"]["guide_constraint_audit"][
                    "guide_equality_polycoef_by_joint"
                ],
                "guide_independent_actuator_present_by_joint": run["runtime"]["guide_constraint_audit"][
                    "guide_independent_actuator_joint_present_by_joint"
                ],
            }
            for mode, run in (
                ("free", free_summary),
                ("same_side_track_speed", coupled_summary),
            )
        },
        "sampling_result": {
            "free": free_summary.get("sampling_result"),
            "same_side_track_speed": coupled_summary.get("sampling_result"),
        },
        "fixed_action_trajectory_metrics": {
            "free": trajectory_metrics(free_summary, free_rows) if rows_usable else None,
            "same_side_track_speed": (
                trajectory_metrics(coupled_summary, coupled_rows) if rows_usable else None
            ),
        },
        "sample_count": min(len(free_rows), len(coupled_rows)),
        "per_step_deltas_path": str(pair_dir / "paired_trajectory_deltas.jsonl"),
        "paired_delta_semantics": (
            "Per-policy-step coupled-minus-free recorded state and guide/step shape-pair snapshots. "
            "These differences are descriptive and do not infer contact force, sustained support, or climbing cause."
        ),
    }

    comparison_path = pair_dir / "pair_comparison.json"
    deltas_path = pair_dir / "paired_trajectory_deltas.jsonl"
    with comparison_path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with deltas_path.open("x", encoding="utf-8") as stream:
        for row in delta_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": results["status"], "checks": checks}, ensure_ascii=False))
    return 0 if results["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

