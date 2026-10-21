"""Pure-CPU summary of already completed side-PI confirmation raw data."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
CONF = ROOT / "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual"
OUT = Path(__file__).resolve().parent / "evidence_metrics_v3.json"
WHEEL_RADIUS_M = 0.045
TRACK_SEPARATION_M = 0.41
EFFORT_CAP_NM = 2.0
EXPECTED_SUBSTEPS_PER_POLICY_STEP = 10


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_tree(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def indexed_cases(value) -> dict[str, dict]:
    if isinstance(value, dict):
        return value
    return {row["case"]: row for row in value}


def last_mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def raw_case_metrics(summary: dict, policy_rows: list[dict], effort_rows: list[dict]) -> dict:
    cases = indexed_cases(summary["cases"])
    result = {}
    for case_name, case_summary in cases.items():
        drive_rows = sorted(
            (row for row in policy_rows if row["case"] == case_name and row["phase"] == "drive"),
            key=lambda row: row["step"],
        )
        brake_rows = sorted(
            (row for row in policy_rows if row["case"] == case_name and row["phase"] == "brake"),
            key=lambda row: row["step"],
        )
        tail = drive_rows[-20:]
        left_qd = last_mean(tail, "left_actual_qd_mean_rad_s")
        right_qd = last_mean(tail, "right_actual_qd_mean_rad_s")
        left_target = last_mean(tail, "left_target_mean_rad_s")
        right_target = last_mean(tail, "right_target_mean_rad_s")
        if tail:
            actual_w = mean(float(row["actual_v_w"][1]) for row in tail)
            nominal_w = mean(float(row["nominal_v_w"][1]) for row in tail)
            yaw_delta = float(tail[-1]["yaw_unwrapped_rad"]) - float(tail[0]["yaw_unwrapped_rad"])
        else:
            nominal_w = float(case_summary.get("nominal_yaw_rate_raw_action_rad_s", 0.0))
            yaw_delta = None
        left_effort = last_mean(tail, "left_applied_effort_mean_nm")
        right_effort = last_mean(tail, "right_applied_effort_mean_nm")
        surface_left = left_qd * WHEEL_RADIUS_M if left_qd is not None else None
        surface_right = right_qd * WHEEL_RADIUS_M if right_qd is not None else None
        no_slip_w = (
            (surface_right - surface_left) / TRACK_SEPARATION_M
            if surface_left is not None and surface_right is not None else None
        )
        ratio_to_command = abs(actual_w / nominal_w) if nominal_w and actual_w is not None else None
        ratio_to_roll_kinematics = abs(actual_w / no_slip_w) if no_slip_w and actual_w is not None else None

        case_effort_rows = [row for row in effort_rows if row["case"] == case_name]
        max_effort = max(
            (abs(float(value)) for row in case_effort_rows for value in row["applied_effort_nm"]),
            default=0.0,
        )
        over_cap = sum(
            abs(float(value)) > EFFORT_CAP_NM + 1e-5
            for row in case_effort_rows for value in row["applied_effort_nm"]
        )
        case_effort_finite = all(finite_tree(row["applied_effort_nm"]) for row in case_effort_rows)
        phase_counts = {}
        for row in case_effort_rows:
            phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1

        brake_tail = brake_rows[-6:]
        brake = None
        if brake_rows:
            brake = {
                "policy_samples": len(brake_rows),
                "tail_samples": len(brake_tail),
                "last_track_command_mps": brake_rows[-1]["track_command_mps"],
                "tail_max_abs_actual_v_mps": max(abs(float(row["actual_v_w"][0])) for row in brake_tail),
                "tail_max_abs_actual_w_rad_s": max(abs(float(row["actual_v_w"][1])) for row in brake_tail),
                "end_actual_v_w": brake_rows[-1]["actual_v_w"],
                "end_wheel_qd_rad_s": brake_rows[-1]["wheel_actual_qd_rad_s"],
                "end_integral_ff_nm_per_wheel": brake_rows[-1]["pi_side_integral_feedforward_nm_per_proxy_wheel"],
            }

        result[case_name] = {
            "nominal_yaw_rate_rad_s": nominal_w,
            "last_1s_actual_yaw_rate_rad_s": actual_w,
            "last_1s_yaw_tracking_fraction": ratio_to_command,
            "drive_yaw_delta_from_raw_rad": yaw_delta,
            "last_1s_side_mean_qd_rad_s": {"left": left_qd, "right": right_qd},
            "last_1s_side_mean_target_qd_rad_s": {"left": left_target, "right": right_target},
            "last_1s_mean_abs_qd_error_rad_s": {
                "left": abs(left_target - left_qd) if left_target is not None and left_qd is not None else None,
                "right": abs(right_target - right_qd) if right_target is not None and right_qd is not None else None,
            },
            "wheel_surface_speed_from_side_mean_qd_mps": {"left": surface_left, "right": surface_right},
            "no_slip_yaw_from_proxy_wheel_qd_rad_s": no_slip_w,
            "body_yaw_vs_proxy_wheel_no_slip_ratio": ratio_to_roll_kinematics,
            "last_1s_side_mean_applied_pd_plus_ff_nm": {"left": left_effort, "right": right_effort},
            "max_abs_applied_effort_nm_physics_substeps": max_effort,
            "physics_substep_rows": len(case_effort_rows),
            "physics_substep_phase_counts": phase_counts,
            "physics_substep_row_count_matches_phase_duration": len(case_effort_rows) == (
                1800 if not brake_rows else 2800
            ),
            "physics_substep_over_cap_count": over_cap,
            "physics_substep_values_finite": case_effort_finite,
            "brake": brake,
        }

    return {
        "controller": summary.get("counterfactual_controller"),
        "base_drive_config": summary.get("base_drive_config"),
        "source_sha256": summary.get("source_sha256"),
        "all_policy_rows_finite": all(finite_tree(row) for row in policy_rows),
        "all_effort_rows_finite": all(finite_tree(row) for row in effort_rows),
        "case_count": len(cases),
        "cases": result,
    }


def old_contact_context() -> dict:
    summary_path = ROOT / "results/isaac_field_diagnosis/ground_contact/ground_contact_tail_pose_summary_20261017.json"
    summary = read_json(summary_path)
    compact = {}
    for name, summary_case in summary["cases"].items():
        raw_path = summary_path.parent / summary_case["source_json"]
        raw = read_json(raw_path)
        raw_cases = indexed_cases(raw.get("cases", {}))
        case = raw_cases.get(name, {})
        phases = case.get("phases", {})
        compact[name] = {
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
            "case_definition": raw.get("setup", {}).get("case_definitions", {}).get(name),
            "setup_physics": raw.get("setup", {}).get("physics"),
            "ground": case.get("ground"),
            "drive_velocity_gain": case.get("drive_velocity_gain"),
            "raw_status": case.get("status"),
            "tail_pose_phase_summaries": summary_case.get("phases", {}),
            "raw_phase_contact_summaries": {
                phase_name: {
                    "max_native_contact_pairs": phase.get("summary", {}).get("max_native_contact_pairs"),
                    "max_wheel_ground_contact_pairs": phase.get("summary", {}).get("max_wheel_ground_contact_pairs"),
                    "max_abs_selected_wheel_contact_force_N": phase.get("summary", {}).get("max_abs_selected_wheel_contact_force_N"),
                    "max_abs_root_forward_velocity_mps": phase.get("summary", {}).get("max_abs_root_forward_velocity_mps"),
                    "max_abs_public_vs_body_spin_rad_s": phase.get("summary", {}).get("max_abs_public_vs_body_spin_rad_s"),
                }
                for phase_name, phase in phases.items()
            },
        }
    return {
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "cases": compact,
        "interpretation": "prior v5-era/earlier contact experiment; useful for deciding whether to repeat a contact-path probe, not an exact confirmation-state comparison",
    }


def old_skid_context() -> dict:
    path = ROOT / "results/isaac_field_diagnosis/skid_response/wheel_response.json"
    doc = read_json(path)
    compact = {}
    for name, case in doc.get("cases", {}).items():
        summary = case.get("summary", {})
        compact[name] = {
            "action": case.get("action"),
            "root_yaw_delta_rad": summary.get("root_yaw_delta_rad"),
            "root_yaw_rate_peak_abs_radps": summary.get("root_yaw_rate_peak_abs_radps"),
            "body_command_v_w_final": summary.get("body_command_v_w_final"),
            "body_actual_v_w_final": summary.get("body_actual_v_w_final"),
            "mean_wheel_target_radps_left": summary.get("mean_wheel_target_radps_left"),
            "mean_wheel_target_radps_right": summary.get("mean_wheel_target_radps_right"),
            "mean_wheel_actual_radps_left": summary.get("mean_wheel_actual_radps_left"),
            "mean_wheel_actual_radps_right": summary.get("mean_wheel_actual_radps_right"),
            "mean_abs_yaw_command_error_radps": summary.get("mean_abs_yaw_command_error_radps"),
        }
    return {"path": path.relative_to(ROOT).as_posix(), "setup": doc.get("setup"), "cases": compact}


def source_facts() -> dict:
    assets_path = ROOT / "isaac_field/assets.py"
    env_path = ROOT / "isaac_field/env.py"
    task_path = ROOT / "isaac_field/task.py"
    return {
        "files": {
            path.relative_to(ROOT).as_posix(): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for path in (assets_path, env_path, task_path)
        },
        "asset_numbers": {
            "total_mass_kg": 30.0,
            "chassis_mass_kg": 26.4,
            "proxy_wheel_mass_kg_each": 0.25,
            "proxy_wheel_count_total": 14,
            "proxy_wheel_count_per_side": 7,
            "proxy_wheel_radius_m": WHEEL_RADIUS_M,
            "proxy_wheel_width_m": 0.06,
            "track_contact_length_m": 0.595,
            "track_center_separation_m": TRACK_SEPARATION_M,
            "wheel_center_z_m": -0.105,
            "nominal_root_z_m_in_confirmation": 0.15396,
        },
        "controller_and_contact_numbers": {
            "track_max_speed_mps": 15.0 / 3.6,
            "track_acceleration_mps2": 1.5,
            "policy_dt_s": 0.05,
            "physics_dt_s": 0.005,
            "decimation": 10,
            "native_wheel_p_gain_nm_per_rad_s": 0.04,
            "native_wheel_effort_cap_nm_each": EFFORT_CAP_NM,
            "newton_default_shape_mu": 0.8,
            "newton_margin_m": 0.002,
            "newton_gap_m": 0.005,
        },
        "modeled_geometry": {
            "roller_cylinder_axis": "Y",
            "wheel_revolute_joint_axis": "Y",
            "contact_shape": "14 individually rotating Y-axis cylinders; no continuous belt",
            "terrain_contact_material": "no explicit per-track/ground physics material in the local authored mesh; Newton default shape mu=.8 is configured",
        },
        "important_code_locations": {
            "mass_and_wheel_dimensions": "isaac_field/assets.py:32-63",
            "wheel_positions_and_spacing": "isaac_field/assets.py:65-103",
            "cylinder_axis_and_collision": "isaac_field/assets.py:186-198,283-306",
            "default_newton_contact_mu": "isaac_field/env.py:76-87",
            "wheel_actuator_limits": "isaac_field/env.py:163-171",
            "wheel_target_conversion_and_nominal_vw": "isaac_field/env.py:389-405",
            "wheel_ground_effort_to_motor_equivalent": "isaac_field/env.py:440-463,611-666",
        },
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol = read_json(CONF / "confirmation_protocol.json")
    runs = {}
    for name in ("confirmation_p", "confirmation_pi"):
        folder = CONF / name
        summary = read_json(folder / "counterfactual_summary.json")
        policy_rows = read_jsonl(folder / "policy_step_samples.jsonl")
        effort_rows = read_jsonl(folder / "physics_substep_effort.jsonl")
        runs[name] = raw_case_metrics(summary, policy_rows, effort_rows)

    p_run = runs["confirmation_p"]
    pi_run = runs["confirmation_pi"]
    pair_metadata = {
        "source_sha256_maps_identical": p_run["source_sha256"] == pi_run["source_sha256"],
        "base_drive_config_identical": p_run["base_drive_config"] == pi_run["base_drive_config"],
        "case_names_match": set(p_run["cases"]) == set(pi_run["cases"]),
        "p_controller": p_run["controller"],
        "pi_controller": pi_run["controller"],
        "p_all_raw_finite": p_run["all_policy_rows_finite"] and p_run["all_effort_rows_finite"],
        "pi_all_raw_finite": pi_run["all_policy_rows_finite"] and pi_run["all_effort_rows_finite"],
        "p_all_physics_substeps_within_2nm_plus_1e5": all(
            row["physics_substep_over_cap_count"] == 0 and row["physics_substep_values_finite"]
            for row in p_run["cases"].values()
        ),
        "pi_all_physics_substeps_within_2nm_plus_1e5": all(
            row["physics_substep_over_cap_count"] == 0 and row["physics_substep_values_finite"]
            for row in pi_run["cases"].values()
        ),
    }

    report = {
        "schema": "side-pi-proxy-track-physics-gap-analysis-v1",
        "scope": "read-only local-source and already completed confirmation raw analysis; no simulator, GPU, evaluation, FPS, training, or 830xx files accessed",
        "protocol": {
            "path": (CONF / "confirmation_protocol.json").relative_to(ROOT).as_posix(),
            "sha256": __import__("hashlib").sha256((CONF / "confirmation_protocol.json").read_bytes()).hexdigest(),
            "case_names": protocol["case_names"],
            "environment": protocol["environment"],
            "braking_gate": protocol["braking_stop_gate"],
        },
        "source_facts": source_facts(),
        "pair_metadata_and_cap_checks": pair_metadata,
        "confirmation_runs": runs,
        "prior_contact_diagnostic": old_contact_context(),
        "prior_skid_diagnostic": old_skid_context(),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "protocol_sha256": report["protocol"]["sha256"],
        "runs": {
            name: {
                case: {
                    "yaw_command": values["nominal_yaw_rate_rad_s"],
                    "yaw_actual_last1": values["last_1s_actual_yaw_rate_rad_s"],
                    "yaw_tracking_fraction": values["last_1s_yaw_tracking_fraction"],
                    "body_vs_wheel_roll_ratio": values["body_yaw_vs_proxy_wheel_no_slip_ratio"],
                    "wheel_qd": values["last_1s_side_mean_qd_rad_s"],
                    "effort_max": values["max_abs_applied_effort_nm_physics_substeps"],
                }
                for case, values in run["cases"].items()
            }
            for name, run in runs.items()
        },
        "prior_contact_default_or_mujoco_case_context": {
            name: {
                "case_definition": case.get("case_definition"),
                "setup_physics": case.get("setup_physics"),
                "ground": case.get("ground"),
                "raw_forward": case.get("raw_phase_contact_summaries", {}).get("forward"),
                "tail_forward_pose": case.get("tail_pose_phase_summaries", {}).get("forward"),
            }
            for name, case in report["prior_contact_diagnostic"]["cases"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
