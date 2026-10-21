#!/usr/bin/env python3
"""Verify exact pairing and summarize the capped-P versus per-side-PI probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_finite_tree(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{location}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite value at {location}: {value!r}")


def flatten_numeric(value, location):
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(flatten_numeric(child, f"{location}[{index}]"))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric data at {location}, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite numeric data at {location}: {value!r}")
    return [float(value)]


def max_abs_difference(left, right, location="state"):
    left_values = flatten_numeric(left, f"{location}.left")
    right_values = flatten_numeric(right, f"{location}.right")
    if len(left_values) != len(right_values):
        raise ValueError(f"numeric shape mismatch at {location}")
    return max((abs(a - b) for a, b in zip(left_values, right_values)), default=0.0)


def compare_initial_state_fields(state_p, state_pi, tolerances, location="state"):
    deltas, passed = {}, {}
    for field, tolerance in tolerances.items():
        delta = max_abs_difference(state_p[field], state_pi[field], f"{location}.{field}")
        deltas[field] = delta
        passed[field] = delta <= float(tolerance)
    return deltas, passed


def evaluate_brake_tail(
    rows, *, tail_count, command_target, speed_limit_mps, yaw_limit_rad_s
):
    tail = sorted(rows, key=lambda row: row["step"])[-int(tail_count):]
    if len(tail) != int(tail_count):
        return {"tail_policy_steps": len(tail), "passed": False, "reason": "incomplete tail"}
    command_zero = all(row["track_command_mps"] == list(command_target) for row in tail)
    speed_ok = all(abs(row["actual_v_w"][0]) < float(speed_limit_mps) for row in tail)
    yaw_ok = all(abs(row["actual_v_w"][1]) < float(yaw_limit_rad_s) for row in tail)
    return {
        "tail_policy_steps": len(tail),
        "track_command_exact_zero": command_zero,
        "max_abs_actual_v_mps": max(abs(row["actual_v_w"][0]) for row in tail),
        "max_abs_actual_w_rad_s": max(abs(row["actual_v_w"][1]) for row in tail),
        "all_abs_actual_v_below_limit": speed_ok,
        "all_abs_actual_w_below_limit": yaw_ok,
        "passed": bool(command_zero and speed_ok and yaw_ok),
    }


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            assert_finite_tree(row, f"{path.name}:{line_number}")
            rows.append(row)
    return rows


def check_source_freeze(root: Path, freeze_path: Path, runs, failures):
    if not freeze_path.is_file():
        failures.append(f"source freeze file missing: {freeze_path}")
        return None
    digests = {}
    for line_number, line in enumerate(freeze_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            failures.append(f"invalid source freeze line {line_number}")
            continue
        expected, relative = parts[0].lower(), parts[1].strip()
        file_path = root / relative
        if not file_path.is_file():
            failures.append(f"frozen source missing: {relative}")
            continue
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        digests[relative] = actual
        if actual != expected:
            failures.append(f"source freeze mismatch: {relative}")
    if not digests:
        failures.append("source freeze manifest contains no source files")
    for run in runs:
        for relative, expected in run.get("source_sha256", {}).items():
            if relative not in digests:
                failures.append(f"summary source is not listed in freeze manifest: {relative}")
            file_path = root / relative
            actual = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.is_file() else None
            if actual != expected:
                failures.append(f"summary source hash no longer matches: {relative}")
    return hashlib.sha256(freeze_path.read_bytes()).hexdigest()


def compare_confirmation(root: Path, out_path: Path, protocol_path: Path, freeze_path: Path):
    out_path = out_path.resolve()
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite {out_path}")
    protocol = read_json(protocol_path)
    p_path = root / "confirmation_p" / "counterfactual_summary.json"
    pi_path = root / "confirmation_pi" / "counterfactual_summary.json"
    p, pi = read_json(p_path), read_json(pi_path)
    assert_finite_tree(p, "capped_p_summary")
    assert_finite_tree(pi, "pi_summary")
    failures = []
    p_ki = p["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"]
    pi_ki = pi["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"]
    if p_ki != 0.0 or pi_ki != 0.1:
        failures.append(f"expected Ki 0.0/0.1, got {p_ki}/{pi_ki}")
    expected_arch = "side_integral_plus_native_per_wheel_P"
    if any(run["counterfactual_controller"].get("controller_architecture") != expected_arch for run in (p, pi)):
        failures.append("controller architecture metadata mismatch")
    same_sources = p.get("source_sha256") == pi.get("source_sha256")
    if not same_sources:
        failures.append("P and PI source_sha256 maps differ")
    same_drive = p.get("base_drive_config") == pi.get("base_drive_config")
    if not same_drive:
        failures.append("base v5 drive configuration differs")
    same_environment = p.get("environment") == pi.get("environment")
    if not same_environment:
        failures.append("environment/reset configuration differs")
    freeze_failures_before = len(failures)
    freeze_sha = check_source_freeze(Path(__file__).resolve().parents[4], freeze_path, (p, pi), failures)
    source_freeze_matches = len(failures) == freeze_failures_before

    expected_names = protocol["case_names"]
    p_cases = {row["case"]: row for row in p["cases"]}
    pi_cases = {row["case"]: row for row in pi["cases"]}
    if p.get("case_count") != 4 or pi.get("case_count") != 4:
        failures.append("confirmation requires exactly four cases per run")
    if set(p_cases) != set(expected_names) or set(pi_cases) != set(expected_names):
        failures.append("confirmation case names do not match the four-case protocol")
    if set(p_cases) != set(pi_cases):
        failures.append("P and PI case identities differ")
    expected_timing = protocol["timing_s"]
    for label, run in (("P", p), ("PI", pi)):
        environment = run["environment"]
        for field in ("settle_s", "drive_s", "brake_observation_s", "policy_dt_s", "physics_dt_s"):
            if not math.isclose(
                float(environment[field]), float(expected_timing[field if field != "settle_s" else "p_only_settle"]),
                rel_tol=0.0, abs_tol=1e-9,
            ):
                failures.append(f"{label}: unexpected {field}={environment[field]}")
        if int(environment["decimation"]) != int(expected_timing["decimation_expected"]):
            failures.append(f"{label}: unexpected decimation={environment['decimation']}")

    tolerances = {
        key: value
        for key, value in protocol["initial_state_pairing_tolerances"].items()
        if key != "comparison_rule"
    }
    max_by_field = {field: 0.0 for field in tolerances}
    max_by_case = {}
    paired = []
    for case in expected_names:
        if case not in p_cases or case not in pi_cases:
            continue
        p_row, pi_row = p_cases[case], pi_cases[case]
        if p_row["policy_normalized_action"] != pi_row["policy_normalized_action"]:
            failures.append(f"{case}: policy action mismatch")
        state_p, state_pi = p_row["initial_state_after_settle"], pi_row["initial_state_after_settle"]
        case_deltas = {}
        for field, tolerance in tolerances.items():
            field_deltas, field_passed = compare_initial_state_fields(
                state_p, state_pi, {field: tolerance}, location=case
            )
            delta = field_deltas[field]
            case_deltas[field] = delta
            max_by_field[field] = max(max_by_field[field], delta)
            if not field_passed[field]:
                failures.append(f"{case}: {field} delta {delta} exceeds {tolerance}")
        max_by_case[case] = case_deltas
        paired.append({
            "case": case,
            "action": p_row["policy_normalized_action"],
            "initial_state_max_abs_delta": case_deltas,
            "drive": {
                "capped_p_end_actual_v_w": p_row["drive"]["end_actual_v_w"],
                "pi_end_actual_v_w": pi_row["drive"]["end_actual_v_w"],
                "capped_p_last_1s_qd_tracking_error": p_row["drive"]["last_1s_mean_abs_qd_tracking_error_rad_s"],
                "pi_last_1s_qd_tracking_error": pi_row["drive"]["last_1s_mean_abs_qd_tracking_error_rad_s"],
                "capped_p_yaw_sign_matches": p_row["drive"]["yaw_sign_matches_raw_command_last_1s"],
                "pi_yaw_sign_matches": pi_row["drive"]["yaw_sign_matches_raw_command_last_1s"],
            },
        })

    tail_count = int(protocol["braking_stop_gate"]["expected_tail_policy_steps"])
    braking_protocol = protocol["braking_stop_gate"]
    effort_limit = float(braking_protocol["actual_applied_effort_nm_per_proxy_wheel_limit"])
    effort_tolerance = float(braking_protocol["actual_applied_effort_tolerance_nm"])
    brake_gate_results = {}
    effort_maxima = {}
    finite_policy_counts = {}
    for label, run in (("confirmation_p", p), ("confirmation_pi", pi)):
        sample_path = Path(run["sample_outputs"]["policy_steps_path"])
        substep_path = Path(run["sample_outputs"]["physics_substeps_path"])
        if not sample_path.is_absolute():
            sample_path = root / sample_path
        if not substep_path.is_absolute():
            substep_path = root / substep_path
        policy_rows, substep_rows = read_jsonl(sample_path), read_jsonl(substep_path)
        finite_policy_counts[label] = len(policy_rows)
        if len(policy_rows) != run.get("sample_count"):
            failures.append(f"{label}: policy sample count mismatch")
        expected_substeps = int(run.get("expected_physics_substeps", -1))
        if len(substep_rows) != expected_substeps:
            failures.append(f"{label}: physics substep sample count mismatch")
        all_efforts = [
            float(value)
            for row in substep_rows
            for value in row["applied_effort_nm"]
        ]
        max_effort = max((abs(value) for value in all_efforts), default=float("inf"))
        effort_maxima[label] = max_effort
        if max_effort > effort_limit + effort_tolerance:
            failures.append(
                f"{label}: physics substep effort {max_effort} exceeds "
                f"{effort_limit} + {effort_tolerance} Nm"
            )
        monitor = run["physics_substep_applied_effort_monitor"]
        if monitor["checked_physics_substeps"] != expected_substeps:
            failures.append(f"{label}: Newton physics iteration coverage mismatch")
        if run["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"] == 0.0:
            if any(
                row["pi_side_integral_feedforward_nm_per_proxy_wheel"] != [0.0, 0.0]
                or any(value != 0.0 for value in row["pi_integral_feedforward_broadcast_nm_per_wheel"])
                for row in policy_rows
            ):
                failures.append(f"{label}: Ki=0 generated integral/feedforward")

        for case in (name for name in expected_names if name.startswith("brake_")):
            rows = sorted(
                (row for row in policy_rows if row["case"] == case and row["phase"] == "brake"),
                key=lambda row: row["step"],
            )
            case_result = evaluate_brake_tail(
                rows,
                tail_count=tail_count,
                command_target=braking_protocol["track_command_mps_exact"],
                speed_limit_mps=braking_protocol["actual_v_mps_abs_strict_limit"],
                yaw_limit_rad_s=braking_protocol["actual_w_rad_s_abs_strict_limit"],
            )
            brake_gate_results[f"{label}/{case}"] = case_result
            if not case_result.get("track_command_exact_zero", False):
                failures.append(f"{label}/{case}: track command not exactly zero during final 0.3 s")
            if not case_result.get("all_abs_actual_v_below_limit", False):
                failures.append(f"{label}/{case}: actual speed stop threshold failed")
            if not case_result.get("all_abs_actual_w_below_limit", False):
                failures.append(f"{label}/{case}: actual yaw-rate stop threshold failed")

    gates = {
        "same_source_hashes": same_sources,
        "source_freeze_matches_current": source_freeze_matches,
        "same_v5_drive_config": same_drive,
        "same_environment": same_environment,
        "four_case_actions_match": not any("policy action mismatch" in item or "case names" in item for item in failures),
        "initial_state_within_protocol_tolerance": not any("delta" in item and "exceeds" in item for item in failures),
        "finite_policy_and_substep_logs": True,
        "all_substeps_within_2nm_plus_1e_5": all(value <= 2.0 + 1e-5 for value in effort_maxima.values()),
        "braking_stop_gate_all_cases": all(row["passed"] for row in brake_gate_results.values()),
    }
    result = {
        "schema": "isaac-field-pi-counterfactual-confirmation-result-v1",
        "status": "passed" if not failures else "failed",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_freeze_path": str(freeze_path.resolve()),
        "source_freeze_sha256": freeze_sha,
        "source_snapshot_manifest": str((Path(__file__).resolve().parent / "source_snapshot" / "source_snapshot_manifest.json")),
        "source_snapshot_manifest_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent / "source_snapshot" / "source_snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        "run_summaries": {"capped_p": str(p_path), "pi": str(pi_path)},
        "initial_state_pairing": {
            "protocol_tolerances": tolerances,
            "max_abs_difference_by_field_across_cases": max_by_field,
            "max_abs_difference_by_case": max_by_case,
            "passed": gates["initial_state_within_protocol_tolerance"],
        },
        "braking_stop_gate": brake_gate_results,
        "physics_effort_gate": {
            "limit_nm_per_proxy_wheel": effort_limit,
            "tolerance_nm": effort_tolerance,
            "max_abs_effort_nm_by_run": effort_maxima,
        },
        "policy_sample_count_by_run": finite_policy_counts,
        "gates": gates,
        "gate_failures": failures,
        "paired_cases": paired,
        "interpretation_guard": [
            "This is a runtime-only v5 simulator confirmation, not a navigation-success trial.",
            "Ki=0.1 remains an uncalibrated assumption; no tuning or threshold changes are permitted.",
            "Native per-wheel IdealPD remains active; the two per-side integrators are diagnostic only.",
            "Simulator applied effort is not measured motor torque or a validated hardware PI loop.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"CONFIRMATION_RESULT_WRITTEN {out_path}")
    print(f"CONFIRMATION_STATUS {result['status']}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-mode", choices=("exact", "confirmation-v1"), default="exact")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--source-freeze", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.protocol_mode == "confirmation-v1":
        if args.protocol is None or args.source_freeze is None:
            parser.error("confirmation-v1 requires --protocol and --source-freeze")
        return compare_confirmation(root, args.output, args.protocol, args.source_freeze)
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
