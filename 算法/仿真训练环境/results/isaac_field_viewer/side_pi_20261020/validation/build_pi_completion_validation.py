#!/usr/bin/env python3
"""Build a read-only completion record for the fresh side-PI v6 trial.

This is deliberately separate from the legacy minimal-inputs completion
builder. Pairing follows ``scripts/train_isaac_foundation_fresh.py``'s
``_paired_dev_conditions`` contract and the frozen
``scripts/compare_isaac_exploration.py::_pair_initial_conditions`` rules:
per-seed plant geometry hash equality, start/goal equality at 1e-8 absolute
tolerance, and trajectory-start/reset consistency. The protocol's seed lists
are authoritative; the old comparator's 24-row legacy rule is not reused.

The report's ``passed`` status means artifact, source, model, process-exit,
and paired-evaluation integrity checks passed. It does not mean navigation
improved or that field capability has been established.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_side_pi_20261020"
EXPECTED_PROTOCOL_SHA256 = "0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8"
EXPECTED_BASELINE_SHA256 = "bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129"
PAIR_ATOL = 1e-8
PAIR_CONTRACT_FIELDS = (
    "action_config", "checkpoint_action_config",
    "damage_metric", "damage_config", "checkpoint_damage_metric",
    "checkpoint_damage_config", "drive_config", "checkpoint_drive_config",
    "observation_config", "checkpoint_observation_config",
    "goal_distance", "goal_angle_rad", "mode", "deterministic",
    "distance_cost", "trajectory_schema",
)
ROW_METRIC_FIELDS = (
    "raw_return", "success", "damage", "collisions", "final_distance",
    "timed_out", "damaged_episodes",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes its allowed root: {path}") from exc
    return resolved


def _float_vector_equal(left: Any, right: Any, *, atol: float = PAIR_ATOL) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != len(right):
        return False
    try:
        return all(
            math.isfinite(float(a)) and math.isfinite(float(b))
            and abs(float(a) - float(b)) <= atol
            for a, b in zip(left, right)
        )
    except (TypeError, ValueError):
        return False


def _expected_eval_contract(protocol: dict[str, Any], domain: str) -> dict[str, Any]:
    evaluation = protocol.get("evaluation", {})
    domain_cfg = evaluation.get(domain, {})
    return {
        "schema": "isaac-field-heldout-v1",
        "action_config": protocol.get("action_config"),
        "checkpoint_action_config": protocol.get("action_config"),
        "mode": evaluation.get("mode"),
        "deterministic": evaluation.get("deterministic"),
        "distance_cost": evaluation.get("distance_cost"),
        "goal_distance": domain_cfg.get("goal_distance_m"),
        "goal_angle_rad": math.radians(float(domain_cfg["goal_angle_deg"])),
    }


def _report_contract_checks(
    report: dict[str, Any], protocol: dict[str, Any], domain: str,
) -> dict[str, bool]:
    expected = _expected_eval_contract(protocol, domain)
    checks = {
        "heldout_schema": report.get("schema") == expected["schema"],
        "action_config_matches_protocol": (
            report.get("action_config") == expected["action_config"]
            and report.get("checkpoint_action_config") == expected["checkpoint_action_config"]
        ),
        "mode_matches_protocol": report.get("mode") == expected["mode"],
        "deterministic_matches_protocol": report.get("deterministic") == expected["deterministic"],
        "distance_cost_matches_protocol": report.get("distance_cost") == expected["distance_cost"],
        "goal_distance_matches_protocol": _float_vector_equal(
            report.get("goal_distance"), expected["goal_distance"], atol=PAIR_ATOL,
        ),
        "goal_angle_matches_protocol": (
            isinstance(report.get("goal_angle_rad"), (int, float))
            and math.isfinite(float(report["goal_angle_rad"]))
            and math.isclose(float(report["goal_angle_rad"]), expected["goal_angle_rad"], rel_tol=0.0, abs_tol=PAIR_ATOL)
        ),
        "trajectory_schema_present": bool(report.get("trajectory_schema")),
    }
    damage_metric = protocol.get("config", {}).get("damage_metric")
    if damage_metric is not None:
        checks["damage_metric_matches_protocol"] = (
            report.get("damage_metric") == damage_metric
            and report.get("checkpoint_damage_metric") == damage_metric
        )
    return checks


def paired_eval_report(
    name: str,
    baseline: dict[str, Any],
    final: dict[str, Any],
    pairing_record: dict[str, Any],
    expected_seeds: list[int],
    protocol: dict[str, Any],
    domain: str,
    assessment: str,
    baseline_file: str,
    final_file: str,
) -> dict[str, Any]:
    expected = sorted(int(seed) for seed in expected_seeds)
    base_rows_list = baseline.get("rows", [])
    final_rows_list = final.get("rows", [])
    base_rows = {int(row["seed"]): row for row in base_rows_list if "seed" in row}
    final_rows = {int(row["seed"]): row for row in final_rows_list if "seed" in row}
    row_seeds_valid = (
        len(base_rows_list) == len(expected)
        and len(final_rows_list) == len(expected)
        and sorted(base_rows) == expected
        and sorted(final_rows) == expected
        and sorted(int(seed) for seed in baseline.get("seeds", [])) == expected
        and sorted(int(seed) for seed in final.get("seeds", [])) == expected
    )

    baseline_contract_checks = _report_contract_checks(baseline, protocol, domain)
    final_contract_checks = _report_contract_checks(final, protocol, domain)
    same_contract = all(
        baseline.get(key) == final.get(key)
        for key in PAIR_CONTRACT_FIELDS
    )
    same_contract = same_contract and all(baseline_contract_checks.values()) and all(final_contract_checks.values())

    mismatches: list[dict[str, Any]] = []
    per_seed = []
    if row_seeds_valid:
        for seed in expected:
            before = base_rows[seed]
            after = final_rows[seed]
            before_initial = before.get("reset_initial_condition", {})
            after_initial = after.get("reset_initial_condition", {})
            differences: list[str] = []
            before_plant_hash = before_initial.get("plant_xy_sha256")
            after_plant_hash = after_initial.get("plant_xy_sha256")
            if (
                not isinstance(before_plant_hash, str) or not before_plant_hash
                or not isinstance(after_plant_hash, str) or not after_plant_hash
                or before_plant_hash != after_plant_hash
            ):
                differences.append("plant_xy_sha256")
            for key in ("start_pose_xy_yaw", "goal_xy"):
                if not _float_vector_equal(before_initial.get(key), after_initial.get(key)):
                    differences.append(key)
            if before.get("trajectory_start_matches_reset") is not True:
                differences.append("baseline_trajectory_start_matches_reset")
            if after.get("trajectory_start_matches_reset") is not True:
                differences.append("final_trajectory_start_matches_reset")
            if differences:
                mismatches.append({"seed": seed, "differences": differences})
            per_seed.append({
                "seed": seed,
                "initial_condition_matches": not differences,
                "initial_condition_differences": differences,
                "baseline": {key: before.get(key) for key in ROW_METRIC_FIELDS},
                "final": {key: after.get(key) for key in ROW_METRIC_FIELDS},
            })
    else:
        mismatches.append({"differences": ["row_or_top_level_seed_set_mismatch"]})

    runner_pair_valid = (
        pairing_record.get("valid") is True
        and pairing_record.get("paired_seed_count") == len(expected)
        and pairing_record.get("expected_seed_count") == len(expected)
        and pairing_record.get("mismatches") == []
    )
    pairing_valid = row_seeds_valid and same_contract and runner_pair_valid and not mismatches
    return {
        "name": name,
        "baseline_report": baseline_file,
        "final_report": final_file,
        "expected_seeds": expected,
        "baseline_seeds": sorted(base_rows),
        "final_seeds": sorted(final_rows),
        "runner_pairing_record": pairing_record,
        "runner_pairing_valid": runner_pair_valid,
        "row_seed_sets_valid": row_seeds_valid,
        "same_evaluation_contract": same_contract,
        "baseline_contract_checks": baseline_contract_checks,
        "final_contract_checks": final_contract_checks,
        "initial_conditions_valid_with_atol": PAIR_ATOL,
        "initial_condition_mismatches": mismatches,
        "pairing_valid": pairing_valid,
        "assessment_from_runner": assessment,
        "baseline_summary": baseline.get("summary"),
        "final_summary": final.get("summary"),
        "per_seed": per_seed,
    }


def _evaluation_files(trial: Path, comparison: dict[str, Any]) -> dict[tuple[str, str], Path]:
    rows = comparison.get("evaluation_reports", [])
    if not isinstance(rows, list):
        raise ValueError("comparison_summary.evaluation_reports must be a list")
    result: dict[tuple[str, str], Path] = {}
    pattern = re.compile(r"^evaluation_(free|field)_(baseline|final)_.+\.json$")
    for raw in rows:
        relative = Path(str(raw))
        candidate = _within(trial / relative, trial)
        match = pattern.match(candidate.name)
        if not match:
            continue
        key = (match.group(1), match.group(2))
        if key in result:
            raise ValueError(f"duplicate evaluation report for {key}: {raw}")
        result[key] = candidate
    expected_keys = {
        (domain, variant)
        for domain in ("free", "field")
        for variant in ("baseline", "final")
    }
    if set(result) != expected_keys:
        raise ValueError(f"expected one report for each free/field baseline/final pair; got {sorted(result)}")
    return result


def _source_snapshot_check(repo_root: Path, trial: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    snapshot = protocol.get("source_snapshot", {})
    entries = snapshot.get("files", [])
    expected_count = snapshot.get("expected_count", len(entries))
    actual_count = len(entries)
    rows = []
    for entry in entries:
        source_relative = Path(entry["path"])
        snapshot_relative = Path(entry["snapshot_path"])
        source = _within(repo_root / source_relative, repo_root)
        frozen = _within(trial / snapshot_relative, trial)
        expected = str(entry["sha256"]).lower()
        source_hash = sha256(source) if source.is_file() else None
        frozen_hash = sha256(frozen) if frozen.is_file() else None
        rows.append({
            "path": source_relative.as_posix(),
            "snapshot_path": snapshot_relative.as_posix(),
            "expected_sha256": expected,
            "live_sha256": source_hash,
            "snapshot_sha256": frozen_hash,
            "live_matches_protocol": source_hash == expected,
            "snapshot_matches_protocol": frozen_hash == expected,
        })
    count_matches = expected_count == actual_count
    return {
        "expected_count_from_protocol": expected_count,
        "actual_count": actual_count,
        "count_matches_protocol": count_matches,
        "all_live_matches_protocol": count_matches and all(row["live_matches_protocol"] for row in rows),
        "all_snapshots_match_protocol": count_matches and all(row["snapshot_matches_protocol"] for row in rows),
        "files": rows,
    }


def _exit_evidence(args: argparse.Namespace) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if args.runner_exit_file is not None:
        evidence["file"] = str(args.runner_exit_file)
        data = read_json(args.runner_exit_file)
        evidence["record"] = data
        file_runner_code = data.get("runner_exit_code", data.get("process_exit_code", data.get("exit_code")))
        file_exec_code = data.get(
            "exec_session_exit_code",
            data.get("launcher_exit_code", data.get("exec_exit_code")),
        )
    else:
        file_runner_code = None
        file_exec_code = None
    runner_code = args.runner_exit_code if args.runner_exit_code is not None else file_runner_code
    exec_code = args.exec_exit_code if args.exec_exit_code is not None else file_exec_code
    if args.runner_exit_code is not None and file_runner_code is not None and args.runner_exit_code != file_runner_code:
        raise ValueError("--runner-exit-code conflicts with runner exit file")
    if args.exec_exit_code is not None and file_exec_code is not None and args.exec_exit_code != file_exec_code:
        raise ValueError("--exec-exit-code conflicts with runner exit file")
    evidence.update({
        "runner_exit_code": runner_code,
        "exec_session_exit_code": exec_code,
        "exec_session_id": args.exec_session_id,
        "runner_pid": args.runner_pid,
        "runner_exit_zero": runner_code == 0,
        "exec_session_exit_zero": exec_code == 0,
    })
    return evidence


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    trial = _within(args.trial_dir, repo_root)
    protocol_path = _within(args.protocol, repo_root)
    manifest_path = _within(args.manifest, repo_root)
    run_status_path = _within(args.run_status, repo_root)
    baseline_path = _within(args.baseline, repo_root)
    model_validation_path = _within(args.model_validation, repo_root)

    protocol = read_json(protocol_path)
    run_status = read_json(run_status_path)
    manifest = read_json(manifest_path)
    comparison = read_json(trial / "comparison_summary.json")
    protocol_digest = sha256(protocol_path)
    training = protocol.get("training", {})
    timesteps_total = int(training["timesteps_total"])
    stages = training.get("stages", [])
    if not stages:
        raise ValueError("protocol.training.stages is empty")
    final_stage = stages[-1]
    final_relative = Path(f"stage_{len(stages):02d}_{final_stage['name']}.zip")
    final_path = _within(trial / final_relative, trial)
    baseline_relative = Path(protocol.get("initialization", {}).get("baseline_path", ""))
    if not baseline_relative.as_posix():
        raise ValueError("protocol.initialization.baseline_path is missing")
    if _within(trial / baseline_relative, trial) != baseline_path:
        raise ValueError("--baseline does not match protocol.initialization.baseline_path")

    checkpoint_rows = manifest.get("checkpoints", [])
    by_path: dict[str, dict[str, Any]] = {}
    for entry in checkpoint_rows:
        rel = Path(entry["path"]).as_posix()
        if rel in by_path:
            raise ValueError(f"duplicate checkpoint manifest path: {rel}")
        by_path[rel] = entry
    baseline_entry = by_path.get(baseline_relative.as_posix(), {})
    final_entry = by_path.get(final_relative.as_posix(), {})
    baseline_digest = sha256(baseline_path) if baseline_path.is_file() else None
    final_digest = sha256(final_path) if final_path.is_file() else None
    source_snapshot = _source_snapshot_check(repo_root, trial, protocol)

    model_validation = read_json(model_validation_path) if model_validation_path.is_file() else {}
    model_checkpoint = model_validation.get("checkpoint") or {}
    model_baseline = model_validation.get("baseline") or {}
    model_checks = model_validation.get("checks") or {}
    model_checkpoint_contract = model_checkpoint.get("contract") or {}
    model_baseline_contract = model_baseline.get("contract") or {}
    model_checkpoint_optimizer = model_checkpoint.get("optimizer") or {}
    model_baseline_optimizer = model_baseline.get("optimizer") or {}
    model_before = model_validation.get("files_before") or {}
    model_protocol_signature = model_before.get("protocol") or {}
    model_baseline_signature = model_before.get("baseline") or {}
    model_final_signature = model_before.get("checkpoint") or {}

    eval_paths = _evaluation_files(trial, comparison)
    evaluation = protocol.get("evaluation", {})
    dev = {}
    for domain in ("free", "field"):
        label = "free_space" if domain == "free" else "field"
        pairing_key = f"{label}_pairing"
        assessment_key = f"{label}_assessment"
        seeds = evaluation.get("free_space" if domain == "free" else "field", {}).get("seeds", [])
        baseline_report_path = eval_paths[(domain, "baseline")]
        final_report_path = eval_paths[(domain, "final")]
        paired = paired_eval_report(
            label,
            read_json(baseline_report_path),
            read_json(final_report_path),
            comparison.get(pairing_key, {}),
            [int(seed) for seed in seeds],
            protocol,
            "free_space" if domain == "free" else "field",
            comparison.get(assessment_key, "unknown"),
            baseline_report_path.relative_to(trial).as_posix(),
            final_report_path.relative_to(trial).as_posix(),
        )
        dev[label] = paired

    exit_evidence = _exit_evidence(args)
    blind = evaluation.get("blind_seeds", {})
    checks = {
        "protocol_sha_matches_frozen_expected": protocol_digest == EXPECTED_PROTOCOL_SHA256,
        "run_status_protocol_sha_matches_file": run_status.get("protocol_sha256") == protocol_digest,
        "protocol_training_budget_is_positive": timesteps_total > 0,
        "run_completed_at_protocol_budget": (
            run_status.get("status") == "completed"
            and run_status.get("phase") == "completed"
            and run_status.get("num_timesteps") == timesteps_total
            and run_status.get("exit_code") == 0
        ),
        "real_runner_process_exit_zero_supplied": exit_evidence["runner_exit_zero"],
        "real_launcher_exec_exit_zero_supplied": exit_evidence["exec_session_exit_zero"],
        "source_snapshot_count_and_hashes_match_protocol": (
            source_snapshot["count_matches_protocol"]
            and source_snapshot["all_live_matches_protocol"]
            and source_snapshot["all_snapshots_match_protocol"]
        ),
        "checkpoint_manifest_paths_unique_and_nonempty": (
            bool(checkpoint_rows) and len(by_path) == len(checkpoint_rows)
        ),
        "baseline_matches_manifest_protocol_and_expected_sha": (
            baseline_path.is_file()
            and baseline_digest == EXPECTED_BASELINE_SHA256
            and baseline_entry.get("sha256") == baseline_digest
            and baseline_entry.get("num_timesteps") == int(protocol.get("initialization", {}).get("baseline_num_timesteps", 0)) == 0
        ),
        "final_checkpoint_matches_manifest_and_protocol_budget": (
            final_path.is_file()
            and final_entry.get("sha256") == final_digest
            and final_entry.get("num_timesteps") == timesteps_total
        ),
        "cpu_model_validation_passed_for_baseline_and_final": (
            model_validation.get("status") == "passed"
            and model_checkpoint.get("num_timesteps") == timesteps_total
            and model_baseline.get("num_timesteps") == 0
            and Path(str(model_checkpoint.get("path", ""))).resolve() == final_path
            and Path(str(model_baseline.get("path", ""))).resolve() == baseline_path
            and model_checkpoint_contract.get("passed") is True
            and model_baseline_contract.get("passed") is True
            and all(value is True for value in model_checks.values())
        ),
        "model_validation_matches_protocol_baseline_and_final_files": (
            model_protocol_signature.get("sha256") == protocol_digest
            and model_baseline_signature.get("sha256") == baseline_digest
            and model_final_signature.get("sha256") == final_digest
        ),
        "final_adam_is_finite_and_has_positive_steps": (
            model_checkpoint_optimizer.get("class") == "Adam"
            and model_checkpoint_optimizer.get("empty") is False
            and model_checkpoint_optimizer.get("all_finite") is True
            and model_checkpoint_optimizer.get("step_entry_count", 0) > 0
            and model_checkpoint_optimizer.get("min_step", 0) > 0
            and model_checkpoint_optimizer.get("max_step", 0) > 0
        ),
        "baseline_adam_is_empty_and_finite": (
            model_baseline_optimizer.get("class") == "Adam"
            and model_baseline_optimizer.get("empty") is True
            and model_baseline_optimizer.get("all_finite") is True
        ),
        "free_space_reports_are_contract_and_seed_paired": dev["free_space"]["pairing_valid"],
        "field_reports_are_contract_and_seed_paired": dev["field"]["pairing_valid"],
        "blind_seeds_830xx_unused_and_unobserved": (
            blind.get("used") is False and blind.get("observed") is False
        ),
    }
    comparison_has_no_blind_claim = (
        str(comparison.get("blind_830xx", "")).strip().lower()
        == "not evaluated or observed"
    )
    checks["runner_summary_preserves_830xx_blind_boundary"] = comparison_has_no_blind_claim

    return {
        "schema": "isaac-field-fresh-side-pi-v6-completion-validation-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "interpretation": "passed means artifact, source, CPU-model, exit-evidence, and paired-development integrity checks passed; it is not a navigation-improvement or field-capability claim.",
        "trial": trial.name,
        "validation_scope": "Read-only post-run verification. No training, environment creation, simulation, GPU evaluation, or blind-seed evaluation is performed by this builder.",
        "completion": {
            "run_status": run_status.get("status"),
            "run_phase": run_status.get("phase"),
            "num_timesteps": run_status.get("num_timesteps"),
            "protocol_budget": timesteps_total,
            "runner_exit_evidence": exit_evidence,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_digest,
            "expected_sha256": EXPECTED_PROTOCOL_SHA256,
            "input_contract": protocol.get("architecture", {}).get("input_contract"),
            "encoder_config": protocol.get("architecture", {}).get("field_encoder_config"),
            "observation_shapes": protocol.get("architecture", {}).get("observation_shapes"),
            "action_config": protocol.get("action_config"),
            "training": training,
            "evaluation": evaluation,
            "source_snapshot": source_snapshot,
        },
        "initialization": {
            "baseline_path": baseline_relative.as_posix(),
            "baseline_sha256": baseline_digest,
            "expected_baseline_sha256": EXPECTED_BASELINE_SHA256,
            "baseline_num_timesteps": model_baseline.get("num_timesteps"),
            "weight_transfer": protocol.get("initialization", {}).get("weight_transfer"),
        },
        "final_model": {
            "path": final_relative.as_posix(),
            "sha256": final_digest,
            "size_bytes": final_path.stat().st_size if final_path.is_file() else None,
            "num_timesteps": model_checkpoint.get("num_timesteps"),
            "manifest_entry": final_entry,
            "contract": model_checkpoint_contract,
            "optimizer": model_checkpoint_optimizer,
            "weight_changes_vs_zero_step_baseline": model_validation.get("weight_changes_vs_zero_step_baseline"),
            "cpu_model_validation_path": str(model_validation_path),
        },
        "checkpoint_manifest": {
            "path": str(manifest_path),
            "actual_entry_count": len(checkpoint_rows),
            "baseline_entry": baseline_entry,
            "final_entry": final_entry,
        },
        "development_evaluations": {
            "free_space": dev["free_space"],
            "field": dev["field"],
            "dataset_role": evaluation.get("dataset_role"),
            "claim_limit": evaluation.get("claim_limit"),
            "blind_830xx": "not evaluated or observed",
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--trial-dir", type=Path, default=TRIAL)
    parser.add_argument("--protocol", type=Path, default=TRIAL / "protocol.json")
    parser.add_argument("--manifest", type=Path, default=TRIAL / "checkpoint_manifest.json")
    parser.add_argument("--run-status", type=Path, default=TRIAL / "run_status.json")
    parser.add_argument("--baseline", type=Path, default=TRIAL / "untrained_baseline.zip")
    parser.add_argument("--model-validation", type=Path, default=TRIAL / "completion_model_validation.json")
    parser.add_argument("--output", type=Path, default=TRIAL / "completion_validation_pi_v6.json")
    parser.add_argument("--runner-exit-file", type=Path)
    parser.add_argument("--runner-exit-code", type=int)
    parser.add_argument("--exec-exit-code", type=int)
    parser.add_argument("--exec-session-id", type=int)
    parser.add_argument("--runner-pid", type=int)
    args = parser.parse_args()

    try:
        output = _within(args.output, args.repo_root.resolve())
    except Exception as exc:
        parser.error(str(exc))
    if output.exists():
        parser.error(f"refusing to overwrite completion record: {output}")
    try:
        report = build_report(args)
    except Exception as exc:
        report = {
            "schema": "isaac-field-fresh-side-pi-v6-completion-validation-v1",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "checks": report.get("checks", {}), "error": report.get("error")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
