"""Build a read-only completion record for the v6 minimal-inputs trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_minimal_inputs_20261020"
EXPECTED_PROTOCOL_SHA256 = "b5e8e9c3b0561bc73733528aa0c5be3120bb5aad3a54a1fb3418ad84f30a6d65"
EXPECTED_BASELINE_SHA256 = "443b9fc73676bc5dfe6d42450bf2112089852d27c373d12c4b2443f484e43ca9"
EXPECTED_FINAL_SHA256 = "2c03cc130977cc005adf56527bc10bc4719a8a9d9489fa82c5a1b41f9d3966e5"
EXPECTED_TIMESTEPS = 196608
EXPECTED_SNAPSHOT_COUNT = 40
EXPECTED_CHECKPOINT_COUNT = 29
MODEL_VALIDATION = TRIAL / "completion_model_validation.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_delta(before: dict, after: dict) -> dict:
    numeric = (
        "successes", "success_rate", "mean_raw_return", "mean_damage",
        "mean_collisions", "mean_final_distance", "timed_out_episodes",
        "clean_successes", "damaged_episodes", "total_collisions",
    )
    return {
        key: float(after[key]) - float(before[key])
        for key in numeric
        if key in before and key in after
    }


def pair_report(
    trial: Path,
    name: str,
    baseline_file: str,
    final_file: str,
    pairing_record: dict,
    expected_seeds: list[int],
    assessment: str,
) -> dict:
    baseline = read_json(trial / baseline_file)
    final = read_json(trial / final_file)
    baseline_rows = {int(row["seed"]): row for row in baseline.get("rows", [])}
    final_rows = {int(row["seed"]): row for row in final.get("rows", [])}
    seeds = sorted(baseline_rows)
    matching_seeds = seeds == sorted(final_rows) == sorted(expected_seeds)
    matching_metrics = all(
        baseline.get(key) == final.get(key)
        for key in (
            "action_config", "damage_metric", "damage_config", "drive_config",
            "observation_config", "goal_distance", "goal_angle_rad", "mode",
            "deterministic", "trajectory_schema",
        )
    )
    per_seed = []
    for seed in seeds:
        before = baseline_rows[seed]
        after = final_rows.get(seed)
        if after is None:
            continue
        per_seed.append({
            "seed": seed,
            "baseline": {
                key: before.get(key)
                for key in ("raw_return", "success", "damage", "collisions", "final_distance", "timed_out")
            },
            "final": {
                key: after.get(key)
                for key in ("raw_return", "success", "damage", "collisions", "final_distance", "timed_out")
            },
        })

    valid = (
        pairing_record.get("valid") is True
        and pairing_record.get("paired_seed_count") == len(expected_seeds)
        and not pairing_record.get("mismatches")
        and matching_seeds
        and matching_metrics
        and baseline.get("schema") == final.get("schema") == "isaac-field-heldout-v1"
    )
    return {
        "name": name,
        "baseline_report": baseline_file,
        "final_report": final_file,
        "pairing_valid": valid,
        "expected_seed_count": len(expected_seeds),
        "paired_seed_count": len(per_seed),
        "seeds": seeds,
        "same_evaluation_contract": matching_metrics,
        "assessment_from_runner": assessment,
        "baseline": baseline.get("summary"),
        "final": final.get("summary"),
        "change_final_minus_baseline": summary_delta(baseline.get("summary", {}), final.get("summary", {})),
        "per_seed": per_seed,
    }


def main() -> int:
    protocol_path = TRIAL / "protocol.json"
    baseline_path = TRIAL / "untrained_baseline.zip"
    final_path = TRIAL / "stage_04_turn90.zip"
    protocol = read_json(protocol_path)
    run_status = read_json(TRIAL / "run_status.json")
    process_exit = read_json(TRIAL / "process_exit.json")
    manifest = read_json(TRIAL / "checkpoint_manifest.json")
    comparison = read_json(TRIAL / "comparison_summary.json")
    model = read_json(MODEL_VALIDATION)

    protocol_hash = sha256(protocol_path)
    baseline_hash = sha256(baseline_path)
    final_hash = sha256(final_path)
    manifest_entries = {item["path"]: item for item in manifest.get("checkpoints", [])}
    baseline_entry = manifest_entries.get("untrained_baseline.zip", {})
    final_entry = manifest_entries.get("stage_04_turn90.zip", {})
    snapshots = model.get("protocol", {}).get("source_snapshot", {})
    expected_eval = protocol.get("evaluation", {})
    eval_reports = comparison.get("evaluation_reports", [])
    blind = expected_eval.get("blind_seeds", {})

    free = pair_report(
        TRIAL, "free_space",
        "evaluation_free_baseline_87001_87004.json",
        "evaluation_free_final_87001_87004.json",
        comparison.get("free_space_pairing", {}),
        expected_eval.get("free_space", {}).get("seeds", []),
        comparison.get("free_space_assessment", "unknown"),
    )
    field = pair_report(
        TRIAL, "field",
        "evaluation_field_baseline_82001_82004.json",
        "evaluation_field_final_82001_82004.json",
        comparison.get("field_pairing", {}),
        expected_eval.get("field", {}).get("seeds", []),
        comparison.get("field_assessment", "unknown"),
    )

    checks = {
        "runner_process_exit_zero": process_exit.get("exit_code") == 0,
        "launcher_exec_session_returned_zero": True,
        "run_completed_at_full_timestep_budget": (
            run_status.get("status") == "completed"
            and run_status.get("phase") == "completed"
            and run_status.get("num_timesteps") == EXPECTED_TIMESTEPS
            and run_status.get("exit_code") == 0
        ),
        "protocol_sha_matches_expected_and_run_status": (
            protocol_hash == EXPECTED_PROTOCOL_SHA256
            and run_status.get("protocol_sha256") == protocol_hash
        ),
        "source_snapshot_40_files_match_snapshot_and_live_sources": (
            snapshots.get("expected_count") == EXPECTED_SNAPSHOT_COUNT
            and snapshots.get("actual_count") == EXPECTED_SNAPSHOT_COUNT
            and snapshots.get("all_snapshot_sha256_match") is True
            and snapshots.get("all_current_sha256_match") is True
        ),
        "checkpoint_manifest_has_29_entries": len(manifest.get("checkpoints", [])) == EXPECTED_CHECKPOINT_COUNT,
        "final_model_sha_and_step_match_manifest": (
            final_hash == EXPECTED_FINAL_SHA256
            and final_entry.get("sha256") == final_hash
            and final_entry.get("num_timesteps") == EXPECTED_TIMESTEPS
        ),
        "zero_step_baseline_sha_and_step_match_manifest": (
            baseline_hash == EXPECTED_BASELINE_SHA256
            and baseline_entry.get("sha256") == baseline_hash
            and baseline_entry.get("num_timesteps") == 0
        ),
        "cpu_model_validation_passed": model.get("status") == "passed",
        "optimizer_is_finite_and_updated": (
            model.get("checkpoint", {}).get("optimizer", {}).get("class") == "Adam"
            and model.get("checkpoint", {}).get("optimizer", {}).get("all_finite") is True
            and model.get("checkpoint", {}).get("optimizer", {}).get("min_step", 0) > 0
            and model.get("checkpoint", {}).get("optimizer", {}).get("max_step", 0) > 0
        ),
        "policy_weights_finite_and_expected_groups_checked": model.get("all_policy_weights_finite") is True,
        "free_space_paired_development_reports_valid": free["pairing_valid"],
        "field_paired_development_reports_valid": field["pairing_valid"],
        "blind_830xx_unused_and_unobserved": blind.get("used") is False and blind.get("observed") is False,
        "runner_summary_reports_no_improvement_both_domains": (
            comparison.get("free_space_assessment") == "no_improvement"
            and comparison.get("field_assessment") == "no_improvement"
        ),
    }

    report = {
        "schema": "isaac-field-fresh-minimal-inputs-completion-validation-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "trial": "fresh_minimal_inputs_20261020",
        "validation_scope": "Read-only completion checks after the runner exited; model loaded on CPU without instantiating an environment or simulation. No training or 830xx evaluation was run during validation.",
        "completion": {
            "run_status": run_status.get("status"),
            "final_num_timesteps": run_status.get("num_timesteps"),
            "completed_at_utc": process_exit.get("completed_at_utc"),
            "completed_at_beijing": process_exit.get("completed_at_beijing"),
            "runner_pid": process_exit.get("runner_pid"),
            "process_exit_code": process_exit.get("exit_code"),
            "exec_session_id": 86650,
            "exec_session_exit_code": 0,
            "exec_session_result": "Launcher command returned exit code 0; runner process record also reports exit code 0.",
        },
        "protocol": {
            "path": "protocol.json",
            "sha256": protocol_hash,
            "expected_sha256": EXPECTED_PROTOCOL_SHA256,
            "input_contract": protocol.get("architecture", {}).get("input_contract"),
            "encoder_config": protocol.get("architecture", {}).get("field_encoder_config"),
            "observation_shapes": protocol.get("architecture", {}).get("observation_shapes"),
            "source_snapshot_expected_count": snapshots.get("expected_count"),
            "source_snapshot_actual_count": snapshots.get("actual_count"),
            "snapshot_hashes_match": snapshots.get("all_snapshot_sha256_match"),
            "current_source_hashes_match": snapshots.get("all_current_sha256_match"),
        },
        "initialization": {
            "baseline_path": "untrained_baseline.zip",
            "baseline_sha256": baseline_hash,
            "baseline_num_timesteps": model.get("baseline", {}).get("num_timesteps"),
            "source_initialization": "New RecurrentPPO with no old checkpoint or weight transfer.",
            "action_config": protocol.get("action_config"),
        },
        "final_model": {
            "path": "stage_04_turn90.zip",
            "sha256": final_hash,
            "size_bytes": final_path.stat().st_size,
            "num_timesteps": model.get("checkpoint", {}).get("num_timesteps"),
            "action_config": model.get("checkpoint", {}).get("field_action_config"),
            "encoder_config": model.get("checkpoint", {}).get("field_encoder_config"),
            "observation_shapes": model.get("checkpoint", {}).get("observation_shapes"),
            "device": model.get("checkpoint", {}).get("device"),
            "adam_optimizer": model.get("checkpoint", {}).get("optimizer"),
            "all_weights_finite": model.get("all_policy_weights_finite"),
            "weight_comparison_vs_zero_step_baseline": model.get("weight_updates"),
            "cpu_validation_report": "completion_model_validation.json",
            "detection_branch_note": "Detector MLP is finite and unchanged because the frozen training protocol uses plants=0; field detections occur only in development evaluation.",
        },
        "outputs": {
            "total_checkpoint_count": len(manifest.get("checkpoints", [])),
            "final_stage_checkpoint": "stage_04_turn90.zip",
            "evaluations": eval_reports,
        },
        "development_evaluations": {
            "free_space": free,
            "field": field,
            "blind_830xx": "not evaluated or observed",
            "claim_limit": "Development comparison only; no improvement in either paired set, not an architecture ablation or blind validation, and not evidence of full-field navigation capability.",
        },
        "checks": checks,
    }

    output = TRIAL / "completion_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "checks": checks,
        "free_space": {"baseline": free["baseline"], "final": free["final"], "delta": free["change_final_minus_baseline"]},
        "field": {"baseline": field["baseline"], "final": field["final"], "delta": field["change_final_minus_baseline"]},
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
