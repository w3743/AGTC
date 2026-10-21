"""Read-only CPU validation for the fresh side-PI v6 training baseline/checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
import zipfile


EXPECTED_INPUT = "sensor-minimal-v6"
EXPECTED_ENCODER = "depth64-lidar-single1d-minimal-v3"
EXPECTED_DRIVE = "independent_left_right_track_speed_explicit_pi_v6"
EXPECTED_KI = 0.1
EXPECTED_STEPS = 8192
EXPECTED_SOURCE_COUNT = 41
EXPECTED_SHAPES = {
    "lidar": [1, 400],
    "depth": [1, 160, 88],
    "detections": [16, 8],
    "state": [33],
    "feedback": [37],
}
KNOWN_OLD_TRIAL_HASHES = {
    "results/isaac_field_improvement/fresh_minimal_inputs_20261020/protocol.json":
        "b5e8e9c3b0561bc73733528aa0c5be3120bb5aad3a54a1fb3418ad84f30a6d65",
    "results/isaac_field_improvement/fresh_minimal_inputs_20261020/untrained_baseline.zip":
        "443b9fc73676bc5dfe6d42450bf2112089852d27c373d12c4b2443f484e43ca9",
    "results/isaac_field_improvement/fresh_minimal_inputs_20261020/stage_04_turn90.zip":
        "2c03cc130977cc005adf56527bc10bc4719a8a9d9489fa82c5a1b41f9d3966e5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def zip_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes expected root: {path}") from exc
    return resolved


def source_snapshot_check(repo_root: Path, trial_dir: Path, protocol: dict) -> dict:
    entries = protocol.get("source_snapshot", {}).get("files", [])
    checked = []
    for entry in entries:
        relative = Path(entry["path"])
        snapshot_relative = Path(entry["snapshot_path"])
        source = _inside(repo_root / relative, repo_root)
        snapshot = _inside(trial_dir / snapshot_relative, trial_dir)
        expected = str(entry["sha256"]).lower()
        current_hash = sha256(source) if source.is_file() else None
        snapshot_hash = sha256(snapshot) if snapshot.is_file() else None
        checked.append({
            "path": relative.as_posix(),
            "snapshot_path": snapshot_relative.as_posix(),
            "expected_sha256": expected,
            "snapshot_sha256": snapshot_hash,
            "current_sha256": current_hash,
            "snapshot_matches_protocol": snapshot_hash == expected,
            "current_matches_protocol": current_hash == expected,
        })
    count_ok = len(entries) == EXPECTED_SOURCE_COUNT
    return {
        "expected_count": EXPECTED_SOURCE_COUNT,
        "actual_count": len(entries),
        "all_snapshot_sha256_match": count_ok and all(row["snapshot_matches_protocol"] for row in checked),
        "all_current_sha256_match": count_ok and all(row["current_matches_protocol"] for row in checked),
        "files": checked,
    }


def tensor_group(name: str) -> str | None:
    if ".feedback.0." in name:
        return "feedback_fc"
    if ".boxes.0." in name or ".boxes.2." in name:
        return "detection_fc"
    if ".proprioception.0." in name:
        return "state_fc"
    if ".laser.0." in name or ".laser.2." in name:
        return "lidar_conv1d"
    if ".depth.0." in name or ".depth.2." in name:
        return "depth_conv2d"
    if "lstm_actor." in name or "lstm_critic." in name:
        return "actor_critic_lstm"
    if "action_net." in name:
        return "action_net"
    return None


def optimizer_summary(model, torch) -> dict:
    states = getattr(model.policy.optimizer, "state", {})
    steps = []
    finite_state = True
    for state in states.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if hasattr(step, "item") else int(step))
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                finite_state = finite_state and bool(torch.isfinite(value).all().item())
    return {
        "class": type(model.policy.optimizer).__name__,
        "state_parameter_count": len(states),
        "step_entry_count": len(steps),
        "min_step": min(steps) if steps else 0,
        "max_step": max(steps) if steps else 0,
        "all_finite": finite_state,
        "empty": not states,
    }


def state_dict_cpu(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}


def compare_weights(baseline_state: dict, checkpoint_state: dict, torch) -> dict:
    families = (
        "feedback_fc", "state_fc", "lidar_conv1d", "depth_conv2d", "detection_fc",
        "actor_critic_lstm", "action_net",
    )
    groups: dict[str, list[str]] = {family: [] for family in families}
    unclassified = []
    for name in sorted(set(baseline_state) & set(checkpoint_state)):
        family = tensor_group(name)
        (groups[family] if family is not None else unclassified).append(name)

    report = {}
    for family in families:
        names = groups[family]
        changed_tensors = changed_elements = 0
        max_abs_delta = 0.0
        all_finite = bool(names)
        shapes_match = bool(names)
        for name in names:
            before, after = baseline_state[name], checkpoint_state[name]
            if tuple(before.shape) != tuple(after.shape):
                shapes_match = False
                continue
            if before.is_floating_point() or after.is_floating_point():
                all_finite = all_finite and bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
            delta = after - before
            count = int(torch.count_nonzero(delta).item())
            if count:
                changed_tensors += 1
                changed_elements += count
                max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        detector_branch = family == "detection_fc"
        report[family] = {
            "parameter_tensor_count": len(names),
            "changed_tensor_count": changed_tensors,
            "changed_element_count": changed_elements,
            "max_abs_delta": max_abs_delta,
            "all_finite": all_finite,
            "shapes_match": shapes_match,
            "update_expected": not detector_branch,
            "plant_free_detection_update_note": (
                "plants=0 training can leave detector weights unchanged; report actual change without treating it as failure"
                if detector_branch else None
            ),
            "passed": bool(names) and all_finite and shapes_match and (detector_branch or changed_tensors > 0),
            "parameter_names": names,
        }
    report["unclassified"] = {"parameter_names": unclassified, "count": len(unclassified)}
    return report


def protected_old_trial_check(repo_root: Path, reference_path: Path) -> dict:
    reference = read_json(reference_path)
    rows = []
    for entry in reference.get("files", []):
        relative = Path(entry["path"])
        path = _inside(repo_root / relative, repo_root)
        actual = signature(path) if path.is_file() else None
        expected_sha = str(entry["sha256"]).lower()
        known_sha = entry.get("known_sha256")
        expected_known = KNOWN_OLD_TRIAL_HASHES.get(relative.as_posix())
        rows.append({
            "path": relative.as_posix(),
            "expected_sha256": expected_sha,
            "actual": actual,
            "known_historical_sha256": expected_known,
            "reference_known_sha256": known_sha,
            "matches_reference": actual is not None and actual["sha256"] == expected_sha,
            "matches_known_historical_sha256": expected_known is None or (
                actual is not None and actual["sha256"] == expected_known
                and (known_sha is None or str(known_sha).lower() == expected_known)
            ),
        })
    ok = bool(rows) and all(row["matches_reference"] and row["matches_known_historical_sha256"] for row in rows)
    return {"reference_path": str(reference_path), "protected_file_count": len(rows), "all_match": ok, "files": rows}


def _model_attr(model, metadata: dict, name: str):
    value = getattr(model, name, None)
    return value if value is not None else metadata.get(name)


def model_contract(model, metadata: dict, protocol: dict, torch) -> dict:
    spaces = getattr(model.observation_space, "spaces", {})
    shapes = {key: list(space.shape) for key, space in spaces.items()}
    architecture = protocol.get("architecture", {})
    encoder_expected = architecture.get("field_encoder_config", {})
    action_expected = protocol.get("action_config", {})
    encoder = _model_attr(model, metadata, "field_encoder_config")
    action = _model_attr(model, metadata, "field_action_config")
    drive = _model_attr(model, metadata, "field_drive_config")
    gain = drive.get("integral_gain_nm_per_proxy_wheel_per_rad_s_s") if isinstance(drive, dict) else None
    checks = {
        "input_schema": architecture.get("input_contract") == EXPECTED_INPUT,
        "encoder_schema": isinstance(encoder, dict) and encoder.get("schema") == EXPECTED_ENCODER,
        "encoder_matches_protocol": encoder == encoder_expected,
        "action_matches_protocol": action == action_expected,
        "drive_interface_pi_v6": isinstance(drive, dict) and drive.get("interface") == EXPECTED_DRIVE,
        "drive_integral_gain_0_1": isinstance(gain, (int, float)) and math.isclose(float(gain), EXPECTED_KI, rel_tol=0.0, abs_tol=1e-9),
        "drive_metadata_matches_between_checkpoint_and_protocol_runtime": (
            not protocol.get("runtime_contract", {}).get("drive_config")
            or drive == protocol["runtime_contract"]["drive_config"]
        ),
        "observation_shapes": shapes == EXPECTED_SHAPES and shapes == architecture.get("observation_shapes"),
        "feature_fusion_dim_192": isinstance(encoder, dict) and encoder.get("features_dim") == 192,
        "policy_weights_finite": all(
            bool(torch.isfinite(value).all().item())
            for value in model.policy.state_dict().values()
            if value.is_floating_point()
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "field_input_contract": architecture.get("input_contract"),
        "field_encoder_config": encoder,
        "field_action_config": action,
        "field_drive_config": drive,
        "observation_shapes": shapes,
    }


def run_validation(args) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    root = args.repo_root.resolve()
    trial = _inside(args.trial_dir, root)
    protocol_path = _inside(args.protocol, root)
    manifest_path = _inside(args.manifest, root)
    status_path = _inside(args.run_status, root)
    baseline_path = _inside(args.baseline, root)
    checkpoint_path = None if args.baseline_only else _inside(args.checkpoint, root)
    if args.baseline_only and args.checkpoint is not None:
        raise ValueError("--checkpoint cannot be used with --baseline-only")
    if not args.baseline_only and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --baseline-only is set")

    initial_signatures = {
        "protocol": signature(protocol_path),
        "baseline": signature(baseline_path),
    }
    if checkpoint_path is not None:
        initial_signatures["checkpoint"] = signature(checkpoint_path)
    protocol = read_json(protocol_path)
    status = read_json(status_path)
    manifest = read_json(manifest_path)
    protocol_digest = initial_signatures["protocol"]["sha256"]
    manifest_by_path = {item["path"]: item for item in manifest.get("checkpoints", [])}
    baseline_rel = baseline_path.relative_to(trial).as_posix()
    baseline_manifest = manifest_by_path.get(baseline_rel)
    checkpoint_rel = checkpoint_path.relative_to(trial).as_posix() if checkpoint_path else None
    checkpoint_manifest = manifest_by_path.get(checkpoint_rel) if checkpoint_rel else None

    expected_digest = args.expected_protocol_sha256.lower()
    expected_baseline_digest = args.expected_baseline_sha256.lower()
    baseline_before = RecurrentPPO.load(str(baseline_path), env=None, device="cpu")
    baseline_metadata = zip_metadata(baseline_path)
    baseline_contract = model_contract(baseline_before, baseline_metadata, protocol, torch)
    baseline_optimizer = optimizer_summary(baseline_before, torch)
    baseline_state = state_dict_cpu(baseline_before)

    checkpoint = checkpoint_metadata = checkpoint_contract = checkpoint_optimizer = checkpoint_state = None
    weight_changes = None
    if checkpoint_path is not None:
        checkpoint = RecurrentPPO.load(str(checkpoint_path), env=None, device="cpu")
        checkpoint_metadata = zip_metadata(checkpoint_path)
        checkpoint_contract = model_contract(checkpoint, checkpoint_metadata, protocol, torch)
        checkpoint_optimizer = optimizer_summary(checkpoint, torch)
        checkpoint_state = state_dict_cpu(checkpoint)
        weight_changes = compare_weights(baseline_state, checkpoint_state, torch)

    snapshot_check = source_snapshot_check(root, trial, protocol)
    old_trial_check = protected_old_trial_check(root, args.protected_reference.resolve())
    final_signatures = {
        "protocol": signature(protocol_path),
        "baseline": signature(baseline_path),
    }
    if checkpoint_path is not None:
        final_signatures["checkpoint"] = signature(checkpoint_path)
    protocol_stable = final_signatures["protocol"] == initial_signatures["protocol"]
    model_files_stable = final_signatures == initial_signatures
    run_status_protocol_ok = status.get("protocol_sha256") == protocol_digest == expected_digest
    manifest_checks = {
        "baseline_entry_present": bool(baseline_manifest),
        "baseline_entry_sha_matches": bool(baseline_manifest) and baseline_manifest.get("sha256") == initial_signatures["baseline"]["sha256"],
        "baseline_manifest_step_zero": bool(baseline_manifest) and baseline_manifest.get("num_timesteps") == 0,
        "checkpoint_entry_present": checkpoint_path is None or bool(checkpoint_manifest),
        "checkpoint_entry_sha_matches": checkpoint_path is None or (
            bool(checkpoint_manifest) and checkpoint_manifest.get("sha256") == initial_signatures["checkpoint"]["sha256"]
        ),
        "checkpoint_manifest_step_matches": checkpoint_path is None or (
            bool(checkpoint_manifest) and checkpoint_manifest.get("num_timesteps") == EXPECTED_STEPS
        ),
    }
    baseline_checks = {
        "baseline_zero_step": baseline_before.num_timesteps == 0,
        "baseline_sha_matches_expected": initial_signatures["baseline"]["sha256"] == expected_baseline_digest,
        "baseline_metadata_contract": baseline_contract["passed"],
        "baseline_adam_is_empty_and_finite": baseline_optimizer["class"] == "Adam" and baseline_optimizer["empty"] and baseline_optimizer["all_finite"],
        "baseline_file_manifest_hash": manifest_checks["baseline_entry_sha_matches"],
    }
    checks = {
        "protocol_sha_matches_expected_and_run_status": run_status_protocol_ok,
        "protocol_unchanged_during_read": protocol_stable,
        "41_source_snapshot_and_live_hashes_match": snapshot_check["all_snapshot_sha256_match"] and snapshot_check["all_current_sha256_match"],
        "baseline_zero_step_metadata_adam_finite_and_manifest": all(baseline_checks.values()),
        "old_trial_original_hashes_preserved": old_trial_check["all_match"],
        "cpu_only_no_cuda": str(baseline_before.device) == "cpu" and not torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    checkpoint_optimizer_ok = False
    if checkpoint_path is not None:
        checkpoint_optimizer_ok = (
            checkpoint_optimizer["class"] == "Adam"
            and not checkpoint_optimizer["empty"]
            and checkpoint_optimizer["step_entry_count"] > 0
            and checkpoint_optimizer["min_step"] > 0
            and checkpoint_optimizer["max_step"] > 0
            and checkpoint_optimizer["all_finite"]
        )
        checks.update({
            "checkpoint_8192_step_metadata_and_finite": (
                checkpoint.num_timesteps == EXPECTED_STEPS and checkpoint_contract["passed"]
            ),
            "baseline_and_checkpoint_contracts_match": (
                baseline_contract["field_encoder_config"] == checkpoint_contract["field_encoder_config"]
                and baseline_contract["field_action_config"] == checkpoint_contract["field_action_config"]
                and baseline_contract["field_drive_config"] == checkpoint_contract["field_drive_config"]
                and baseline_contract["observation_shapes"] == checkpoint_contract["observation_shapes"]
            ),
            "checkpoint_adam_has_finite_positive_steps": checkpoint_optimizer_ok,
            "checkpoint_manifest_hash_and_step": all(
                manifest_checks[key] for key in (
                    "checkpoint_entry_present", "checkpoint_entry_sha_matches", "checkpoint_manifest_step_matches",
                )
            ),
            "baseline_and_checkpoint_unchanged_during_read": model_files_stable,
            "all_expected_training_branches_finite_and_changed": all(
                value["passed"] for key, value in weight_changes.items() if key != "unclassified"
            ),
        })

    report = {
        "schema": "fresh-side-pi-v6-readonly-checkpoint-validation-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "CPU-only RecurrentPPO checkpoint inspection; no environment, simulation, GPU evaluation, or training was created.",
        "trial_dir": str(trial),
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_digest,
            "expected_sha256": expected_digest,
            "run_status_protocol_sha256": status.get("protocol_sha256"),
            "source_snapshot": snapshot_check,
        },
        "manifest": {
            "path": str(manifest_path),
            "baseline_entry": baseline_manifest,
            "checkpoint_entry": checkpoint_manifest,
            "checks": manifest_checks,
        },
        "files_before": initial_signatures,
        "files_after": final_signatures,
        "baseline": {
            "path": str(baseline_path),
            "num_timesteps": baseline_before.num_timesteps,
            "contract": baseline_contract,
            "optimizer": baseline_optimizer,
        },
        "checkpoint": None if checkpoint_path is None else {
            "path": str(checkpoint_path),
            "num_timesteps": checkpoint.num_timesteps,
            "contract": checkpoint_contract,
            "optimizer": checkpoint_optimizer,
        },
        "weight_changes_vs_zero_step_baseline": weight_changes,
        "training_context": {
            "plants": protocol.get("config", {}).get("plants"),
            "detection_branch_note": (
                "The detector branch is finite; whether its weights changed is reported directly. "
                "With plants=0 training observations, no detector positives may be present."
            ),
        },
        "protected_old_trial": old_trial_check,
        "checks": checks,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--protected-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = None
    try:
        report = run_validation(args)
    except Exception as exc:
        report = {
            "schema": "fresh-side-pi-v6-readonly-checkpoint-validation-v1",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "checks": report.get("checks", {})}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
