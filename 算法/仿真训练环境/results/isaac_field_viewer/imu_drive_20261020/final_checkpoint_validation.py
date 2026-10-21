"""Read-only CPU validation of the final 196608-step IMU-drive checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_imu_drive_20261020"
EXPECTED_INPUT_SCHEMA = "sensor-imu-drive-v5"
EXPECTED_PROTOCOL_PREFIX = "7f42"
EXPECTED_PROTOCOL_SUFFIX = "a7a39"
EXPECTED_BASELINE_SHA256 = "afb53b326b8fadc7928363a34a63ea5585ff07b8d746c3909ea0784f23751129"
CHECKPOINT_RELATIVE = "stage_04_turn90.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_from_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def tensor_group(name: str) -> str | None:
    if ".feedback.0." in name:
        return "feedback_fc"
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


def optimizer_summary(model) -> dict:
    states = getattr(model.policy.optimizer, "state", {})
    steps = []
    finite_state = True
    for state in states.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if hasattr(step, "item") else int(step))
        for value in state.values():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                import torch

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


def observation_shapes(model) -> dict[str, list[int]]:
    spaces = model.observation_space.spaces
    return {key: list(space.shape) for key, space in spaces.items()}


def source_snapshot_check(protocol: dict) -> dict:
    entries = protocol.get("source_snapshot", {}).get("files", [])
    result = []
    for entry in entries:
        source = ROOT / entry["path"]
        snapshot = TRIAL / entry["snapshot_path"]
        expected = entry["sha256"]
        source_hash = sha256(source) if source.is_file() else None
        snapshot_hash = sha256(snapshot) if snapshot.is_file() else None
        result.append({
            "path": entry["path"],
            "snapshot_path": entry["snapshot_path"],
            "expected_sha256": expected,
            "snapshot_sha256": snapshot_hash,
            "current_sha256": source_hash,
            "snapshot_matches_protocol": snapshot_hash == expected,
            "current_matches_protocol": source_hash == expected,
        })
    return {
        "expected_count": 40,
        "actual_count": len(entries),
        "all_snapshot_sha256_match": len(entries) == 40 and all(item["snapshot_matches_protocol"] for item in result),
        "all_current_sha256_match": len(entries) == 40 and all(item["current_matches_protocol"] for item in result),
        "files": result,
    }


def compare_groups(baseline_state: dict, checkpoint_state: dict, torch) -> dict:
    groups: dict[str, list[str]] = {}
    for name in sorted(set(baseline_state) & set(checkpoint_state)):
        family = tensor_group(name)
        if family is not None:
            groups.setdefault(family, []).append(name)

    report = {}
    for family in (
        "feedback_fc", "state_fc", "lidar_conv1d", "depth_conv2d",
        "actor_critic_lstm", "action_net",
    ):
        names = groups.get(family, [])
        changed_tensors = 0
        changed_elements = 0
        max_abs_delta = 0.0
        all_finite = bool(names)
        shapes_match = bool(names)
        for name in names:
            before = baseline_state[name].detach().cpu()
            after = checkpoint_state[name].detach().cpu()
            if tuple(before.shape) != tuple(after.shape):
                shapes_match = False
                continue
            all_finite = all_finite and bool(torch.isfinite(before).all().item()) and bool(torch.isfinite(after).all().item())
            delta = after - before
            if torch.count_nonzero(delta).item():
                changed_tensors += 1
                changed_elements += int(torch.count_nonzero(delta).item())
                max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        report[family] = {
            "parameter_tensor_count": len(names),
            "changed_tensor_count": changed_tensors,
            "changed_element_count": changed_elements,
            "max_abs_delta": max_abs_delta,
            "all_finite": all_finite,
            "shapes_match": shapes_match,
            "parameter_names": names,
            "passed": bool(names) and all_finite and shapes_match and changed_tensors > 0,
        }
    return report


def run_validation(output: Path) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    protocol_path = TRIAL / "protocol.json"
    protocol_hash_before = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint_path = TRIAL / CHECKPOINT_RELATIVE
    baseline_path = TRIAL / "untrained_baseline.zip"
    manifest = json.loads((TRIAL / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest_by_path = {item["path"]: item for item in manifest.get("checkpoints", [])}
    checkpoint_manifest = manifest_by_path.get(CHECKPOINT_RELATIVE)
    baseline_manifest = manifest_by_path.get("untrained_baseline.zip")
    before_files = {
        "checkpoint": {"size": checkpoint_path.stat().st_size, "mtime_ns": checkpoint_path.stat().st_mtime_ns, "sha256": sha256(checkpoint_path)},
        "baseline": {"size": baseline_path.stat().st_size, "mtime_ns": baseline_path.stat().st_mtime_ns, "sha256": sha256(baseline_path)},
    }

    expected_architecture = protocol.get("architecture", {})
    expected_encoder = expected_architecture.get("field_encoder_config")
    expected_shapes = expected_architecture.get("observation_shapes")
    expected_action = protocol.get("action_config")
    protocol_digest_valid = (
        protocol_hash_before.startswith(EXPECTED_PROTOCOL_PREFIX)
        and protocol_hash_before.endswith(EXPECTED_PROTOCOL_SUFFIX)
    )
    snapshot_check = source_snapshot_check(protocol)

    baseline = RecurrentPPO.load(str(baseline_path), env=None, device="cpu")
    checkpoint = RecurrentPPO.load(str(checkpoint_path), env=None, device="cpu")
    baseline_state = {key: value.detach().cpu().clone() for key, value in baseline.policy.state_dict().items()}
    checkpoint_state = {key: value.detach().cpu().clone() for key, value in checkpoint.policy.state_dict().items()}

    baseline_meta = json_from_zip(baseline_path)
    checkpoint_meta = json_from_zip(checkpoint_path)
    baseline_shapes = observation_shapes(baseline)
    checkpoint_shapes = observation_shapes(checkpoint)
    expected_shapes_lists = {key: list(value) for key, value in expected_shapes.items()}
    baseline_optimizer = optimizer_summary(baseline)
    checkpoint_optimizer = optimizer_summary(checkpoint)
    groups = compare_groups(baseline_state, checkpoint_state, torch)
    all_weights_finite = all(
        bool(torch.isfinite(value).all().item())
        for state in (baseline_state, checkpoint_state)
        for value in state.values()
        if value.is_floating_point()
    )

    after_files = {
        "checkpoint": {"size": checkpoint_path.stat().st_size, "mtime_ns": checkpoint_path.stat().st_mtime_ns, "sha256": sha256(checkpoint_path)},
        "baseline": {"size": baseline_path.stat().st_size, "mtime_ns": baseline_path.stat().st_mtime_ns, "sha256": sha256(baseline_path)},
    }
    protocol_hash_after = sha256(protocol_path)

    checkpoint_sha_matches = bool(checkpoint_manifest) and before_files["checkpoint"]["sha256"] == checkpoint_manifest["sha256"]
    baseline_sha_matches = bool(baseline_manifest) and before_files["baseline"]["sha256"] == baseline_manifest["sha256"] == EXPECTED_BASELINE_SHA256
    metadata_consistent = (
        baseline.num_timesteps == 0
        and checkpoint.num_timesteps == 196608
        and baseline.field_action_config == checkpoint.field_action_config == expected_action
        and baseline.field_encoder_config == checkpoint.field_encoder_config == expected_encoder
        and baseline_shapes == checkpoint_shapes == expected_shapes_lists
        and protocol.get("architecture", {}).get("input_contract") == EXPECTED_INPUT_SCHEMA
        and baseline_meta.get("field_action_config") == checkpoint_meta.get("field_action_config") == expected_action
        and baseline_meta.get("field_encoder_config") == checkpoint_meta.get("field_encoder_config") == expected_encoder
    )
    files_stable = before_files == after_files
    protocol_stable = protocol_hash_before == protocol_hash_after
    optimizer_ok = (
        baseline_optimizer["class"] == "Adam"
        and baseline_optimizer["empty"]
        and checkpoint_optimizer["class"] == "Adam"
        and not checkpoint_optimizer["empty"]
        and checkpoint_optimizer["step_entry_count"] > 0
        and checkpoint_optimizer["min_step"] > 0
        and checkpoint_optimizer["all_finite"]
    )

    checks = {
        "protocol_sha_matches_expected_prefix_suffix": protocol_digest_valid,
        "protocol_unchanged_during_read": protocol_stable,
        "checkpoint_manifest_sha_matches_file": checkpoint_sha_matches,
        "baseline_manifest_and_expected_sha_match_file": baseline_sha_matches,
        "checkpoint_and_baseline_unchanged_during_read": files_stable,
        "baseline_is_zero_step_and_checkpoint_is_196608": baseline.num_timesteps == 0 and checkpoint.num_timesteps == 196608,
        "metadata_contracts_and_observation_shapes_match": metadata_consistent,
        "optimizer_is_adam_with_positive_steps_and_finite_state": optimizer_ok,
        "all_policy_weights_finite": all_weights_finite,
        "source_snapshot_40_files_match_protocol_and_current_source": snapshot_check["all_snapshot_sha256_match"] and snapshot_check["all_current_sha256_match"],
        **{f"{family}_finite_and_updated": value["passed"] for family, value in groups.items()},
        "cpu_only_no_cuda": str(checkpoint.device) == "cpu" and not torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    report = {
        "schema": "imu-drive-final-readonly-validation-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "read_only_scope": "Loaded only fixed baseline and final 196608-step checkpoint; did not instantiate an environment or simulation.",
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_hash_before,
            "expected_prefix": EXPECTED_PROTOCOL_PREFIX,
            "expected_suffix": EXPECTED_PROTOCOL_SUFFIX,
            "input_schema": protocol.get("architecture", {}).get("input_contract"),
            "source_snapshot": snapshot_check,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "manifest": checkpoint_manifest,
            "file_before": before_files["checkpoint"],
            "file_after": after_files["checkpoint"],
            "num_timesteps": checkpoint.num_timesteps,
            "device": str(checkpoint.device),
            "field_action_config": checkpoint.field_action_config,
            "field_encoder_config": checkpoint.field_encoder_config,
            "observation_shapes": checkpoint_shapes,
            "optimizer": checkpoint_optimizer,
        },
        "baseline": {
            "path": str(baseline_path),
            "manifest": baseline_manifest,
            "file_before": before_files["baseline"],
            "file_after": after_files["baseline"],
            "num_timesteps": baseline.num_timesteps,
            "field_action_config": baseline.field_action_config,
            "field_encoder_config": baseline.field_encoder_config,
            "observation_shapes": baseline_shapes,
            "optimizer": baseline_optimizer,
        },
        "weight_updates": groups,
        "all_policy_weights_finite": all_weights_finite,
        "metadata_consistent": metadata_consistent,
        "files_stable_during_read": files_stable,
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    try:
        report = run_validation(args.output)
    except Exception as exc:
        report = {
            "schema": "imu-drive-final-readonly-validation-v1",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "checks": report.get("checks", {}), "weight_updates": {key: {"changed_tensor_count": value.get("changed_tensor_count"), "changed_element_count": value.get("changed_element_count"), "all_finite": value.get("all_finite"), "passed": value.get("passed")} for key, value in report.get("weight_updates", {}).items()}}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

