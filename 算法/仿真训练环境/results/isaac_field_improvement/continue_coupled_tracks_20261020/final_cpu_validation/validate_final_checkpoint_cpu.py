"""Read-only CPU integrity check for the completed coupled-track continuation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[4]
TRIAL = ROOT / "results/isaac_field_improvement/continue_coupled_tracks_20261020"
OUTPUT_DIR = TRIAL / "final_cpu_validation"
EXPECTED_PROTOCOL_SHA256 = "fe54ce6233e10705a9bbc393784eeca8cbe4d459ac467af1b2fd3656fb4baa18"
EXPECTED_BASELINE_SHA256 = "6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5"
BASELINE_RELATIVE = "parent_final_baseline.zip"
CHECKPOINT_RELATIVE = "stage_04_turn90.zip"
BASELINE_TIMESTEPS = 196_608
FINAL_TIMESTEPS = 393_216
SOURCE_COUNT = 43
INITIAL_ADAM_PARAMETERS = 37
INITIAL_ADAM_STEP = 7607


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_summary(model, torch) -> dict:
    optimizer = model.policy.optimizer
    states = getattr(optimizer, "state", {})
    steps, finite, floating_count = [], True, 0
    for state in states.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if hasattr(step, "item") else int(step))
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                floating_count += 1
                finite = finite and bool(torch.isfinite(value).all().item())
    return {
        "class": type(optimizer).__name__, "state_parameter_count": len(states),
        "step_entry_count": len(steps), "min_step": min(steps) if steps else 0,
        "max_step": max(steps) if steps else 0,
        "all_steps_positive": bool(steps) and all(value > 0 for value in steps),
        "floating_state_tensor_count": floating_count, "all_finite": finite,
        "empty": not states,
    }


def source_hashes(protocol: dict) -> dict:
    rows = []
    for entry in protocol.get("source_snapshot", {}).get("files", []):
        expected = entry["sha256"]
        live, snapshot = ROOT / entry["path"], TRIAL / entry["snapshot_path"]
        live_sha = sha256(live) if live.is_file() else None
        snapshot_sha = sha256(snapshot) if snapshot.is_file() else None
        rows.append({"path": entry["path"], "expected_sha256": expected,
                     "live_sha256": live_sha, "snapshot_sha256": snapshot_sha,
                     "matches": live_sha == expected and snapshot_sha == expected})
    return {
        "count": len(rows), "matching_count": sum(row["matches"] for row in rows),
        "all_match": len(rows) == SOURCE_COUNT and all(row["matches"] for row in rows),
        "files": rows,
    }


def compare_weights(baseline, checkpoint, torch) -> dict:
    before = {key: value.detach().cpu() for key, value in baseline.policy.state_dict().items()}
    after = {key: value.detach().cpu() for key, value in checkpoint.policy.state_dict().items()}
    same_keys = set(before) == set(after)
    same_shapes = same_keys and all(before[key].shape == after[key].shape for key in before)
    finite = all(bool(torch.isfinite(value).all().item())
                 for state in (before, after) for value in state.values()
                 if value.is_floating_point())
    changed_tensors = changed_elements = compared_elements = 0
    max_abs_delta = 0.0
    changed_names, unchanged_names = [], []
    if same_shapes:
        for name in sorted(before):
            left, right = before[name], after[name]
            compared_elements += left.numel()
            changed = int(torch.count_nonzero(right - left).item()) if left.is_floating_point() else int(
                torch.count_nonzero(right != left).item())
            changed_elements += changed
            changed_tensors += int(changed > 0)
            (changed_names if changed else unchanged_names).append(name)
            if changed and left.is_floating_point():
                max_abs_delta = max(max_abs_delta, float((right - left).abs().max().item()))
    return {
        "changed_tensors": changed_tensors, "unchanged_tensors": len(unchanged_names),
        "changed_elements": changed_elements, "compared_elements": compared_elements,
        "max_abs_delta": max_abs_delta, "all_weights_finite": finite,
        "same_keys": same_keys, "same_shapes": same_shapes,
        "changed_tensor_names": changed_names, "unchanged_tensor_names": unchanged_names,
    }


def _checkpoint_data(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def run() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    protocol_path = TRIAL / "protocol.json"
    baseline_path = TRIAL / BASELINE_RELATIVE
    checkpoint_path = TRIAL / CHECKPOINT_RELATIVE
    manifest_path = TRIAL / "checkpoint_manifest.json"
    status_path = TRIAL / "run_status.json"
    training_exit_path = TRIAL / "process_exit.json"

    protocol_sha_before = sha256(protocol_path)
    baseline_sha_before = sha256(baseline_path)
    checkpoint_sha_before = sha256(checkpoint_path)
    manifest_sha_before = sha256(manifest_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    training_exit = json.loads(training_exit_path.read_text(encoding="utf-8"))
    python_exit = (TRIAL / "python.exit_code").read_text(encoding="utf-8").strip()
    initialization = json.loads((TRIAL / "initialization.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = {row["path"]: row for row in manifest.get("checkpoints", [])}
    checkpoint_manifest = manifest_rows.get(CHECKPOINT_RELATIVE)
    baseline_manifest = manifest_rows.get(BASELINE_RELATIVE)

    baseline = RecurrentPPO.load(str(baseline_path), env=None, device="cpu")
    checkpoint = RecurrentPPO.load(str(checkpoint_path), env=None, device="cpu")
    weights = compare_weights(baseline, checkpoint, torch)
    baseline_optimizer = optimizer_summary(baseline, torch)
    checkpoint_optimizer = optimizer_summary(checkpoint, torch)
    snapshots = source_hashes(protocol)
    baseline_meta = _checkpoint_data(baseline_path)
    checkpoint_meta = _checkpoint_data(checkpoint_path)

    protocol_sha_after = sha256(protocol_path)
    baseline_sha_after = sha256(baseline_path)
    checkpoint_sha_after = sha256(checkpoint_path)
    manifest_sha_after = sha256(manifest_path)
    final_sha_from_manifest = None if checkpoint_manifest is None else checkpoint_manifest.get("sha256")

    action = protocol.get("action_config")
    expected_drive = initialization.get("live_drive_config", {})
    expected_observation = initialization.get("live_observation_config", {})
    drive = checkpoint_meta.get("field_drive_config", {})
    coupling = drive.get("same_side_wheel_coupling", {})
    contact = drive.get("physics_contact", {})
    observation = checkpoint_meta.get("field_observation_config", {})
    expected_timesteps = protocol.get("initialization", {}).get("expected_final_num_timesteps")
    checks = {
        "protocol_sha_matches_expected_and_stable": (
            protocol_sha_before == EXPECTED_PROTOCOL_SHA256 == protocol_sha_after
        ),
        "training_real_exit_zero_and_completed": (
            status.get("status") == "completed"
            and status.get("exit_code") == 0
            and status.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
            and status.get("num_timesteps") == FINAL_TIMESTEPS
            and training_exit.get("exit_code") == 0
            and python_exit == "0"
        ),
        "parent_baseline_sha_and_steps_stable": (
            baseline_sha_before == EXPECTED_BASELINE_SHA256 == baseline_sha_after
            and protocol.get("initialization", {}).get("baseline_sha256") == EXPECTED_BASELINE_SHA256
            and baseline.num_timesteps == BASELINE_TIMESTEPS
            and baseline_meta.get("num_timesteps") == BASELINE_TIMESTEPS
            and baseline_manifest is not None
            and baseline_manifest.get("sha256") == EXPECTED_BASELINE_SHA256
        ),
        "final_checkpoint_sha_is_manifest_derived_and_stable": (
            checkpoint_sha_before == checkpoint_sha_after
            and final_sha_from_manifest == checkpoint_sha_before
            and checkpoint_manifest is not None
            and checkpoint_manifest.get("num_timesteps") == FINAL_TIMESTEPS
            and checkpoint_manifest.get("adam_optimizer_state_present") is True
        ),
        "checkpoint_manifest_stable": manifest_sha_before == manifest_sha_after,
        "cumulative_steps_match_child_protocol": (
            checkpoint.num_timesteps == FINAL_TIMESTEPS
            and checkpoint.num_timesteps == expected_timesteps
            and checkpoint_meta.get("num_timesteps") == FINAL_TIMESTEPS
        ),
        "43_live_sources_and_snapshot_copies_match": snapshots["all_match"],
        "baseline_and_final_weights_finite_and_shapes_match": (
            weights["all_weights_finite"] and weights["same_keys"] and weights["same_shapes"]
        ),
        "weights_updated_from_parent_baseline": (
            weights["changed_tensors"] > 0 and weights["changed_elements"] > 0
        ),
        "baseline_and_final_adam_finite_and_continued": (
            baseline_optimizer["class"] == checkpoint_optimizer["class"] == "Adam"
            and baseline_optimizer["state_parameter_count"] == INITIAL_ADAM_PARAMETERS
            and baseline_optimizer["step_entry_count"] == INITIAL_ADAM_PARAMETERS
            and baseline_optimizer["min_step"] == baseline_optimizer["max_step"] == INITIAL_ADAM_STEP
            and baseline_optimizer["all_finite"] and not baseline_optimizer["empty"]
            and checkpoint_optimizer["state_parameter_count"] == INITIAL_ADAM_PARAMETERS
            and checkpoint_optimizer["step_entry_count"] == INITIAL_ADAM_PARAMETERS
            and checkpoint_optimizer["min_step"] > baseline_optimizer["max_step"]
            and checkpoint_optimizer["all_steps_positive"]
            and checkpoint_optimizer["all_finite"] and not checkpoint_optimizer["empty"]
        ),
        "loaded_cpu_only_and_cuda_unavailable": (
            str(checkpoint.device) == "cpu"
            and str(baseline.device) == "cpu"
            and not torch.cuda.is_available()
            and torch.cuda.device_count() == 0
        ),
        "action_observation_and_v7_drive_contract_match": (
            baseline.field_action_config == checkpoint.field_action_config == action
            and baseline.field_drive_config == checkpoint.field_drive_config == expected_drive
            and baseline.field_observation_config == checkpoint.field_observation_config == expected_observation
            and checkpoint_meta.get("field_action_config") == action
            and drive.get("interface") == "independent_left_right_track_speed_coupled_pi_v7"
            and coupling.get("enabled") is True
            and observation.get("schema") == "sensor-minimal-v6"
            and contact.get("physics_dt_s") == 0.001
            and contact.get("decimation") == 50
            and contact.get("policy_dt_s") == 0.05
            and contact.get("cone") == "elliptic"
            and contact.get("impratio") == 1.0
        ),
    }
    return {
        "schema": "coupled-track-continuation-final-checkpoint-cpu-validation-v1",
        "status": "passed" if all(checks.values()) else "failed", "read_only": True,
        "device": str(checkpoint.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": {"path": str(protocol_path), "sha256_before": protocol_sha_before,
                     "sha256_after": protocol_sha_after},
        "training_exit": {"status": status.get("status"), "status_num_timesteps": status.get("num_timesteps"),
                          "python_exit_code": python_exit, "process_exit": training_exit},
        "sources": snapshots,
        "baseline": {"path": str(baseline_path), "sha256_before": baseline_sha_before,
                     "sha256_after": baseline_sha_after, "num_timesteps": baseline.num_timesteps,
                     "manifest": baseline_manifest, "optimizer": baseline_optimizer},
        "checkpoint": {"path": str(checkpoint_path), "sha256_before": checkpoint_sha_before,
                       "sha256_after": checkpoint_sha_after,
                       "sha256_from_manifest": final_sha_from_manifest,
                       "num_timesteps": checkpoint.num_timesteps, "manifest": checkpoint_manifest,
                       "optimizer": checkpoint_optimizer},
        "weight_updates": weights,
        "live_contract": {"action": checkpoint.field_action_config,
                          "observation": observation, "drive": drive},
        "checks": checks,
        "cuda_available_after_mask": bool(torch.cuda.is_available()),
        "cuda_device_count_after_mask": int(torch.cuda.device_count()),
    }


def main() -> int:
    try:
        report = run()
    except Exception as exc:
        report = {"schema": "coupled-track-continuation-final-checkpoint-cpu-validation-v1",
                  "status": "failed", "read_only": True, "error": f"{type(exc).__name__}: {exc}"}
    report["exit_code"] = 0 if report["status"] == "passed" else 1
    output = OUTPUT_DIR / "final_cpu_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": report["status"], "exit_code": report["exit_code"],
                      "checks": report.get("checks", {}), "output": str(output)}, ensure_ascii=False, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
