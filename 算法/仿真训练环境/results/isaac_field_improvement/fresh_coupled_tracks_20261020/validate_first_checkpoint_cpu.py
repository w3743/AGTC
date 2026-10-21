"""Read-only CPU integrity check for the first fresh v7 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_coupled_tracks_20261020"
EXPECTED_PROTOCOL_SHA256 = "2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6"
EXPECTED_BASELINE_SHA256 = "beda8fde92236b558433a148245094caeefb88d7c23b9ea73ed94c4cd596f1d5"
CHECKPOINT_RELATIVE = "stage_1_near/checkpoints/field_policy_near_8192_steps.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_summary(model, torch) -> dict:
    optimizer = model.policy.optimizer
    states = getattr(optimizer, "state", {})
    steps = []
    finite = True
    for state in states.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if hasattr(step, "item") else int(step))
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                finite = finite and bool(torch.isfinite(value).all().item())
    return {
        "class": type(optimizer).__name__,
        "state_parameter_count": len(states),
        "step_entry_count": len(steps),
        "min_step": min(steps) if steps else 0,
        "max_step": max(steps) if steps else 0,
        "all_finite": finite,
        "empty": not states,
    }


def source_hashes(protocol: dict) -> dict:
    files = protocol.get("source_snapshot", {}).get("files", [])
    rows = []
    for entry in files:
        expected = entry["sha256"]
        live = ROOT / entry["path"]
        snapshot = TRIAL / entry["snapshot_path"]
        live_sha = sha256(live) if live.is_file() else None
        snapshot_sha = sha256(snapshot) if snapshot.is_file() else None
        rows.append({
            "path": entry["path"],
            "expected_sha256": expected,
            "live_sha256": live_sha,
            "snapshot_sha256": snapshot_sha,
            "matches": live_sha == expected and snapshot_sha == expected,
        })
    return {
        "count": len(files),
        "matching_count": sum(row["matches"] for row in rows),
        "all_match": len(files) == 42 and all(row["matches"] for row in rows),
        "files": rows,
    }


def run() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    protocol_path = TRIAL / "protocol.json"
    baseline_path = TRIAL / "untrained_baseline.zip"
    checkpoint_path = TRIAL / CHECKPOINT_RELATIVE
    protocol_sha_before = sha256(protocol_path)
    baseline_sha_before = sha256(baseline_path)
    checkpoint_sha_before = sha256(checkpoint_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    initialization = json.loads((TRIAL / "initialization.json").read_text(encoding="utf-8"))
    manifest = json.loads((TRIAL / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest_rows = {row["path"]: row for row in manifest.get("checkpoints", [])}
    checkpoint_manifest = manifest_rows.get(CHECKPOINT_RELATIVE)
    baseline_manifest = manifest_rows.get("untrained_baseline.zip")

    baseline = RecurrentPPO.load(str(baseline_path), env=None, device="cpu")
    checkpoint = RecurrentPPO.load(str(checkpoint_path), env=None, device="cpu")
    baseline_state = {key: value.detach().cpu() for key, value in baseline.policy.state_dict().items()}
    checkpoint_state = {key: value.detach().cpu() for key, value in checkpoint.policy.state_dict().items()}

    same_keys = set(baseline_state) == set(checkpoint_state)
    finite_weights = all(
        bool(torch.isfinite(value).all().item())
        for state in (baseline_state, checkpoint_state)
        for value in state.values()
        if value.is_floating_point()
    )
    changed_tensors = 0
    changed_elements = 0
    max_abs_delta = 0.0
    same_shapes = same_keys
    for name in sorted(set(baseline_state) & set(checkpoint_state)):
        before, after = baseline_state[name], checkpoint_state[name]
        if before.shape != after.shape:
            same_shapes = False
            continue
        if before.is_floating_point() and after.is_floating_point():
            delta = after - before
            changed = int(torch.count_nonzero(delta).item())
            changed_elements += changed
            changed_tensors += int(changed > 0)
            if changed:
                max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        else:
            changed = int(torch.count_nonzero(before != after).item())
            changed_elements += changed
            changed_tensors += int(changed > 0)

    baseline_optimizer = optimizer_summary(baseline, torch)
    checkpoint_optimizer = optimizer_summary(checkpoint, torch)
    snapshots = source_hashes(protocol)
    checkpoint_meta = json.loads(zipfile.ZipFile(checkpoint_path).read("data"))

    protocol_sha_after = sha256(protocol_path)
    baseline_sha_after = sha256(baseline_path)
    checkpoint_sha_after = sha256(checkpoint_path)
    current_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    init_drive = initialization.get("drive_config", {})
    contact = init_drive.get("physics_contact", {})
    coupling = init_drive.get("same_side_wheel_coupling", {})
    checks = {
        "protocol_sha_matches_and_stable": protocol_sha_before == EXPECTED_PROTOCOL_SHA256 == protocol_sha_after,
        "42_live_sources_and_snapshot_copies_match": snapshots["all_match"],
        "baseline_is_formal_zero_step_expected_sha": (
            baseline.num_timesteps == 0
            and baseline_sha_before == EXPECTED_BASELINE_SHA256 == baseline_sha_after
            and initialization.get("sha256") == EXPECTED_BASELINE_SHA256
            and initialization.get("num_timesteps") == 0
        ),
        "checkpoint_is_8192_and_manifest_sha_matches": (
            checkpoint.num_timesteps == 8192
            and checkpoint_sha_before == checkpoint_sha_after
            and checkpoint_manifest is not None
            and checkpoint_manifest.get("sha256") == checkpoint_sha_before
            and checkpoint_manifest.get("num_timesteps") == 8192
        ),
        "checkpoint_and_baseline_metadata_match_protocol": (
            baseline.field_action_config == checkpoint.field_action_config == protocol["action_config"]
            and checkpoint_meta.get("field_action_config") == protocol["action_config"]
            and current_protocol["initialization"].get("checkpoint") is None
            and current_protocol["initialization"].get("weight_transfer") is False
        ),
        "all_weights_finite_and_shapes_match": finite_weights and same_keys and same_shapes,
        "parameters_updated_from_zero_step_baseline": changed_tensors > 0 and changed_elements > 0,
        "adam_optimizer_has_positive_finite_steps": (
            checkpoint_optimizer["class"] == "Adam"
            and not checkpoint_optimizer["empty"]
            and checkpoint_optimizer["step_entry_count"] > 0
            and checkpoint_optimizer["min_step"] > 0
            and checkpoint_optimizer["all_finite"]
        ),
        "loaded_cpu_only": str(checkpoint.device) == "cpu" and not torch.cuda.is_available(),
        "live_runtime_contract_matches_v7": (
            init_drive.get("interface") == "independent_left_right_track_speed_coupled_pi_v7"
            and coupling.get("enabled") is True
            and contact.get("physics_dt_s") == 0.001
            and contact.get("decimation") == 50
            and contact.get("policy_dt_s") == 0.05
            and contact.get("cone") == "elliptic"
            and contact.get("impratio") == 1.0
        ),
    }
    return {
        "schema": "fresh-coupled-tracks-first-checkpoint-cpu-validation-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "read_only": True,
        "device": str(checkpoint.device),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha_before},
        "sources": snapshots,
        "baseline": {
            "path": str(baseline_path),
            "sha256": baseline_sha_before,
            "num_timesteps": baseline.num_timesteps,
            "optimizer": baseline_optimizer,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha_before,
            "num_timesteps": checkpoint.num_timesteps,
            "manifest": checkpoint_manifest,
            "optimizer": checkpoint_optimizer,
        },
        "weight_updates": {
            "changed_tensors": changed_tensors,
            "changed_elements": changed_elements,
            "max_abs_delta": max_abs_delta,
            "all_weights_finite": finite_weights,
            "same_keys": same_keys,
            "same_shapes": same_shapes,
        },
        "live_contract": {
            "drive_interface": init_drive.get("interface"),
            "coupling_enabled": coupling.get("enabled"),
            "coupling_relations_per_world": coupling.get("relations_per_world"),
            "physics_contact": contact,
        },
        "checks": checks,
        "cuda_available_after_mask": bool(torch.cuda.is_available()),
    }


def main() -> int:
    try:
        report = run()
    except Exception as exc:
        report = {
            "schema": "fresh-coupled-tracks-first-checkpoint-cpu-validation-v1",
            "status": "failed",
            "read_only": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    report["exit_code"] = 0 if report["status"] == "passed" else 1
    output = TRIAL / "first_checkpoint_cpu_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "exit_code": report["exit_code"], "checks": report.get("checks", {}), "output": str(output)}, ensure_ascii=False, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
