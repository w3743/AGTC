#!/usr/bin/env python
"""Read-only CPU audit for the frozen a02 source snapshot and latest complete RSL checkpoint.
Writes only checkpoint_audit.json beside this script. Does not launch Isaac Sim/Kit or initialize CUDA.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import hashlib
import importlib.metadata as importlib_metadata
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "checkpoint_audit.json"
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[5]
TRIAL = ROOT / "算法" / "仿真训练环境" / "results" / "isaac_field_improvement" / "physx_pointmove_flat12_20261024_a02"
SNAPSHOT = TRIAL / "source_snapshot"
TRAIN_DIR = TRIAL / "logs" / "rsl_rl" / "physx_pointmove_flat12_20261024_a02" / "training_a02"
BASELINE = TRIAL / "logs" / "rsl_rl" / "physx_pointmove_flat12_20261024_a02" / "baseline_a02" / "model_0.pt"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def stat_signature(path: Path) -> dict[str, int]:
    s = path.stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}

def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

def checkpoint_number(path: Path) -> int | None:
    m = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(m.group(1)) if m else None

def scan_checkpoints() -> list[Path]:
    return sorted((p for p in TRAIN_DIR.glob("model_*.pt") if checkpoint_number(p) is not None), key=lambda p: checkpoint_number(p), reverse=True)

def stable_signature(path: Path, polls: int = 3, interval_s: float = 0.25) -> tuple[bool, list[dict[str, int]]]:
    observations = []
    for i in range(polls):
        try:
            observations.append(stat_signature(path))
        except FileNotFoundError:
            return False, observations
        if i + 1 < polls:
            time.sleep(interval_s)
    return len(observations) == polls and all(x == observations[0] for x in observations), observations

def tensor_stats(value: Any, torch: Any) -> dict[str, Any]:
    leaves = []
    def walk(x: Any, key: str) -> None:
        if torch.is_tensor(x):
            finite = bool(torch.isfinite(x).all().item()) if x.is_floating_point() or x.is_complex() else True
            leaves.append({"path": key, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device), "finite": finite, "numel": x.numel()})
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(x, (tuple, list)):
            for i, v in enumerate(x):
                walk(v, f"{key}[{i}]")
    walk(value, "")
    return {
        "tensor_count": len(leaves),
        "finite_tensor_count": sum(1 for x in leaves if x["finite"]),
        "all_finite": all(x["finite"] for x in leaves),
        "total_tensor_elements": sum(x["numel"] for x in leaves),
        "devices": sorted({x["device"] for x in leaves}),
        "tensors": leaves,
    }

def numeric_scalar(x: Any, torch: Any) -> float | None:
    if torch.is_tensor(x):
        if x.numel() != 1:
            return None
        x = x.detach().cpu().item()
    if isinstance(x, (int, float)):
        return float(x)
    return None

def inspect_payload(payload: Any, path: Path, digest_before: str, digest_after: str, sig_before: dict[str, int], sig_after: dict[str, int], torch: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint root must be dict, got {type(payload).__name__}")
    required = ["actor_state_dict", "critic_state_dict", "optimizer_state_dict", "field_policy_contract", "num_timesteps", "iter"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"checkpoint missing required RSL resume fields: {missing}")
    actor = payload["actor_state_dict"]
    critic = payload["critic_state_dict"]
    optimizer = payload["optimizer_state_dict"]
    contract = payload["field_policy_contract"]
    if not isinstance(actor, dict) or not isinstance(critic, dict) or not isinstance(optimizer, dict) or not isinstance(contract, dict):
        raise TypeError("unexpected types for actor, critic, optimizer, or field_policy_contract")
    actor_stats = tensor_stats(actor, torch)
    critic_stats = tensor_stats(critic, torch)
    optimizer_stats = tensor_stats(optimizer, torch)
    opt_state = optimizer.get("state", {})
    adam_steps = []
    if isinstance(opt_state, dict):
        for state in opt_state.values():
            if isinstance(state, dict) and "step" in state:
                step = numeric_scalar(state["step"], torch)
                if step is not None:
                    adam_steps.append(step)
    param_groups = optimizer.get("param_groups", [])
    group_summary = []
    if isinstance(param_groups, list):
        for g in param_groups:
            if isinstance(g, dict):
                group_summary.append({k: json_safe(g.get(k)) for k in ("lr", "eps", "betas", "weight_decay", "amsgrad", "maximize") if k in g})
    canonical_contract = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract_sha = hashlib.sha256(canonical_contract).hexdigest()
    c = {
        "sha256_canonical_json_sorted_compact_utf8": contract_sha,
        "backend": contract.get("backend"),
        "simulator": contract.get("simulator"),
        "physics_backend": contract.get("physics_backend"),
        "learning_backend": contract.get("learning_backend"),
        "format": contract.get("format"),
        "version": contract.get("version"),
        "input_schema": contract.get("input_schema"),
        "encoder_schema": (contract.get("encoder_config") or {}).get("schema") if isinstance(contract.get("encoder_config"), dict) else None,
        "observation_shapes": contract.get("observation_shapes"),
        "action_config": contract.get("action_config"),
        "actor_config": contract.get("actor_config"),
        "critic_config": contract.get("critic_config"),
        "lstm": contract.get("lstm"),
        "full_contract": json_safe(contract),
    }
    return {
        "path": str(path),
        "filename_iteration_suffix": checkpoint_number(path),
        "bytes": sig_before["bytes"],
        "mtime_ns": sig_before["mtime_ns"],
        "sha256_before_load": digest_before,
        "sha256_after_load": digest_after,
        "file_unchanged_during_load": sig_before == sig_after and digest_before == digest_after,
        "top_level_keys": list(payload.keys()),
        "iteration": int(payload["iter"]),
        "num_timesteps": int(payload["num_timesteps"]),
        "infos_value": json_safe(payload.get("infos")),
        "actor_state_dict": {"entries": len(actor), **actor_stats},
        "critic_state_dict": {"entries": len(critic), **critic_stats},
        "model_tensors_all_finite": actor_stats["all_finite"] and critic_stats["all_finite"],
        "optimizer": {
            "state_entries": len(opt_state) if isinstance(opt_state, dict) else None,
            "adam_step_values_unique": sorted(set(adam_steps)),
            "adam_step_min": min(adam_steps) if adam_steps else None,
            "adam_step_max": max(adam_steps) if adam_steps else None,
            "step_entry_count": len(adam_steps),
            "optimizer_tensor_state": optimizer_stats,
            "all_optimizer_tensors_finite": optimizer_stats["all_finite"],
            "param_groups": group_summary,
        },
        "field_policy_contract": c,
        "payload_has_explicit_rng_state": any("rng" in str(k).lower() for k in payload),
        "payload_has_explicit_rollout_or_simulator_state": any(any(s in str(k).lower() for s in ("rollout", "simulator", "env_state", "hidden_state")) for k in payload),
    }

def main() -> None:
    # Importing torch is sufficient for CPU deserialization. No Isaac/Omniverse modules are imported.
    import torch
    torch_version = torch.__version__
    manifest_path = TRIAL / "source_manifest.json"
    manifest = json_read(manifest_path)
    entries = manifest.get("files", [])
    row_checks = []
    for row in entries:
        rel = Path(str(row.get("snapshot_path", "")))
        file_path = TRIAL / rel
        expected = str(row.get("sha256", "")).lower()
        exists = file_path.is_file()
        actual = sha256(file_path) if exists else None
        row_checks.append({
            "source_path": row.get("path"),
            "snapshot_path": row.get("snapshot_path"),
            "origin": row.get("origin"),
            "expected_sha256": expected,
            "actual_snapshot_sha256": actual,
            "exists": exists,
            "snapshot_hash_matches_manifest": bool(exists and expected and actual == expected),
        })
    snap_ok = len(row_checks) == len(entries) and all(r["snapshot_hash_matches_manifest"] for r in row_checks)

    protocol_path = TRIAL / "protocol.json"
    protocol = json_read(protocol_path)
    protocol_sha = sha256(protocol_path)
    status_path = TRIAL / "run_status.json"
    status = json_read(status_path) if status_path.exists() else {}
    launch_path = TRIAL / "controller_launch.json"
    launch = json_read(launch_path) if launch_path.exists() else {}
    baseline_validation_path = TRIAL / "baseline_validation.json"
    baseline_validation = json_read(baseline_validation_path) if baseline_validation_path.exists() else {}
    protocol_expected = status.get("protocol_sha256") or launch.get("protocol_sha256")

    baseline_sig_before = stat_signature(BASELINE)
    baseline_sha_before = sha256(BASELINE)
    baseline_payload = torch.load(BASELINE, map_location="cpu", weights_only=False)
    baseline_sha_after = sha256(BASELINE)
    baseline_sig_after = stat_signature(BASELINE)
    baseline_contract = baseline_payload.get("field_policy_contract") if isinstance(baseline_payload, dict) else None
    baseline_optimizer = baseline_payload.get("optimizer_state_dict", {}) if isinstance(baseline_payload, dict) else {}
    baseline_opt_state = baseline_optimizer.get("state", {}) if isinstance(baseline_optimizer, dict) else {}
    baseline_recorded_sha = baseline_validation.get("checkpoint_sha256")
    baseline = {
        "path": str(BASELINE),
        "bytes": baseline_sig_before["bytes"],
        "sha256_before_load": baseline_sha_before,
        "sha256_after_load": baseline_sha_after,
        "file_unchanged_during_load": baseline_sig_before == baseline_sig_after and baseline_sha_before == baseline_sha_after,
        "sha256_recorded_in_baseline_validation": baseline_recorded_sha,
        "sha256_matches_recorded_validation": baseline_sha_before == baseline_recorded_sha,
        "iteration": baseline_payload.get("iter") if isinstance(baseline_payload, dict) else None,
        "num_timesteps": baseline_payload.get("num_timesteps") if isinstance(baseline_payload, dict) else None,
        "optimizer_state_entries": len(baseline_opt_state) if isinstance(baseline_opt_state, dict) else None,
        "contract": baseline_contract,
    }

    checkpoint_attempts = []
    selected_payload = None
    selected_path = None
    selected_sig = None
    selected_hash_before = None
    selected_hash_after = None
    # Retry if a new numbered file appears while the current candidate is being read.
    for round_no in range(1, 5):
        candidates = scan_checkpoints()
        initial_numbers = {checkpoint_number(p) for p in candidates}
        selection_this_round = None
        for candidate in candidates:
            stable, observations = stable_signature(candidate)
            attempt = {"round": round_no, "path": str(candidate), "iteration_suffix": checkpoint_number(candidate), "stable_poll_observations": observations, "stable_before_load": stable}
            if not stable:
                attempt["result"] = "skipped_unstable_during_poll"
                checkpoint_attempts.append(attempt)
                continue
            sig_before = stat_signature(candidate)
            sha_before = sha256(candidate)
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                sha_after = sha256(candidate)
                sig_after = stat_signature(candidate)
                if sig_before != sig_after or sha_before != sha_after:
                    attempt["result"] = "changed_during_load"
                    checkpoint_attempts.append(attempt)
                    continue
                # Validate expected official RSL resume payload fields without creating an env.
                if not isinstance(payload, dict) or not all(k in payload for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "field_policy_contract", "num_timesteps", "iter")):
                    attempt["result"] = "not_a_complete_rsl_resume_payload"
                    checkpoint_attempts.append(attempt)
                    continue
                attempt["result"] = "loaded_complete_cpu_payload"
                checkpoint_attempts.append(attempt)
                selection_this_round = (candidate, payload, sig_before, sha_before, sha_after)
                break
            except Exception as e:
                attempt["result"] = "cpu_load_failed"
                attempt["error_type"] = type(e).__name__
                attempt["error"] = str(e)[:500]
                checkpoint_attempts.append(attempt)
        if selection_this_round is None:
            time.sleep(0.4)
            continue
        candidate, payload, sig_before, sha_before, sha_after = selection_this_round
        after_numbers = {checkpoint_number(p) for p in scan_checkpoints()}
        appeared_newer = any(n is not None and n > checkpoint_number(candidate) and n not in initial_numbers for n in after_numbers)
        failed_higher_initial = any(n is not None and n > checkpoint_number(candidate) for n in initial_numbers) and any(
            a.get("round") == round_no and a.get("iteration_suffix") > checkpoint_number(candidate) and a.get("result") != "loaded_complete_cpu_payload" for a in checkpoint_attempts
        )
        if appeared_newer or (failed_higher_initial and round_no < 4):
            time.sleep(0.4)
            continue
        selected_path, selected_payload, selected_sig, selected_hash_before, selected_hash_after = candidate, payload, sig_before, sha_before, sha_after
        break
    if selected_payload is None:
        raise RuntimeError("No stable, complete RSL checkpoint could be CPU-loaded from the training run directory.")

    selected = inspect_payload(selected_payload, selected_path, selected_hash_before, selected_hash_after, selected_sig, stat_signature(selected_path), torch)
    selected["stability_observation_count_before_load"] = next((len(a["stable_poll_observations"]) for a in reversed(checkpoint_attempts) if a.get("path") == str(selected_path) and a.get("result") == "loaded_complete_cpu_payload"), None)
    selected["checkpoint_selection_rule"] = "highest numeric model_N.pt that was stable across pre-load polls, CPU-loadable with required RSL resume fields, and unchanged during load; candidate directory rescanned after load"
    selected["contract_matches_zero_step_baseline"] = selected_payload.get("field_policy_contract") == baseline_contract
    reference_contract_sha = baseline_validation.get("field_policy_contract_sha256")
    selected["field_policy_contract"]["sha256_matches_first_checkpoint_audit_reference"] = selected["field_policy_contract"]["sha256_canonical_json_sorted_compact_utf8"] == reference_contract_sha
    selected["field_policy_contract"]["sha256_reference_from_first_checkpoint_audit"] = reference_contract_sha

    # Presence of only a pyproject.toml is not a complete runtime lock or self-contained simulator distribution.
    lock_names = {"uv.lock", "poetry.lock", "pdm.lock", "requirements.txt", "environment.yml", "conda.yaml", "Pipfile.lock"}
    present_lock_files = sorted(str(p.relative_to(SNAPSHOT)) for p in SNAPSHOT.rglob("*") if p.is_file() and p.name in lock_names)
    project_pyproject = SNAPSHOT / "pyproject.toml"
    runtime_versions = protocol.get("runtime_versions", {})
    expected_versions = protocol.get("expected_runtime_versions", {})
    runtime_package_names = {"isaacsim": "isaacsim", "isaaclab": "isaaclab", "isaaclab-physx": "isaaclab-physx", "isaaclab-rl": "isaaclab-rl", "rsl-rl-lib": "rsl-rl-lib", "torch": "torch", "viser": "viser", "vqf": "vqf"}
    installed_versions = {}
    for label, package in runtime_package_names.items():
        try:
            installed_versions[label] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            installed_versions[label] = None
        except Exception as e:
            installed_versions[label] = f"metadata_error:{type(e).__name__}"
    contract = selected_payload["field_policy_contract"]
    has_rng = bool(selected["payload_has_explicit_rng_state"])
    has_runtime_state = bool(selected["payload_has_explicit_rollout_or_simulator_state"])

    result = {
        "schema": "agtc-a02-sampling-live-handoff-cpu-checkpoint-audit-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "trial_id": "physx_pointmove_flat12_20261024_a02",
            "operation": "read-only checkpoint and source snapshot audit",
            "deserialization": "torch.load(map_location='cpu', weights_only=False) for trusted local repository checkpoints",
            "torch_version": torch_version,
            "torch_cuda_build": getattr(torch.version, "cuda", None),
            "cuda_visible_devices_for_audit_process": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "isaac_sim_or_kit_imported_or_started": False,
            "source_files_or_checkpoints_modified": False,
        },
        "source_snapshot": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "manifest_schema": manifest.get("manifest_schema"),
            "manifest_row_count": len(entries),
            "checked_rows": len(row_checks),
            "all_snapshot_files_exist_and_match_manifest": snap_ok,
            "live_source_tree_equality_required": False,
            "live_source_tree_checked": False,
            "live_source_tree_note": "Live files were intentionally not used as the integrity oracle because env.py may be concurrently edited; each frozen source_snapshot row is checked only against its manifest SHA-256.",
            "rows": row_checks,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha,
            "sha256_reference_from_run_status_or_launch": protocol_expected,
            "sha256_matches_reference": protocol_sha == protocol_expected,
            "schema": protocol.get("schema"),
            "status": protocol.get("status"),
            "backend": protocol.get("backend"),
            "task": protocol.get("task"),
            "training": protocol.get("training"),
            "runtime_versions_recorded": runtime_versions,
            "expected_runtime_versions_recorded": expected_versions,
        },
        "run_status_snapshot": {
            "path": str(status_path),
            "status": status.get("status"),
            "phase": status.get("phase"),
            "parent_pid": status.get("parent_pid"),
            "updated_at_utc": status.get("updated_at_utc"),
            "current_stage": {k: status.get("current_stage", {}).get(k) for k in ("name", "status", "child_pid", "child_still_running", "last_heartbeat_at_utc")},
            "resume_child": next(({k: child.get(k) for k in ("name", "status", "child_pid", "child_still_running", "last_heartbeat_at_utc")} for child in status.get("children", []) if child.get("name") == "resume_training_cli"), None),
            "is_snapshot_only_not_live_process_verification": True,
        },
        "baseline": baseline,
        "checkpoint_directory_inventory_at_audit": [
            {"path": str(p), "iteration_suffix": checkpoint_number(p), **stat_signature(p)} for p in scan_checkpoints()
        ],
        "checkpoint_load_attempts": checkpoint_attempts,
        "latest_complete_checkpoint": selected,
        "resume_boundary": {
            "policy_and_critic_weights_present": True,
            "adam_optimizer_state_present": selected["optimizer"]["state_entries"] is not None and selected["optimizer"]["state_entries"] > 0,
            "iteration_and_num_timesteps_present": True,
            "field_policy_contract_present": True,
            "explicit_rng_state_serialized": has_rng,
            "explicit_rollout_or_simulator_state_serialized": has_runtime_state,
            "live_simulator_physics_state_available_from_checkpoint": False,
            "live_lstm_rollout_hidden_state_available_from_checkpoint": False,
            "exact_continuation_possible": False,
            "reason": "Checkpoint contains the learned actor/critic parameters and Adam state, but not the live PhysX environment state, random-generator states, rollout buffer, or the in-flight LSTM hidden state. Resuming starts a new rollout from restored weights/optimizer and saved progress; it is a training resume, not bitwise exact continuation of the stopped live session.",
        },
        "snapshot_reproducibility_assessment": {
            "manifested_trial_source_snapshot_integrity": "complete" if snap_ok else "failed",
            "snapshot_source_row_count": len(entries),
            "pyproject_present": project_pyproject.is_file(),
            "dependency_lock_files_found_in_snapshot": present_lock_files,
            "runtime_versions_pinned_in_protocol": expected_versions,
            "installed_runtime_package_versions_seen_by_audit_interpreter": installed_versions,
            "snapshot_is_self_contained_for_independent_training_reproduction": False,
            "reason": "The snapshot preserves the manifested project/trial sources and pyproject, but does not contain an Isaac Sim/Kit distribution, Isaac Lab/PhysX/RSL-RL source/runtime, or a complete dependency lock/environment. The protocol records expected versions and the launcher uses a machine-local C:/pioneer/isaac-training/.venv. Reproduction requires that external runtime, compatible GPU driver/hardware, installed packages, and the run's assets/configuration; the source_snapshot directory alone is insufficient.",
            "runtime_dependencies_not_embedded_in_source_snapshot": ["Isaac Sim 6.2.0.0 / Kit", "Isaac Lab 28.0.0", "Isaac Lab PhysX 8.0.0", "Isaac Lab RL 1.5.1", "RSL-RL 5.5.1", "PyTorch 2.11.0+cu130", "CUDA-compatible GPU driver/runtime and GPU hardware", "project assets and the configured local Isaac training environment"],
        },
        "limitations": [
            "This is a CPU checkpoint/source-integrity audit, not a simulation, GUI, policy evaluation, training completion, or learning-performance acceptance.",
            "The training process/run_status may advance after the recorded audit timestamp; the checkpoint selected here is the latest stable and loadable candidate observed by this script.",
            "No live source hash comparison was made; the immutable source_snapshot hashes are the relevant frozen evidence.",
        ],
    }
    OUT_JSON.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "output": str(OUT_JSON),
        "source_snapshot_rows": len(row_checks),
        "snapshot_hashes_match": snap_ok,
        "protocol_sha256": protocol_sha,
        "baseline_sha256": baseline_sha_before,
        "selected_checkpoint": selected["path"],
        "checkpoint_sha256": selected["sha256_before_load"],
        "iteration": selected["iteration"],
        "num_timesteps": selected["num_timesteps"],
        "adam_step_min": selected["optimizer"]["adam_step_min"],
        "adam_step_max": selected["optimizer"]["adam_step_max"],
        "contract_matches_baseline": selected["contract_matches_zero_step_baseline"],
        "all_model_and_adam_tensors_finite": selected["model_tensors_all_finite"] and selected["optimizer"]["all_optimizer_tensors_finite"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
