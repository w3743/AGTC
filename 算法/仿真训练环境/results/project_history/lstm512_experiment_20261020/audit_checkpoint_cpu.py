"""Read-only CPU audit for a prepared LSTM-512 recurrent PPO checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "lstm512-checkpoint-cpu-audit-v1"
EXPECTED_UNIQUE_POLICY_PARAMETERS = 3_065_801


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def source_hashes(protocol: dict, trial: Path) -> dict:
    entries = protocol.get("source_snapshot", {}).get("files", [])
    rows = []
    for entry in entries:
        relative = entry.get("path")
        snapshot_relative = entry.get("snapshot_path")
        expected = entry.get("sha256")
        if not relative or not snapshot_relative or not expected:
            rows.append({"path": relative, "matches": False, "error": "incomplete source manifest entry"})
            continue
        source_path = resolve_path(relative, ROOT)
        snapshot_path = resolve_path(snapshot_relative, trial)
        try:
            source_sha = sha256(source_path) if source_path.is_file() else None
            snapshot_sha = sha256(snapshot_path) if snapshot_path.is_file() else None
        except OSError as exc:
            source_sha = snapshot_sha = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        rows.append({
            "path": relative,
            "snapshot_path": snapshot_relative,
            "expected_sha256": expected,
            "source_sha256": source_sha,
            "snapshot_sha256": snapshot_sha,
            "matches": source_sha == expected and snapshot_sha == expected,
            "error": error,
        })
    return {
        "count": len(entries),
        "matching_count": sum(row.get("matches", False) for row in rows),
        "all_match": bool(entries) and all(row.get("matches", False) for row in rows),
        "files": rows,
    }


def optimizer_summary(model, torch) -> dict:
    optimizer = getattr(model.policy, "optimizer", None)
    states = getattr(optimizer, "state", {}) if optimizer is not None else {}
    steps: list[int] = []
    finite = True
    floating_tensor_count = 0
    for state in states.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if hasattr(step, "item") else int(step))
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                floating_tensor_count += 1
                finite = finite and bool(torch.isfinite(value).all().item())
    return {
        "class": type(optimizer).__name__ if optimizer is not None else None,
        "state_parameter_count": len(states),
        "step_entry_count": len(steps),
        "min_step": min(steps) if steps else 0,
        "max_step": max(steps) if steps else 0,
        "all_steps_zero": all(value == 0 for value in steps),
        "all_steps_positive": bool(steps) and all(value > 0 for value in steps),
        "floating_state_tensor_count": floating_tensor_count,
        "all_finite": finite,
        "empty": not states,
    }


def policy_summary(model) -> dict:
    policy = model.policy
    actor = getattr(policy, "lstm_actor", None)
    critic = getattr(policy, "lstm_critic", None)
    named_parameters = list(policy.named_parameters())
    return {
        "unique_parameter_name_count": len(named_parameters),
        "unique_parameter_count": sum(parameter.numel() for _, parameter in named_parameters),
        "actor_lstm": None if actor is None else {
            "input_size": int(actor.input_size),
            "hidden_size": int(actor.hidden_size),
            "num_layers": int(actor.num_layers),
        },
        "critic_lstm": None if critic is None else {
            "input_size": int(critic.input_size),
            "hidden_size": int(critic.hidden_size),
            "num_layers": int(critic.num_layers),
        },
        "features_dim": int(getattr(getattr(policy, "features_extractor", None), "features_dim", -1)),
    }


def state_comparison(baseline, checkpoint, torch) -> dict:
    before = {key: value.detach().cpu() for key, value in baseline.policy.state_dict().items()}
    after = {key: value.detach().cpu() for key, value in checkpoint.policy.state_dict().items()}
    same_keys = set(before) == set(after)
    same_shapes = same_keys
    finite = True
    changed_tensors = 0
    changed_elements = 0
    compared_elements = 0
    max_abs_delta = 0.0
    changed_names = []
    for state in (before, after):
        finite = finite and all(
            bool(torch.isfinite(value).all().item())
            for value in state.values()
            if value.is_floating_point()
        )
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if left.shape != right.shape:
            same_shapes = False
            continue
        compared_elements += left.numel()
        changed = int(torch.count_nonzero(left != right).item())
        if changed:
            changed_tensors += 1
            changed_elements += changed
            changed_names.append(name)
            if left.is_floating_point() and right.is_floating_point():
                max_abs_delta = max(max_abs_delta, float((right - left).abs().max().item()))
    return {
        "same_keys": same_keys,
        "same_shapes": same_shapes,
        "all_weights_finite": finite,
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "compared_element_count": compared_elements,
        "max_abs_delta": max_abs_delta,
        "changed_tensor_names": changed_names,
    }


def run(args: argparse.Namespace) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    trial = resolve_path(args.trial, ROOT)
    checkpoint_path = resolve_path(args.checkpoint, trial)
    protocol_path = trial / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing prepared trial protocol: {protocol_path}")
    protocol_sha_before = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    initialization = json.loads((trial / "initialization.json").read_text(encoding="utf-8"))
    baseline_relative = protocol.get("initialization", {}).get("baseline_path")
    if not baseline_relative:
        raise ValueError("protocol is missing initialization.baseline_path")
    baseline_path = resolve_path(baseline_relative, trial)
    if not baseline_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("baseline or requested checkpoint does not exist")

    baseline_sha_before = sha256(baseline_path)
    checkpoint_sha_before = sha256(checkpoint_path)
    sources_before = source_hashes(protocol, trial)

    baseline = RecurrentPPO.load(str(baseline_path), env=None, device="cpu")
    checkpoint = RecurrentPPO.load(str(checkpoint_path), env=None, device="cpu")
    baseline_policy = policy_summary(baseline)
    checkpoint_policy = policy_summary(checkpoint)
    baseline_optimizer = optimizer_summary(baseline, torch)
    checkpoint_optimizer = optimizer_summary(checkpoint, torch)
    weights = state_comparison(baseline, checkpoint, torch)

    protocol_sha_after = sha256(protocol_path)
    baseline_sha_after = sha256(baseline_path)
    checkpoint_sha_after = sha256(checkpoint_path)
    protocol_after = json.loads(protocol_path.read_text(encoding="utf-8"))
    sources_after = source_hashes(protocol_after, trial)
    architecture = protocol.get("architecture", {})
    protocol_encoder_dim = architecture.get("field_encoder_config", {}).get("features_dim")
    protocol_hidden_size = architecture.get("lstm_hidden_size")
    actor = checkpoint_policy["actor_lstm"] or {}
    critic = checkpoint_policy["critic_lstm"] or {}
    cpu_only = (
        str(baseline.device) == "cpu"
        and str(checkpoint.device) == "cpu"
        and not torch.cuda.is_available()
        and torch.cuda.device_count() == 0
    )
    checks = {
        "initialization_protocol_and_baseline_sha_match": (
            initialization.get("protocol_sha256") == protocol_sha_before
            and initialization.get("sha256") == baseline_sha_before
        ),
        "saved_encoder_and_environment_contracts_match": all(
            getattr(model, "field_encoder_config", None) == architecture.get("field_encoder_config")
            and getattr(model, "field_action_config", None) == protocol.get("action_config")
            and getattr(model, "field_drive_config", None) == initialization.get("drive_config")
            and getattr(model, "field_observation_config", None) == initialization.get("observation_config")
            for model in (baseline, checkpoint)
        ),
        "protocol_sha_stable": protocol_sha_before == protocol_sha_after,
        "protocol_architecture_is_lstm512_encoder192": protocol_hidden_size == 512 and protocol_encoder_dim == 192,
        "live_and_snapshot_sources_match_and_remain_stable": (
            sources_before["all_match"] and sources_after["all_match"]
            and sources_before["files"] == sources_after["files"]
        ),
        "baseline_and_checkpoint_sha_stable": (
            baseline_sha_before == baseline_sha_after
            and checkpoint_sha_before == checkpoint_sha_after
        ),
        "checkpoint_actor_critic_lstm_contract": (
            actor.get("input_size") == 192 and actor.get("hidden_size") == 512 and actor.get("num_layers") == 1
            and critic.get("input_size") == 192 and critic.get("hidden_size") == 512 and critic.get("num_layers") == 1
            and checkpoint_policy["features_dim"] == 192
        ),
        "unique_policy_parameter_count_matches_expected": (
            baseline_policy["unique_parameter_count"] == EXPECTED_UNIQUE_POLICY_PARAMETERS
            and checkpoint_policy["unique_parameter_count"] == EXPECTED_UNIQUE_POLICY_PARAMETERS
        ),
        "all_weights_finite_and_baseline_shapes_match": (
            weights["all_weights_finite"] and weights["same_keys"] and weights["same_shapes"]
        ),
        "baseline_zero_step_and_checkpoint_positive_steps": (
            baseline.num_timesteps == 0 and checkpoint.num_timesteps > 0
            and baseline_optimizer["all_steps_zero"]
            and checkpoint_optimizer["all_steps_positive"]
        ),
        "baseline_and_checkpoint_adam_states_finite": (
            baseline_optimizer["class"] == "Adam"
            and baseline_optimizer["all_finite"]
            and checkpoint_optimizer["class"] == "Adam"
            and checkpoint_optimizer["all_finite"]
        ),
        "checkpoint_has_weight_updates": weights["changed_tensor_count"] > 0 and weights["changed_element_count"] > 0,
        "loaded_cpu_only": cpu_only,
    }
    return {
        "schema": SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "read_only": True,
        "training_or_simulation_run": False,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": {
            "path": str(protocol_path),
            "sha256_before": protocol_sha_before,
            "sha256_after": protocol_sha_after,
            "hidden_size": protocol_hidden_size,
            "encoder_features_dim": protocol_encoder_dim,
            "source_hashes_before": sources_before,
            "source_hashes_after": sources_after,
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256_before": baseline_sha_before,
            "sha256_after": baseline_sha_after,
            "num_timesteps": int(baseline.num_timesteps),
            "policy": baseline_policy,
            "optimizer": baseline_optimizer,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "num_timesteps": int(checkpoint.num_timesteps),
            "policy": checkpoint_policy,
            "optimizer": checkpoint_optimizer,
        },
        "weight_comparison": weights,
        "checks": checks,
        "cuda_available_after_mask": bool(torch.cuda.is_available()),
        "cuda_device_count_after_mask": int(torch.cuda.device_count()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, help="prepared training trial directory; relative paths use repository root")
    parser.add_argument("--checkpoint", required=True, help="positive-timestep checkpoint; relative paths use --trial")
    parser.add_argument("--output", required=True, help="new JSON report path; relative paths use repository root")
    args = parser.parse_args()
    output = resolve_path(args.output, ROOT)
    if output.exists():
        print(f"Refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "failed",
            "read_only": True,
            "training_or_simulation_run": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    report["exit_code"] = 0 if report["status"] == "passed" else 1
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "exit_code": report["exit_code"], "checks": report.get("checks", {}), "output": str(output)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
