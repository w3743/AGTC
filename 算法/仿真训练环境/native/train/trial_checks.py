#!/usr/bin/env python3
"""Standard-library gates for the frozen native V1.1 training/evaluation runners."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from contract_identity import (
    IDENTITY_SCHEMA_VERSION,
    canonical_contract_identity,
    contract_identity_sha256,
)

PROTOCOL_VERSION = "V1.1"
EXPECTED_PROTOCOL_SHA = "7ff8d6fcba79defbbc907b09527b244fb6e27134ed6c43de13ed1bdc197f2998"
EXPECTED_CHANNELS = {
    "lidar": "warp_kernel",
    "depth": "analytic_warp",
    "detections": "analytic_detector_warp",
    "feedback": "vqf_reference",
}
FORMAL_ITERATIONS = 300
FORMAL_SAMPLES = 64 * 32
METRIC_FLOAT_FIELDS = (
    "wall_s", "collection_s", "policy_act_s", "env_step_s", "learning_s",
    "samples_per_s", "end_to_end_samples_per_s", "mean_reward", "ppo_loss",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _same_number(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def require_official_physics(sim_dt: Any, decimation: Any) -> None:
    if not _same_number(sim_dt, 0.001) or str(decimation) != "50":
        raise ValueError(
            f"formal runner requires sim_dt=0.001 and decimation=50; got {sim_dt!r}/{decimation!r}"
        )


def _contract_errors(contract: dict[str, Any], identity: dict[str, Any]) -> tuple[list[str], str | None]:
    errors: list[str] = []
    try:
        canonical = canonical_contract_identity(contract)
        digest = contract_identity_sha256(contract)
    except (TypeError, ValueError) as exc:
        return [f"invalid task contract: {exc}"], None

    if contract.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
        errors.append("contract identity_schema_version is missing or unsupported")
    if contract.get("task_identity") != canonical:
        errors.append("contract task_identity does not equal canonical contract identity")
    if contract.get("task_identity_sha256") != digest:
        errors.append("contract task_identity_sha256 does not match canonical identity")
    if contract.get("observation_contract_sha256") != digest:
        errors.append("contract observation_contract_sha256 alias does not match task identity")
    if contract.get("task_backend") != "fast_stop_v2":
        errors.append("contract task_backend is not fast_stop_v2")
    if contract.get("task", {}).get("backend") != "fast_stop_v2":
        errors.append("contract task.backend is not fast_stop_v2")

    env = identity.get("env") if isinstance(identity.get("env"), dict) else {}
    if identity.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
        errors.append("identity.json schema is missing or unsupported")
    if identity.get("task_backend") != "fast_stop_v2":
        errors.append("identity.json task_backend is not fast_stop_v2")
    if identity.get("task_identity") != canonical:
        errors.append("identity.json task_identity does not equal canonical contract identity")
    if identity.get("task_identity_sha256") != digest:
        errors.append("identity.json task_identity_sha256 does not match contract")
    if env.get("task_backend") != "fast_stop_v2":
        errors.append("identity.env.task_backend is not fast_stop_v2")
    if env.get("task_identity_sha256") != digest:
        errors.append("identity.env.task_identity_sha256 does not match contract")
    if env.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
        errors.append("identity.env identity schema is missing or unsupported")
    if identity.get("physics") != "newton_xpbd" or identity.get("device") != "cuda":
        errors.append("identity.json does not record the formal Newton/CUDA recipe")
    policy = identity.get("policy") if isinstance(identity.get("policy"), dict) else {}
    if policy.get("policy_type") != "lstm" or int(policy.get("hidden_size", -1)) != 512:
        errors.append("identity.json policy is not LSTM-512")

    physics = contract.get("physics") if isinstance(contract.get("physics"), dict) else {}
    env_physics = env.get("physics") if isinstance(env.get("physics"), dict) else {}
    for label, item in (("contract.physics", physics), ("identity.env.physics", env_physics)):
        if item.get("engine") != "newton_xpbd":
            errors.append(f"{label}.engine is not newton_xpbd")
        if str(item.get("device")) != "cuda:0":
            errors.append(f"{label}.device is not cuda:0")
        if not _same_number(item.get("dt_s"), 0.001):
            errors.append(f"{label}.dt_s is not 0.001")
        if str(item.get("decimation")) != "50":
            errors.append(f"{label}.decimation is not 50")
        if not _same_number(item.get("policy_dt_s"), 0.05):
            errors.append(f"{label}.policy_dt_s is not 0.05")

    if not _same_number(contract.get("physics_dt_s"), 0.001):
        errors.append("contract.physics_dt_s is not 0.001")
    if str(contract.get("decimation")) != "50":
        errors.append("contract.decimation is not 50")
    if not _same_number(contract.get("policy_dt_s"), 0.05):
        errors.append("contract.policy_dt_s is not 0.05")
    if contract.get("physics_experimental") is not False:
        errors.append("contract.physics_experimental is not false")
    if not _same_number(env.get("sim_dt_s"), 0.001) or str(env.get("decimation")) != "50":
        errors.append("identity.env does not record official 1 ms/50 physics")
    if env.get("experimental_physics") is not False:
        errors.append("identity.env.experimental_physics is not false")

    channels = contract.get("channels") if isinstance(contract.get("channels"), dict) else {}
    expected = {**EXPECTED_CHANNELS, "state": None}
    for name, backend in expected.items():
        channel = channels.get(name) if isinstance(channels.get(name), dict) else {}
        if channel.get("mode") != "real":
            errors.append(f"observation channel {name} is not real")
        if backend is not None and channel.get("backend") != backend:
            errors.append(f"observation channel {name} backend is not {backend}")
    return errors, digest


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(key) or _contains_nonfinite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _trial_evidence_errors(trial: Path, digest: str | None) -> list[str]:
    errors: list[str] = []
    summary_path = trial / "summary.json"
    if not summary_path.is_file():
        errors.append("summary.json is missing")
    else:
        summary_read = False
        try:
            summary = _read_json(summary_path)
            summary_read = True
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            summary = {}
            errors.append(f"summary.json is unreadable: {exc}")
        if summary_read:
            if _contains_nonfinite(summary):
                errors.append("summary.json contains non-finite fields")
            expected_summary = {
                "iterations": FORMAL_ITERATIONS,
                "start_iteration": 0,
                "end_iteration": FORMAL_ITERATIONS,
                "num_envs": 64,
                "num_steps": 32,
                "device": "cuda",
                "physics": "newton_xpbd",
            }
            for name, expected in expected_summary.items():
                actual = summary.get(name)
                if type(actual) is not type(expected) or actual != expected:
                    errors.append(f"summary.json {name} is not {expected}")
            if summary.get("observation_contract_sha256") != digest:
                errors.append("summary.json observation contract SHA does not match the trial")
            wall = summary.get("total_wall_s")
            if (not isinstance(wall, (int, float)) or isinstance(wall, bool)
                    or not math.isfinite(float(wall)) or wall <= 0):
                errors.append("summary.json total_wall_s is missing, non-finite, or non-positive")

    metrics_path = trial / "metrics.jsonl"
    rows: list[Any] = []
    if not metrics_path.is_file():
        errors.append("metrics.jsonl is missing")
    else:
        try:
            with metrics_path.open("r", encoding="utf-8-sig") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        errors.append(f"metrics.jsonl line {line_number} is blank")
                        continue
                    try:
                        rows.append(json.loads(line))
                    except (json.JSONDecodeError, TypeError) as exc:
                        errors.append(f"metrics.jsonl line {line_number} is invalid JSON: {exc}")
        except (OSError, UnicodeError) as exc:
            errors.append(f"metrics.jsonl is unreadable: {exc}")
    if len(rows) != FORMAL_ITERATIONS:
        errors.append(f"metrics.jsonl has {len(rows)} rows; expected {FORMAL_ITERATIONS}")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"metrics.jsonl row {index} is not an object")
            continue
        if _contains_nonfinite(row):
            errors.append(f"metrics.jsonl row {index} contains non-finite fields")
        if type(row.get("iter")) is not int or row.get("iter") != index:
            errors.append(f"metrics.jsonl row {index} has unexpected iteration {row.get('iter')!r}")
        if type(row.get("samples")) is not int or row.get("samples") != FORMAL_SAMPLES:
            errors.append(f"metrics.jsonl row {index} samples is not {FORMAL_SAMPLES}")
        if row.get("observation_contract_sha256") != digest:
            errors.append(f"metrics.jsonl row {index} observation contract SHA does not match the trial")
        for name in METRIC_FLOAT_FIELDS:
            value = row.get(name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                errors.append(f"metrics.jsonl row {index} {name} is missing or non-finite")
        for name in ("wall_s", "collection_s", "policy_act_s", "env_step_s", "learning_s"):
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value < 0:
                errors.append(f"metrics.jsonl row {index} {name} is negative")
        for name in ("samples_per_s", "end_to_end_samples_per_s"):
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value <= 0:
                errors.append(f"metrics.jsonl row {index} {name} is non-positive")
        for name in ("nonfinite_env_resets", "nonfinite_actions"):
            value = row.get(name)
            if type(value) is not int:
                errors.append(f"metrics.jsonl row {index} {name} is missing or not an integer")
            elif value != 0:
                errors.append(f"metrics.jsonl row {index} reports {name}={value}")

    events_path = trial / "nonfinite_events.jsonl"
    if events_path.exists():
        try:
            event_bytes = events_path.read_bytes()
            if event_bytes:
                errors.append(f"nonfinite_events.jsonl is non-empty ({len(event_bytes)} bytes)")
        except OSError as exc:
            errors.append(f"nonfinite_events.jsonl is unreadable: {exc}")

    failure_path = trial / "failure.json"
    if failure_path.exists():
        try:
            failure = _read_json(failure_path)
            reason = failure.get("reason")
            errors.append(f"failure.json is present (reason={reason!r})")
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            errors.append(f"failure.json is present but unreadable: {exc}")
    return errors


def _protocol_sha(path: Path) -> str:
    info = _read_json(path)
    digest = info.get("protocol_sha256")
    if digest != EXPECTED_PROTOCOL_SHA:
        raise ValueError(f"{path} does not identify frozen V1.1 protocol {EXPECTED_PROTOCOL_SHA}")
    return str(digest)


def write_manifest(trial: Path, train_exit: int, protocol_sha_path: Path) -> int:
    protocol_sha = _protocol_sha(protocol_sha_path)
    contract_path = trial / "observation_contract.json"
    identity_path = trial / "identity.json"
    errors: list[str] = []
    contract: dict[str, Any] = {}
    identity: dict[str, Any] = {}
    if not contract_path.is_file():
        errors.append("observation_contract.json is missing")
    else:
        contract = _read_json(contract_path)
    if not identity_path.is_file():
        errors.append("identity.json is missing")
    else:
        identity = _read_json(identity_path)
    digest = None
    if contract and identity:
        contract_errors, digest = _contract_errors(contract, identity)
        errors.extend(contract_errors)
    errors.extend(_trial_evidence_errors(trial, digest))

    checkpoint_names = sorted(p.name for p in trial.glob("*.pt") if p.is_file())
    if train_exit == 0 and not checkpoint_names:
        errors.append("training exited 0 but produced no checkpoint")
    physics = contract.get("physics") if isinstance(contract.get("physics"), dict) else {}
    dt = physics.get("dt_s", contract.get("physics_dt_s"))
    decimation = physics.get("decimation", contract.get("decimation"))
    official = _same_number(dt, 0.001) and str(decimation) == "50"
    formal = train_exit == 0 and not errors and official
    manifest = {
        "protocol_sha256": protocol_sha,
        "protocol_file": "PROTOCOL_NATIVE_V1.1.md",
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cli_args_file": "cli_args.txt",
        "exit_code": train_exit,
        "task_backend": contract.get("task_backend"),
        "identity_schema_version": contract.get("identity_schema_version"),
        "task_identity_sha256": digest,
        "observation_contract_sha256": contract.get("observation_contract_sha256"),
        "capability_metrics_allowed": False,
        "phase": "formal_native_v1_trial",
        "formal_eligible": bool(formal),
        "validation_errors": errors,
        "physics": {
            "sim_dt": dt,
            "decimation": decimation,
            "experimental": not official,
        },
        "channels_all_real": not any("not real" in e for e in errors),
        "channels": {
            name: {
                "mode": ((contract.get("channels") or {}).get(name) or {}).get("mode"),
                "backend": ((contract.get("channels") or {}).get(name) or {}).get("backend"),
            }
            for name in ("lidar", "depth", "detections", "state", "feedback")
        },
        "checkpoints": checkpoint_names,
    }
    target = trial / "trial_manifest.json"
    tmp = trial / "trial_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    print("TRIAL_MANIFEST " + json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 3 if train_exit == 0 and not formal else 0


def preflight_checkpoint(checkpoint: Path, out: Path, protocol_sha_path: Path) -> int:
    checkpoint = checkpoint.resolve()
    trial = checkpoint.parent
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    current_sha = _protocol_sha(protocol_sha_path)
    trial_sha = _protocol_sha(trial / "PROTOCOL_NATIVE_V1.1.sha256")
    if current_sha != trial_sha:
        raise ValueError("checkpoint trial uses a different frozen protocol SHA")
    contract = _read_json(trial / "observation_contract.json")
    identity = _read_json(trial / "identity.json")
    errors, digest = _contract_errors(contract, identity)
    if errors:
        raise ValueError("checkpoint sidecar contract rejected: " + "; ".join(errors))
    manifest = _read_json(trial / "trial_manifest.json")
    if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("protocol_sha256") != current_sha:
        raise ValueError("checkpoint trial manifest protocol identity is invalid")
    if manifest.get("exit_code") != 0 or manifest.get("formal_eligible") is not True:
        raise ValueError("checkpoint trial is not a successful formal V1.1 trial")
    evidence_errors = _trial_evidence_errors(trial, digest)
    if evidence_errors:
        raise ValueError("checkpoint trial run evidence rejected: " + "; ".join(evidence_errors))
    if manifest.get("task_backend") != "fast_stop_v2" or manifest.get("task_identity_sha256") != digest:
        raise ValueError("checkpoint trial manifest task identity does not match its contract")
    if not _same_number((manifest.get("physics") or {}).get("sim_dt"), 0.001):
        raise ValueError("checkpoint trial manifest is not 1 ms")
    if str((manifest.get("physics") or {}).get("decimation")) != "50":
        raise ValueError("checkpoint trial manifest is not decimation 50")
    if (manifest.get("physics") or {}).get("experimental") is not False:
        raise ValueError("checkpoint trial manifest marks physics experimental")
    if checkpoint.name not in manifest.get("checkpoints", []):
        raise ValueError("checkpoint is not listed in trial_manifest.json")

    out.mkdir(parents=True, exist_ok=True)
    proof = {
        "checkpoint": str(checkpoint),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": current_sha,
        "task_backend": "fast_stop_v2",
        "task_identity_sha256": digest,
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "sidecar_contract_valid": True,
        "checkpoint_payload_validation": "delegated_to_eval_native_preflight_and_strict_match",
    }
    (out / "checkpoint_preflight.json").write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("CHECKPOINT_PREFLIGHT " + json.dumps(proof, ensure_ascii=False, sort_keys=True))
    return 0


def check_evaluation(out: Path, expected_seeds: list[int]) -> int:
    proof = _read_json(out / "checkpoint_preflight.json")
    evaluation = _read_json(out / "evaluation.json")
    errors: list[str] = []
    expected_sha = proof.get("task_identity_sha256")
    policy = evaluation.get("policy") if isinstance(evaluation.get("policy"), dict) else {}
    config = evaluation.get("config") if isinstance(evaluation.get("config"), dict) else {}
    physics = evaluation.get("physics") if isinstance(evaluation.get("physics"), dict) else {}

    if evaluation.get("task_backend") != "fast_stop_v2":
        errors.append("evaluation task_backend is not fast_stop_v2")
    if policy.get("task_identity_sha256") != expected_sha or evaluation.get("task_identity_sha256") != expected_sha:
        errors.append("evaluation task identity SHA does not match checkpoint trial")
    if evaluation.get("checkpoint_contract_validation") != "strict_match":
        errors.append("eval_native did not report checkpoint contract strict_match")
    if policy.get("checkpoint_contract_validation") != "strict_match":
        errors.append("evaluation policy metadata did not report checkpoint strict_match")
    if not _same_number(physics.get("dt_s"), 0.001) or str(physics.get("decimation")) != "50":
        errors.append("evaluation physics is not 1 ms/50")
    if physics.get("experimental") is not False:
        errors.append("evaluation physics is marked experimental")
    if config.get("task_backend") != "fast_stop_v2":
        errors.append("evaluation config task_backend is not fast_stop_v2")
    if config.get("physics") != "newton_xpbd" or config.get("device") != "cuda":
        errors.append("evaluation config is not Newton/CUDA")
    for key, expected in (("lidar", "warp"), ("feedback", "vqf"),
                          ("depth", "analytic"), ("detections", "analytic")):
        if config.get(key) != expected:
            errors.append(f"evaluation config {key} is not {expected}")
    if config.get("detections_warp") is not True:
        errors.append("evaluation detections_warp is not enabled")
    if not _same_number(config.get("sim_dt_s"), 0.001) or str(config.get("decimation")) != "50":
        errors.append("evaluation CLI physics is not 1 ms/50")
    if config.get("max_steps") != 1200:
        errors.append("evaluation max_steps is not 1200")
    if config.get("seeds") != expected_seeds:
        errors.append(f"evaluation seeds do not match required set {expected_seeds}")
    if evaluation.get("capability_metrics_allowed") is not False:
        errors.append("evaluation incorrectly enables capability metrics")

    rows = evaluation.get("rows")
    if not isinstance(rows, list):
        errors.append("evaluation rows are missing or malformed")
        rows = []
    if [row.get("seed") for row in rows if isinstance(row, dict)] != expected_seeds:
        errors.append("evaluation rows do not cover the required seed set exactly once in order")
    invalid_reasons = {"runtime_error", "non_finite_state", "non_finite_reward"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"evaluation row {index} is malformed")
            continue
        if row.get("reason") in invalid_reasons:
            errors.append(f"evaluation row {index} reports {row.get('reason')}")
        if row.get("error"):
            errors.append(f"evaluation row {index} contains an error")
        if _contains_nonfinite(row):
            errors.append(f"evaluation row {index} contains a non-finite field")
    diagnostics = evaluation.get("env_diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            errors.append("env_diagnostics is malformed")
        else:
            if int(diagnostics.get("count", 0) or 0) > 0:
                errors.append("env_diagnostics reports non-finite events")
            if diagnostics.get("events"):
                errors.append("env_diagnostics contains non-finite events")
            if int(diagnostics.get("sensor_frames", 0) or 0) > 0:
                errors.append("env_diagnostics reports non-finite sensor frames")
            if diagnostics.get("last_mask"):
                errors.append("env_diagnostics reports non-finite environments")
            if _contains_nonfinite(diagnostics):
                errors.append("env_diagnostics contains non-finite fields")
    verdict = {
        "match": not errors,
        "protocol_sha256": proof.get("protocol_sha256"),
        "task_backend": evaluation.get("task_backend"),
        "checkpoint_task_identity_sha256": expected_sha,
        "evaluation_task_identity_sha256": evaluation.get("task_identity_sha256"),
        "checkpoint_contract_validation": evaluation.get("checkpoint_contract_validation"),
        "physics": physics,
        "seeds": config.get("seeds"),
        "validation_errors": errors,
        "capability_metrics_allowed": False,
    }
    target = out / "sha_check.json"
    tmp = out / "sha_check.json.tmp"
    tmp.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    print("EVALUATION_CHECK " + json.dumps(verdict, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    physics_parser = sub.add_parser("require-official-physics")
    physics_parser.add_argument("--sim-dt", required=True)
    physics_parser.add_argument("--decimation", required=True)
    manifest_parser = sub.add_parser("write-manifest")
    manifest_parser.add_argument("--trial-dir", type=Path, required=True)
    manifest_parser.add_argument("--train-exit", type=int, required=True)
    manifest_parser.add_argument("--protocol-sha", type=Path, required=True)
    preflight_parser = sub.add_parser("preflight-checkpoint")
    preflight_parser.add_argument("--checkpoint", type=Path, required=True)
    preflight_parser.add_argument("--out", type=Path, required=True)
    preflight_parser.add_argument("--protocol-sha", type=Path, required=True)
    eval_parser = sub.add_parser("check-evaluation")
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument("--seeds", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "require-official-physics":
            require_official_physics(args.sim_dt, args.decimation)
            return 0
        if args.command == "write-manifest":
            return write_manifest(args.trial_dir, args.train_exit, args.protocol_sha)
        if args.command == "preflight-checkpoint":
            return preflight_checkpoint(args.checkpoint, args.out, args.protocol_sha)
        if args.command == "check-evaluation":
            seeds = [int(value) for value in args.seeds.split(",") if value]
            return check_evaluation(args.out, seeds)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"TRIAL_CHECK_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
