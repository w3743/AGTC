"""Freeze or launch a paired RSL-RL GAE-lambda comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from typing import Any

from scripts.compare_isaac_exploration import (
    _evaluate, _json, _open_env, _pair_initial_conditions, _sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/isaacsim_physx_rsl_gae_20261024"
SOURCE_FILES = (
    "isaac_field/runtime.py", "isaac_field/learning.py", "isaac_field/policy_contract.py",
    "isaac_field/rsl_rl_cfg.py", "isaac_field/rsl_rl_policy.py", "isaac_field/rsl_rl_inference.py",
    "scripts/compare_isaac_exploration.py", "scripts/compare_isaac_gae.py",
    "scripts/improve_isaac_policy.py",
)
SEEDS = tuple(range(82_001, 82_025))
BASE_TRAINING = {
    "timesteps_per_variant": 16_384, "num_envs": 4,
    "environment_seed": 68_001, "initialization_seed": 68_001,
    "n_steps": 128, "batch_size": 128, "n_epochs": 5,
    "learning_rate": 1e-4, "gamma": 0.995, "target_kl": 0.03,
    "goal_gain": 1.0, "log_std_init": -2.0,
    "variants": {"gae_095": {"gae_lambda": 0.95}, "gae_099": {"gae_lambda": 0.99}},
}


def _snapshot(output: Path) -> list[dict[str, str]]:
    rows = []
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        snapshot_relative = f"source_snapshot/{relative}"
        target = output / snapshot_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": relative, "snapshot_path": snapshot_relative, "sha256": _sha256(source)})
    return rows


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse experiment output: {output}")
    output.mkdir(parents=True)
    from isaac_field.action_contract import action_config
    from isaac_field.policy_contract import stamp_backend_identity
    from isaac_field.task import Config

    config = asdict(Config(plants=16, layout="rows", row_spacing=0.8, plant_spacing=0.3,
                           terrain_enabled=False, max_time=20.0, damage_penalty=20.0,
                           goal_progress_budget=5.0))
    protocol = {
        "schema": "isaac-field-rsl-gae-pair-v1", "status": "prepared",
        "action_config": action_config(), "config": config,
        "task": {"mode": "goto", "goal_distance_m": [1.0, 3.0],
                 "goal_angle_rad": math.pi / 3.0, "distance_cost": 0.0},
        "training": BASE_TRAINING,
        "experiment": {
            "only_variant_change": "GAE lambda", "source_policy": "fresh random RSL-RL actor/critic",
            "source_optimizer": "fresh official RSL-RL optimizer for each variant",
        },
        "evaluation": {"environment_seed": 80_003, "seeds": list(SEEDS),
                       "goal_distance_m": [1.0, 3.0], "goal_angle_rad": math.pi / 3.0,
                       "distance_cost": 0.0, "deterministic": True,
                       "record_trajectories": True,
                       "claim_limit": "Paired 24-seed development comparison; no general competence claim."},
        "source_snapshot": {"files": _snapshot(output)},
    }
    stamp_backend_identity(protocol)
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": _sha256(output / "protocol.json"),
        "num_timesteps": 0,
    })
    return {"output": str(output), "source_files": len(SOURCE_FILES),
            "variants": list(BASE_TRAINING["variants"]), "heldout_seeds": len(SEEDS)}


def preflight(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --launch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (status.get("status") != "prepared"
            or status.get("protocol_sha256") != _sha256(protocol_path)):
        raise ValueError("experiment is not an unchanged prepared run")
    from isaac_field.policy_contract import require_backend_identity

    require_backend_identity(protocol)
    if (protocol.get("schema") != "isaac-field-rsl-gae-pair-v1"
            or protocol.get("training") != BASE_TRAINING
            or protocol.get("evaluation", {}).get("seeds") != list(SEEDS)):
        raise ValueError("GAE protocol differs from this runner")
    rows = protocol.get("source_snapshot", {}).get("files", [])
    if {row.get("path") for row in rows} != set(SOURCE_FILES):
        raise ValueError("source snapshot inventory differs from this runner")
    for row in rows:
        live, copied = ROOT / row["path"], output / row["snapshot_path"]
        if _sha256(live) != row["sha256"] or _sha256(copied) != row["sha256"]:
            raise ValueError(f"source snapshot mismatch: {row['path']}")
    return protocol


def launch(output: Path = DEFAULT_OUTPUT, *, device: str = "cuda:0") -> dict:
    from isaac_field.runtime import launch_app

    protocol = preflight(output)
    output = output.resolve()
    with launch_app(device=device, headless=True):
        return _launch_preflighted(output, protocol, device)


def _launch_preflighted(output: Path, protocol: dict, device: str) -> dict:
    from isaac_field.learning import train_stage
    from isaac_field.policy_contract import require_backend_identity
    from isaac_field.task import Config
    from scripts.improve_isaac_policy import _safe_goto_assessment

    runtime = {"num_timesteps": 0, "protocol_sha256": _sha256(output / "protocol.json")}
    _json(output / "run_status.json", {"status": "running", **runtime, "phase": "training"})
    try:
        cfg = Config(**protocol["config"])
        reports, training_results = {}, {}
        hp = protocol["training"]
        for label, variant in hp["variants"].items():
            env = _open_env(cfg, protocol, hp["environment_seed"], hp["num_envs"], True)
            checkpoint = output / label / "field_policy_after.pt"
            try:
                result = train_stage(
                    env, None, hp["timesteps_per_variant"], checkpoint,
                    seed=hp["initialization_seed"], device=device,
                    goal_gain=hp["goal_gain"], n_steps=hp["n_steps"],
                    batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                    learning_rate=hp["learning_rate"], gamma=hp["gamma"],
                    gae_lambda=variant["gae_lambda"], target_kl=hp["target_kl"],
                    log_std_init=hp["log_std_init"],
                )
            finally:
                env.close()
            require_backend_identity(result.model.field_policy_contract)
            runtime["num_timesteps"] += result.trained_timesteps
            training_results[label] = {
                "checkpoint": str(result.checkpoint), "checkpoint_sha256": _sha256(result.checkpoint),
                "requested_timesteps": result.requested_timesteps,
                "trained_timesteps": result.trained_timesteps,
                "total_timesteps": result.total_timesteps, "seed": result.seed,
                "gae_lambda": variant["gae_lambda"],
                "backend_identity": {
                    key: result.model.field_policy_contract[key]
                    for key in ("simulator", "physics_backend", "learning_backend")
                },
            }
            _json(output / "run_status.json", {"status": "running", **runtime, "phase": f"evaluating_{label}"})
            reports[label] = _evaluate(result.checkpoint, cfg, protocol, label, output, device)
        pairing = _pair_initial_conditions(reports)
        if not pairing["valid"]:
            raise ValueError(f"evaluation initial conditions do not pair: {pairing['mismatches'][:5]}")
        summary = {
            "schema": "isaac-field-rsl-gae-pair-summary-v1",
            "backend_identity": {"simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl"},
            "training": training_results,
            "evaluation_summaries": {label: report["summary"] for label, report in reports.items()},
            "paired_initial_conditions": pairing,
            "assessment_gae_099_vs_095": _safe_goto_assessment(reports["gae_095"], reports["gae_099"]),
            "claim_limit": protocol["evaluation"]["claim_limit"],
        }
        _json(output / "comparison_summary.json", summary)
        _json(output / "run_status.json", {"status": "completed", **runtime,
                                             "summary": "comparison_summary.json"})
        return summary
    except BaseException as exc:
        _json(output / "run_status.json", {"status": "failed", **runtime,
            "exception": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = prepare(args.output) if args.prepare else launch(args.output, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
