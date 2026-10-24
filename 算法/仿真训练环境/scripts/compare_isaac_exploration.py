"""Prepare or launch a paired RSL-RL action-noise-scale experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/isaacsim_physx_rsl_exploration_20261024"
SOURCE_FILES = tuple(sorted(
    {path.relative_to(ROOT).as_posix() for path in (ROOT / "isaac_field").rglob("*.py")}
    | {"scripts/compare_isaac_exploration.py", "scripts/improve_isaac_policy.py"}
))
# The archived Newton/SB3 foundation validator imports this original source
# manifest ABI for read-only forensic checks. Keep its exact historical shape
# separate from new PhysX/RSL trial snapshots.
LEGACY_FOUNDATION_SOURCE_FILES = (
    tuple(sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "isaac_field").glob("*.py")))
    + ("scripts/isaaclab_python.sh", "scripts/compare_isaac_exploration.py",
       "scripts/improve_isaac_policy.py")
)
TRAINING = {
    "timesteps_per_variant": 16_384,
    "num_envs": 4,
    "environment_seed": 68_001,
    "initialization_seed": 68_001,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 5,
    "learning_rate": 1e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "target_kl": 0.03,
    "goal_gain": 1.0,
    "variants": {"narrow": {"log_std_init": -2.0}, "wide": {"log_std_init": -1.5}},
}
SEEDS = tuple(range(82_001, 82_025))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _checkpoint_steps(path: Path) -> int:
    if path.suffix.lower() == ".zip":
        # Historical SB3 checkpoints remain inspectable without loading SB3 or
        # converting the original archive.
        with zipfile.ZipFile(path) as archive:
            return int(json.loads(archive.read("data"))["num_timesteps"])
    from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

    if path.suffix.lower() != ".pt":
        raise ValueError("checkpoint inspection supports historical .zip or native RSL-RL .pt files")
    return int(load_rsl_checkpoint_payload(path, map_location="cpu")["num_timesteps"])


def _source_manifest(
    output: Path, *, copy_files: bool, source_files: tuple[str, ...] | None = None,
) -> dict:
    """Return the long-standing source snapshot ABI for offline protocol readers."""
    rows = []
    for relative in (SOURCE_FILES if source_files is None else source_files):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        snapshot_relative = f"source_snapshot/{relative}"
        digest = _sha256(source)
        if copy_files:
            target = output / snapshot_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rows.append({"path": relative, "snapshot_path": snapshot_relative, "sha256": digest})
    return {"files": rows}


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
    """Freeze a new trial; never reuse the archived Newton/SB3 exploration run."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse experiment output: {output}")
    output.mkdir(parents=True)
    from isaac_field.action_contract import action_config
    from isaac_field.policy_contract import stamp_backend_identity
    from isaac_field.task import Config

    task = asdict(Config(plants=16, layout="rows", row_spacing=0.8, plant_spacing=0.3,
                         terrain_enabled=False, max_time=20.0, damage_penalty=20.0,
                         goal_progress_budget=5.0))
    protocol = {
        "schema": "isaac-field-rsl-action-noise-pair-v1",
        "status": "prepared",
        "action_config": action_config(),
        "config": task,
        "task": {"mode": "goto", "goal_distance_m": [1.0, 3.0],
                 "goal_angle_rad": math.pi / 3.0, "distance_cost": 0.0},
        "training": TRAINING,
        "evaluation": {"environment_seed": 80_003, "seeds": list(SEEDS),
                       "goal_distance_m": [1.0, 3.0], "goal_angle_rad": math.pi / 3.0,
                       "distance_cost": 0.0, "deterministic": True,
                       "record_trajectories": True,
                       "claim_limit": "Paired 24-seed development comparison; no general competence claim."},
        "exploration": {
            "method": "RSL-RL diagonal Gaussian actor",
            "only_variant_change": "initial log standard deviation",
            "variants": TRAINING["variants"],
            "source_policy": "fresh random actor/critic with matched initialization seed",
            "gSDE": "not available in this RSL field model; no SB3 fallback",
        },
        "source_snapshot": {"files": _snapshot(output)},
    }
    stamp_backend_identity(protocol)
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": _sha256(output / "protocol.json"),
        "num_timesteps": 0,
    })
    return {"output": str(output), "source_files": len(SOURCE_FILES),
            "variants": list(TRAINING["variants"]), "heldout_seeds": len(SEEDS)}


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
    if protocol.get("schema") != "isaac-field-rsl-action-noise-pair-v1":
        raise ValueError("unsupported RSL exploration protocol schema")
    if protocol.get("training") != TRAINING or protocol.get("evaluation", {}).get("seeds") != list(SEEDS):
        raise ValueError("training or held-out conditions differ from the frozen protocol")
    rows = protocol.get("source_snapshot", {}).get("files", [])
    if {row.get("path") for row in rows} != set(SOURCE_FILES):
        raise ValueError("source snapshot inventory differs from this runner")
    for row in rows:
        live, copied = ROOT / row["path"], output / row["snapshot_path"]
        if _sha256(live) != row["sha256"] or _sha256(copied) != row["sha256"]:
            raise ValueError(f"source snapshot mismatch: {row['path']}")
    return protocol


def _open_env(cfg, protocol: dict, seed: int, num_envs: int, auto_reset: bool):
    from isaac_field.runtime import create_env

    task = protocol["task"]
    return create_env(
        cfg=cfg, mode=task["mode"], num_envs=num_envs, seed=seed,
        auto_reset=auto_reset, goal_distance=tuple(task["goal_distance_m"]),
        goal_angle=task["goal_angle_rad"], distance_cost=task["distance_cost"], rsl=True,
    )


def _evaluate(checkpoint: Path, cfg, protocol: dict, label: str, output: Path, device: str) -> dict:
    from isaac_field.env import SingleFieldAdapter
    from isaac_field.learning import evaluate_model
    from scripts.improve_isaac_policy import _add_safety_summary

    spec = protocol["evaluation"]
    engine = _open_env(cfg, protocol, spec["environment_seed"], 1, False)
    try:
        report = evaluate_model(
            SingleFieldAdapter(engine), checkpoint, spec["seeds"],
            goal_distance=tuple(spec["goal_distance_m"]), goal_angle=spec["goal_angle_rad"],
            mode=protocol["task"]["mode"], deterministic=spec["deterministic"],
            device=device, record_trajectories=spec["record_trajectories"],
        )
    finally:
        engine.close()
    report = _add_safety_summary(report)
    _json(output / f"evaluation_{label}.json", report)
    return report


def _pair_initial_conditions(reports: dict[str, dict]) -> dict:
    import numpy as np

    rows = {label: {row["seed"]: row for row in report["rows"]}
            for label, report in reports.items()}
    baseline = rows[next(iter(rows))]
    mismatches = []
    for label, variants in rows.items():
        for seed in sorted(set(baseline) | set(variants)):
            reference, row = baseline.get(seed), variants.get(seed)
            if reference is None or row is None:
                mismatches.append({"seed": seed, "variant": label, "differences": ["missing_seed"]})
                continue
            a, b = reference["reset_initial_condition"], row["reset_initial_condition"]
            differences = []
            if a["plant_xy_sha256"] != b["plant_xy_sha256"]:
                differences.append("plant_xy_sha256")
            for key in ("start_pose_xy_yaw", "goal_xy"):
                if not np.allclose(a[key], b[key], rtol=0.0, atol=1e-8):
                    differences.append(key)
            if not row.get("trajectory_start_matches_reset", False):
                differences.append("trajectory_start_matches_reset")
            if differences:
                mismatches.append({"seed": seed, "variant": label, "differences": differences})
    count = len(SEEDS)
    return {"valid": not mismatches and all(len(variants) == count for variants in rows.values()),
            "paired_seed_count": len(baseline), "mismatches": mismatches}


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
                    gae_lambda=hp["gae_lambda"], target_kl=hp["target_kl"],
                    log_std_init=variant["log_std_init"],
                )
            finally:
                env.close()
            require_backend_identity(result.model.field_policy_contract)
            runtime["num_timesteps"] += result.trained_timesteps
            training_results[label] = {
                "checkpoint": str(result.checkpoint),
                "checkpoint_sha256": _sha256(result.checkpoint),
                "requested_timesteps": result.requested_timesteps,
                "trained_timesteps": result.trained_timesteps,
                "total_timesteps": result.total_timesteps,
                "seed": result.seed,
                "log_std_init": variant["log_std_init"],
                "optimizer": "fresh official RSL-RL optimizer",
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
        labels = tuple(reports)
        summary = {
            "schema": "isaac-field-rsl-action-noise-pair-summary-v1",
            "backend_identity": {"simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl"},
            "training": training_results,
            "evaluation_summaries": {label: report["summary"] for label, report in reports.items()},
            "paired_initial_conditions": pairing,
            "assessment_wide_vs_narrow": _safe_goto_assessment(reports[labels[0]], reports[labels[1]]),
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
