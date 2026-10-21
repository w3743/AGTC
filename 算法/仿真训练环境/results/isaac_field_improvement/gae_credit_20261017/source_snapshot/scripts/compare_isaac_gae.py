"""Freeze or launch a paired GAE-lambda RecurrentPPO comparison."""
from __future__ import annotations

import argparse, json, math, shutil
from dataclasses import asdict
from pathlib import Path

from scripts.compare_isaac_exploration import (
    SOURCE_FILES as DEPENDENCIES, _evaluate, _json, _open_env, _pair_initial_conditions,
    _sha256, _checkpoint_steps,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/isaac_field_improvement/gae_credit_20261017"
PREV = ROOT / "results/isaac_field_improvement/exploration_gsde_20261017"
SOURCE = ROOT / "results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip"
CONTROL = PREV / "gaussian/field_policy_after.zip"
BASELINE_EVAL, GAUSSIAN_EVAL = PREV / "evaluation_baseline.json", PREV / "evaluation_gaussian.json"
FILES = tuple(sorted(set(DEPENDENCIES + ("scripts/compare_isaac_gae.py",))))
SEEDS = list(range(82001, 82025))


def _ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path),
            "num_timesteps": _checkpoint_steps(path) if path.suffix == ".zip" else None}


def _previous():
    path = PREV / "protocol.json"
    p = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["path"]: r for r in p["source_snapshot"]["files"]}
    if set(rows) != set(DEPENDENCIES):
        raise ValueError("previous protocol source inventory mismatch")
    for name, row in rows.items():
        if (_sha256(ROOT / name) != row["sha256"] or
                _sha256(PREV / row["snapshot_path"]) != row["sha256"]):
            raise ValueError(f"previous source snapshot mismatch: {name}")
    from isaac_field.task import Config
    expected = asdict(Config(plants=16, layout="rows", row_spacing=.8, plant_spacing=.3,
                             terrain_enabled=False, max_time=20., damage_penalty=20., goal_progress_budget=5.))
    hp = {"n_steps": 128, "batch_size": 128, "n_epochs": 5, "learning_rate": 1e-4,
          "gamma": .995, "target_kl": .03, "goal_gain": 100.}
    if (p["config"] != expected or p["source_checkpoint"]["path"] != SOURCE.relative_to(ROOT).as_posix()
            or p["source_checkpoint"]["num_timesteps"] != 53248 or p["training"]["hyperparameters"] != hp
            or p["training"]["timesteps_per_variant"] != 16384 or p["training"]["num_envs"] != 4
            or p["training"]["environment_seed"] != 68001 or p["training"]["initialization_seed"] != 68001
            or p["evaluation"]["environment_seed"] != 80003 or p["evaluation"]["seeds"] != SEEDS):
        raise ValueError("previous protocol differs from requested paired conditions")
    return p, _sha256(path)


def _reports():
    baseline, gaussian = (json.loads(p.read_text(encoding="utf-8")) for p in (BASELINE_EVAL, GAUSSIAN_EVAL))
    for report, checkpoint in ((baseline, SOURCE), (gaussian, CONTROL)):
        if (Path(report["checkpoint"]).resolve() != checkpoint.resolve()
                or report["distance_cost"] != 0.0
                or report["reward"] != "raw_task_reward_distance_cost_zero"
                or [int(r["seed"]) for r in report["rows"]] != SEEDS):
            raise ValueError("reused evaluation does not match checkpoint/reward/seed protocol")
    pairing = _pair_initial_conditions({"baseline": baseline, "gaussian": gaussian})
    if not pairing["valid"]:
        raise ValueError(f"reused evaluations are not paired: {pairing['mismatches'][:3]}")
    return baseline, gaussian, pairing


def _snapshot(output):
    rows = []
    for name in FILES:
        source, digest, rel = ROOT / name, _sha256(ROOT / name), f"source_snapshot/{name}"
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": name, "snapshot_path": rel, "sha256": digest})
    return rows


def prepare():
    if OUT.exists():
        raise FileExistsError(f"refusing to reuse experiment output: {OUT}")
    old, old_hash = _previous()
    _, _, pairing = _reports()
    if (_checkpoint_steps(SOURCE) != 53248 or _checkpoint_steps(CONTROL) != 69632):
        raise ValueError("source/control checkpoint timestep markers mismatch")
    from isaac_field.task import Config
    OUT.mkdir(parents=True)
    protocol = {
        "schema": "isaac-field-gaelambda-paired-v1", "status": "prepared",
        "source_snapshot": {"files": _snapshot(OUT)},
        "previous_protocol": {"path": "results/isaac_field_improvement/exploration_gsde_20261017/protocol.json",
                              "sha256": old_hash},
        "checkpoints": {"source_53248": _ref(SOURCE), "gaussian_control_69632": _ref(CONTROL)},
        "reused_evaluations": {"baseline": _ref(BASELINE_EVAL), "gaussian_control": _ref(GAUSSIAN_EVAL),
                               "paired_initial_conditions": pairing},
        "config": asdict(Config(plants=16, layout="rows", row_spacing=.8, plant_spacing=.3,
                                 terrain_enabled=False, max_time=20., damage_penalty=20., goal_progress_budget=5.)),
        "task": old["task"],
        "training": {"timesteps": 16384, "num_envs": 4, "environment_seed": 68001,
                     "initialization_seed": 68001, "hyperparameters": old["training"]["hyperparameters"],
                     "variant": {"use_sde": False, "sde_sample_freq": -1},
                     "optimizer": "fresh Adam; no source optimizer state copied",
                     "gae_lambda": {"source": .95, "model": .99, "rollout_buffer": .99,
                                    "set_both_before_train_stage": True},
                     "only_treatment_change": "GAE lambda .95 to .99; otherwise match the previous Gaussian training conditions",
                     "checkpoint_interval_new_timesteps": 4096},
        "evaluation": {**old["evaluation"], "dataset_role": "previously observed development set, not blind validation",
                       "claim_limit": "if treatment improves over Gaussian here, validate separately on unseen 83k seeds"},
        "assessment": {"helper": "scripts.improve_isaac_policy._safe_goto_assessment",
                       "comparisons": ["treatment vs original baseline", "treatment vs Gaussian control"]},
    }
    _json(OUT / "protocol.json", protocol)
    _json(OUT / "run_status.json", {"status": "prepared", "protocol_sha256": _sha256(OUT / "protocol.json")})
    return {"output": str(OUT), "source_sha256": protocol["checkpoints"]["source_53248"]["sha256"],
            "control_sha256": protocol["checkpoints"]["gaussian_control_69632"]["sha256"],
            "frozen_source_files": len(FILES), "paired_development_seeds": 24}


def preflight():
    path, status_path = OUT / "protocol.json", OUT / "run_status.json"
    p, status = json.loads(path.read_text(encoding="utf-8")), json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "prepared" or status.get("protocol_sha256") != _sha256(path):
        raise FileExistsError(f"refusing launch; prepared protocol/status required, got {status.get('status')!r}")
    if {r["path"] for r in p["source_snapshot"]["files"]} != set(FILES):
        raise ValueError("frozen source inventory changed")
    for r in p["source_snapshot"]["files"]:
        if _sha256(ROOT / r["path"]) != r["sha256"] or _sha256(OUT / r["snapshot_path"]) != r["sha256"]:
            raise ValueError(f"frozen source mismatch: {r['path']}")
    _, old_hash = _previous()
    if old_hash != p["previous_protocol"]["sha256"]:
        raise ValueError("previous protocol changed")
    for ref, file in ((p["checkpoints"]["source_53248"], SOURCE),
                      (p["checkpoints"]["gaussian_control_69632"], CONTROL),
                      (p["reused_evaluations"]["baseline"], BASELINE_EVAL),
                      (p["reused_evaluations"]["gaussian_control"], GAUSSIAN_EVAL)):
        if _sha256(file) != ref["sha256"] or (ref["num_timesteps"] is not None and _checkpoint_steps(file) != ref["num_timesteps"]):
            raise ValueError(f"frozen evidence changed: {file}")
    _reports()
    return p


def launch(device="cuda:0"):
    p = preflight()
    _json(OUT / "run_status.json", {"status": "running", "phase": "training_gae_lambda_099"})
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from isaac_field.learning import train_stage
        from isaac_field.policy import make_exploration_variant
        from isaac_field.task import Config
        from isaac_field.training import _sb3_wrapper
        from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment
        baseline, gaussian, _ = _reports()
        source = RecurrentPPO.load(str(SOURCE), device=device)
        tr, hp = p["training"], p["training"]["hyperparameters"]
        env = _open_env(Config(**p["config"]), p, tr["environment_seed"], tr["num_envs"], True)
        try:
            model = make_exploration_variant(source, _sb3_wrapper(env), use_sde=False,
                                            seed=tr["initialization_seed"], sde_sample_freq=-1)
            if float(model.gae_lambda) != .95 or float(model.rollout_buffer.gae_lambda) != .95:
                raise ValueError("unexpected Gaussian source GAE settings")
            model.gae_lambda = model.rollout_buffer.gae_lambda = .99
            if float(model.gae_lambda) != .99 or float(model.rollout_buffer.gae_lambda) != .99:
                raise RuntimeError("could not set both GAE lambda values")
            ckpt_dir = OUT / "treatment/checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            periodic = CheckpointCallback(save_freq=4096 // tr["num_envs"], save_path=str(ckpt_dir),
                                          name_prefix="field_policy_gae_lambda_099")
            progress = _make_progress_callback(tr["num_envs"], env, interval=4096)
            stage = train_stage(env, model, tr["timesteps"], OUT / "treatment/field_policy_after",
                                callback=CallbackList([periodic, progress]), seed=tr["environment_seed"], device=device,
                                goal_gain=hp["goal_gain"], n_steps=hp["n_steps"], batch_size=hp["batch_size"],
                                n_epochs=hp["n_epochs"], learning_rate=hp["learning_rate"], gamma=hp["gamma"],
                                target_kl=hp["target_kl"], verbose=0)
        finally:
            env.close()
        _json(OUT / "training_result.json", {"checkpoint": str(stage.checkpoint), "requested_timesteps": stage.requested_timesteps,
              "trained_timesteps": stage.trained_timesteps, "total_timesteps": stage.total_timesteps,
              "gae_lambda": .99, "rollout_buffer_gae_lambda": .99, "seed": stage.seed,
              "periodic_checkpoints": sorted(str(f) for f in ckpt_dir.glob("*.zip")), "aggregate_progress": progress.log_rows})
        _json(OUT / "run_status.json", {"status": "running", "phase": "evaluating_treatment"})
        treatment = _evaluate(stage.checkpoint, Config(**p["config"]), p, "gae_lambda_099", OUT, device)
        pairing = _pair_initial_conditions({"baseline": baseline, "gaussian": gaussian, "treatment": treatment})
        if not pairing["valid"]:
            raise ValueError(f"initial conditions not paired: {pairing['mismatches'][:3]}")
        summary = {"paired_initial_conditions": pairing,
                   "evaluation_summaries": {"baseline": baseline["summary"], "gaussian_control": gaussian["summary"],
                                            "gae_lambda_099": treatment["summary"]},
                   "assessment_vs_original_baseline": _safe_goto_assessment(baseline, treatment),
                   "assessment_vs_gaussian_control": _safe_goto_assessment(gaussian, treatment),
                   "claim_limit": p["evaluation"]["claim_limit"]}
        _json(OUT / "comparison_summary.json", summary)
        _json(OUT / "run_status.json", {"status": "completed", "pairing_valid": True,
                                          "summary": "comparison_summary.json"})
        return summary
    except Exception as exc:
        _json(OUT / "run_status.json", {"status": "failed", "exception": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--launch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = prepare() if args.prepare else launch(args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
