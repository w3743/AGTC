"""Freeze and run controlled, instrumented foundation experiments.

One native SimulationContext is open at a time. Training, mastery probes and
the fixed global selection suite use separate contexts and explicit seeds.
No scheduler, viewer deployment, reward shaping or privileged actor inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_isaac_foundation_fresh as foundation
from isaac_field.curriculum_control import (
    CurriculumControlConfig, CurriculumController, ProbeSpec,
    compare_best_checkpoints,
)
from isaac_field.task import Config

SCHEMA = "isaac-field-mastery-experiment-v1"
CONTRACT_KEYS = (
    "action_config", "drive_config", "observation_config", "damage_config",
    "damage_metric", "distance_cost", "goal_distance", "goal_angle_rad",
    "mode", "deterministic", "evaluation_config",
)
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/mastery_method_20261020"
STAGES = tuple(
    {"name": name, "mode": "goto", "goal_distance": distance,
     "goal_angle": math.radians(angle), "plants": plants}
    for name, distance, angle, plants in (
        ("near", [0.6, 1.2], 15, 0),
        ("turn30", [0.6, 1.2], 30, 0),
        ("turn60", [0.6, 1.2], 60, 0),
        ("turn90", [0.6, 1.2], 90, 0),
        ("range", [1.0, 3.0], 90, 0),
        ("sparse2", [1.0, 3.0], 90, 2),
        ("field8", [1.0, 3.0], 90, 8),
        ("field16", [1.0, 3.0], 90, 16),
    )
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _sources() -> tuple[str, ...]:
    return tuple(sorted(set(foundation._source_paths()) | {
        "scripts/train_isaac_mastery.py", "scripts/run_isaac_mastery.sh",
    }))


def prepare(output: Path, *, method: str = "mastery", budget: int = 196_608,
            block_steps: int = 8_192, gamma: float = 0.995,
            gae_lambda: float = 0.95) -> dict:
    """Freeze the complete protocol without creating a simulator or training."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse experiment directory: {output}")
    if method not in ("fixed", "mastery"):
        raise ValueError("method must be fixed or mastery")
    # SB3 rounds every learn call to a complete rollout: reject budgets that
    # could silently overshoot the protocol (128 steps * 4 environments).
    for name, value in (("budget", budget), ("block_steps", block_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 512 or value % 512:
            raise ValueError(f"{name} must be a positive multiple of 512")
    if method == "fixed" and budget % (512 * len(STAGES)):
        raise ValueError("fixed budget must divide into equal, complete stage rollouts")
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite in (0, 1]")
    # All arms share a fixed 60-second input AND termination contract. Historical
    # 20-second reports are not baselines for this experiment.
    cfg = Config(plants=0, terrain_enabled=False, max_time=60.0,
                 damage_penalty=20.0, goal_progress_budget=5.0,
                 damage_metric="stem-contact-v2")
    paths = _sources()
    output.mkdir(parents=True)
    rows = []
    for name in paths:
        source, target = ROOT / name, output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": name, "snapshot_path": target.relative_to(output).as_posix(),
                     "sha256": foundation._sha256(source)})
    protocol = {
        "schema": SCHEMA, "method": method, "config": asdict(cfg),
        "stages": STAGES, "budget": budget, "block_steps": block_steps,
        "initialization": {"seed": 96_001, "num_timesteps": 0,
                           "source": "new random policy and Adam; no trained checkpoint"},
        "hyperparameters": {"n_steps": 128, "batch_size": 128, "n_epochs": 5,
                            "learning_rate": 3e-4, "gamma": gamma, "gae_lambda": gae_lambda,
                            "log_std_init": -2.0, "target_kl": 0.03},
        "mastery": {"episodes_per_stage": 16, "min_clean_success_rate": 0.8,
                    "required_consecutive_passes": 2, "probe_interval_timesteps": block_steps,
                    "max_damaged_episodes": 0, "max_mean_damage": 0.0,
                    "max_total_collisions": 0, "active_blocks_per_replay": 3,
                    "seed_banks": {stage["name"]: list(range(88_100 + i * 100, 88_116 + i * 100))
                                   for i, stage in enumerate(STAGES)}},
        "selection": {"stage_indices": list(range(len(STAGES))), "episodes_per_stage": 4,
                      "seed_banks": [list(range(89_000 + i * 100, 89_004 + i * 100)) for i in range(len(STAGES))],
                      "every_probes": 4, "also_on_promotion": True,
                      "criterion": "clean successes increase; damaged episodes, mean damage and collisions do not increase"},
        "evaluation_environment_seed": 90_003,
        "training_seed_base": 97_000,
        "dataset_role": "development for curriculum and checkpoint selection, never held-out",
        "blind_seeds": {"range": [83_000, 83_999], "used": False},
        "source_snapshot": {"files": rows}, "runtime": foundation._package_versions(),
        "isaaclab_commit": foundation._isaac_commit(), "robot_geometry": foundation._robot_geometry(),
        "action_config": foundation._action_config(),
        "claim_limit": "Software validation and training completion do not establish a learned capability improvement.",
    }
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {"status": "prepared", "num_timesteps": 0,
          "protocol_sha256": foundation._sha256(output / "protocol.json")})
    return {"status": "prepared", "output": str(output), "method": method,
            "source_files": len(rows), "training_started": False}


def preflight(output: Path) -> dict:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA or status.get("status") != "prepared":
        raise ValueError("only a prepared, never-launched mastery experiment may launch")
    if status["protocol_sha256"] != foundation._sha256(output / "protocol.json"):
        raise ValueError("prepared protocol hash changed")
    verify_sources(output, protocol)
    if protocol["action_config"] != foundation._action_config():
        raise ValueError("current action contract differs from the protocol")
    if (protocol["runtime"] != foundation._package_versions()
            or protocol["isaaclab_commit"] != foundation._isaac_commit()
            or protocol["robot_geometry"] != foundation._robot_geometry()):
        raise ValueError("runtime dependencies, upstream revision or robot geometry changed after prepare")
    return protocol


def verify_sources(output: Path, protocol: dict) -> None:
    rows = protocol["source_snapshot"]["files"]
    if {row["path"] for row in rows} != set(_sources()):
        raise ValueError("source inventory differs from prepared experiment")
    for row in rows:
        if (foundation._sha256(ROOT / row["path"]) != row["sha256"]
                or foundation._sha256(output / row["snapshot_path"]) != row["sha256"]):
            raise ValueError(f"frozen source mismatch: {row['path']}")


def _contract(report: dict) -> dict:
    return {key: report[key] for key in CONTRACT_KEYS}


def _attach_probe(report: dict, *, purpose: str, stage: str) -> dict:
    report["probe"] = {"schema": "isaac-field-fixed-probe-v1", "purpose": purpose,
                       "probe_stage": stage, "seeds": report["seeds"], "config": _contract(report)}
    return report


def _open_env(output: Path, protocol: dict, stage: dict, *, training: bool, seed: int):
    from isaac_field.env import make_env

    verify_sources(output, protocol)
    cfg = replace(Config(**protocol["config"]), plants=stage["plants"])
    expected_task = asdict(cfg)
    engine = make_env(cfg=cfg, mode=stage["mode"], num_envs=4 if training else 1,
                      seed=seed, auto_reset=training, goal_distance=tuple(stage["goal_distance"]),
                      goal_angle=stage["goal_angle"], distance_cost=0.0)
    try:
        foundation._live_environment_contract(engine)
        if (asdict(engine.field_cfg) != expected_task or engine.cfg.mode != stage["mode"]
                or list(engine.cfg.goal_distance) != stage["goal_distance"]
                or engine.cfg.goal_angle != stage["goal_angle"]
                or engine.cfg.distance_cost != 0.0 or engine.num_envs != (4 if training else 1)):
            raise ValueError("actual environment task or sampling differs from the frozen stage")
        _append(output / "runtime_assets.jsonl", {"training": training, "seed": seed,
                "stage": stage, "assets": foundation._runtime_assets(engine)})
    except BaseException:
        engine.close()
        raise
    return engine


def _evaluate_stage(output: Path, protocol: dict, checkpoint: Path, stage: dict,
                    seeds: list[int]) -> dict:
    from isaac_field.env import SingleFieldAdapter
    from isaac_field.learning import evaluate_model

    engine = _open_env(output, protocol, stage, training=False,
                       seed=protocol["evaluation_environment_seed"])
    try:
        report = evaluate_model(SingleFieldAdapter(engine), checkpoint, seeds,
            goal_distance=tuple(stage["goal_distance"]), goal_angle=stage["goal_angle"],
            mode=stage["mode"], record_initial_conditions=True, device=protocol["device"])
        report["evaluation_config"] = {"task": asdict(engine.field_cfg),
            "environment_seed": protocol["evaluation_environment_seed"], "stage": stage}
    finally:
        engine.close()
    return report


def _global_probe(output: Path, protocol: dict, checkpoint: Path) -> dict:
    reports = [_evaluate_stage(output, protocol, checkpoint, protocol["stages"][index], seeds)
               for index, seeds in zip(protocol["selection"]["stage_indices"],
                                       protocol["selection"]["seed_banks"])]
    # Record the actual contract of every suite member; never relabel a single
    # scene's report as if all suite episodes had that same scene.
    result = {key: reports[0][key] for key in CONTRACT_KEYS}
    for report in reports[1:]:
        for key in ("action_config", "drive_config", "observation_config", "damage_config", "damage_metric"):
            if report[key] != result[key]:
                raise ValueError(f"global suite has inconsistent {key}")
    result.update(schema="isaac-field-development-suite-v1", checkpoint=str(checkpoint),
                  goal_distance=None, goal_angle_rad=None, mode="goto_suite",
                  seeds=[seed for report in reports for seed in report["seeds"]],
                  rows=[row for report in reports for row in report["rows"]],
                  evaluation_config={"members": [_contract(report) for report in reports]})
    return _attach_probe(result, purpose="best_checkpoint", stage="global_suite")


def _mastery_specs(protocol: dict, baseline_report: dict) -> dict[str, ProbeSpec]:
    """Declare expected contracts; these are not simulated probe results."""
    specs = {}
    for stage in protocol["stages"]:
        contract = {key: baseline_report[key] for key in CONTRACT_KEYS}
        cfg = replace(Config(**protocol["config"]), plants=stage["plants"])
        contract.update(goal_distance=stage["goal_distance"], goal_angle_rad=stage["goal_angle"],
                        mode=stage["mode"], evaluation_config={"task": asdict(cfg),
                            "environment_seed": protocol["evaluation_environment_seed"], "stage": stage})
        specs[stage["name"]] = ProbeSpec(protocol["mastery"]["seed_banks"][stage["name"]], contract)
    return specs


def _controller(protocol: dict, mastery_specs: dict, baseline_report: dict) -> CurriculumController:
    return CurriculumController(protocol["stages"], CurriculumControlConfig(
        total_timesteps=protocol["budget"],
        probe_interval_timesteps=protocol["mastery"]["probe_interval_timesteps"],
        training_seeds=[protocol["training_seed_base"] + i for i in range(protocol["budget"] // 512 + 1)],
        mastery_probes=mastery_specs,
        best_probe=ProbeSpec(baseline_report["seeds"], _contract(baseline_report)),
        required_consecutive_passes=protocol["mastery"]["required_consecutive_passes"],
        min_clean_success_rate=protocol["mastery"]["min_clean_success_rate"],
        minimum_stage_timesteps=min(protocol["budget"], protocol["block_steps"]
            + protocol["mastery"]["required_consecutive_passes"]
            * protocol["mastery"]["probe_interval_timesteps"]),
        active_blocks_per_replay=protocol["mastery"]["active_blocks_per_replay"],
    ))


def _policy_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_best(protocol: dict, candidate: dict, best: dict) -> dict:
    """Prevent gains on easy scenes from concealing regressions on plant scenes."""
    comparison = compare_best_checkpoints(candidate, best)
    if not comparison["comparable"]:
        return comparison
    candidates = {row["seed"]: row for row in candidate["rows"]}
    incumbents = {row["seed"]: row for row in best["rows"]}
    members = []

    def metrics(rows):
        return {"clean_successes": sum(row["success"] and row["damage"] == 0 and row["collisions"] == 0 for row in rows),
                "damaged_episodes": sum(row["damage"] > 0 for row in rows),
                "total_damage": sum(row["damage"] for row in rows),
                "total_collisions": sum(row["collisions"] for row in rows)}

    for index, seeds in zip(protocol["selection"]["stage_indices"], protocol["selection"]["seed_banks"]):
        current = metrics([candidates[seed] for seed in seeds])
        previous = metrics([incumbents[seed] for seed in seeds])
        safe = (current["clean_successes"] >= previous["clean_successes"]
                and all(current[key] <= previous[key] for key in
                        ("damaged_episodes", "total_damage", "total_collisions")))
        members.append({"stage": protocol["stages"][index]["name"], "candidate": current,
                        "best": previous, "non_regression": safe})
    comparison["member_checks"] = members
    if not all(member["non_regression"] for member in members):
        comparison.update(replace_best=False, assessment="suite_member_regression",
                          selected_checkpoint=best.get("checkpoint"))
    return comparison


def launch(output: Path, *, device: str = "cuda:0") -> dict:
    protocol = preflight(output)
    # Runtime-only device is recorded separately, never mutate the frozen file.
    protocol["device"] = device
    consumed = 0
    status_base = {"protocol_sha256": foundation._sha256(output / "protocol.json"), "device": device}

    def status(phase, **extra):
        _json(output / "run_status.json", {**status_base, "status": "running", "phase": phase,
              "num_timesteps": consumed, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **extra})

    status("initializing")
    try:
        from isaac_field.policy import make_model
        from isaac_field.training import _sb3_wrapper
        from isaac_field.learning import train_stage
        from isaac_field.training_diagnostics import make_sb3_diagnostics_callback

        stages = protocol["stages"]
        hp = protocol["hyperparameters"]
        baseline = output / "untrained_baseline.zip"
        env = _open_env(output, protocol, stages[0], training=True,
                        seed=protocol["initialization"]["seed"])
        try:
            model = make_model(_sb3_wrapper(env), seed=protocol["initialization"]["seed"],
                               device=device, **hp)
            model.field_drive_config = dict(env.drive_config)
            model.field_observation_config = dict(env.observation_config)
            model.field_damage_config = dict(env.tasks[0].damage_config)
            model.save(str(baseline))
            _json(output / "initialization.json", {"num_timesteps": model.num_timesteps,
                  "policy_tensor_sha256": _policy_digest(model),
                  "checkpoint_sha256": foundation._sha256(baseline),
                  "optimizer_state_entries": len(model.policy.optimizer.state),
                  "initialization_seed": protocol["initialization"]["seed"]})
            del model
        finally:
            env.close()
        status("initial_probes")
        best_report = _global_probe(output, protocol, baseline)
        _json(output / "probes" / "global_000000.json", best_report)
        mastery_specs = _mastery_specs(protocol, best_report)
        _json(output / "probe_contracts.json", {"status": "declared_expected_contracts_not_results",
              "mastery": {name: asdict(spec) for name, spec in mastery_specs.items()},
              "global": _contract(best_report)})
        controller = _controller(protocol, mastery_specs, best_report)
        controller.validate_best_probe(best_report)
        checkpoint, best_checkpoint = baseline, baseline
        blocks, probe_count = [], 0
        while consumed < protocol["budget"]:
            verify_sources(output, protocol)
            if protocol["method"] == "mastery":
                plan = controller.next_training_plan(max_timesteps=protocol["block_steps"])
                stage = next(item for item in stages if item["name"] == plan["sampled_stage"])
                block_steps = plan["timesteps"]
            else:
                stage_size = protocol["budget"] // len(stages)
                index = min(consumed // stage_size, len(stages) - 1)
                stage = stages[index]
                block_steps = min(protocol["block_steps"], (index + 1) * stage_size - consumed)
                plan = {"sampled_stage": stage["name"], "timesteps": block_steps,
                        "sampling": {key: stage[key] for key in ("mode", "goal_distance", "goal_angle", "plants")}}
            block_index = len(blocks)
            status("training", sampled_stage=stage["name"], planned_timesteps=block_steps)
            env = _open_env(output, protocol, stage, training=True,
                            seed=protocol["training_seed_base"] + block_index)
            try:
                callback = make_sb3_diagnostics_callback(output / "diagnostics.jsonl", stage=stage["name"],
                              info_provider=lambda: env.diagnostic_step_infos)
                result = train_stage(env, checkpoint, block_steps,
                    output / "checkpoints" / f"steps_{consumed + block_steps:07d}_{stage['name']}.zip",
                    seed=protocol["training_seed_base"] + block_index,
                    device=device, callback=callback, **hp)
                actual_sampling = {"mode": env.cfg.mode,
                    "goal_distance": list(env.cfg.goal_distance), "goal_angle": env.cfg.goal_angle,
                    "plants": env.field_cfg.plants}
            finally:
                env.close()
            if result.trained_timesteps != block_steps or result.total_timesteps != consumed + block_steps:
                raise RuntimeError("actual SB3 timestep count differs from the frozen global budget")
            consumed += result.trained_timesteps
            checkpoint = result.checkpoint
            if protocol["method"] == "mastery":
                decision = controller.record_training(result.trained_timesteps,
                    sampled_stage=stage["name"], actual_sampling=actual_sampling)
            else:
                decision = {"probe_due": True, "active_stage": stage, "active_stage_index": stages.index(stage)}
            blocks.append({"plan": plan, "actual_sampling": actual_sampling,
                           "checkpoint": str(checkpoint), "sha256": foundation._sha256(checkpoint),
                           "num_timesteps": consumed})
            promoted = False
            did_probe = decision["probe_due"]
            if decision["probe_due"]:
                status("mastery_probe")
                active = next(item for item in stages if item["name"] == decision["active_stage"]["name"])
                report = _evaluate_stage(output, protocol, checkpoint, active,
                                         protocol["mastery"]["seed_banks"][active["name"]])
                report = _attach_probe(report, purpose="mastery", stage=active["name"])
                _json(output / "probes" / f"mastery_{consumed:07d}_{active['name']}.json", report)
                probe_count += 1
                if protocol["method"] == "mastery":
                    decision = controller.record_probe(report)
                    promoted = decision["last_event"]["promotion"] == "advanced"
                _append(output / "decisions.jsonl", {"num_timesteps": consumed, "decision": decision})
            final_mastered = protocol["method"] == "mastery" and controller.decision()["mastered"]
            if promoted or final_mastered or (did_probe and probe_count % protocol["selection"]["every_probes"] == 0) or consumed == protocol["budget"]:
                status("global_selection_probe")
                report = _global_probe(output, protocol, checkpoint)
                controller.validate_best_probe(report)
                comparison = _select_best(protocol, report, best_report)
                if comparison["replace_best"]:
                    best_report, best_checkpoint = report, checkpoint
                _json(output / "probes" / f"global_{consumed:07d}.json", report)
                _append(output / "selection.jsonl", {"num_timesteps": consumed, **comparison})
            _json(output / "controller_state.json", controller.state_dict())
            _json(output / "checkpoint_manifest.json", {"baseline": str(baseline), "blocks": blocks,
                  "latest_checkpoint": str(checkpoint), "best_checkpoint": str(best_checkpoint)})
            # Stop a fully mastered final stage without spending the remaining
            # budget. A curriculum that runs out of budget stays incomplete.
            if protocol["method"] == "mastery" and controller.decision()["mastered"]:
                break
        summary = {"schema": SCHEMA, "method": protocol["method"], "num_timesteps": consumed,
            "budget": protocol["budget"], "latest_checkpoint": str(checkpoint),
            "best_checkpoint": str(best_checkpoint), "best_checkpoint_sha256": foundation._sha256(best_checkpoint),
            "controller": controller.decision() if protocol["method"] == "mastery" else None,
            "training_finished": True, "capability_improvement_established": False,
            "claim_limit": protocol["claim_limit"]}
        _json(output / "summary.json", summary)
        status("finished", status="completed", exit_code=0)
        return summary
    except BaseException as exc:
        status("failed", status="failed", exit_code=1, exception=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        # This is the Python outcome, not an observed operating-system exit.
        # run_isaac_mastery.sh independently writes the actual child exit code.
        exit_status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
        _json(output / "python_outcome.json", {"intended_exit_code": exit_status.get("exit_code", 1),
              "num_timesteps": consumed, "status": exit_status["status"],
              "timestamp_utc": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("fixed", "mastery"), default="mastery")
    parser.add_argument("--budget", type=int, default=196_608)
    parser.add_argument("--block-steps", type=int, default=8_192)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.output, method=args.method, budget=args.budget,
                         block_steps=args.block_steps, gamma=args.gamma, gae_lambda=args.gae_lambda)
    elif args.preflight:
        protocol = preflight(args.output)
        result = {"status": "preflight_passed", "method": protocol["method"], "training_started": False}
    else:
        result = launch(args.output, device=args.device)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
