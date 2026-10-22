"""Evaluate an RSL fast-stop checkpoint or explicitly selected archived SB3 trial."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2"
DEFAULT_OUTPUT = TRIAL.with_name(TRIAL.name + "_evaluation")
FAST_STOP_SMOKE_SEED = 87_001
FAST_STOP_SMOKE_MAX_SCENARIOS = 2
FAST_STOP_SMOKE_MAX_STEPS = 2
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_fast_stop_resume as trajectory_runner
from scripts import train_isaac_mastery as mastery
from scripts import view_fast_stop_goal as goal_viewer


def _write(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_rsl_task_contract(contract: dict[str, Any]):
    """Rebuild the complete saved task and verify its strict fast-stop semantics."""
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    task, front_guides, body_stability = goal_viewer._rsl_environment_configs(contract)

    # Reuse the objective source of truth while preserving every other saved
    # Config field exactly as recorded in the checkpoint contract.
    from scripts import train_isaac_foundation_fresh as fresh

    expected_semantics = {
        **fresh.FAST_STOP_CONFIG,
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "max_time": 60.0,
        "size": 100.0,
        "plants": 0,
        "terrain_enabled": False,
        "rock_density": 0.0,
    }
    actual_task = asdict(task)
    differences = sorted(
        key for key, expected in expected_semantics.items()
        if actual_task.get(key) != expected
    )
    if differences:
        raise ValueError(
            "RSL checkpoint task_config is not the strict fast-stop 60 s absolute-goal full-map profile; "
            f"mismatched fields: {differences}"
        )
    if contract.get("input_schema") != ABSOLUTE_GOAL_INPUT_SCHEMA:
        raise ValueError("RSL checkpoint input_schema must be the absolute-goal schema")
    return task, front_guides, body_stability


def _create_rsl_evaluation_adapter(contract: dict[str, Any], seed: int):
    """Create a one-environment task from checkpoint metadata, without SB3 fallback."""
    from isaac_field.env import SingleFieldAdapter, make_env

    task, front_guides, body_stability = _validate_rsl_task_contract(contract)
    engine = make_env(
        cfg=task,
        mode="goto",
        num_envs=1,
        seed=seed,
        auto_reset=False,
        goal_distance=None,
        goal_angle=math.pi,
        distance_cost=0.0,
        front_guide_spec=front_guides,
        body_stability_config=body_stability,
    )
    return engine, SingleFieldAdapter(engine)


def _evaluate_rsl_checkpoint(
    checkpoint: str | Path,
    *,
    seed: int,
    scenario_count: int,
    max_steps: int | None,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any], Path, list[int]]:
    path = Path(checkpoint).expanduser().resolve()
    if path.suffix.lower() != ".pt":
        raise ValueError("RSL evaluation requires an RSL .pt checkpoint")
    if not path.is_file():
        raise FileNotFoundError(path)
    if type(scenario_count) is not int or scenario_count < 1:
        raise ValueError("scenario_count must be a positive integer")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer when provided")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.learning import evaluate_model
    from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

    checkpoint_sha256 = _checkpoint_sha256(path)
    checkpoint_stat = path.stat()
    payload = load_rsl_checkpoint_payload(path, map_location="cpu")
    contract = payload["field_policy_contract"]
    seeds = list(range(seed, seed + scenario_count))
    _assert_no_parallel_simulation()
    engine, adapter = _create_rsl_evaluation_adapter(contract, seed)
    try:
        goal_viewer._verify_rsl_environment_contract(engine, contract)
        evaluated = evaluate_model(
            adapter,
            path,
            seeds,
            goal_distance=None,
            goal_angle=math.pi,
            mode="goto",
            deterministic=True,
            max_steps=max_steps,
            device=device,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            expected_observation_config=contract.get("observation_config"),
        )
    finally:
        engine.close()
    after_stat = path.stat()
    if ((checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
            != (after_stat.st_size, after_stat.st_mtime_ns)
            or _checkpoint_sha256(path) != checkpoint_sha256):
        raise RuntimeError(f"checkpoint changed during evaluation: {path}")
    evaluated.update(
        checkpoint=str(path),
        checkpoint_sha256=checkpoint_sha256,
        backend="rsl_rl",
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        task_config=contract["task_config"],
        front_guide_spec=contract["front_guide_spec"],
        body_stability_config=contract["body_stability_config"],
        evaluation_scope={
            "scenario_seeds": seeds,
            "max_steps_per_scenario": max_steps,
            "episode_cap_applied": max_steps is not None,
        },
    )
    if max_steps is not None:
        evaluated["assessment"] = "not_assessed"
        evaluated["claim_limit"] = (
            "A max_steps cap was applied; capped episodes are diagnostic and do not establish task competence."
        )
    return evaluated, contract, path, seeds


def run_rsl_evaluation(
    checkpoint: str | Path,
    *,
    seed: int = 100_007,
    scenario_count: int = 10,
    max_steps: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the existing full episode evaluator on a native RSL checkpoint."""
    evaluated, _, _, _ = _evaluate_rsl_checkpoint(
        checkpoint,
        seed=seed,
        scenario_count=scenario_count,
        max_steps=max_steps,
        device=device,
    )
    return evaluated


def run_rsl_smoke(
    checkpoint: str | Path,
    *,
    seed: int = FAST_STOP_SMOKE_SEED,
    scenario_count: int = 1,
    max_steps: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run an explicitly bounded native smoke check; it does not assess competence."""
    if type(scenario_count) is not int or not 1 <= scenario_count <= FAST_STOP_SMOKE_MAX_SCENARIOS:
        raise ValueError(f"smoke scenario_count must be between 1 and {FAST_STOP_SMOKE_MAX_SCENARIOS}")
    if type(max_steps) is not int or not 1 <= max_steps <= FAST_STOP_SMOKE_MAX_STEPS:
        raise ValueError(f"smoke max_steps must be between 1 and {FAST_STOP_SMOKE_MAX_STEPS}")
    evaluated, contract, path, seeds = _evaluate_rsl_checkpoint(
        checkpoint,
        seed=seed,
        scenario_count=scenario_count,
        max_steps=max_steps,
        device=device,
    )

    rows = [
        {
            key: row.get(key)
            for key in (
                "seed", "steps", "reason", "terminated", "truncated", "timed_out",
                "final_distance", "episode_time_s", "damage", "damage_metric", "collisions",
            )
        }
        for row in evaluated["rows"]
    ]
    return {
        "schema": "isaac-field-rsl-native-smoke-v1",
        "status": "smoke_completed",
        "assessment": "not_assessed",
        "claim_limit": (
            "At most two scenarios and two steps per scenario. This checks checkpoint loading, "
            "finite policy actions, and the native evaluation path only; task competence and "
            "success rate are not assessed."
        ),
        "backend": "rsl_rl",
        "model_class": evaluated["model_class"],
        "checkpoint": str(path),
        "checkpoint_sha256": evaluated["checkpoint_sha256"],
        "input_schema": evaluated["input_schema"],
        "task_profile": "fast_stop_v1",
        "task_config": contract["task_config"],
        "front_guide_spec": contract["front_guide_spec"],
        "body_stability_config": contract["body_stability_config"],
        "scenario_seeds": seeds,
        "max_steps_per_scenario": max_steps,
        "rows": rows,
    }


def _root_path(value: str | Path, label: str) -> Path:
    path = goal_viewer._resolve(value)
    if not goal_viewer._inside(path, ROOT.resolve()):
        raise ValueError(f"{label} resolves outside the workspace")
    return path


def _transfer_config(protocol: dict[str, Any]) -> dict[str, Any]:
    transfer = protocol.get("transfer_probe")
    expected = mastery._fast_stop_transfer_probe()
    if goal_viewer._stable_hash(transfer) != goal_viewer._stable_hash(expected):
        raise ValueError("transfer probe differs from the frozen fast-stop rough profile")
    if transfer["environment_config"].get("terrain_enabled") is not True:
        raise ValueError("rough transfer evaluation must keep terrain enabled")
    return transfer["environment_config"]


def check_trial(trial: str | Path = TRIAL, *, selection: str = "latest") -> dict[str, Any]:
    """Validate frozen sources and the selected checkpoint without launching Isaac."""
    trial = _root_path(trial, "trial")
    protocol_path = trial / "protocol.json"
    protocol = goal_viewer._json(protocol_path)
    goal_viewer._validate_protocol(protocol)
    mastery._validate_profile_protocol(protocol)
    transfer_config = _transfer_config(protocol)
    sources = goal_viewer._verify_sources(trial, protocol)
    protocol_sha = goal_viewer._sha256(protocol_path)
    status = goal_viewer._json(trial / "run_status.json")
    if status.get("protocol_sha256") != protocol_sha:
        raise ValueError("run_status protocol SHA differs from protocol.json")
    selected = goal_viewer._resolve_checkpoint(trial, protocol, selection)
    path = Path(selected["path"]).resolve()
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data").decode("utf-8"))
    if data.get("field_action_config") != protocol["action_config"]:
        raise ValueError("selected checkpoint action contract differs from protocol")
    members = protocol["transfer_probe"]["members"]
    if [member.get("name") for member in members] != ["rough_a", "rough_b"]:
        raise ValueError("frozen transfer probe must contain rough_a and rough_b")
    if any(len(member.get("seeds", [])) != 4 or len(set(member["seeds"])) != 4
           for member in members):
        raise ValueError("each rough transfer member must have four unique frozen seeds")
    return {"trial": trial, "protocol": protocol, "protocol_sha256": protocol_sha,
            "source_file_count": sources["file_count"], "source_sha256": sources["source_sha256"],
            "run_status": status, "transfer_config": transfer_config,
            "checkpoint": {**selected, "path": str(path)}}


def _evaluate_member(context: dict[str, Any], output: Path, member: dict[str, Any],
                     device: str) -> dict[str, Any]:
    """Reuse the existing trace adapter with the full frozen rough config and seeds."""
    protocol = context["protocol"]
    return trajectory_runner._evaluate_checkpoint(
        output, mastery.Config(**context["transfer_config"]), Path(context["checkpoint"]["path"]),
        member["name"], tuple(member["seeds"]), int(member["plants"]),
        math.degrees(float(member["goal_angle"])), int(member["environment_seed"]), device,
        protocol["profile_physics_contract"], protocol["goal_objective_contract"],
    )


def _validate_report(report: dict[str, Any], context: dict[str, Any],
                     member: dict[str, Any]) -> None:
    protocol = context["protocol"]
    cfg = report.get("evaluation_config", {})
    if (report.get("seeds") != member["seeds"]
            or report.get("action_config") != protocol["action_config"]
            or report.get("checkpoint_action_config") != protocol["action_config"]
            or report.get("distance_cost") != 0.0 or report.get("goal_distance") is not None
            or report.get("mode") != "goto" or report.get("deterministic") is not True
            or cfg.get("environment_seed") != member["environment_seed"]
            or cfg.get("plants") != member["plants"] or cfg.get("goal_distance_m") is not None
            or not math.isclose(float(cfg.get("goal_angle_deg", math.nan)),
                                math.degrees(member["goal_angle"]), rel_tol=0, abs_tol=1e-10)
            or cfg.get("goal_sampling_profile") != mastery.foundation.PROFILE_RANDOM_ROUGH
            or cfg.get("terrain_config") != protocol["profile_physics_contract"]["terrain_config"]
            or cfg.get("goal_objective_contract") != protocol["goal_objective_contract"]):
        raise ValueError(f"{member['name']} report differs from frozen config/objective/seeds")
    if (report.get("checkpoint_drive_config") != report.get("drive_config")
            or report.get("checkpoint_observation_config") != report.get("observation_config")
            or report.get("checkpoint_damage_config") != report.get("damage_config")
            or report.get("damage_metric") != protocol["config"]["damage_metric"]
            or Path(report.get("checkpoint", "")).resolve()
            != Path(context["checkpoint"]["path"]).resolve()):
        raise ValueError(f"{member['name']} report checkpoint/runtime contract mismatch")
    rows = report.get("rows")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != member["seeds"]:
        raise ValueError(f"{member['name']} report does not contain each frozen seed once")
    for row in rows:
        recorded_metrics = row.get("path_metrics")
        trajectory = row.get("trajectory", [])
        trace = [step.get("root_com_kinematics") for step in trajectory]
        computed_metrics = trajectory_runner.summarize_trajectory(
            row, trace, distance_tolerance_m=protocol["goal_objective_contract"]["goal_stop"]["distance_tolerance_m"],
        )
        if json.dumps(recorded_metrics, sort_keys=True) != json.dumps(computed_metrics, sort_keys=True):
            raise ValueError(f"{member['name']} episode path metrics do not match its step samples")
        if not row.get("reason"):
            raise ValueError(f"{member['name']} episode lacks a terminal reason")


def _assert_no_parallel_simulation() -> None:
    active = []
    if os.name == "posix" and Path("/proc").is_dir():
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = [part.decode("utf-8", "replace") for part in
                        (proc / "cmdline").read_bytes().split(b"\0") if part]
            except OSError:
                continue
            if _is_parallel_simulation_argv(argv, int(proc.name)):
                active.append(f"PID {proc.name}: {' '.join(argv)}")
    else:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("cannot verify parallel Isaac processes on this host") from exc
        for process in psutil.process_iter(("pid", "cmdline")):
            argv = process.info.get("cmdline") or []
            if _is_parallel_simulation_argv(argv, process.info["pid"]):
                command = " ".join(argv)
                active.append(f"PID {process.info['pid']}: {command}")
    if active:
        raise RuntimeError("parallel Isaac training/evaluation/viewer process detected: "
                           + "; ".join(active))


def _is_parallel_simulation_argv(argv: list[str], pid: int, *, current_pid: int | None = None) -> bool:
    """Match Python workers by executable and script/module tokens, never shell text."""
    if pid == (os.getpid() if current_pid is None else current_pid) or not argv:
        return False
    if not Path(argv[0]).name.lower().startswith("python"):
        return False
    for token in argv[1:]:
        name = Path(token.split("=", 1)[0]).name.lower()
        if name.endswith(".py") and name.startswith(("train_", "evaluate_")):
            return True
        if name == "view_fast_stop_goal.py":
            return True
    for index, token in enumerate(argv[:-1]):
        if token != "-m":
            continue
        module = argv[index + 1]
        if module == "isaac_field.viewer":
            return True
        if (module == "isaac_field" and index + 2 < len(argv)
                and argv[index + 2] in {
                    "viewer", "smoke", "train", "resume", "train-sb3", "evaluate",
                    "adversary", "adversary-sb3",
                }):
            return True
        if module.startswith("scripts."):
            script_module = module.rsplit(".", 1)[-1]
            if script_module.startswith(("train_", "evaluate_")) or script_module == "view_fast_stop_goal":
                return True
    return False


def run(trial: str | Path = TRIAL, output: str | Path = DEFAULT_OUTPUT, *,
        selection: str = "latest", device: str = "cuda:0", launch: bool = False) -> dict[str, Any]:
    context = check_trial(trial, selection=selection)
    output = _root_path(output, "evaluation output")
    if (output == context["trial"] or output.is_relative_to(context["trial"])
            or context["trial"].is_relative_to(output)):
        raise ValueError("evaluation output must be separate from the trial directory")
    if not launch:
        return {"status": "check_passed", "launch_allowed": False,
                "note": "CPU-only read-only check; sampling requires --launch after real exit 0.",
                "trial": context["trial"].relative_to(ROOT.resolve()).as_posix(),
                "protocol_sha256": context["protocol_sha256"],
                "source_file_count": context["source_file_count"],
                "source_sha256": context["source_sha256"], "run_status": context["run_status"],
                "checkpoint": context["checkpoint"]}
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    gate = goal_viewer.validate_completed_trial(context["trial"], check_process=True)
    if gate.get("protocol_sha256") != context["protocol_sha256"]:
        raise ValueError("trial protocol changed between preflight and completion gate")
    current = check_trial(context["trial"], selection=selection)
    if current["checkpoint"]["sha256"] != context["checkpoint"]["sha256"]:
        raise ValueError("selected checkpoint changed between preflight and launch")
    _assert_no_parallel_simulation()

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir()
    protocol = context["protocol"]
    manifest = {"schema": "fast-stop-mastery-process-sampling-v1", "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "trial": context["trial"].relative_to(ROOT.resolve()).as_posix(),
                "protocol_sha256": context["protocol_sha256"],
                "source_file_count": context["source_file_count"],
                "source_sha256": context["source_sha256"],
                "completion_gate": gate, "checkpoint": context["checkpoint"],
                "transfer_probe": protocol["transfer_probe"],
                "goal_objective_contract": protocol["goal_objective_contract"],
                "profile_physics_contract": protocol["profile_physics_contract"],
                "evaluation_runner_sha256": goal_viewer._sha256(Path(__file__).resolve()),
                "trajectory_runner_sha256": goal_viewer._sha256(Path(trajectory_runner.__file__).resolve()),
                "claim_limit": "Process diagnosis on development transfer seeds; no capability-improvement claim."}
    _write(output / "run_manifest.json", manifest)
    try:
        for member in protocol["transfer_probe"]["members"]:
            report = _evaluate_member(context, output, member, device)
            _validate_report(report, context, member)
            _write(output / f"evaluation_{member['name']}.json", report)
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        _write(output / "run_manifest.json", manifest)
        return {"status": "completed", "output": str(output),
                "checkpoint": context["checkpoint"], "members": [m["name"] for m in protocol["transfer_probe"]["members"]],
                "claim_limit": manifest["claim_limit"]}
    except BaseException as exc:
        manifest.update(status="failed", exception=repr(exc),
                        failed_at_utc=datetime.now(timezone.utc).isoformat())
        _write(output / "run_manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="CPU-only check for an archived SB3 trial")
    actions.add_argument("--launch", action="store_true", help="sample a completed archived SB3 trial")
    parser.add_argument("--trial", type=Path, default=TRIAL)
    parser.add_argument("--output", type=Path, default=None,
                        help="legacy mode: output directory; RSL mode: report JSON file")
    parser.add_argument("--selection", choices=("latest", "near-best"), default="latest")
    parser.add_argument("--device", default=None, help="policy inference device (default: cpu for RSL)")
    parser.add_argument("--legacy-sb3", action="store_true",
                        help="explicitly enable the historical SB3 trial evaluation path")
    parser.add_argument("--checkpoint", type=Path,
                        help="explicit native RSL-RL .pt checkpoint")
    parser.add_argument("--smoke", action="store_true",
                        help="limit an RSL run to at most two scenarios and two steps; not a competence assessment")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="optional positive episode step cap; --smoke restricts it to 1 or 2")
    args = parser.parse_args(argv)
    try:
        if args.checkpoint is not None:
            if args.check or args.launch:
                parser.error("--check/--launch apply to archived SB3 trial mode, not RSL checkpoints")
            if args.legacy_sb3:
                parser.error("--legacy-sb3 trial mode does not accept a direct --checkpoint; use --trial/--selection")
            if args.checkpoint.suffix.lower() != ".pt":
                parser.error("RSL evaluation requires .pt; use --legacy-sb3 for the archived SB3 trial path")
            if args.max_steps is not None and args.max_steps < 1:
                parser.error("--max-steps must be positive")
            if args.scenario_count is not None and args.scenario_count < 1:
                parser.error("--scenario-count must be positive")
            if args.seed is not None and args.seed < 0:
                parser.error("--seed must be non-negative")
            report_path = None
            if args.output is not None:
                report_path = args.output.expanduser().resolve()
                if report_path.exists():
                    raise FileExistsError(f"refusing to overwrite RSL evaluation report: {report_path}")
                report_path.parent.mkdir(parents=True, exist_ok=True)
            if args.smoke:
                if args.max_steps is None:
                    parser.error("--smoke requires --max-steps 1 or 2")
                result = run_rsl_smoke(
                    args.checkpoint,
                    seed=FAST_STOP_SMOKE_SEED if args.seed is None else args.seed,
                    scenario_count=1 if args.scenario_count is None else args.scenario_count,
                    max_steps=args.max_steps,
                    device=args.device or "cpu",
                )
            else:
                result = run_rsl_evaluation(
                    args.checkpoint,
                    seed=100_007 if args.seed is None else args.seed,
                    scenario_count=10 if args.scenario_count is None else args.scenario_count,
                    max_steps=args.max_steps,
                    device=args.device or "cpu",
                )
            if report_path is not None:
                result["report"] = str(report_path)
                _write(report_path, result)
        else:
            if not args.legacy_sb3:
                parser.error("provide --checkpoint <RSL .pt>; archived trials require explicit --legacy-sb3")
            if args.smoke or args.max_steps is not None or args.scenario_count is not None or args.seed is not None:
                parser.error("--smoke/--seed/--scenario-count/--max-steps require an RSL .pt checkpoint")
            result = run(args.trial, args.output or DEFAULT_OUTPUT, selection=args.selection,
                         device=args.device or "cuda:0", launch=args.launch)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(f"MASTERY EVALUATION REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
