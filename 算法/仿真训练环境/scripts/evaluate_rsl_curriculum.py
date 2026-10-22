"""Full-episode development evaluation for a native RSL field checkpoint.

This runs the existing learning.evaluate_model loop on the task and physics
configuration recorded in the checkpoint. Results are development data only;
this entrypoint never triggers a curriculum controller or acceptance action.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SEED_START = 87_010
DEFAULT_SCENARIO_COUNT = 16
STRICTPASS_SCENARIOS = 16
STRICTPASS_CLEAN_SUCCESS_MIN = 13
RESERVED_CURRICULUM_SEEDS = (83_000, 83_999)
TRACE_SPEED_SOURCE = "root_com_vel_w"
TRACK_SIGN_EPSILON_MPS = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _validate_sampling(
    seed_start: int,
    scenario_count: int,
    goal_distance: tuple[float, float] | None,
    goal_angle_deg: float,
    field_size: float,
) -> tuple[tuple[float, float] | None, float]:
    if type(seed_start) is not int or seed_start < 0:
        raise ValueError("seed_start must be a non-negative integer")
    if type(scenario_count) is not int or scenario_count < 1:
        raise ValueError("scenario_count must be a positive integer")
    seed_end = seed_start + scenario_count - 1
    reserved_start, reserved_end = RESERVED_CURRICULUM_SEEDS
    if seed_start <= reserved_end and seed_end >= reserved_start:
        raise ValueError(
            "development evaluation seeds must not intersect reserved curriculum seed bank "
            f"[{reserved_start}, {reserved_end}]"
        )
    if goal_distance is not None:
        if (len(goal_distance) != 2
                or not all(math.isfinite(float(value)) for value in goal_distance)
                or float(goal_distance[0]) < 0
                or float(goal_distance[1]) < float(goal_distance[0])):
            raise ValueError("goal_distance must be a finite non-negative MIN/MAX pair")
        if float(goal_distance[1]) > math.sqrt(2.0) * float(field_size):
            raise ValueError("goal_distance maximum cannot exceed the field diagonal")
        goal_distance = (float(goal_distance[0]), float(goal_distance[1]))
    angle = float(goal_angle_deg)
    if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
        raise ValueError("goal_angle_deg must be in [0, 180]")
    return goal_distance, math.radians(angle)


def _goal_stop_trace_adapter(base_adapter):
    """Extend SingleFieldAdapter with one measured stop-state sample per step."""

    class GoalStopTraceAdapter(base_adapter):
        def __init__(self, env):
            super().__init__(env)
            self._active_stop_trace: list[dict[str, Any]] | None = None
            self._stop_traces: list[list[dict[str, Any]]] = []

        def reset(self, *, seed=None, options=None):
            if self._active_stop_trace is not None:
                self._stop_traces.append(self._active_stop_trace)
            self._active_stop_trace = []
            return super().reset(seed=seed, options=options)

        def step(self, action):
            result = super().step(action)
            if self._active_stop_trace is None:
                raise RuntimeError("evaluation stepped before reset initialized the stop trace")
            task = self.task
            stop = task._goal_stop_state()
            if stop.get("speed_source") != TRACE_SPEED_SOURCE:
                raise ValueError(
                    "RSL curriculum evaluation requires per-step root COM linear and full angular speed; "
                    f"got {stop.get('speed_source')!r}"
                )
            goal = task.goal
            robot = task.robot
            sample = {
                "time_s": float(task.time),
                "distance_m": float(math.hypot(
                    float(goal[0]) - float(robot[0]), float(goal[1]) - float(robot[1])
                )),
                "linear_speed_xy_norm_m_s": float(stop["linear_speed_m_s"]),
                "angular_speed_xyz_norm_rad_s": float(stop["angular_speed_rad_s"]),
                "hold_s": float(stop["hold_s"]),
                "linear_speed_limit_m_s": float(stop["stop_speed_limit_m_s"]),
                "angular_speed_limit_rad_s": float(stop["stop_angular_speed_limit_rad_s"]),
                "hold_required_s": float(stop["hold_required_s"]),
                "distance_tolerance_m": float(stop["distance_tolerance_m"]),
                "within_stop_limits": bool(stop["within_stop_limits"]),
                "speed_source": str(stop["speed_source"]),
            }
            numeric = (
                "time_s", "distance_m", "linear_speed_xy_norm_m_s",
                "angular_speed_xyz_norm_rad_s", "hold_s", "linear_speed_limit_m_s",
                "angular_speed_limit_rad_s", "hold_required_s", "distance_tolerance_m",
            )
            if any(not math.isfinite(sample[key]) or sample[key] < 0 for key in numeric):
                raise ValueError("per-step goal stop trace contains invalid kinematics")
            self._active_stop_trace.append(sample)
            return result

        def finish_stop_traces(self) -> list[list[dict[str, Any]]]:
            if self._active_stop_trace is not None:
                self._stop_traces.append(self._active_stop_trace)
                self._active_stop_trace = None
            return self._stop_traces

    return GoalStopTraceAdapter


def _track_reversals(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    commands = []
    for step in trajectory:
        value = step.get("track_command_lr_mps")
        if value is None:
            raise ValueError("recorded trajectory is missing physical left/right track commands")
        pair = [float(item) for item in value]
        if len(pair) != 2 or not all(math.isfinite(item) for item in pair):
            raise ValueError("physical left/right track command must contain two finite speeds")
        commands.append(pair)

    reversals = {"left": 0, "right": 0}
    prior = {"left": 0, "right": 0}
    for left, right in commands:
        for side, value in (("left", left), ("right", right)):
            sign = 1 if value > TRACK_SIGN_EPSILON_MPS else (
                -1 if value < -TRACK_SIGN_EPSILON_MPS else 0
            )
            if sign and prior[side] and sign != prior[side]:
                reversals[side] += 1
            if sign:
                prior[side] = sign
    return {
        "left_track": reversals["left"],
        "right_track": reversals["right"],
        "total": reversals["left"] + reversals["right"],
        "definition": (
            "sign changes between consecutive nonzero physical track commands; "
            f"commands within ±{TRACK_SIGN_EPSILON_MPS:g} m/s are ignored"
        ),
    }


def _finalize_episode_rows(
    evaluated: dict[str, Any], stop_traces: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = evaluated.get("rows")
    if not isinstance(rows, list) or len(rows) != len(stop_traces):
        raise ValueError("evaluator rows and measured stop traces differ in episode count")
    unsafe_count = 0
    clean_count = 0
    damage_total = 0
    collision_total = 0
    reversal_total = {"left_track": 0, "right_track": 0, "total": 0}
    max_linear_speed = 0.0
    max_angular_speed = 0.0

    for row, trace in zip(rows, stop_traces):
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != len(trace) or not trace:
            raise ValueError(f"seed {row.get('seed')} has incomplete trajectory/stop-trace rows")
        for step, sample in zip(trajectory, trace):
            step["goal_stop_trace"] = sample
            max_linear_speed = max(max_linear_speed, sample["linear_speed_xy_norm_m_s"])
            max_angular_speed = max(max_angular_speed, sample["angular_speed_xyz_norm_rad_s"])
        final_stop = trace[-1]
        row["goal_stop_trace"] = trace
        row["terminal_stop_state"] = final_stop
        reversals = _track_reversals(trajectory)
        row["track_direction_reversals"] = reversals
        for key in reversal_total:
            reversal_total[key] += reversals[key]

        missing_counts = [key for key in ("damage", "collisions") if key not in row]
        if missing_counts:
            raise ValueError(
                f"seed {row.get('seed')} is missing native episode safety metrics: {missing_counts}"
            )
        damage = row["damage"]
        collisions = row["collisions"]
        if (type(damage) is not int or damage < 0
                or type(collisions) is not int or collisions < 0):
            raise ValueError(
                f"seed {row.get('seed')} damage and collisions must be non-negative integers"
            )
        damage_total += damage
        collision_total += collisions
        safe_terminal_stop = bool(
            final_stop["within_stop_limits"]
            and final_stop["hold_s"] + 1e-9 >= final_stop["hold_required_s"]
            and final_stop["distance_m"] <= final_stop["distance_tolerance_m"]
        )
        successful_goal = bool(row.get("success")) and row.get("reason") == "goal_reached"
        clean_success = successful_goal and damage == 0 and collisions == 0 and safe_terminal_stop
        unsafe_reasons = []
        if damage:
            unsafe_reasons.append("damage")
        if collisions:
            unsafe_reasons.append("collision")
        if successful_goal and not safe_terminal_stop:
            unsafe_reasons.append("goal_reached_without_measured_stop")
        unsafe = bool(unsafe_reasons)
        row["terminal_stop_safe"] = safe_terminal_stop
        row["clean_success"] = bool(clean_success)
        row["unsafe"] = unsafe
        row["unsafe_reasons"] = unsafe_reasons
        unsafe_count += int(unsafe)
        clean_count += int(clean_success)

    scenario_count = len(rows)
    strictpass_eligible = scenario_count == STRICTPASS_SCENARIOS
    strictpass = bool(
        strictpass_eligible
        and clean_count >= STRICTPASS_CLEAN_SUCCESS_MIN
        and damage_total == 0
        and collision_total == 0
    )
    summary = evaluated.setdefault("summary", {})
    summary.update({
        "clean_successes": clean_count,
        "clean_success_rate": clean_count / scenario_count if scenario_count else None,
        "damage_total": damage_total,
        "collision_total": collision_total,
        "unsafe_episodes": unsafe_count,
        "strictpass_required_scenarios": STRICTPASS_SCENARIOS,
        "strictpass_required_clean_successes": STRICTPASS_CLEAN_SUCCESS_MIN,
        "strictpass_eligible": strictpass_eligible,
        "strictpass": strictpass,
        "total_track_direction_reversals": reversal_total,
        "max_measured_linear_speed_xy_m_s": max_linear_speed,
        "max_measured_angular_speed_xyz_rad_s": max_angular_speed,
    })
    evaluated["assessment"] = (
        "strictpass" if strictpass else "strictfail" if strictpass_eligible else "not_assessed"
    )
    evaluated["claim_limit"] = (
        "Development evaluation data only. No course controller, 830xx curriculum, or training action is triggered."
    )
    return evaluated


def run_evaluation(
    checkpoint: str | Path,
    output: str | Path,
    *,
    seed_start: int = DEFAULT_SEED_START,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    goal_distance: tuple[float, float] | None = None,
    goal_angle_deg: float = 180.0,
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"RSL curriculum evaluation requires an existing .pt checkpoint: {path}")
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {output_dir}")
    if output_dir == path or output_dir in path.parents:
        raise ValueError("evaluation output must not contain or equal the checkpoint path")

    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.rsl_rl_inference import RslPredictionModel
    from scripts import view_fast_stop_goal as goal_viewer

    checkpoint_stat = path.stat()
    checkpoint_sha = _sha256(path)
    model = RslPredictionModel(
        path, device="cpu", input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    if (model.backend != "rsl_rl" or model.input_schema != ABSOLUTE_GOAL_INPUT_SCHEMA
            or model.checkpoint_sha256 != checkpoint_sha):
        raise ValueError("checkpoint did not load as the expected native ABS-schema RSL policy")
    contract = model.contract
    if contract.get("backend") != "rsl_rl" or contract.get("input_schema") != ABSOLUTE_GOAL_INPUT_SCHEMA:
        raise ValueError("RSL checkpoint contract must explicitly declare the ABS schema")
    task_config = contract.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError("RSL checkpoint must contain the complete task_config")
    episode_duration_s = float(task_config["max_time"])
    task_dt_s = float(task_config["dt"])
    goal_distance, goal_angle = _validate_sampling(
        seed_start, scenario_count, goal_distance, goal_angle_deg, float(task_config["size"]),
    )
    seeds = list(range(seed_start, seed_start + scenario_count))
    expected_steps = max(1, int(math.ceil(episode_duration_s / task_dt_s)))

    from scripts import evaluate_fast_stop_mastery as mastery
    mastery._assert_no_parallel_simulation()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {output_dir}")
    output_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "isaac-field-rsl-curriculum-evaluation-run-v1",
        "status": "running",
        "started_at_utc": started,
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_num_timesteps": model.num_timesteps,
        "contract_sha256": goal_viewer._stable_hash(contract),
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "scenario_seeds": seeds,
        "goal_distance_m": None if goal_distance is None else list(goal_distance),
        "goal_sampling": "global" if goal_distance is None else "uniform_distance_range",
        "goal_angle_deg": float(goal_angle_deg),
        "episode_duration_s": episode_duration_s,
        "max_steps_per_episode": expected_steps,
        "max_steps_cap_applied": False,
        "device": "cpu",
        "record_trajectories": True,
        "record_initial_conditions": True,
        "claim_limit": "Development data only; no curriculum controller is invoked.",
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    engine = None
    adapter = None
    try:
        engine, base_adapter = mastery._create_rsl_evaluation_adapter(contract, seed_start)
        goal_viewer._verify_rsl_environment_contract(engine, contract)
        adapter_class = _goal_stop_trace_adapter(type(base_adapter))
        adapter = adapter_class(engine)
        del base_adapter
        from isaac_field.learning import evaluate_model

        evaluated = evaluate_model(
            adapter,
            model,
            seeds,
            goal_distance=goal_distance,
            goal_angle=goal_angle,
            mode="goto",
            deterministic=True,
            max_steps=None,
            device="cpu",
            record_trajectories=True,
            record_initial_conditions=True,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            expected_observation_config=contract.get("observation_config"),
        )
        traces = adapter.finish_stop_traces()
        if evaluated.get("model_class") != "RslPredictionModel":
            raise ValueError("learning.evaluate_model did not load the native RSL predictor")
        if evaluated.get("input_schema") != ABSOLUTE_GOAL_INPUT_SCHEMA:
            raise ValueError("learning.evaluate_model returned a non-ABS policy schema")
        native_metrics = dict(evaluated.get("summary", {}))
        report = _finalize_episode_rows(evaluated, traces)
        report.update({
            "schema": "isaac-field-rsl-curriculum-evaluation-v1",
            "backend": "rsl_rl",
            "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_num_timesteps": model.num_timesteps,
            "contract_sha256": goal_viewer._stable_hash(contract),
            "task_config": contract["task_config"],
            "front_guide_spec": contract["front_guide_spec"],
            "body_stability_config": contract["body_stability_config"],
            "initial_conditions": [
                {
                    "seed": int(row["seed"]),
                    "reset_initial_condition": row["reset_initial_condition"],
                    "trajectory_start_matches_reset": bool(
                        row.get("trajectory_start_matches_reset", False)
                    ),
                }
                for row in report["rows"]
            ],
            "sampling": {
                "scenario_seeds": seeds,
                "seed_start": seed_start,
                "scenario_count": scenario_count,
                "goal_distance_m": None if goal_distance is None else list(goal_distance),
                "goal_sampling": "global" if goal_distance is None else "uniform_distance_range",
                "goal_angle_deg": float(goal_angle_deg),
                "goal_angle_rad": goal_angle,
            "episode_duration_s": episode_duration_s,
                "max_steps_per_episode": expected_steps,
                "max_steps_cap_applied": False,
                "record_trajectories": True,
                "record_initial_conditions": True,
                "deterministic": True,
                "device": "cpu",
            },
            "native_metrics": native_metrics,
            "task_contract": {
                "task_config": contract["task_config"],
                "front_guide_spec": contract["front_guide_spec"],
                "body_stability_config": contract["body_stability_config"],
            },
        })
        _write_json(output_dir / "evaluation.json", report)

        after_stat = path.stat()
        if ((checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
                != (after_stat.st_size, after_stat.st_mtime_ns)
                or _sha256(path) != checkpoint_sha):
            raise RuntimeError(f"checkpoint changed during evaluation: {path}")
        manifest.update(
            status="completed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            assessment=report["assessment"],
            evaluation_report="evaluation.json",
            checkpoint_sha256_after=_sha256(path),
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        return {
            "status": "completed",
            "assessment": report["assessment"],
            "output": str(output_dir),
            "checkpoint_sha256": checkpoint_sha,
            "summary": report["summary"],
        }
    except BaseException as exc:
        manifest.update(
            status="failed",
            failed_at_utc=datetime.now(timezone.utc).isoformat(),
            exception=repr(exc),
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        elif engine is not None:
            engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="existing native RSL-RL .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="new, non-existing directory for the evaluation report")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--scenario-count", type=int, default=DEFAULT_SCENARIO_COUNT)
    parser.add_argument("--goal-distance", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=None,
                        help="optional distance range in metres; omitted means global goal sampling")
    parser.add_argument("--goal-angle-deg", type=float, default=180.0,
                        help="half-angle of goal sampling in degrees, from 0 to 180")
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(
            args.checkpoint,
            args.output,
            seed_start=args.seed_start,
            scenario_count=args.scenario_count,
            goal_distance=None if args.goal_distance is None else tuple(args.goal_distance),
            goal_angle_deg=args.goal_angle_deg,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
