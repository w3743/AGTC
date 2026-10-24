"""Run one paired rough-field evaluation of the verified fast-stop continuation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / (
    "results/isaac_field_improvement/"
    "random_global_rough_fast_stop_lstm512_resume_20261021_attempt2"
)
DEFAULT_OUTPUT = CONTROL.with_name(CONTROL.name + "_evaluation")
BASELINE_SHA256 = "db473f5e78038c066226f61cb6a5ee02ea71615823f28e519366cb0809ecbb8a"
EXPECTED_TIMESTEPS = 196_608
EXPECTED_START_TIMESTEPS = 57_344
SPLIT_SEEDS = {
    "rough_a": (87_001, 87_002, 87_003, 87_004),
    "rough_b": (82_001, 82_002, 82_003, 82_004),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _workspace_path(value: str | Path, *, label: str) -> Path:
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError(f"{label} must not traverse outside the workspace")
    resolved = (raw if raw.is_absolute() else ROOT / raw).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the workspace") from exc
    return resolved


def _child_path(control: Path, value: str | Path, *, label: str) -> Path:
    raw = Path(value)
    resolved = (raw if raw.is_absolute() else control / raw).resolve()
    try:
        resolved.relative_to(control.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the control trial") from exc
    return resolved


def _checkpoint_info(path: Path, expected_timesteps: int,
                     parent_protocol: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    before = path.stat()
    digest_before = _sha256(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"checkpoint ZIP integrity failed at {bad_member}")
        names = set(archive.namelist())
        required = {"data", "policy.pth", "policy.optimizer.pth"}
        if not required.issubset(names):
            raise ValueError("checkpoint must contain data, policy tensors, and Adam state")
        if not archive.getinfo("policy.pth").file_size or not archive.getinfo("policy.optimizer.pth").file_size:
            raise ValueError("checkpoint policy tensors and Adam state must be nonempty")
        data = json.loads(archive.read("data"))
    expected_encoder = parent_protocol["architecture"]["field_encoder_config"]
    if (data.get("num_timesteps") != expected_timesteps
            or data.get("n_steps") != 128 or data.get("n_envs") != 4
            or data.get("gamma") != 1.0
            or data.get("field_action_config") != parent_protocol["action_config"]
            or data.get("field_encoder_config") != expected_encoder
            or data.get("policy_kwargs", {}).get("lstm_hidden_size") != 512
            or data.get("policy_kwargs", {}).get("n_lstm_layers") != 1):
        raise ValueError(f"checkpoint metadata does not match the pinned {expected_timesteps}-step policy")
    after = path.stat()
    digest_after = _sha256(path)
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or digest_before != digest_after):
        raise ValueError(f"checkpoint changed while being inspected: {path}")
    return {
        "path": str(path), "sha256": digest_after, "num_timesteps": expected_timesteps,
        "adam_optimizer_state_present": True, "zip_integrity": True,
    }


def select_final_checkpoint(control: Path, summary: dict[str, Any],
                            manifest: dict[str, Any],
                            parent_protocol: dict[str, Any]) -> dict[str, Any]:
    """Select only the runner-declared final artifact, never a same-step periodic ZIP."""
    final_raw = summary.get("checkpoint_final")
    final_sha = summary.get("checkpoint_final_sha256")
    if not isinstance(final_raw, str) or not final_raw or not isinstance(final_sha, str):
        raise ValueError("resume summary lacks checkpoint_final and checkpoint_final_sha256")
    final_path = _child_path(control, final_raw, label="final checkpoint")
    relative = final_path.relative_to(control.resolve()).as_posix()
    if ("checkpoints" in Path(relative).parts
            or re.search(r"_\d+_steps\.zip$", final_path.name, flags=re.IGNORECASE)):
        raise ValueError("resume_summary final checkpoint must not be a periodic checkpoint")
    if summary.get("actual_final_num_timesteps") != EXPECTED_TIMESTEPS:
        raise ValueError("resume summary does not record exactly 196608 completed timesteps")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError("checkpoint manifest has no checkpoint list")
    matching = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        if _child_path(control, row["path"], label="manifest checkpoint") == final_path:
            matching.append(row)
    if len(matching) != 1:
        raise ValueError("resume summary final artifact must match exactly one manifest row")
    row = matching[0]
    if (row.get("num_timesteps") != EXPECTED_TIMESTEPS
            or row.get("sha256") != final_sha
            or row.get("adam_optimizer_state_present") is not True):
        raise ValueError("final manifest row differs from the resume summary or lacks Adam state")
    info = _checkpoint_info(final_path, EXPECTED_TIMESTEPS, parent_protocol)
    if info["sha256"] != final_sha:
        raise ValueError("final checkpoint bytes differ from the resume summary SHA")
    return info


def validate_completed_trial(control: Path) -> dict[str, Any]:
    """Require real successful completion and verify the frozen trial lineage."""
    from scripts import resume_fast_stop as resume
    from scripts import train_isaac_foundation_fresh as fresh

    control = _workspace_path(control, label="control trial")
    process_exit_path = control / "process_exit.json"
    python_exit_path = control / "python.exit_code"
    status_path = control / "run_status.json"
    summary_path = control / "resume_summary.json"
    protocol_path = control / "protocol.json"
    manifest_path = control / "checkpoint_manifest.json"
    initialization_path = control / "initialization.json"
    for path in (process_exit_path, python_exit_path, status_path, summary_path,
                 protocol_path, manifest_path, initialization_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed control trial evidence is missing: {path.name}")

    protocol_sha = _sha256(protocol_path)
    protocol = _load_json(protocol_path)
    status = _load_json(status_path)
    process_exit = _load_json(process_exit_path)
    try:
        python_exit = int(python_exit_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise ValueError("python.exit_code is missing or malformed") from exc
    if (process_exit.get("source") != "resume_fast_stop_supervisor_waited_for_worker"
            or process_exit.get("actual_exit_code") != 0
            or process_exit.get("exit_code") != 0 or python_exit != 0):
        raise ValueError("control trial lacks a real supervisor-recorded exit code 0")
    if (status.get("status") != "completed" or status.get("phase") != "completed"
            or status.get("exit_code") != 0 or status.get("num_timesteps") != EXPECTED_TIMESTEPS
            or status.get("protocol_sha256") != protocol_sha):
        raise ValueError("control run_status is not completed at exactly 196608 steps")
    if protocol.get("schema") != resume.SCHEMA:
        raise ValueError("control protocol schema is not the fast-stop continuation schema")
    summary = _load_json(summary_path)
    if (summary.get("status") != "training_completed_no_evaluation"
            or summary.get("actual_final_num_timesteps") != EXPECTED_TIMESTEPS
            or summary.get("starting_num_timesteps") != EXPECTED_START_TIMESTEPS
            or summary.get("parent_protocol_sha256") != protocol.get("parent", {}).get("protocol_sha256")):
        raise ValueError("resume summary does not attest the expected completed continuation")
    if (status.get("checkpoint_final") != summary.get("checkpoint_final")
            or status.get("checkpoint_final_sha256") != summary.get("checkpoint_final_sha256")):
        raise ValueError("completed run_status final checkpoint differs from resume_summary")

    # Verify the immutable original protocol and source snapshot, and then all 46
    # current child sources against both their live files and frozen copies.
    parent_ref = protocol.get("parent", {})
    parent_path = _workspace_path(parent_ref.get("path", ""), label="parent trial")
    parent_protocol_path = parent_path / "protocol.json"
    parent_protocol_sha = _sha256(parent_protocol_path)
    if parent_protocol_sha != parent_ref.get("protocol_sha256"):
        raise ValueError("pinned parent protocol SHA changed")
    parent_protocol = _load_json(parent_protocol_path)
    resume._validate_parent_contract(parent_protocol)
    resume._verify_parent_snapshot(parent_path, parent_protocol)
    parent_status_path = _workspace_path(parent_ref.get("status_path", ""), label="parent status")
    if _sha256(parent_status_path) != parent_ref.get("status_sha256"):
        raise ValueError("pinned parent run-status evidence changed")
    source_manifest = protocol.get("source_snapshot", {})
    if (source_manifest.get("base_source_count") != resume.SOURCE_COUNT
            or source_manifest.get("extra_runner") != resume.RUNNER
            or source_manifest.get("total_source_count") != resume.SOURCE_COUNT + 1):
        raise ValueError("control source snapshot is not the frozen 45+resume-runner set")
    resume._verify_child_sources(control, source_manifest)

    recovery_ref = protocol.get("recovery_manifest", {})
    if recovery_ref.get("path") != "recovery_manifest.json":
        raise ValueError("control protocol has an unexpected recovery-manifest reference")
    recovery_path = control / "recovery_manifest.json"
    if not recovery_path.is_file() or _sha256(recovery_path) != recovery_ref.get("sha256"):
        raise ValueError("recovery manifest is missing or changed")
    recovery = _load_json(recovery_path)
    if recovery.get("parent", {}).get("protocol_sha256") != parent_protocol_sha:
        raise ValueError("recovery manifest parent protocol differs from the pinned parent")

    baseline_path = control / "parent_checkpoint_baseline.zip"
    expected_baseline_sha = protocol.get("initialization", {}).get("checkpoint_sha256")
    if expected_baseline_sha != BASELINE_SHA256 or _sha256(baseline_path) != BASELINE_SHA256:
        raise ValueError("copied 57344-step baseline SHA is not the approved checkpoint")
    baseline_info = _checkpoint_info(baseline_path, EXPECTED_START_TIMESTEPS, parent_protocol)
    initialization = _load_json(initialization_path)
    if (initialization.get("status") != "passed"
            or initialization.get("num_timesteps") != EXPECTED_START_TIMESTEPS
            or initialization.get("checkpoint_sha256") != BASELINE_SHA256
            or initialization.get("policy_weights_equal_checkpoint") is not True
            or initialization.get("adam_optimizer") != "Adam"
            or initialization.get("adam_state_equal_checkpoint") is not True):
        raise ValueError("continuation initialization audit did not verify loaded policy and Adam state")

    manifest = _load_json(manifest_path)
    final_info = select_final_checkpoint(control, summary, manifest, parent_protocol)
    baseline_rows = [row for row in manifest.get("checkpoints", [])
                     if isinstance(row, dict)
                     and _child_path(control, row.get("path", ""), label="manifest checkpoint") == baseline_path]
    if (len(baseline_rows) != 1 or baseline_rows[0].get("num_timesteps") != EXPECTED_START_TIMESTEPS
            or baseline_rows[0].get("sha256") != BASELINE_SHA256
            or baseline_rows[0].get("adam_optimizer_state_present") is not True):
        raise ValueError("checkpoint manifest does not contain the pinned 57344-step baseline")

    protocol_parent_sha = _sha256(parent_protocol_path)
    return {
        "control": control,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "parent_protocol": parent_protocol,
        "parent_protocol_sha256": protocol_parent_sha,
        "run_status": status,
        "process_exit": process_exit,
        "resume_summary": summary,
        "checkpoint_manifest_sha256": _sha256(manifest_path),
        "initialization": initialization,
        "baseline": baseline_info,
        "final": final_info,
        "child_source_hashes": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in sorted(source_manifest["files"], key=lambda item: item["path"])
        ],
        "parent_source_hashes": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in sorted(parent_protocol["source_snapshot"]["files"], key=lambda item: item["path"])
        ],
        "recovery_manifest_sha256": recovery_ref["sha256"],
        "_manifest": manifest,
    }


def summarize_trajectory(row: dict[str, Any], step_kinematics: list[dict[str, Any]],
                         *, distance_tolerance_m: float) -> dict[str, Any]:
    """Validate step-complete trajectory/physics samples and derive path metrics."""
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("episode lacks a nonempty recorded trajectory")
    if not isinstance(step_kinematics, list) or len(step_kinematics) != len(trajectory):
        raise ValueError("per-step root kinematics do not match the recorded trajectory length")
    if (isinstance(distance_tolerance_m, bool) or not isinstance(distance_tolerance_m, (int, float))
            or not math.isfinite(float(distance_tolerance_m)) or float(distance_tolerance_m) <= 0):
        raise ValueError("distance tolerance must be finite and positive")
    distances: list[float] = []
    holds: list[float] = []
    for index, (step, kinematics) in enumerate(zip(trajectory, step_kinematics)):
        if not isinstance(step, dict) or not isinstance(kinematics, dict):
            raise ValueError(f"trajectory step {index} is malformed")
        for phase in ("before", "after"):
            snapshot = step.get(phase)
            if not isinstance(snapshot, dict):
                raise ValueError(f"trajectory step {index} is missing {phase} snapshot")
            distance = snapshot.get("goal_distance_m")
            hold = snapshot.get("hold_s")
            time_s = snapshot.get("time_s")
            if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                    or not math.isfinite(float(distance)) or float(distance) < 0
                    or isinstance(hold, bool) or not isinstance(hold, (int, float))
                    or not math.isfinite(float(hold)) or float(hold) < 0
                    or isinstance(time_s, bool) or not isinstance(time_s, (int, float))
                    or not math.isfinite(float(time_s)) or float(time_s) < 0):
                raise ValueError(f"trajectory step {index} has invalid distance, hold, or time data")
            distances.append(float(distance))
            holds.append(float(hold))
        for key in ("time_s", "linear_speed_xy_norm_m_s", "angular_speed_xyz_norm_rad_s", "hold_s", "distance_m"):
            value = kinematics.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"trajectory step {index} lacks finite physical {key}")
        if kinematics.get("speed_source") != "root_com_vel_w":
            raise ValueError(f"trajectory step {index} did not use native root COM velocity")
        after_distance = float(step["after"]["goal_distance_m"])
        if not math.isclose(float(kinematics["distance_m"]), after_distance,
                            rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"trajectory step {index} kinematics are misaligned with its after snapshot")
        for key in ("time_s", "hold_s"):
            if not math.isclose(float(kinematics[key]), float(step["after"][key]),
                                rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"trajectory step {index} kinematics {key} are misaligned with its after snapshot")
        step["root_com_kinematics"] = dict(kinematics)
    stop = row.get("goal_stop")
    if (not isinstance(stop, dict) or stop.get("schema") != "goal_stop_v1"
            or stop.get("profile") != "fast_stop_v1" or not step_kinematics):
        raise ValueError("episode lacks its terminal fast-stop record")
    final_kinematics = step_kinematics[-1]
    for source_key, stop_key in (
        ("linear_speed_xy_norm_m_s", "linear_speed_m_s"),
        ("angular_speed_xyz_norm_rad_s", "angular_speed_rad_s"),
        ("hold_s", "hold_s"),
    ):
        if not math.isclose(float(final_kinematics[source_key]), float(stop[stop_key]),
                            rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"terminal goal-stop {stop_key} differs from the final physical sample")
    result = {
        "min_distance_m": min(distances),
        "entered_radius": min(distances) <= float(distance_tolerance_m),
        "distance_tolerance_m": float(distance_tolerance_m),
        "max_hold_s": max(holds),
        "kinematic_samples": len(step_kinematics),
        "speed_source": "root_com_vel_w (XY norm and full XYZ angular norm)",
    }
    row["path_metrics"] = result
    return result


def _require_valid_pairing(pairing: dict[str, Any], seeds: tuple[int, ...],
                           baseline: dict[str, Any], final: dict[str, Any]) -> None:
    if pairing.get("valid") is not True or pairing.get("mismatches"):
        raise ValueError(f"paired evaluation conditions are invalid: {pairing.get('mismatches')}")
    if pairing.get("paired_seed_count") != len(seeds):
        raise ValueError("paired evaluation did not cover the expected number of seeds")
    expected = set(seeds)
    for label, report in (("baseline", baseline), ("final", final)):
        rows = report.get("rows")
        observed = [row.get("seed") for row in rows] if isinstance(rows, list) else []
        if len(observed) != len(seeds) or set(observed) != expected:
            raise ValueError(f"{label} report does not contain exactly the paired development seeds")


def compare_split(split: str, baseline: dict[str, Any], final: dict[str, Any],
                  pairing: dict[str, Any]) -> dict[str, Any]:
    from scripts import improve_isaac_policy as improve
    from scripts import train_isaac_foundation_fresh as fresh

    seeds = SPLIT_SEEDS[split]
    _require_valid_pairing(pairing, seeds, baseline, final)
    before_rows = {row["seed"]: row for row in baseline["rows"]}
    after_rows = {row["seed"]: row for row in final["rows"]}
    outcomes = []
    common_clean_times = []
    for seed in seeds:
        before = before_rows[seed]
        after = after_rows[seed]
        before_clean = (before.get("success") is True and before.get("damage") == 0
                        and before.get("collisions") == 0)
        after_clean = (after.get("success") is True and after.get("damage") == 0
                       and after.get("collisions") == 0)
        common_clean = before_clean and after_clean
        if common_clean:
            common_clean_times.append({
                "seed": seed,
                "baseline_s": before["episode_time_s"],
                "final_s": after["episode_time_s"],
                "delta_s": float(after["episode_time_s"]) - float(before["episode_time_s"]),
            })
        outcomes.append({
            "seed": seed,
            "baseline": {key: before.get(key) for key in (
                "success", "reason", "timed_out", "episode_time_s", "final_distance",
                "damage", "collisions", "goal_stop", "path_metrics")},
            "final": {key: after.get(key) for key in (
                "success", "reason", "timed_out", "episode_time_s", "final_distance",
                "damage", "collisions", "goal_stop", "path_metrics")},
            "common_clean_success": common_clean,
        })
    termination_safety = fresh._rough_termination_safety(baseline, final)
    safe_assessment = improve._safe_goto_assessment(baseline, final)
    strict_time = fresh._fast_stop_time_assessment(
        baseline, final, pairing_valid=True,
        termination_safety_not_increased=termination_safety["not_increased"],
    )
    return {
        "split": split,
        "pairing": pairing,
        "safety": {
            "baseline": {key: baseline["summary"].get(key) for key in (
                "clean_successes", "damaged_episodes", "mean_damage", "total_collisions")},
            "final": {key: final["summary"].get(key) for key in (
                "clean_successes", "damaged_episodes", "mean_damage", "total_collisions")},
            "rough_termination_events": termination_safety,
            "predeclared_assessment": safe_assessment,
            "gate_passed": (safe_assessment == "improved_under_predeclared_safety_criterion"
                            and termination_safety["not_increased"]),
        },
        "paired_common_clean_success_time": {
            "episodes": common_clean_times,
            "count": len(common_clean_times),
            "median_delta_s": (None if not common_clean_times else
                               float(statistics.median(item["delta_s"]
                                                       for item in common_clean_times))),
            "interpretation": "descriptive paired time among seeds clean-successful in both models",
        },
        "strict_paired_time_assessment": strict_time,
        "episode_outcomes": outcomes,
    }


def _evaluate_checkpoint(output: Path, cfg, checkpoint: Path, label: str,
                         seeds: tuple[int, ...], plants: int, angle_deg: float,
                         environment_seed: int, device: str,
                         profile_physics_contract: dict,
                         goal_objective_contract: dict) -> dict[str, Any]:
    """Use fresh's real evaluator with a local adapter for per-step root speeds."""
    from isaac_field.env import SingleFieldAdapter
    from isaac_field.runtime import create_env
    from isaac_field.learning import evaluate_model
    from scripts import improve_isaac_policy as improve
    from scripts import train_isaac_foundation_fresh as fresh

    class GoalStopTraceAdapter(SingleFieldAdapter):
        def __init__(self, env):
            super().__init__(env)
            self._active_trace: list[dict[str, Any]] | None = None
            self._traces: list[list[dict[str, Any]]] = []

        def reset(self, *, seed=None, options=None):
            if self._active_trace is not None:
                self._traces.append(self._active_trace)
            self._active_trace = []
            return super().reset(seed=seed, options=options)

        def step(self, action):
            result = super().step(action)
            task = self.task
            stop = task._goal_stop_state()
            if stop.get("speed_source") != "root_com_vel_w":
                raise ValueError("per-step speed sample fell back from native root COM velocity")
            goal = task.goal
            robot = task.robot
            distance = math.hypot(float(goal[0]) - float(robot[0]),
                                  float(goal[1]) - float(robot[1]))
            sample = {
                "time_s": float(task.time),
                "distance_m": distance,
                "linear_speed_xy_norm_m_s": float(stop["linear_speed_m_s"]),
                "angular_speed_xyz_norm_rad_s": float(stop["angular_speed_rad_s"]),
                "hold_s": float(stop["hold_s"]),
                "speed_source": str(stop["speed_source"]),
            }
            if not all(math.isfinite(sample[key]) and sample[key] >= 0 for key in (
                    "time_s", "distance_m", "linear_speed_xy_norm_m_s",
                    "angular_speed_xyz_norm_rad_s", "hold_s")):
                raise ValueError("per-step root kinematics contain invalid values")
            if self._active_trace is None:
                raise RuntimeError("evaluation stepped before resetting the trace adapter")
            self._active_trace.append(sample)
            return result

        def finish_traces(self) -> list[list[dict[str, Any]]]:
            if self._active_trace is not None:
                self._traces.append(self._active_trace)
                self._active_trace = None
            return self._traces

    eval_cfg = fresh.replace(cfg, plants=plants)
    angle = math.radians(angle_deg)
    front_guide_spec, body_stability_config = fresh._rough_profile_components(
        fresh.PROFILE_RANDOM_ROUGH_FAST_STOP
    )
    engine = create_env(
        cfg=eval_cfg, mode="goto", num_envs=1, seed=environment_seed,
        auto_reset=False, goal_distance=None, goal_angle=angle, distance_cost=0.0,
        front_guide_spec=front_guide_spec,
        body_stability_config=body_stability_config,
    )
    adapter = GoalStopTraceAdapter(engine)
    try:
        fresh._validate_rough_profile_environment(engine, profile_physics_contract)
        assets = fresh._runtime_assets(engine)
        report = evaluate_model(
            adapter, checkpoint, seeds, goal_distance=None, goal_angle=angle,
            mode="goto", deterministic=True, device=device, record_trajectories=True,
            legacy_sb3=Path(checkpoint).suffix.lower() == ".zip",
        )
        traces = adapter.finish_traces()
    finally:
        adapter.close()
    rows = report.get("rows", [])
    if len(rows) != len(seeds) or len(traces) != len(rows):
        raise ValueError("evaluator and physical trace adapter returned different episode counts")
    tolerance = float(goal_objective_contract["goal_stop"]["distance_tolerance_m"])
    for row, trace in zip(rows, traces):
        if len(row.get("trajectory", [])) != len(trace):
            raise ValueError(f"seed {row.get('seed')} has incomplete per-step root kinematics")
        summarize_trajectory(row, trace, distance_tolerance_m=tolerance)

    report = improve._add_safety_summary(report)
    report = fresh._add_rough_evaluation_summaries(report)
    report = fresh._add_fast_stop_evaluation_summaries(report, goal_objective_contract)
    report["runtime_assets"] = assets
    terrain_config = {key: getattr(eval_cfg, key) for key in fresh.RANDOM_ROUGH_TERRAIN_CONFIG}
    if terrain_config != profile_physics_contract.get("terrain_config"):
        raise ValueError("evaluation terrain differs from the pinned rough profile")
    report["evaluation_config"] = {
        "environment_seed": int(environment_seed), "plants": int(plants),
        "goal_distance_m": None, "goal_angle_deg": float(angle_deg),
        "goal_sampling_profile": fresh.PROFILE_RANDOM_ROUGH,
        "terrain_config": terrain_config,
        "goal_objective_contract": goal_objective_contract,
    }
    report["evaluation_model_label"] = label
    return report


def run_evaluation(control: Path, output: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    verified = validate_completed_trial(control)
    if (Path(verified["baseline"]["path"]).suffix.lower() == ".zip"
            or Path(verified["final"]["path"]).suffix.lower() == ".zip"):
        raise RuntimeError(
            "this frozen evaluation is pinned to Newton-era SB3 ZIP checkpoints; "
            "it cannot sample them in the current Isaac Sim PhysX environment without "
            "creating a new, explicitly identified cross-simulator evaluation protocol"
        )
    control = verified["control"]
    output = _workspace_path(output, label="evaluation output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    from scripts import train_isaac_foundation_fresh as fresh
    from isaac_field.task import Config

    protocol = verified["protocol"]
    cfg = Config(**protocol["config"])
    profile = fresh.PROFILE_RANDOM_ROUGH_FAST_STOP
    evaluation = fresh._profile_evaluation(profile)
    if evaluation.get("goal_objective_contract") != protocol.get("goal_objective_contract"):
        raise ValueError("live fast-stop objective differs from the frozen control protocol")
    output.mkdir(parents=True)
    report_manifest = {
        "schema": "fast-stop-resume-paired-development-evaluation-v1",
        "status": "running",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_trial": control.relative_to(ROOT.resolve()).as_posix(),
        "output": output.relative_to(ROOT.resolve()).as_posix(),
        "device": device,
        "trial_protocol_sha256": verified["protocol_sha256"],
        "parent_protocol_sha256": verified["parent_protocol_sha256"],
        "recovery_manifest_sha256": verified["recovery_manifest_sha256"],
        "checkpoint_manifest_sha256": verified["checkpoint_manifest_sha256"],
        "run_status": verified["run_status"],
        "process_exit": verified["process_exit"],
        "resume_summary": verified["resume_summary"],
        "initialization": verified["initialization"],
        "checkpoints": {"57344_baseline": verified["baseline"],
                        "196608_final": verified["final"]},
        "development_splits": {
            split: {"environment_seed": evaluation[split]["environment_seed"],
                    "episode_seeds": list(SPLIT_SEEDS[split]),
                    "plants": evaluation[split]["plants"],
                    "goal_angle_deg": evaluation[split]["goal_angle_deg"],
                    "goal_distance_m": evaluation[split]["goal_distance_m"]}
            for split in ("rough_a", "rough_b")
        },
        "child_source_hashes": verified["child_source_hashes"],
        "parent_source_hashes": verified["parent_source_hashes"],
        "evaluation_runner_sha256": _sha256(Path(__file__).resolve()),
        "blind_830xx": {"used": False, "observed": False},
    }
    _write_json(output / "run_manifest.json", report_manifest)
    try:
        split_comparisons = {}
        objective = protocol["goal_objective_contract"]
        physics = protocol["profile_physics_contract"]
        for split in ("rough_a", "rough_b"):
            plants, angle_deg, goal_distance, environment_seed = fresh._evaluation_settings(
                evaluation, split
            )
            if goal_distance is not None:
                raise ValueError(f"{split} must preserve full random-rough goal sampling")
            seeds = SPLIT_SEEDS[split]
            reports = {}
            for model_label, checkpoint_info in (
                ("baseline_57344", verified["baseline"]),
                ("final_196608", verified["final"]),
            ):
                report = _evaluate_checkpoint(
                    output, cfg, Path(checkpoint_info["path"]), model_label, seeds,
                    plants, angle_deg, environment_seed, device, physics, objective,
                )
                reports[model_label] = report
                _write_json(output / f"evaluation_{split}_{model_label}.json", report)
            pairing = fresh._paired_dev_conditions(
                {"baseline": reports["baseline_57344"], "final": reports["final_196608"]},
                seeds, require_rough_terrain_pair=True,
            )
            _write_json(output / f"pairing_{split}.json", pairing)
            split_comparisons[split] = compare_split(
                split, reports["baseline_57344"], reports["final_196608"], pairing,
            )
        comparison = {
            "schema": "fast-stop-resume-paired-development-comparison-v1",
            "baseline_steps": EXPECTED_START_TIMESTEPS,
            "final_steps": EXPECTED_TIMESTEPS,
            "claim_limit": "Paired development evidence only; not held-out generalization or real-vehicle safety calibration.",
            "splits": split_comparisons,
        }
        _write_json(output / "comparison.json", comparison)
        report_manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output / "run_manifest.json", report_manifest)
        return comparison
    except BaseException as exc:
        report_manifest.update(status="failed", failed_at_utc=datetime.now(timezone.utc).isoformat(),
                               exception=repr(exc))
        _write_json(output / "run_manifest.json", report_manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_evaluation(args.control, args.output, device=args.device)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
