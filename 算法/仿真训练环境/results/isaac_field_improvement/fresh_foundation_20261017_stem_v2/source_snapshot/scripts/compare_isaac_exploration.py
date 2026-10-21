"""Prepare or launch a paired Gaussian-vs-gSDE RecurrentPPO experiment."""
from __future__ import annotations

import argparse, hashlib, json, math, shutil, zipfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/exploration_gsde_20261017"
DEFAULT_CHECKPOINT = ROOT / "results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip"
SOURCE_FILES = tuple(sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "isaac_field").glob("*.py"))) + (
    "scripts/isaaclab_python.sh", "scripts/compare_isaac_exploration.py", "scripts/improve_isaac_policy.py",
)

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def _checkpoint_steps(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read("data"))["num_timesteps"])

def _source_manifest(output: Path, *, copy_files: bool) -> dict:
    rows = []
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target_relative = f"source_snapshot/{relative}"
        digest = _sha256(source)
        if copy_files:
            target = output / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rows.append({"path": relative, "snapshot_path": target_relative, "sha256": digest})
    return {"files": rows}

def prepare(output: Path, checkpoint: Path) -> dict:
    """Freeze one protocol and source snapshot; never overwrite an existing run."""
    output, checkpoint = output.resolve(), checkpoint.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse experiment output: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    steps = _checkpoint_steps(checkpoint)
    if steps != 53_248:
        raise ValueError(f"expected the 53,248-step source checkpoint, got {steps}: {checkpoint}")
    from isaac_field.task import Config
    config = asdict(Config(plants=16, layout="rows", row_spacing=.8, plant_spacing=.3,
                           terrain_enabled=False, max_time=20., damage_penalty=20., goal_progress_budget=5.))
    output.mkdir(parents=True)
    manifest = _source_manifest(output, copy_files=True)
    protocol = {
        "schema": "isaac-field-exploration-pair-v1", "status": "prepared",
        "source_checkpoint": {"path": checkpoint.relative_to(ROOT).as_posix(), "sha256": _sha256(checkpoint), "num_timesteps": steps},
        "source_snapshot": manifest,
        "config": config,
        "task": {"mode": "goto", "goal_distance_m": [1., 3.], "goal_angle_rad": math.pi / 3., "distance_cost": 0.},
        "calibration": {"environment_seed": 78001, "steps": 128, "deterministic_source_actions": True,
                        "latent": "policy_net forward-hook output after recurrent policy path"},
        "training": {"source_checkpoint_timesteps": 53248, "timesteps_per_variant": 16384, "num_envs": 4,
                     "environment_seed": 68001, "initialization_seed": 68001,
                     "hyperparameters": {"n_steps": 128, "batch_size": 128, "n_epochs": 5,
                                          "learning_rate": 1e-4, "gamma": .995, "target_kl": .03, "goal_gain": 100.},
                     "optimizer": "fresh Adam for both variants; no state copied",
                     "variants": {"gaussian": {"use_sde": False, "sde_sample_freq": -1},
                                  "gsde": {"use_sde": True, "sde_sample_freq": 8}}},
        "evaluation": {"environment_seed": 80003, "seeds": list(range(82001, 82025)),
                       "goal_distance_m": [1., 3.], "goal_angle_rad": math.pi / 3.,
                       "distance_cost": 0., "deterministic": True, "record_trajectories": True},
        "noise_schedule_note": "gSDE resamples its matrix every 8 policy steps; actions still use each observation/LSTM state each tick. At 20 Hz this is 0.4 s logical time, not a measured frame rate.",
        "assessment": {"helper": "scripts.improve_isaac_policy._safe_goto_assessment",
                       "pairing": "identical held-out reset conditions", "claim_limit": "one paired 24-seed run is not a general competence guarantee"},
    }
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {"status": "prepared", "protocol_sha256": _sha256(output / "protocol.json")})
    return {"output": str(output), "checkpoint_sha256": protocol["source_checkpoint"]["sha256"], "source_files": len(manifest["files"])}

def preflight(output: Path) -> dict:
    output = output.resolve()
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare first; protocol and prepared status are required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "prepared":
        raise FileExistsError(f"refusing to launch output with status={status.get('status')!r}")
    if status.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("protocol changed after prepare")
    checkpoint = ROOT / protocol["source_checkpoint"]["path"]
    if not checkpoint.is_file() or _sha256(checkpoint) != protocol["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint hash changed after prepare")
    if _checkpoint_steps(checkpoint) != protocol["source_checkpoint"]["num_timesteps"]:
        raise ValueError("source checkpoint timestep marker changed after prepare")
    manifest = protocol["source_snapshot"]
    if set(SOURCE_FILES) != {row["path"] for row in manifest["files"]}:
        raise ValueError("source file inventory changed after prepare")
    for row in manifest["files"]:
        live, snapshot = ROOT / row["path"], output / row["snapshot_path"]
        if _sha256(live) != row["sha256"] or _sha256(snapshot) != row["sha256"]:
            raise ValueError(f"source snapshot mismatch: {row['path']}")
    return protocol

def _open_env(cfg, protocol: dict, seed: int, num_envs: int, auto_reset: bool):
    from isaac_field.env import make_env
    task = protocol["task"]
    return make_env(
        cfg=cfg, mode=task["mode"], num_envs=num_envs, seed=seed,
        auto_reset=auto_reset, goal_distance=tuple(task["goal_distance_m"]),
        goal_angle=task["goal_angle_rad"], distance_cost=task["distance_cost"],
    )

def _calibrate_latent_norm(model, cfg, protocol: dict) -> float:
    import numpy as np
    from isaac_field.env import SingleFieldAdapter
    from isaac_field.policy import capture_actor_latent_pi
    spec = protocol["calibration"]
    engine = _open_env(cfg, protocol, spec["environment_seed"], 1, False)
    try:
        adapter = SingleFieldAdapter(engine)
        observation, _ = adapter.reset(seed=spec["environment_seed"])
        state, episode_start = None, np.ones(1, dtype=bool)
        with capture_actor_latent_pi(model.policy) as latent_rows:
            for _ in range(spec["steps"]):
                action, state = model.predict(
                    observation, state=state, episode_start=episode_start, deterministic=True
                )
                observation, _, terminated, truncated, _ = adapter.step(action)
                done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
                episode_start = np.asarray([done], dtype=bool)
                if done:
                    observation, _ = adapter.reset(seed=spec["environment_seed"])
        norms = []
        for latent in latent_rows:
            if hasattr(latent, "detach"):
                latent = latent.detach().cpu().numpy()
            array = np.asarray(latent, dtype=np.float64)
            norms.extend(np.square(array).sum(axis=-1).reshape(-1).tolist())
        if len(norms) != spec["steps"]:
            raise RuntimeError(f"expected {spec['steps']} policy latents, captured {len(norms)}")
        mean_sq = float(np.mean(norms))
        if not math.isfinite(mean_sq) or mean_sq <= 0.0:
            raise ValueError(f"invalid mean latent norm squared: {mean_sq}")
        return mean_sq
    finally:
        engine.close()

def _evaluate(checkpoint, cfg, protocol: dict, label: str, output: Path, device: str) -> dict:
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
    baseline = rows["baseline"]
    mismatches = []
    for seed in sorted(baseline):
        ref = baseline[seed]["reset_initial_condition"]
        for label, by_seed in rows.items():
            row = by_seed.get(seed)
            if row is None:
                mismatches.append({"seed": seed, "variant": label, "differences": ["missing_seed"]})
                continue
            initial = row["reset_initial_condition"]
            differences = []
            if initial["plant_xy_sha256"] != ref["plant_xy_sha256"]:
                differences.append("plant_xy_sha256")
            for key in ("start_pose_xy_yaw", "goal_xy"):
                if not np.allclose(initial[key], ref[key], rtol=0.0, atol=1e-8):
                    differences.append(key)
            if not row.get("trajectory_start_matches_reset", False):
                differences.append("trajectory_start_matches_reset")
            if differences:
                mismatches.append({"seed": seed, "variant": label, "differences": differences})
    return {"valid": not mismatches and all(len(rows[name]) == 24 for name in rows),
            "paired_seed_count": len(baseline), "mismatches": mismatches}

def launch(output: Path, *, device: str = "cuda:0") -> dict:
    protocol = preflight(output)
    output = output.resolve()
    _json(output / "run_status.json", {"status": "running", "phase": "loading_source"})
    try:
        from sb3_contrib import RecurrentPPO
        from isaac_field.task import Config
        from isaac_field.learning import train_stage
        from isaac_field.policy import make_exploration_variant
        from isaac_field.training import _sb3_wrapper
        from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment
        cfg = Config.from_checkpoint_config(protocol["config"])
        checkpoint = ROOT / protocol["source_checkpoint"]["path"]
        source = RecurrentPPO.load(str(checkpoint), device=device)
        baseline = _evaluate(checkpoint, cfg, protocol, "baseline", output, device)
        _json(output / "run_status.json", {"status": "running", "phase": "latent_calibration"})
        latent_norm_sq_mean = _calibrate_latent_norm(source, cfg, protocol)
        source_log_std = source.policy.log_std.detach().cpu().tolist()
        _json(output / "calibration.json", {
            "environment_seed": protocol["calibration"]["environment_seed"],
            "steps": protocol["calibration"]["steps"],
            "mean_latent_norm_sq": latent_norm_sq_mean,
            "source_gaussian_log_std": source_log_std,
            "matched_gsde_log_std_per_feature": [x - 0.5 * math.log(latent_norm_sq_mean) for x in source_log_std],
        })
        reports = {"baseline": baseline}
        training_results = {}
        for label in ("gaussian", "gsde"):
            spec = protocol["training"]
            variant_cfg = spec["variants"][label]
            _json(output / "run_status.json", {"status": "running", "phase": f"training_{label}"})
            env = _open_env(cfg, protocol, spec["environment_seed"], spec["num_envs"], True)
            try:
                wrapped = _sb3_wrapper(env)
                model = make_exploration_variant(
                    source, wrapped, use_sde=variant_cfg["use_sde"],
                    latent_norm_sq_mean=(latent_norm_sq_mean if variant_cfg["use_sde"] else None),
                    seed=spec["initialization_seed"], sde_sample_freq=variant_cfg["sde_sample_freq"],
                )
                hp = spec["hyperparameters"]
                callback = _make_progress_callback(spec["num_envs"], env, interval=2048)
                result = train_stage(
                    env, model, spec["timesteps_per_variant"], output / label / "field_policy_after",
                    callback=callback, seed=spec["environment_seed"], device=device,
                    goal_gain=hp["goal_gain"], n_steps=hp["n_steps"], batch_size=hp["batch_size"],
                    n_epochs=hp["n_epochs"], learning_rate=hp["learning_rate"], gamma=hp["gamma"],
                    target_kl=hp["target_kl"], verbose=0,
                )
            finally:
                env.close()
            training_results[label] = {
                "checkpoint": str(result.checkpoint),
                "requested_timesteps": result.requested_timesteps,
                "trained_timesteps": result.trained_timesteps,
                "total_timesteps": result.total_timesteps,
                "seed": result.seed,
                "source_checkpoint_timesteps": spec["source_checkpoint_timesteps"],
                "use_sde": variant_cfg["use_sde"],
                "sde_sample_freq": variant_cfg["sde_sample_freq"],
                "fresh_optimizer": True,
            }
            _json(output / "run_status.json", {"status": "running", "phase": f"evaluating_{label}"})
            reports[label] = _evaluate(result.checkpoint, cfg, protocol, label, output, device)
        pairing = _pair_initial_conditions(reports)
        if not pairing["valid"]:
            raise ValueError(f"held-out initial conditions do not pair: {pairing['mismatches'][:5]}")
        summary = {
            "schema": "isaac-field-exploration-pair-summary-v1",
            "protocol": str(output / "protocol.json"),
            "source_checkpoint": protocol["source_checkpoint"],
            "calibration": json.loads((output / "calibration.json").read_text(encoding="utf-8")),
            "training": training_results,
            "evaluation_summaries": {name: report["summary"] for name, report in reports.items()},
            "paired_initial_conditions": pairing,
            "assessment_vs_baseline": {
                label: _safe_goto_assessment(baseline, reports[label])
                for label in ("gaussian", "gsde")
            },
            "assessment_gsde_vs_gaussian": _safe_goto_assessment(reports["gaussian"], reports["gsde"]),
            "claim_limit": protocol["assessment"]["claim_limit"],
        }
        _json(output / "comparison_summary.json", summary)
        _json(output / "run_status.json", {"status": "completed", "pairing_valid": True,
                                            "summary": "comparison_summary.json"})
        return summary
    except Exception as exc:
        _json(output / "run_status.json", {"status": "failed", "exception": f"{type(exc).__name__}: {exc}"})
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.output, args.checkpoint)
    else:
        result = launch(args.output, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
