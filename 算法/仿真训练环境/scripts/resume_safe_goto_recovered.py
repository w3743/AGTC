"""One-off recovery continuation for the failed safe-goto run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/isaac_field_improvement/safe_goto_20261017_reward_v1_resume1"
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checkpoint_steps(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read("data"))["num_timesteps"])

def preflight() -> dict:
    protocol = read_json(OUT / "protocol.json")
    for source in protocol["source_checkpoints"].values():
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"] or checkpoint_steps(path) != source["num_timesteps"]:
            raise ValueError(f"source checkpoint changed: {path}")
    parent_protocol = ROOT / protocol["source_protocol"]["path"]
    parent_status = ROOT / "results/isaac_field_improvement/safe_goto_20261017_reward_v1/run_status.json"
    if sha256(parent_protocol) != protocol["source_protocol"]["sha256"]:
        raise ValueError("parent protocol hash changed")
    if sha256(parent_status) != protocol["parent_run"]["run_status_sha256"]:
        raise ValueError("parent run status hash changed")
    return protocol

def snapshot_hashes(protocol: dict) -> dict:
    """Verify the root's prelaunch source snapshot with its ordinary file hashes."""
    snapshot = OUT / "source_snapshot"
    manifest = read_json(snapshot / "source_manifest.json")
    records = {row["source_path"]: row for row in manifest["files"]}
    hashes = {}
    for rel in protocol["sampling_recovery"]["required_source_files"]:
        row = records[rel]
        live = ROOT / rel
        copied = snapshot / row["snapshot_path"]
        digest = sha256(live)
        if digest != row["sha256"] or sha256(copied) != digest:
            raise ValueError(f"snapshot/source hash mismatch: {rel}")
        hashes[rel] = digest
    return {"manifest_sha256": sha256(snapshot / "source_manifest.json"), "files_sha256": hashes}

def make_cfg(protocol: dict):
    from isaac_field.task import Config

    return Config.from_checkpoint_config(protocol["config"])

def add_reset_observer(adapter) -> dict:
    """Capture actual generated plant positions and final start/goal after each reset."""
    import numpy as np

    reset = adapter.reset
    observed = {}

    def reset_and_record(*, seed=None, options=None):
        observation, info = reset(seed=seed, options=options)
        task = adapter.task
        xy = np.ascontiguousarray(task.scene.xy, dtype="<f4")
        digest = hashlib.sha256(np.asarray(xy.shape, dtype="<i8").tobytes() + xy.tobytes()).hexdigest()
        observed[int(seed)] = {
            "seed": int(seed),
            "start_pose_xy_yaw": np.asarray(task.robot[:3], dtype=np.float64).tolist(),
            "goal_xy": np.asarray(task.goal[:2], dtype=np.float64).tolist(),
            "plant_count": int(len(xy)),
            "plant_xy_sha256": digest,
        }
        return observation, info

    adapter.reset = reset_and_record
    return observed

def evaluate(checkpoint: Path, protocol: dict) -> dict:
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.learning import evaluate_model
    from scripts.improve_isaac_policy import _add_safety_summary

    spec = protocol["evaluation"]
    cfg = make_cfg(protocol)
    engine = make_env(
        cfg=cfg, mode="goto", num_envs=1, seed=spec["environment_seed"],
        auto_reset=False, goal_distance=tuple(spec["goal_distance_m"]),
        goal_angle=spec["goal_angle_rad"], distance_cost=spec["distance_cost"],
    )
    try:
        adapter = SingleFieldAdapter(engine)
        resets = add_reset_observer(adapter)
        report = evaluate_model(
            adapter, checkpoint, spec["heldout_seeds"],
            goal_distance=tuple(spec["goal_distance_m"]), goal_angle=spec["goal_angle_rad"],
            mode="goto", deterministic=spec["deterministic"],
            device=protocol["training"]["hyperparameters"]["device"],
            record_trajectories=spec["record_trajectories"],
        )
    finally:
        engine.close()
    import numpy as np

    for row in report["rows"]:
        initial = resets[row["seed"]]
        row["reset_initial_condition"] = initial
        trace_start = row["trajectory"][0]["before"]["pose_xy_yaw"]
        row["trajectory_start_matches_reset"] = bool(np.allclose(
            trace_start, initial["start_pose_xy_yaw"], rtol=0, atol=1e-8
        ))
    return _add_safety_summary(report)

def compare_initial_conditions(before: dict, after: dict) -> dict:
    import numpy as np

    left = {row["seed"]: row for row in before["rows"]}
    right = {row["seed"]: row for row in after["rows"]}
    mismatches, hashes_complete = [], True
    for seed in sorted(set(left) | set(right)):
        a = left.get(seed, {}).get("reset_initial_condition", {})
        b = right.get(seed, {}).get("reset_initial_condition", {})
        diff = []
        for key in ("start_pose_xy_yaw", "goal_xy"):
            if key not in a or key not in b or not np.allclose(a[key], b[key], rtol=0, atol=1e-8):
                diff.append(key)
        ah, bh = a.get("plant_xy_sha256"), b.get("plant_xy_sha256")
        if ah is None or bh is None:
            hashes_complete = False
        elif ah != bh:
            diff.append("plant_xy_sha256")
        for label, row in (("before", left.get(seed)), ("after", right.get(seed))):
            if row is None or not row.get("trajectory_start_matches_reset"):
                diff.append(f"{label}_trajectory_start")
        if diff:
            mismatches.append({"seed": seed, "differences": diff})
    return {
        "valid": len(mismatches) == 0 and len(left) == len(right) == 24,
        "paired_seed_count": len(set(left) & set(right)),
        "plant_xy_hashes_complete": hashes_complete,
        "mismatches": mismatches,
    }

def run(protocol: dict) -> dict:
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from isaac_field.env import make_env
    from isaac_field.learning import train_stage
    from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment

    if (OUT / "run_status.json").exists():
        raise FileExistsError("recovery run already started; refusing to overwrite its outputs")
    source_record = snapshot_hashes(protocol)
    write_json(OUT / "launch_source_hashes.json", source_record)
    write_json(OUT / "run_status.json", {"status": "running", "source_snapshot": source_record})

    spec, stage = protocol["evaluation"], protocol["training"]["stages"][0]
    original = ROOT / protocol["source_checkpoints"]["original_53248"]["path"]
    before = evaluate(original, protocol)
    write_json(OUT / "before_recovered.json", before)

    h = protocol["training"]["hyperparameters"]
    cfg = make_cfg(protocol)
    env = make_env(
        cfg=cfg, mode="goto", num_envs=h["num_envs"], seed=stage["seed"], auto_reset=True,
        goal_distance=tuple(stage["goal_distance_m"]), goal_angle=stage["goal_angle_rad"],
        distance_cost=stage["distance_cost"],
    )
    stage_dir = OUT / "stage_01_medium"
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    progress = _make_progress_callback(h["num_envs"], env, h["progress_aggregate_interval"])
    checkpoint_cb = CheckpointCallback(
        save_freq=h["checkpoint_callback_save_freq_calls"], save_path=str(checkpoint_dir),
        name_prefix="field_policy_medium", verbose=0,
    )
    final_path = OUT / "field_policy_after_resume1.zip"
    try:
        result = train_stage(
            env, ROOT / protocol["source_checkpoints"]["resume_110592"]["path"],
            stage["timesteps"], final_path, callback=CallbackList([checkpoint_cb, progress]),
            seed=stage["seed"], device=h["device"], goal_gain=h["goal_gain"],
            n_steps=h["n_steps"], batch_size=h["batch_size"], n_epochs=h["n_epochs"],
            learning_rate=h["learning_rate"], gamma=h["gamma"], target_kl=h["target_kl"],
            verbose=h["verbose"],
        )
    finally:
        env.close()
    if result.trained_timesteps != stage["timesteps"] or result.total_timesteps != 118_784:
        raise RuntimeError(f"unexpected counters: trained={result.trained_timesteps}, total={result.total_timesteps}")
    if checkpoint_steps(result.checkpoint) != 118_784:
        raise RuntimeError("final checkpoint num_timesteps is not 118784")

    write_json(OUT / "stages_report.json", {"stages": [{
        "name": stage["name"], "seed": stage["seed"], "checkpoint_in": str(ROOT / protocol["source_checkpoints"]["resume_110592"]["path"]),
        "checkpoint_out": str(result.checkpoint), "trained_timesteps": result.trained_timesteps,
        "total_timesteps": result.total_timesteps, "hyperparameters": h,
        "aggregate_progress": progress.log_rows, "periodic_checkpoints": sorted(map(str, checkpoint_dir.glob("*.zip"))),
    }]})
    after = evaluate(result.checkpoint, protocol)
    write_json(OUT / "after_recovered.json", after)
    pairing = compare_initial_conditions(before, after)
    assessment = _safe_goto_assessment(before, after) if pairing["valid"] else "invalid_initial_conditions_differ"
    before_rows = {row["seed"]: row for row in before["rows"]}
    after_rows = {row["seed"]: row for row in after["rows"]}
    pairs = [{"seed": seed,
              "before": {k: v for k, v in before_rows[seed].items() if k != "trajectory"},
              "after": {k: v for k, v in after_rows[seed].items() if k != "trajectory"}}
             for seed in spec["heldout_seeds"]]
    summary = {
        "schema": "isaac-field-safe-goto-recovered-comparison-v1",
        "protocol": str(OUT / "protocol.json"), "checkpoint_before": str(original),
        "checkpoint_after": str(result.checkpoint), "before_report": str(OUT / "before_recovered.json"),
        "after_report": str(OUT / "after_recovered.json"), "stages_report": str(OUT / "stages_report.json"),
        "heldout_before_summary": before["summary"], "heldout_after_summary": after["summary"],
        "paired_initial_conditions": pairing, "assessment": assessment,
        "heldout_before_after_rows": pairs, "claim_limit": spec["claim_limit"],
    }
    write_json(OUT / "comparison_summary.json", summary)
    write_json(OUT / "run_status.json", {"status": "completed", "assessment": assessment,
                                         "pairing_valid": pairing["valid"], "final_checkpoint": str(result.checkpoint)})
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    protocol = preflight()
    if args.preflight_only:
        print(json.dumps({"status": protocol["status"], "checkpoint_steps": {
            key: value["num_timesteps"] for key, value in protocol["source_checkpoints"].items()
        }}, indent=2))
        return
    try:
        print(json.dumps(run(protocol), ensure_ascii=False, indent=2))
    except Exception as exc:
        if (OUT / "run_status.json").exists():
            write_json(OUT / "run_status.json", {"status": "failed", "exception": f"{type(exc).__name__}: {exc}"})
        raise
if __name__ == "__main__":
    main()
