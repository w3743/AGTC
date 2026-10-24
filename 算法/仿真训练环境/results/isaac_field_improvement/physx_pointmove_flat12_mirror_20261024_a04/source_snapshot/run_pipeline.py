"""Resume the latest complete a03 checkpoint headlessly and audit paired results."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


TRIAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = TRIAL_DIR.parents[2]
REPO_ROOT = TRAINING_ROOT.parents[1]
A03_DIR = TRIAL_DIR.parent / "physx_pointmove_flat12_live_20261024_a03"
A03_PIPELINE = A03_DIR / "run_pipeline.py"
A03_RUN_DIR = A03_DIR / "logs/rsl_rl" / A03_DIR.name / "training_live_a03"
A02_DIR = TRIAL_DIR.parent / "physx_pointmove_flat12_20261024_a02"
A02_PIPELINE = A02_DIR / "run_pipeline.py"
HANDOFF_PATH = TRIAL_DIR / "handoff.json"
SOURCE_SNAPSHOT = TRIAL_DIR / "source_snapshot"
PROTOCOL_PATH = TRIAL_DIR / "protocol.json"
MANIFEST_PATH = TRIAL_DIR / "source_manifest.json"
STATUS_PATH = TRIAL_DIR / "run_status.json"
PREFLIGHT_PATH = TRIAL_DIR / "preflight.json"
LOCK_PATH = TRIAL_DIR / ".run_pipeline.lock"
LIVE_STATE_PATH = TRIAL_DIR / "live_state.json"
PYTHON_EXE = Path(r"C:\pioneer\isaac-training\.venv\Scripts\python.exe")
EXPERIMENT_NAME = "physx_pointmove_flat12_mirror_20261024_a04"
RUN_TIMESTAMP = "training_live_a04"
NUM_ENVS, ROLLOUT_STEPS, TARGET_TRANSITIONS = 4, 128, 196608
ADAM_STEPS_PER_ITERATION = 20
EVAL_SEED_START, EVAL_SCENARIOS, EVAL_GOAL_ANGLE_DEG = 87010, 16, 180.0
CORE_PROCESS_ROLES = ("training_native", "resume_training_cli", "controller", "watcher")


def _load_a03():
    spec = importlib.util.spec_from_file_location("isaac_field_a03_pipeline", A03_PIPELINE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reusable a03 helpers from {A03_PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    names = ("TRIAL_DIR", "TRAINING_ROOT", "REPO_ROOT", "HANDOFF_PATH", "SOURCE_SNAPSHOT",
             "PROTOCOL_PATH", "MANIFEST_PATH", "STATUS_PATH", "PREFLIGHT_PATH", "LOCK_PATH",
             "EXPERIMENT_NAME", "RUN_TIMESTAMP", "TARGET_TRANSITIONS", "NUM_ENVS", "ROLLOUT_STEPS",
             "EVAL_SEED_START", "EVAL_SCENARIOS", "EVAL_GOAL_ANGLE_DEG")
    for name in names:
        setattr(module, name, globals()[name])
    return module


A03 = _load_a03()
utc_now, sha256_file = A03.utc_now, A03.sha256_file
read_json, write_json_atomic = A03.read_json, A03.write_json_atomic


def _helper():
    return A03._old_pipeline()


def resume_plan(iteration: int, transitions: int, adam_step: int, optimizer_entries: int) -> dict[str, int]:
    batch = NUM_ENVS * ROLLOUT_STEPS
    if type(iteration) is not int or iteration < 0 or type(adam_step) is not int or adam_step < 0:
        raise ValueError("checkpoint iteration and Adam step must be non-negative integers")
    if type(transitions) is not int or transitions < 0 or transitions >= TARGET_TRANSITIONS or transitions % batch:
        raise ValueError(f"checkpoint transitions must be an aligned value below {TARGET_TRANSITIONS}")
    if type(optimizer_entries) is not int or optimizer_entries <= 0:
        raise ValueError("resumable checkpoint must contain optimizer parameter states")
    remaining = (TARGET_TRANSITIONS - transitions) // batch
    if remaining <= 0:
        raise ValueError("checkpoint already reached the target")
    return {
        "remaining_iterations": remaining,
        "expected_num_timesteps": TARGET_TRANSITIONS,
        "expected_final_payload_iter": iteration + remaining - 1,
        "expected_adam_step": adam_step + remaining * ADAM_STEPS_PER_ITERATION,
    }


def training_argv(checkpoint: Path, plan: dict[str, int]) -> list[str]:
    return [
        str(PYTHON_EXE), "-m", "isaac_field", "resume", "--checkpoint", str(checkpoint.resolve()),
        "--num_envs", str(NUM_ENVS), "--max_iterations", str(plan["remaining_iterations"]),
        "--device", "cuda:0", "--seed", "69041", "--logger", "tensorboard",
        "--experiment_name", EXPERIMENT_NAME, "--run_timestamp", RUN_TIMESTAMP,
        "--resume_optimizer_state", "--viz", "none", "--live-state-file", str(LIVE_STATE_PATH.resolve()),
        "env.task.size=12.0", "env.scene.env_spacing=22.0",
    ]


def _process_roles(handoff: dict[str, Any]) -> tuple[str, ...]:
    records = handoff.get("old_processes", {})
    return (*CORE_PROCESS_ROLES, "training_gui" if "training_gui" in records else "static_gui")


def _require_stopped_handoff() -> tuple[dict[str, Any], list[dict[str, str]]]:
    A03.PROCESS_ROLES = _process_roles(read_json(HANDOFF_PATH))
    handoff = A03._require_stopped_handoff(HANDOFF_PATH)
    if handoff.get("source_trial") != A03_DIR.name:
        raise ValueError("handoff must identify a03 as the source trial")
    if handoff.get("source_protocol_sha256") != sha256_file(A03_DIR / "protocol.json"):
        raise ValueError("handoff source protocol SHA differs from frozen a03 protocol")
    return handoff, A03._verify_process_evidence(handoff)


def _snapshot_dependencies(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest["files"]
    for root, names in ((A03_DIR, ("run_pipeline.py", "protocol.json", "source_manifest.json", "baseline_validation.json")),
                        (A02_DIR, ("run_pipeline.py", "protocol.json", "baseline_validation.json"))):
        for name in names:
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(f"required parent/helper evidence is missing: {source}")
            snapshot_rel = Path("source_snapshot/inherited") / root.name / name
            destination = TRIAL_DIR / snapshot_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            rows.append({"path": source.relative_to(TRAINING_ROOT).as_posix(), "origin": "project",
                         "snapshot_path": snapshot_rel.as_posix(), "sha256": sha256_file(source)})
    rows.sort(key=lambda row: (row["origin"], row["path"]))
    manifest["builder"] += "; a03 protocol/source manifest/baseline/wrapper and pinned a02 helper inputs"
    return manifest


def _validate_a03_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = read_json(A03_DIR / "protocol.json")
    manifest = read_json(A03_DIR / "source_manifest.json")
    baseline = read_json(A03_DIR / "baseline_validation.json")
    if manifest != protocol.get("source_snapshot"):
        raise RuntimeError("a03 source_manifest.json differs from its frozen protocol manifest")
    bad = _snapshot_mismatches(A03_DIR, manifest)
    if bad:
        raise RuntimeError(f"a03 frozen source snapshot mismatch: {bad[:5]}")
    old_status = read_json(A03_DIR / "run_status.json")
    if (old_status.get("protocol_sha256") != sha256_file(A03_DIR / "protocol.json")
            or old_status.get("status") != "superseded"
        or old_status.get("superseded_by") not in (TRIAL_DIR.name, str(TRIAL_DIR.resolve()))):
        raise RuntimeError("a03 status is not an unchanged superseded trial for a04")
    old_helper = A03_DIR / "source_snapshot/a02/run_pipeline.py"
    if not old_helper.is_file() or sha256_file(old_helper) != sha256_file(A02_PIPELINE):
        raise RuntimeError("live a02 helper differs from the helper pinned by a03")
    if baseline.get("status") != "passed":
        raise RuntimeError("reused a02 zero-step baseline audit is not passed")
    baseline_path = Path(protocol["training"]["baseline"]["checkpoint"])
    if not baseline_path.is_file() or sha256_file(baseline_path) != protocol["training"]["baseline"]["checkpoint_sha256"]:
        raise RuntimeError("reused a02 baseline checkpoint SHA changed")
    return protocol, manifest, baseline


def _snapshot_mismatches(trial: Path, manifest: dict[str, Any]) -> list[str]:
    return [row.get("path", "<unknown>") for row in manifest.get("files", [])
            if sha256_file(trial / row["snapshot_path"]) != row["sha256"]]


def prepare(checkpoint_argument: str | Path) -> dict[str, Any]:
    if PROTOCOL_PATH.exists() or MANIFEST_PATH.exists() or SOURCE_SNAPSHOT.exists():
        raise FileExistsError(f"refusing to freeze over existing a04 data: {TRIAL_DIR}")
    handoff, evidence = _require_stopped_handoff()
    helper = _helper()
    protocol_a03, _, baseline = _validate_a03_inputs()
    checkpoint, payload, safe = A03.select_latest_complete_checkpoint(A03_RUN_DIR)
    if Path(checkpoint_argument).resolve() != checkpoint.resolve():
        raise RuntimeError(f"--checkpoint must select the latest complete a03 checkpoint: {checkpoint}")
    source = handoff.get("checkpoint", {})
    if (Path(source.get("path", "")).resolve() != checkpoint.resolve()
            or source.get("sha256") != sha256_file(checkpoint)
            or source.get("iter") != payload["iter"] or source.get("num_timesteps") != payload["num_timesteps"]
            or source.get("adam_step") != safe["adam_step"]):
        raise RuntimeError("handoff checkpoint identity/counts differ from latest complete a03 checkpoint")
    plan = resume_plan(payload["iter"], payload["num_timesteps"], safe["adam_step"], safe["optimizer_state_entries"])
    manifest = _snapshot_dependencies(helper.build_source_manifest())
    requirements = helper.expected_contract_requirements()
    protocol = copy.deepcopy(protocol_a03)
    protocol.update({
        "schema": "isaac-field-mirror-resume-protocol-v1", "status": "prepared",
        "created_at_utc": utc_now(), "trial_id": TRIAL_DIR.name,
        "objective": "resume the latest complete a03 checkpoint to 196608 cumulative transitions, then paired a02-baseline development evaluation",
        "git_identity": helper.repository_identity(), "runtime_versions": helper.installed_versions(),
        "source_snapshot": manifest,
        "source_parent": {"trial": A03_DIR.name, "protocol_sha256": sha256_file(A03_DIR / "protocol.json"),
                          "source_manifest_sha256": sha256_file(A03_DIR / "source_manifest.json"),
                          "baseline_validation_sha256": sha256_file(A03_DIR / "baseline_validation.json")},
        "source_handoff": {"path": str(HANDOFF_PATH.resolve()), "sha256": sha256_file(HANDOFF_PATH)},
        "stopped_process_evidence": evidence,
        "expected_checkpoint_contract_requirements": requirements,
        "live_observer": {
            "state_file": str(LIVE_STATE_PATH.resolve()), "expected_run_id": TRIAL_DIR.name,
            "reader_argv": [str(PYTHON_EXE), str(TRAINING_ROOT / "scripts/view_training_state.py"), "--state-file",
                            str(LIVE_STATE_PATH.resolve()), "--output", str((TRIAL_DIR / "live_viewer").resolve()),
                            "--expected-run-id", TRIAL_DIR.name],
            "viewer_started_by_pipeline": False, "simulator_or_policy_instance": False,
        },
        "completion_watcher": {"script": "scripts/watch_training_completion.py", "interval_seconds": 30,
                               "delivery_deduplicated": True, "started_by_pipeline": False},
        "physx_scene_query_support": {"enabled": False, "reason": "headless CLI matches a02; scene-query manager only, not a precision claim"},
    })
    train = protocol["training"]
    train.update({
        "max_iterations": plan["remaining_iterations"], "target_transitions": TARGET_TRANSITIONS,
        "remaining_iterations": plan["remaining_iterations"],
        "transition_arithmetic": f"{payload['num_timesteps']} + {plan['remaining_iterations']} * {NUM_ENVS} * {ROLLOUT_STEPS} = {TARGET_TRANSITIONS}",
        "optimizer_resume": True, "headless": {"selected": True, "visualizer": "none", "reason": "keep training UI work in an independent live-state viewer"},
    })
    train.pop("gui", None)
    train["resume"] = {
        "argv_template": training_argv(checkpoint, plan),
        "run_dir": str(TRIAL_DIR / "logs/rsl_rl" / EXPERIMENT_NAME / RUN_TIMESTAMP),
        "checkpoint_selection": "latest stable, CPU-loadable numeric a03 checkpoint, verified against stopped-process handoff",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iter": payload["iter"], "checkpoint_num_timesteps": payload["num_timesteps"],
        "checkpoint_adam_step": safe["adam_step"], "checkpoint_optimizer_state_entries": safe["optimizer_state_entries"],
        **plan, "expected_iteration_filename_not_hardcoded": True, "resume_optimizer_state": True,
    }
    train["baseline"].update({"reused_from_trial": A02_DIR.name, "argv": [], "retrained": False})
    protocol["evaluation"]["outputs"] = {"baseline": str(TRIAL_DIR / "eval_baseline"), "final": str(TRIAL_DIR / "eval_final")}
    protocol["evaluation"].update({"seed_start": EVAL_SEED_START, "scenario_count": EVAL_SCENARIOS,
                                   "scenario_seeds": list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SCENARIOS))})
    protocol["export"]["output"] = str(TRIAL_DIR / "final_policy_torchscript.pt")
    protocol["output_policy"] = "The zero-step a02 baseline remains reused; a04 checkpoints, live state and reports are rooted in this trial."
    protocol.setdefault("subprocess", {})["cwd"] = str(TRIAL_DIR)
    write_json_atomic(MANIFEST_PATH, manifest)
    write_json_atomic(PROTOCOL_PATH, protocol)
    status = {"schema": "isaac-field-one-shot-run-status-v1", "status": "prepared", "phase": "prepared",
              "created_at_utc": utc_now(), "updated_at_utc": utc_now(), "protocol_sha256": sha256_file(PROTOCOL_PATH),
              "parent_pid": os.getpid(), "children": [], "current_stage": None, "source_trial": A03_DIR.name,
              "resume_checkpoint": train["resume"], "export": {"status": "pending_optional"}}
    write_json_atomic(STATUS_PATH, status)
    baseline_audit = helper.audit_baseline(protocol, status)
    status = read_json(STATUS_PATH)
    status.update({"status": "prepared", "phase": "prepared", "current_stage": None,
                   "baseline_checkpoint": {"path": baseline_audit["checkpoint"], "sha256": baseline_audit["checkpoint_sha256"],
                                           "audit": "passed", "reused": True}})
    helper._stage_status(status, status)
    result = {"schema": "isaac-field-mirror-resume-prepare-v1", "status": "prepared", "trial_dir": str(TRIAL_DIR),
              "protocol_sha256": sha256_file(PROTOCOL_PATH), "source_file_count": len(manifest["files"]),
              "resume_checkpoint": train["resume"], "native_processes_started": False, "training_started": False,
              "viewer_started": False, "kit_imported": False}
    write_json_atomic(TRIAL_DIR / "prepare.json", result)
    return result


def preflight(*, write_report: bool = True) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not STATUS_PATH.is_file():
        raise FileNotFoundError("run --prepare --checkpoint <latest-complete-a03-checkpoint> first")
    helper = _helper()
    handoff, evidence = _require_stopped_handoff()
    protocol_sha = sha256_file(PROTOCOL_PATH)
    status, protocol = read_json(STATUS_PATH), read_json(PROTOCOL_PATH)
    if status.get("status") != "prepared" or status.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("a04 is not an unchanged prepared run")
    if protocol.get("source_handoff", {}).get("sha256") != sha256_file(HANDOFF_PATH):
        raise RuntimeError("handoff changed after prepare")
    if protocol.get("stopped_process_evidence") != evidence:
        raise RuntimeError("stopped-process evidence changed after prepare")
    if read_json(MANIFEST_PATH) != protocol.get("source_snapshot"):
        raise RuntimeError("source_manifest.json differs from protocol source snapshot")
    source = helper._verify_manifest(protocol["source_snapshot"])
    if not source["valid"]:
        raise RuntimeError(f"a04 source snapshot mismatch: {source['failures'][:5]}")
    if helper.repository_identity() != protocol["git_identity"] or helper.installed_versions() != protocol["runtime_versions"]:
        raise RuntimeError("repository/Isaac Lab identity or runtime versions changed after prepare")
    _validate_a03_inputs()
    resume = protocol["training"]["resume"]
    checkpoint = Path(resume["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != resume["checkpoint_sha256"]:
        raise RuntimeError("selected source checkpoint is missing or changed")
    latest, payload, safe = A03.select_latest_complete_checkpoint(A03_RUN_DIR)
    if latest.resolve() != checkpoint.resolve():
        raise RuntimeError("a newer complete a03 checkpoint appeared after prepare; prepare from it")
    if (payload["iter"] != resume["checkpoint_iter"] or payload["num_timesteps"] != resume["checkpoint_num_timesteps"]
            or safe["adam_step"] != resume["checkpoint_adam_step"]
            or safe["optimizer_state_entries"] != resume["checkpoint_optimizer_state_entries"]):
        raise RuntimeError("source checkpoint counters changed after prepare")
    baseline = protocol["training"]["baseline"]
    if sha256_file(Path(baseline["checkpoint"])) != baseline["checkpoint_sha256"]:
        raise RuntimeError("reused a02 baseline checkpoint changed")
    planned = [Path(resume["run_dir"]), TRIAL_DIR / "eval_baseline", TRIAL_DIR / "eval_final", LIVE_STATE_PATH]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"a04 output paths already exist: {existing}")
    report = {"schema": "isaac-field-mirror-resume-preflight-v1", "status": "passed", "checked_at_utc": utc_now(),
              "protocol_sha256": protocol_sha, "source_manifest": source, "a03_status": "superseded",
              "old_processes_stopped": True, "selected_checkpoint": resume, "remaining_iterations": resume["remaining_iterations"],
              "target_transitions": TARGET_TRANSITIONS, "training_argv": resume["argv_template"],
              "visualizer": "none", "scene_query_support": False, "planned_outputs_absent": True,
              "native_processes_started": False, "viewer_started": False, "training_started": False,
              "parent_source_manifest_sha256": protocol["source_parent"]["source_manifest_sha256"]}
    if write_report:
        write_json_atomic(PREFLIGHT_PATH, report)
    return report


def _headless_child_environment(environment: dict[str, str], *, training: bool) -> dict[str, str]:
    result = dict(environment)
    result.update({"HEADLESS": "1", "LIVESTREAM": "0", "XR": "0"})
    if training:
        result["ISAAC_FIELD_LIVE_STATE_RUN_ID"] = TRIAL_DIR.name
    else:
        result.pop("ISAAC_FIELD_LIVE_STATE_RUN_ID", None)
    return result


def execute_pipeline(*, export: bool = False) -> dict[str, Any]:
    # Keep a03's proven lock, sleep-prevention, process evidence, and ordered stages.
    A03.preflight = preflight
    A03.apply_training_gui_environment = _headless_child_environment
    return A03.execute_pipeline(export=export)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="freeze a04 from a stopped a03 handoff; CPU only")
    actions.add_argument("--preflight", action="store_true", help="verify the frozen source/checkpoint; CPU only")
    actions.add_argument("--dry-run", action="store_true", help="print the launch/evaluation plan; CPU only")
    parser.add_argument("--checkpoint", help="latest complete a03 checkpoint required by --prepare")
    parser.add_argument("--export", action="store_true", help="also attempt optional TorchScript export")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if not args.checkpoint:
                parser.error("--prepare requires --checkpoint")
            result = prepare(args.checkpoint)
        elif args.preflight or args.dry_run:
            result = preflight()
            if args.dry_run:
                result["ordered_stages"] = ["resume_training_cli --viz none", "audit final transitions/Adam", "evaluate_baseline 16 seeds", "evaluate_final same 16 seeds", "paired comparison"]
        else:
            result = execute_pipeline(export=args.export)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"a04 pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
