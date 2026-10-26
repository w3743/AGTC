#!/usr/bin/env python
"""CPU-only gate for the first resumed checkpoint in the recovered a05 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent
FIELD = HERE.parents[2]
A05_DEFAULT = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_recover_20261024_a05"
A04 = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_mirror_20261024_a04"
A04_ENV = (A04 / "logs/rsl_rl/physx_pointmove_flat12_mirror_20261024_a04"
           / "training_live_a04/params/env.yaml")
A04_RUN_DIR = (A04 / "logs/rsl_rl/physx_pointmove_flat12_mirror_20261024_a04"
               / "training_live_a04")
BASE_VERIFIER = (FIELD / "results/isaac_field_diagnosis/sampling_live_handoff_20261024"
                 / "verify_resumed_training.py")
A04_ADAPTER = (FIELD / "results/isaac_field_diagnosis/live_state_mirror_handoff_20261024"
               / "verify_first_checkpoint.py")
A04_PIPELINE = A04 / "run_pipeline.py"
EXPECTED_A05_NAME = "physx_pointmove_flat12_recover_20261024_a05"
EXPECTED_A04_SHA256 = "26c55ba2c1fc9e3da09ed174ec01b20b08c78186ee390657b400dde6152a9025"
EXPECTED_A04_STEPS, EXPECTED_A04_ADAM, EXPECTED_A04_ITER = 26112, 1020, 48
ROLLOUT_TRANSITIONS, ADAM_PER_ROLLOUT = 512, 20
TARGET_TRANSITIONS, TARGET_ADAM, TARGET_FINAL_ITER = 196608, 7680, 380
EXPECTED_FIRST_STEPS, EXPECTED_FIRST_ADAM = 26624, 1040
OUTPUT_NAME = "first_resumed_checkpoint_cpu.json"


class Pending(RuntimeError):
    """The a05 wrapper has not written enough evidence to audit yet."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_source_handoff(trial: Path) -> dict:
    handoff_path = trial / "handoff.json"
    if not handoff_path.is_file():
        raise Pending(f"waiting for {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
    checkpoint = handoff.get("checkpoint", {})
    expected_path = (A04_RUN_DIR / "model_48.pt").resolve()
    if Path(checkpoint.get("path", "")).resolve() != expected_path:
        raise ValueError(f"a05 handoff must use the authorized a04 model_48: {expected_path}")
    if (checkpoint.get("sha256", "").lower() != EXPECTED_A04_SHA256
            or int(checkpoint.get("iter", -1)) != EXPECTED_A04_ITER
            or int(checkpoint.get("num_timesteps", -1)) != EXPECTED_A04_STEPS
            or int(checkpoint.get("adam_step", -1)) != EXPECTED_A04_ADAM):
        raise ValueError("a05 handoff metadata does not match the verified a04 model_48")
    return handoff


def check_wrapper_preflight(trial: Path, wrapper, optimizer_entries: int) -> dict:
    path = trial / "preflight.json"
    if not path.is_file():
        raise Pending(f"waiting for {path}")
    preflight = json.loads(path.read_text(encoding="utf-8-sig"))
    protocol_path = trial / "protocol.json"
    if not protocol_path.is_file():
        raise Pending(f"waiting for {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    if preflight.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("a05 preflight protocol SHA does not match the saved protocol")
    selected = preflight.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise ValueError("a05 preflight must include selected_checkpoint")

    expected = {
        "checkpoint": str((A04_RUN_DIR / "model_48.pt").resolve()),
        "checkpoint_sha256": EXPECTED_A04_SHA256,
        "checkpoint_iter": EXPECTED_A04_ITER,
        "checkpoint_num_timesteps": EXPECTED_A04_STEPS,
        "checkpoint_adam_step": EXPECTED_A04_ADAM,
        "checkpoint_optimizer_state_entries": optimizer_entries,
        "remaining_iterations": (TARGET_TRANSITIONS - EXPECTED_A04_STEPS) // ROLLOUT_TRANSITIONS,
        "expected_num_timesteps": TARGET_TRANSITIONS,
        "expected_final_payload_iter": TARGET_FINAL_ITER,
        "expected_adam_step": TARGET_ADAM,
        "resume_optimizer_state": True,
    }
    for key, value in expected.items():
        actual = selected.get(key)
        if key == "checkpoint" and actual is not None:
            actual = str(Path(actual).resolve())
        if actual != value:
            raise ValueError(f"a05 preflight {key}={actual!r}, expected {value!r}")

    argv = selected.get("argv_template")
    if not isinstance(argv, list):
        raise ValueError("a05 preflight must retain selected_checkpoint.argv_template")
    required_args = {
        "--max_iterations": "333",
        "--device": "cuda:0",
        "--run_timestamp": "training_recover_a05",
        "--experiment_name": EXPECTED_A05_NAME,
        "--viz": "none",
    }
    for flag, value in required_args.items():
        if flag not in argv or argv[argv.index(flag) + 1] != value:
            raise ValueError(f"a05 argv must contain {flag} {value}")
    if "--resume_optimizer_state" not in argv:
        raise ValueError("a05 argv must preserve Adam optimizer state")
    if preflight.get("target_transitions") != TARGET_TRANSITIONS:
        raise ValueError("a05 preflight target_transitions must remain 196608")
    if preflight.get("remaining_iterations") != 333:
        raise ValueError("a05 preflight remaining_iterations must remain 333")
    if preflight.get("visualizer") != "none" or preflight.get("headless_environment") != {
        "HEADLESS": "1", "LIVESTREAM": "0", "XR": "0",
    }:
        raise ValueError("a05 training must remain headless with scene query support disabled")
    training = protocol.get("training", {})
    resume = training.get("resume", {})
    if (protocol.get("physx_scene_query_support", {}).get("enabled") is not False
            or training.get("max_iterations") != 333
            or training.get("target_transitions") != TARGET_TRANSITIONS
            or training.get("remaining_iterations") != 333
            or training.get("optimizer_resume") is not True
            or training.get("headless", {}).get("selected") is not True
            or training.get("headless", {}).get("visualizer") != "none"
            or resume.get("expected_num_timesteps") != TARGET_TRANSITIONS
            or resume.get("expected_final_payload_iter") != TARGET_FINAL_ITER
            or resume.get("expected_adam_step") != TARGET_ADAM
            or resume.get("checkpoint_num_timesteps") != EXPECTED_A04_STEPS
            or resume.get("checkpoint_adam_step") != EXPECTED_A04_ADAM
            or resume.get("resume_optimizer_state") is not True):
        raise ValueError("a05 protocol weakens the headless, resume, or completion gate")
    observer = protocol.get("live_observer", {})
    reader_argv = observer.get("reader_argv", [])
    viewer_source = FIELD / "scripts/view_training_state_web.py"
    if (observer.get("simulator_or_policy_instance") is not False
            or observer.get("viewer_started_by_pipeline") is not False
            or not any(str(value).endswith("view_training_state_web.py") for value in reader_argv)):
        raise ValueError("a05 web observer must read the live state without a simulator or policy instance")
    if not viewer_source.is_file():
        raise Pending(f"waiting for the lightweight web observer source: {viewer_source}")
    if observer.get("source_sha256") != sha256(viewer_source):
        raise ValueError("a05 protocol does not pin the current lightweight web observer source")

    planner = getattr(wrapper, "resume_plan", None)
    if not callable(planner):
        raise ValueError("a05 run_pipeline.py must expose resume_plan for the CPU gate")
    plan = planner(EXPECTED_A04_ITER, EXPECTED_A04_STEPS, EXPECTED_A04_ADAM, optimizer_entries)
    for key, value in {
        "remaining_iterations": 333,
        "expected_num_timesteps": TARGET_TRANSITIONS,
        "expected_final_payload_iter": TARGET_FINAL_ITER,
        "expected_adam_step": TARGET_ADAM,
    }.items():
        if plan.get(key) != value:
            raise ValueError(f"a05 resume_plan {key}={plan.get(key)!r}, expected {value!r}")
    return {"selected_checkpoint": selected, "resume_plan": plan,
            "training_argv": argv, "status": preflight.get("status"),
            "protocol_sha256": preflight["protocol_sha256"],
            "scene_query_support_enabled": protocol["physx_scene_query_support"]["enabled"],
            "live_observer": observer, "viewer_source": str(viewer_source),
            "viewer_source_sha256": observer["source_sha256"]}


def verify(trial: Path, torch) -> dict:
    if trial.name != EXPECTED_A05_NAME:
        raise ValueError(f"refusing to audit/write a non-a05 directory: {trial}")
    if not trial.is_dir():
        raise Pending(f"waiting for a05 trial directory: {trial}")
    existing_report = trial / OUTPUT_NAME
    if existing_report.is_file():
        try:
            saved = json.loads(existing_report.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        if saved.get("status") == "passed":
            return {"schema": "agtc-a05-recovery-first-checkpoint-cpu-v1",
                    "status": "already_passed", "report": str(existing_report),
                    "resumed_checkpoint": saved.get("resumed_checkpoint")}

    check_source_handoff(trial)
    a04_pipeline = load_module(A04_PIPELINE, "a05_recovery_a04_pipeline_cpu_helpers")
    selected_path, _, safe = a04_pipeline.A03.select_latest_complete_checkpoint(A04_RUN_DIR)
    expected_path = (A04_RUN_DIR / "model_48.pt").resolve()
    if selected_path.resolve() != expected_path:
        raise ValueError(f"a04 latest complete checkpoint changed: {selected_path}")
    if sha256(expected_path) != EXPECTED_A04_SHA256:
        raise ValueError("a04 model_48 bytes no longer match the authorized checkpoint hash")
    optimizer_entries = int(safe["optimizer_state_entries"])

    wrapper = load_module(trial / "run_pipeline.py", "a05_recovery_pipeline_for_cpu_verify")
    preflight = check_wrapper_preflight(trial, wrapper, optimizer_entries)

    verifier = load_module(BASE_VERIFIER, "a05_reused_resume_checkpoint_verifier")
    a04_adapter = load_module(A04_ADAPTER, "a05_reused_a04_headless_config_adapter")
    verifier.A02 = A04
    verifier.A02_ENV = A04_ENV
    pipeline = SimpleNamespace(
        select_latest_complete_checkpoint=a04_pipeline.A03.select_latest_complete_checkpoint,
        _adam_step=a04_pipeline.A03._adam_step,
        _old_pipeline=a04_pipeline._helper,
    )
    verifier.import_pipeline = lambda _trial: pipeline
    verifier.compare_configs = lambda old, new: a04_adapter.compare_headless_configs(
        old, new, verifier)
    torch.set_num_threads(1)
    result = verifier.verify(trial, torch)

    source = result["handoff_checkpoint"]
    resumed = result["resumed_checkpoint"]
    progress = result["resume_progress"]
    adam = result["adam_resume"]
    if (source["sha256_after_load"] != EXPECTED_A04_SHA256
            or source["num_timesteps"] != EXPECTED_A04_STEPS
            or source["adam_step"] != EXPECTED_A04_ADAM):
        raise ValueError("CPU-loaded handoff does not match the authorized a04 checkpoint")
    if (resumed["num_timesteps"] != EXPECTED_FIRST_STEPS
            or resumed["adam_step"] != EXPECTED_FIRST_ADAM
            or resumed["iter"] != EXPECTED_A04_ITER):
        raise ValueError("first a05 checkpoint must be iter48 / 26624 transitions / Adam1040")
    if (progress != {"additional_num_timesteps": ROLLOUT_TRANSITIONS,
                     "rollouts": 1, "transitions_per_rollout": ROLLOUT_TRANSITIONS}
            or adam != {"step_at_handoff": EXPECTED_A04_ADAM,
                        "step_at_checkpoint": EXPECTED_FIRST_ADAM,
                        "steps_per_rollout": ADAM_PER_ROLLOUT}):
        raise ValueError("first a05 checkpoint must add exactly one 512-transition / 20-Adam rollout")
    if not result["physical_config"].get("passed") or not result["policy_contract"].get("identical"):
        raise ValueError("a05 physical config and policy contract must match the a04 handoff")
    if any(result["weight_changes"].get("changed_by_group", {}).get(group, 0) <= 0
           for group in ("actor_state_dict", "critic_state_dict")):
        raise ValueError("both actor and critic weights must change after the resumed rollout")

    return {
        "schema": "agtc-a05-recovery-first-checkpoint-cpu-v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_trial": A04.name,
        "handoff_checkpoint": source,
        "resumed_checkpoint": resumed,
        "resume_progress": progress,
        "adam_resume": adam,
        "weight_changes": result["weight_changes"],
        "policy_contract": result["policy_contract"],
        "physical_config": result["physical_config"],
        "wrapper_preflight": preflight,
        "completion_target": {
            "remaining_iterations": 333,
            "expected_final_payload_iter": TARGET_FINAL_ITER,
            "expected_num_timesteps": TARGET_TRANSITIONS,
            "expected_adam_step": TARGET_ADAM,
        },
        "verification_reuse": {
            "base_verifier": str(BASE_VERIFIER), "base_verifier_sha256": sha256(BASE_VERIFIER),
            "base_schema": "agtc-a03-resume-first-checkpoint-cpu-v1",
            "headless_config_adapter": str(A04_ADAPTER),
            "headless_config_adapter_sha256": sha256(A04_ADAPTER),
            "a04_pipeline_sha256": sha256(A04_PIPELINE),
            "a05_pipeline_sha256": sha256(trial / "run_pipeline.py"),
        },
        "scope": {
            "device": "cpu", "cuda_visible_devices": "",
            "native_runtime_disabled": True, "kit_imported": False,
            "training_or_simulation_started_by_verifier": False,
            "checkpoint_bytes_modified": False,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=A05_DEFAULT)
    trial_arg = parser.parse_args(argv).trial
    trial = trial_arg.resolve() if trial_arg.is_absolute() else (FIELD / trial_arg).resolve()
    try:
        import torch
        report = verify(trial, torch)
        code = 0 if report.get("status") in ("passed", "already_passed") else 2
    except Pending as exc:
        report = {"schema": "agtc-a05-recovery-first-checkpoint-cpu-v1",
                  "status": "pending", "message": str(exc),
                  "created_at_utc": datetime.now(timezone.utc).isoformat()}
        code = 2
    except Exception as exc:
        report = {"schema": "agtc-a05-recovery-first-checkpoint-cpu-v1",
                  "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "created_at_utc": datetime.now(timezone.utc).isoformat()}
        code = 1
    if trial.is_dir() and report.get("status") != "already_passed":
        output = trial / OUTPUT_NAME
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "resumed_checkpoint", "completion_target", "error", "message")
                      if key in report}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
