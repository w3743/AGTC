#!/usr/bin/env python
"""Verify the first a03 resumed checkpoint on CPU; never launches Kit or training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
FIELD = SCRIPT_DIR.parents[2]
DEFAULT_TRIAL = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_live_20261024_a03"
A02 = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_20261024_a02"
A02_ENV = A02 / "logs/rsl_rl/physx_pointmove_flat12_20261024_a02/training_a02/params/env.yaml"
ROLLOUT = 4 * 128
ADAM_PER_ROLLOUT = 20
SENSORS = ("depth", "detector", "lidar", "rangefinder", "imu")
OUTPUT = "first_resumed_checkpoint_cpu.json"


class Pending(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def import_pipeline(trial: Path):
    path = trial / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("a03_live_pipeline_cpu_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load a03 pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def latest_complete(pipeline, trial: Path):
    root = trial / "logs/rsl_rl"
    if not root.is_dir():
        return None
    run_dirs = {path.parent for path in root.rglob("model_*.pt")}
    candidates = []
    for run_dir in sorted(run_dirs):
        try:
            path, payload, safe = pipeline.select_latest_complete_checkpoint(run_dir)
            candidates.append((int(safe["filename_iteration"]), path, payload, safe))
        except (OSError, RuntimeError, ValueError, EOFError):
            continue
    if not candidates:
        return None
    # Numeric suffix is used only to select the latest complete file. Progress
    # is read exclusively from payload.num_timesteps.
    return max(candidates, key=lambda row: (row[0], row[1].stat().st_mtime_ns))


def load_same(helper, path: Path):
    before_sig, before_sha = signature(path), sha256(path)
    payload, safe = helper.cpu_load_checkpoint(path)
    after_sig, after_sha = signature(path), sha256(path)
    if before_sig != after_sig or before_sha != after_sha:
        raise Pending(f"checkpoint changed while CPU-loading: {path}")
    return payload, safe, {
        "path": str(path), "bytes": after_sig[0], "sha256_before_load": before_sha,
        "sha256_after_load": after_sha, "file_unchanged_during_load": True,
    }


def yaml_base(path: Path):
    import yaml
    return yaml.load(path.read_text(encoding="utf-8-sig"), Loader=yaml.BaseLoader)


def strip_sim_ui(value):
    return {k: v for k, v in value.items()
            if k not in {"default_visualizer_cfg", "visualizer_cfgs", "log_dir"}}


def differences(old, new, path="", limit=60):
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new), key=str):
            here = f"{path}.{key}" if path else str(key)
            if key not in old:
                out.append(f"{here}: missing in old")
            elif key not in new:
                out.append(f"{here}: missing in new")
            else:
                out.extend(differences(old[key], new[key], here, limit - len(out)))
            if len(out) >= limit:
                break
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: list length {len(old)} != {len(new)}"]
        out = []
        for i, (a, b) in enumerate(zip(old, new)):
            out.extend(differences(a, b, f"{path}[{i}]", limit - len(out)))
            if len(out) >= limit:
                break
        return out
    return [] if old == new else [f"{path}: {old!r} != {new!r}"]


def compare_configs(old_path: Path, new_path: Path):
    old, new = yaml_base(old_path), yaml_base(new_path)
    old_sim, new_sim = strip_sim_ui(old["sim"]), strip_sim_ui(new["sim"])
    visual = new["sim"].get("default_visualizer_cfg")
    if not isinstance(visual, dict):
        raise ValueError("new sim.default_visualizer_cfg is missing")
    visual_check = {
        "origin_type": visual.get("origin_type"),
        "origin_track_path": visual.get("origin_track_path"),
        "origin_env_index": visual.get("origin_env_index"),
        "max_visible_envs": visual.get("max_visible_envs", "<unset>"),
        "passed": (
            visual.get("origin_type") == "asset"
            and visual.get("origin_track_path") == "robot"
            and str(visual.get("origin_env_index")) == "0"
            and visual.get("max_visible_envs") in (None, "null", "None", "~")
        ),
        "renderer_frame_settings_compared": False,
    }
    old_query = old_sim.get("physics", {}).get("enable_scene_query_support")
    new_query = new_sim.get("physics", {}).get("enable_scene_query_support")
    query_exception = None
    sim_compare = new_sim
    if old_query == "false" and new_query == "true" and visual_check["passed"]:
        sim_compare = dict(new_sim)
        sim_compare["physics"] = dict(new_sim["physics"])
        sim_compare["physics"]["enable_scene_query_support"] = old_query
        query_exception = {
            "path": "sim.physics.enable_scene_query_support",
            "old": old_query,
            "new": new_query,
            "reason": "PhysxManager enables the scene query manager when /isaaclab/has_gui is true; this does not change simulation precision.",
            "official_source": "external/IsaacLab/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py:852-856",
        }
    checks = {
        "sim_physics": differences(old_sim, sim_compare),
        "decimation": differences(old["decimation"], new["decimation"]),
        "scene_num_envs": differences(old["scene"]["num_envs"], new["scene"]["num_envs"]),
        "task_full_dict": differences(old["task"], new["task"]),
    }
    checks.update({f"scene_{name}": differences(old["scene"][name], new["scene"][name])
                   for name in SENSORS})
    failed = {name: values for name, values in checks.items() if values}
    if failed or not visual_check["passed"]:
        raise ValueError(f"physical config differs: {failed}; visual_cfg={visual_check}")
    check_status = {name: "identical" for name in checks}
    if query_exception:
        check_status["sim_physics"] = "identical_except_documented_gui_query_toggle"
    return {
        "passed": True,
        "checks": check_status,
        "sim_gui_query_exception": query_exception,
        "sim_ui_fields_ignored": ["default_visualizer_cfg", "visualizer_cfgs", "log_dir"],
        "rendered_pixel_frames_compared": False,
        "old_env_yaml": str(old_path), "new_env_yaml": str(new_path),
        "visual_cfg": visual_check,
    }


def verify(trial: Path, torch):
    handoff_path = trial / "handoff.json"
    if not handoff_path.is_file():
        raise Pending(f"waiting for {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
    meta = handoff.get("checkpoint", {})
    required = ("path", "sha256", "iter", "num_timesteps", "adam_step")
    if any(key not in meta for key in required):
        raise ValueError("handoff.json checkpoint must include path/sha256/iter/num_timesteps/adam_step")
    old_path = Path(meta["path"]).resolve()
    if A02.resolve() not in old_path.parents:
        raise ValueError(f"handoff checkpoint is outside a02: {old_path}")

    pipeline = import_pipeline(trial)
    helper = pipeline._old_pipeline()
    old_payload, old_safe, old_file = load_same(helper, old_path)
    if old_file["sha256_after_load"] != str(meta["sha256"]).lower():
        raise ValueError("handoff checkpoint SHA does not match handoff.json")
    old_iter = int(old_payload["iter"])
    old_steps = int(old_payload["num_timesteps"])
    old_adam = pipeline._adam_step(old_payload)
    if (old_iter, old_steps, old_adam) != (
        int(meta["iter"]), int(meta["num_timesteps"]), int(meta["adam_step"]),
    ):
        raise ValueError("handoff metadata does not match the CPU-loaded a02 checkpoint")

    selected = latest_complete(pipeline, trial)
    if selected is None:
        raise Pending("no complete non-baseline a03 checkpoint is present yet")
    suffix, new_path, _, _ = selected
    new_payload, new_safe, new_file = load_same(helper, new_path)
    selected_after = latest_complete(pipeline, trial)
    if selected_after is None or selected_after[1] != new_path:
        raise Pending("a newer complete checkpoint appeared during the audit; retry")

    new_steps = int(new_payload["num_timesteps"])
    delta = new_steps - old_steps
    if delta < ROLLOUT:
        raise Pending(f"only {delta} additional transitions; waiting for at least {ROLLOUT}")
    if delta % ROLLOUT:
        raise ValueError(f"additional transitions {delta} are not a multiple of {ROLLOUT}")
    rollouts = delta // ROLLOUT
    new_adam = pipeline._adam_step(new_payload)
    expected_adam = old_adam + rollouts * ADAM_PER_ROLLOUT
    if new_adam != expected_adam:
        raise ValueError(f"Adam step {new_adam} != {expected_adam} ({ADAM_PER_ROLLOUT}/rollout)")
    if len(old_payload["optimizer_state_dict"]["state"]) != len(new_payload["optimizer_state_dict"]["state"]):
        raise ValueError("Adam optimizer state entry count changed during resume")

    changes = helper._state_dict_changes(old_payload, new_payload, torch)
    if any(changes["changed_by_group"].get(group, 0) == 0
           for group in ("actor_state_dict", "critic_state_dict")):
        raise ValueError(f"actor and critic must both have updated weights: {changes}")
    requirements = helper.expected_contract_requirements()
    old_contract = helper._assert_checkpoint_contract(old_payload.get("field_policy_contract"), requirements)
    new_contract = helper._assert_checkpoint_contract(new_payload.get("field_policy_contract"), requirements)
    if old_contract["contract"] != new_contract["contract"]:
        raise ValueError("full field_policy_contract differs between a02 handoff and a03 checkpoint")

    new_env = new_path.parent / "params" / "env.yaml"
    config = compare_configs(A02_ENV, new_env)
    return {
        "schema": "agtc-a03-resume-first-checkpoint-cpu-v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "handoff_checkpoint": {**old_file, "iter": old_iter, "num_timesteps": old_steps,
                               "adam_step": old_adam, "cpu_finite_tree": old_safe["finite_tensor_tree"]},
        "resumed_checkpoint": {**new_file, "iter": int(new_payload["iter"]),
                               "num_timesteps": new_steps, "adam_step": new_adam,
                               "filename_suffix_used_only_for_selection": suffix,
                               "cpu_finite_tree": new_safe["finite_tensor_tree"]},
        "resume_progress": {"additional_num_timesteps": delta, "rollouts": rollouts,
                            "transitions_per_rollout": ROLLOUT},
        "adam_resume": {"step_at_handoff": old_adam, "step_at_checkpoint": new_adam,
                        "steps_per_rollout": ADAM_PER_ROLLOUT},
        "weight_changes": changes,
        "policy_contract": {"identical": True, "sha256": new_contract["contract_sha256"]},
        "physical_config": config,
        "scope": {"device": "cpu", "cuda_visible_devices": "", "kit_imported": False,
                  "training_or_simulation_started": False, "checkpoint_bytes_modified": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    trial_arg = parser.parse_args(argv).trial
    trial = trial_arg.resolve() if trial_arg.is_absolute() else (FIELD / trial_arg).resolve()
    output = trial / OUTPUT
    try:
        import torch
        torch.set_num_threads(1)
        report = verify(trial, torch)
        code = 0
    except Pending as exc:
        report = {"schema": "agtc-a03-resume-first-checkpoint-cpu-v1", "status": "pending",
                  "message": str(exc), "created_at_utc": datetime.now(timezone.utc).isoformat()}
        code = 2
    except Exception as exc:
        report = {"schema": "agtc-a03-resume-first-checkpoint-cpu-v1", "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}",
                  "created_at_utc": datetime.now(timezone.utc).isoformat()}
        code = 1
    if trial.is_dir():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "resumed_checkpoint", "resume_progress", "adam_resume", "error", "message")
                      if key in report}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
