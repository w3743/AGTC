#!/usr/bin/env python
"""CPU-only adapter for the a04 first resumed-checkpoint audit."""
from __future__ import annotations

import hashlib, importlib.util, json, os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = Path(__file__).resolve().parent
FIELD = HERE.parents[2]
A04 = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_mirror_20261024_a04"
A03 = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_live_20261024_a03"
A02 = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_20261024_a02"
A02_ENV = A02 / "logs/rsl_rl/physx_pointmove_flat12_20261024_a02/training_a02/params/env.yaml"
VERIFIER = HERE.parent / "sampling_live_handoff_20261024/verify_resumed_training.py"
OUTPUT = A04 / "first_resumed_checkpoint_cpu.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier input: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_headless_configs(old_path: Path, new_path: Path, verifier):
    old, new = verifier.yaml_base(old_path), verifier.yaml_base(new_path)
    diff = verifier.differences
    checks = {"sim_physics": diff(verifier.strip_sim_ui(old["sim"]), verifier.strip_sim_ui(new["sim"])),
              "decimation": diff(old["decimation"], new["decimation"]),
              "scene_num_envs": diff(old["scene"]["num_envs"], new["scene"]["num_envs"]),
              "task_full_dict": diff(old["task"], new["task"]),
              **{f"scene_{n}": diff(old["scene"][n], new["scene"][n]) for n in verifier.SENSORS}}
    failed = {name: values for name, values in checks.items() if values}
    if failed:
        raise ValueError(f"a04 headless physical/sensor YAML differs from a02: {failed}")
    return {"passed": True, "checks": {name: "identical" for name in checks}, "sim_gui_query_exception": None,
            "sim_ui_fields_ignored": ["default_visualizer_cfg", "visualizer_cfgs", "log_dir"],
            "rendered_pixel_frames_compared": False, "old_env_yaml": str(old_path), "new_env_yaml": str(new_path)}


def verify(torch):
    verifier = load(VERIFIER, "a04_reused_first_checkpoint_verifier")
    verifier.A02 = A03  # Reuse its original source-checkpoint containment guard for a03.
    verifier.A02_ENV = A02_ENV  # Physical reference remains the original a02 headless run.
    adapter = load(A04 / "run_pipeline.py", "a04_checkpoint_pipeline_adapter")
    pipeline = SimpleNamespace(
        select_latest_complete_checkpoint=adapter.A03.select_latest_complete_checkpoint,
        _adam_step=adapter.A03._adam_step,
        _old_pipeline=adapter._helper,
    )
    verifier.import_pipeline = lambda _trial: pipeline
    verifier.compare_configs = lambda old, new: compare_headless_configs(old, new, verifier)
    result = verifier.verify(A04, torch)
    fields = ("resumed_checkpoint", "resume_progress", "adam_resume", "weight_changes", "policy_contract",
              "physical_config", "scope")
    return {"schema": "agtc-a04-mirror-first-checkpoint-cpu-v1", "status": result["status"],
            "created_at_utc": result["created_at_utc"], "source_trial": A03.name,
            "source_checkpoint": result["handoff_checkpoint"], **{k: result[k] for k in fields},
            "baseline_physical_reference": {"trial": A02.name, "env_yaml": str(A02_ENV),
                                             "env_yaml_sha256": sha256(A02_ENV)},
            "verification_reuse": {"verifier_path": str(VERIFIER), "verifier_sha256": sha256(VERIFIER),
                                   "base_schema": "agtc-a03-resume-first-checkpoint-cpu-v1",
                                   "checkpoint_counts_from": str(A04 / "handoff.json")}}


def main() -> int:
    from datetime import datetime, timezone
    try:
        import torch
        torch.set_num_threads(1)
        report = verify(torch)
        code = 0
    except Exception as exc:
        status = "pending" if type(exc).__name__ == "Pending" else "failed"
        report = {"schema": "agtc-a04-mirror-first-checkpoint-cpu-v1", "status": status,
                  "error": f"{type(exc).__name__}: {exc}", "created_at_utc": datetime.now(timezone.utc).isoformat()}
        code = 2 if status == "pending" else 1
    if A04.is_dir():
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "source_checkpoint", "resumed_checkpoint", "error") if key in report},
                     ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
