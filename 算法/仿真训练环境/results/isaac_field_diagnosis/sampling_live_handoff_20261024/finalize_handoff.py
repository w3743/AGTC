"""Bind observed stopped-process evidence to the last audited complete checkpoint."""
from pathlib import Path
import datetime as dt
import hashlib
import json

import psutil

ROOT = Path(__file__).resolve().parent
FIELD = ROOT.parents[2]
OLD = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_20261024_a02"
NEW = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_live_20261024_a03"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    result_path = NEW / "handoff.json"
    if result_path.exists():
        raise FileExistsError("handoff already bound; preserve the existing evidence")
    regression = read(ROOT / "cpu_regression/verification.json")
    if regression.get("status") != "passed":
        raise RuntimeError("CPU regression has not passed")
    evidence = read(ROOT / "process_handoff.json")
    if not evidence.get("all_owned_processes_absent"):
        raise RuntimeError("process handoff incomplete")
    records = {row["pid"]: row for row in evidence["processes"]}
    roles = {"training_native": 7520, "resume_training_cli": 22492,
             "controller": 7316, "watcher": 3120, "static_gui": 39828}
    old_processes = {}
    for role, pid in roles.items():
        record = records[pid]
        if not record.get("confirmed_absent") or psutil.pid_exists(pid):
            raise RuntimeError(f"old {role} process still present or PID reused")
        old_processes[role] = {"pid": pid, "state": "stopped", "verified": True,
            "created": record["created"], "observed_exit_code": record.get("observed_exit_code"),
            "argv_sha256": hashlib.sha256(json.dumps(record["cmdline"]).encode()).hexdigest(),
            "evidence_path": str(ROOT / "process_handoff.json")}
    audit = read(ROOT / "checkpoint_audit.json")
    checkpoint = audit["latest_complete_checkpoint"]
    if not (checkpoint["file_unchanged_during_load"] and checkpoint["model_tensors_all_finite"]
            and checkpoint["contract_matches_zero_step_baseline"]):
        raise RuntimeError("checkpoint audit incomplete")
    steps = checkpoint["optimizer"]["adam_step_values_unique"]
    if len(steps) != 1 or int(steps[0]) != steps[0]:
        raise RuntimeError("inconsistent Adam step counters")
    status = read(OLD / "run_status.json")
    if status.get("status") != "superseded" or status.get("superseded_by") != NEW.name:
        raise RuntimeError("old trial is not superseded by the intended successor")
    value = {"schema": "isaac-field-live-resume-handoff-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_trial": OLD.name, "source_protocol_sha256": audit["protocol"]["sha256"],
        "old_run_status": {"status": "superseded", "superseded_by": NEW.name},
        "old_processes": old_processes,
        "checkpoint": {"path": checkpoint["path"], "sha256": checkpoint["sha256_after_load"],
                       "iter": checkpoint["iteration"], "num_timesteps": checkpoint["num_timesteps"],
                       "adam_step": int(steps[0])},
        "checkpoint_audit_path": str(ROOT / "checkpoint_audit.json"),
        "resume_boundary": "weights/Adam/counters restored; simulator/RNG/in-flight rollout/LSTM state reinitialized",
        "native_training_started_by_this_script": False}
    result_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "bound", "num_timesteps": checkpoint["num_timesteps"], "adam_step": int(steps[0])}))


if __name__ == "__main__":
    main()
