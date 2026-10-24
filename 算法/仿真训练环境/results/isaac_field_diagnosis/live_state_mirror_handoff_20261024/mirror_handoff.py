"""Identity-checked, checkpoint-boundary handoff authorized by the user.

No simulator or training process is started here. The source run's next saved
checkpoint is model_48 (the latest complete file at authorization was model_32).
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parent
FIELD = ROOT.parents[2]
OLD = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_live_20261024_a03"
NEW = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_mirror_20261024_a04"
WATCH = Path("C:/pioneer/isaac-training/watchers/pointmove-live-a03")
ROLES = {"training_native": 29076, "resume_training_cli": 7332,
         "controller": 34644, "watcher": 18132, "training_gui": 29076}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checked_process(pid, marker):
    process = psutil.Process(pid)
    command = process.cmdline()
    if marker.casefold() not in " ".join(command).replace("\\", "/").casefold():
        raise RuntimeError(f"PID {pid} is not the expected owned process")
    if OLD.name not in " ".join(command):
        raise RuntimeError(f"PID {pid} is not bound to the source trial")
    return {"pid": pid, "created": process.create_time(), "cmdline": command,
            "parent_pid": process.ppid(), "expected_marker": marker}


def load_pipeline():
    spec = importlib.util.spec_from_file_location("mirror_checkpoint_handoff", NEW / "run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pipeline = load_pipeline()
    source_run = OLD / "logs/rsl_rl" / OLD.name / "training_live_a03"
    next_path = source_run / "model_48.pt"
    if not next_path.is_file():
        print(json.dumps({"status": "waiting_for_next_complete_checkpoint", "expected": str(next_path)}))
        return 2
    path, payload, safe = pipeline.A03.select_latest_complete_checkpoint(source_run)
    if path != next_path or payload["num_timesteps"] != 25600 or safe["adam_step"] != 1000:
        raise RuntimeError("next-checkpoint identity/counters differ; inspect instead of guessing")
    metadata = {"path": str(path), "sha256": sha(path), "iter": payload["iter"],
                "num_timesteps": payload["num_timesteps"], "adam_step": safe["adam_step"]}
    if not args.execute:
        print(json.dumps({"status": "checkpoint_ready", "checkpoint": metadata}, ensure_ascii=False))
        return 0
    gate = read(ROOT / "cpu_gate.json")
    if gate.get("status") != "passed":
        raise RuntimeError("CPU gate has not passed")
    for source, expected in gate["source_sha256"].items():
        if sha(Path(source)) != expected:
            raise RuntimeError(f"source changed after CPU gate: {source}")
    receipt_path = ROOT / "process_handoff.json"
    if receipt_path.exists() or (NEW / "handoff.json").exists():
        raise FileExistsError("handoff already attempted; inspect the existing evidence")
    # Stop the event listener first, then the controller, then its native child.
    identities = [(18132, "watch_training_completion.py"), (34644, "run_pipeline.py"),
                  (29076, "isaac_field resume"), (25232, "watch_training_completion.py"),
                  (38320, "run_pipeline.py"), (7332, "isaac_field resume")]
    records = [checked_process(pid, marker) for pid, marker in identities]
    report = {"schema": "isaac-field-mirror-checkpoint-handoff-v1", "created_at_utc": now(),
        "user_authorization": "independent live observer; switch at next complete checkpoint; preserve physics precision",
        "checkpoint": metadata, "processes": records,
        "stop_method": "identity-checked TerminateProcess after complete checkpoint; in-flight rollout not saved"}
    shutil.copyfile(OLD / "run_status.json", ROOT / "a03_run_status_before.json")
    shutil.copyfile(WATCH / "watch_status.json", ROOT / "watch_status_before.json")
    write(receipt_path, report)
    for record in records:
        try:
            process = psutil.Process(record["pid"])
            if process.create_time() != record["created"]:
                raise RuntimeError("PID reuse detected")
            checked_process(record["pid"], record["expected_marker"])
            process.terminate()
            record["observed_exit_code"] = process.wait(timeout=10)
        except psutil.NoSuchProcess:
            record["already_exited_during_handoff"] = True
        record["confirmed_absent"] = not psutil.pid_exists(record["pid"])
        if not record["confirmed_absent"]:
            raise RuntimeError("owned process remains alive")
        write(receipt_path, report)
    if sha(path) != metadata["sha256"]:
        raise RuntimeError("selected checkpoint bytes changed during handoff")
    status = read(OLD / "run_status.json")
    status.update(status="superseded", phase="authorized_mirror_handoff", superseded_by=NEW.name,
                  updated_at_utc=now(), training_completed=False, evaluations_completed=False)
    for child in [*status.get("children", []), status.get("current_stage")]:
        if isinstance(child, dict) and child.get("name") == "resume_training_cli":
            child.update(status="interrupted_at_saved_checkpoint", child_still_running=False, completed_at_utc=now())
    write(OLD / "run_status.json", status)
    watch = read(WATCH / "watch_status.json")
    watch.update(running=False, stopped_utc=now(), successor_trial=NEW.name)
    write(WATCH / "watch_status.json", watch)
    report.update(all_owned_processes_absent=True, completed_at_utc=now(), checkpoint_bytes_unchanged=True)
    write(receipt_path, report)
    by_pid = {row["pid"]: row for row in records}
    old_processes = {role: {"pid": pid, "state": "stopped", "verified": True,
        "created": by_pid[pid]["created"], "evidence_path": str(receipt_path)} for role, pid in ROLES.items()}
    write(NEW / "handoff.json", {"source_trial": OLD.name,
        "source_protocol_sha256": sha(OLD / "protocol.json"), "checkpoint": metadata,
        "old_processes": old_processes, "old_run_status": {"status": "superseded", "superseded_by": NEW.name}})
    print(json.dumps({"status": "handoff_complete", "checkpoint": metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
