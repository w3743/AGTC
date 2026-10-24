"""One-time user-authorized handoff; terminate only identity-checked owned processes.

This script never starts training. Persisted checkpoints and logs are untouched.
Windows TerminateProcess cannot save the in-flight rollout: resume uses the last
complete checkpoint. An explicit --execute is required after CPU verification.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil

import psutil

ROOT = Path(__file__).resolve().parent
FIELD = ROOT.parents[2]
OLD = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_20261024_a02"
NEW = FIELD / "results/isaac_field_improvement/physx_pointmove_flat12_live_20261024_a03"
WATCH = Path("C:/pioneer/isaac-training/watchers/pointmove-a02")
GUI = FIELD / "results/isaac_field_viewer/physx_pointmove_flat12_gui_20261024_a03"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def capture(pid, expected):
    proc = psutil.Process(pid)
    cmd = proc.cmdline()
    if expected.casefold() not in " ".join(cmd).replace("\\", "/").casefold():
        raise RuntimeError(f"PID {pid} identity mismatch; refusing to stop it: {cmd}")
    if "watch_training_completion.py" in expected:
        if OLD.as_posix().casefold() not in " ".join(cmd).replace("\\", "/").casefold():
            raise RuntimeError("watcher is attached to another trial")
    return {"pid": pid, "created": proc.create_time(), "cmdline": cmd,
            "expected": expected, "parent_pid": proc.ppid()}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--execute", action="store_true")
    options = args.parse_args()
    expected = [
        (3120, "watch_training_completion.py"),
        (7316, (OLD / "run_pipeline.py").as_posix()),
        (7520, "physx_pointmove_flat12_20261024_a02"),
        (39828, (GUI / "open_gui.py").as_posix()),
        (22324, "watch_training_completion.py"),
        (33012, (OLD / "run_pipeline.py").as_posix()),
        (22492, "physx_pointmove_flat12_20261024_a02"),
    ]
    records = [capture(pid, marker) for pid, marker in expected]
    report = {"schema": "isaac-field-authorized-process-handoff-v1", "created_at_utc": now(),
              "user_authorization": "optimize copy/repeated calculations; keep physics; resume checkpoint with live view",
              "successor_trial": str(NEW), "processes": records,
              "stop_method": "identity-checked Windows TerminateProcess; no in-flight checkpoint save",
              "executed": options.execute}
    if not options.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if (ROOT / "process_handoff.json").exists():
        raise FileExistsError("handoff already attempted; inspect evidence before doing anything")
    shutil.copyfile(OLD / "run_status.json", ROOT / "a02_run_status_before.json")
    shutil.copyfile(WATCH / "watch_status.json", ROOT / "watch_status_before.json")
    write(ROOT / "process_handoff.json", report)
    for record in records:
        try:
            proc = psutil.Process(record["pid"])
            if proc.create_time() != record["created"]:
                raise RuntimeError("PID was reused; refusing termination")
            capture(record["pid"], record["expected"])
            proc.terminate()
            record.update(stop_requested_utc=now(), observed_exit_code=proc.wait(timeout=10))
        except psutil.NoSuchProcess:
            record["already_exited_during_handoff"] = True
        record["confirmed_absent"] = not psutil.pid_exists(record["pid"])
        if not record["confirmed_absent"]:
            raise RuntimeError(f"process {record['pid']} still exists")
        write(ROOT / "process_handoff.json", report)
    state = json.loads((OLD / "run_status.json").read_text(encoding="utf-8"))
    state.update(status="superseded", phase="authorized_checkpoint_handoff", updated_at_utc=now(),
                 superseded_by=NEW.name, successor_path=str(NEW), reason=report["user_authorization"],
                 termination_evidence=str(ROOT / "process_handoff.json"),
                 training_completed=False, evaluations_completed=False)
    current = state.get("current_stage")
    if isinstance(current, dict):
        current.update(status="interrupted_by_user_authorized_handoff", child_still_running=False,
                       completed_at_utc=now())
    for child in state.get("children", []):
        if child.get("name") == "resume_training_cli":
            child.update(status="interrupted_by_user_authorized_handoff", child_still_running=False,
                         completed_at_utc=now(), native_training_pid=7520,
                         native_observed_exit_code=next(r.get("observed_exit_code")
                             for r in records if r["pid"] == 7520))
    write(OLD / "run_status.json", state)
    watch = json.loads((WATCH / "watch_status.json").read_text(encoding="utf-8"))
    watch.update(running=False, stopped_utc=now(), reason="authorized successor trial handoff",
                 successor_trial=str(NEW))
    write(WATCH / "watch_status.json", watch)
    evidence = []
    for path in sorted(OLD.glob("*.log")):
        evidence.append({"path": str(path), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report.update(completed_at_utc=now(), all_owned_processes_absent=True,
                  old_raw_logs=evidence, logs_or_checkpoints_modified=False)
    write(ROOT / "process_handoff.json", report)
    print(json.dumps({"status": "handoff_complete", "stopped_pids": [r['pid'] for r in records]}))


if __name__ == "__main__":
    main()
