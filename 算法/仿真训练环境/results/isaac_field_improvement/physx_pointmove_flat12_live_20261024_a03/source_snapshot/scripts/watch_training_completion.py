"""Watch a trial's JSON status and queue one Codex message per completion event."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
QUEUED = re.compile(rf"Queued message ({UUID}) for thread ({UUID})", re.I)

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None

def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def lock_state(path):
    handle = path.open("a+b")
    try:
        handle.seek(0)
        if not handle.read(1):
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError(f"watcher already holds {path}")
    return handle

def parent_identity(pid, expected_start):
    if os.name != "nt" or not isinstance(pid, int):
        return "unknown"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000 | 0x00100000, False, pid)
    if not handle:
        return "missing" if ctypes.get_last_error() == 87 else "unknown"
    created, exited, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
    try:
        wait = kernel.WaitForSingleObject(handle, 0)
        if wait == 0:
            return "missing"
        if wait != 0x102:
            return "unknown"
        ok = kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                   ctypes.byref(kernel_time), ctypes.byref(user_time))
        if not ok:
            return "unknown"
        ticks = (created.high << 32) | created.low
        actual = datetime.fromtimestamp(ticks / 10_000_000 - 11_644_473_600, timezone.utc)
        if expected_start == "capture":
            return actual.isoformat()
        expected = datetime.fromisoformat(str(expected_start).replace("Z", "+00:00"))
        if expected.tzinfo is None:
            expected = expected.replace(tzinfo=timezone.utc)
        return "match" if abs((actual - expected).total_seconds()) <= .001 else "mismatch"
    except (ValueError, OverflowError, OSError):
        return "unknown"
    finally:
        kernel.CloseHandle(handle)

def event_for(status, trial, mismatch_count):
    children = status.get("children", [])
    required_failure = any(isinstance(c, dict) and not c.get("optional")
                           and isinstance(c.get("actual_exit_code"), int)
                           and c["actual_exit_code"] != 0 for c in children)
    if status.get("status") == "failed" or required_failure:
        return "pipeline_failed"
    if status.get("status") == "completed":
        evals = {c.get("name"): c for c in children if isinstance(c, dict)}
        comparison = trial / "comparison.json"
        required = ("evaluate_baseline", "evaluate_final")
        return "pipeline_finished" if all(evals.get(name, {}).get("actual_exit_code") == 0 for name in required) and comparison.is_file() else None
    if status.get("status") == "running" and mismatch_count >= 2:
        return "parent_crashed"
    if status.get("status") in {"completed", "failed"}:
        return None
    training = next((c for c in children if isinstance(c, dict) and c.get("name") == "resume_training_cli"), {})
    validation = read_json(trial / "final_validation.json")
    passed = validation and (validation.get("passed") is True or validation.get("validation_passed") is True
                             or validation.get("status") == "passed")
    return "train_finished" if training.get("actual_exit_code") == 0 and passed else None

def prompt_for(event, trial):
    instructions = {
        "train_finished": "训练子进程实际 exit 0，终局 checkpoint 的 final_validation 已通过。父 pipeline 会自动串行运行 baseline/final 评估和 comparison；只读取真实状态、进程、checkpoint 与报告并简要汇报当前自动评估状态，不要重复启动任何 GPU 训练或评估。",
        "pipeline_finished": "父 pipeline 与评估子进程已正常完成且 comparison.json 已落盘。核对真实进程、checkpoint、评估报告和 comparison，简要解释实际效果；不要把流程完成等同于控制质量通过，也不要重跑 GPU。",
        "pipeline_failed": "检测到父 pipeline failed 或必需子进程非零退出。读取真实状态、进程、checkpoint 和报告，简要说明失败证据；不要重跑训练/评估，不改冻结源。",
        "parent_crashed": "run_status 仍为 running，但 parent PID 的创建时间连续两次缺失或不匹配。核查父进程和原生子进程（子进程可能仍活着）及磁盘证据；不要重跑训练/评估，不改冻结源。",
    }
    return f"{instructions[event]}\n\n受信任的 trial 路径：{trial}\n请在当前聊天中核查并汇报。"

def tick(trial, thread, codex, state_dir, runner=None):
    state_dir.mkdir(parents=True, exist_ok=True)
    binding_path = state_dir / "binding.json"
    binding = {"trial": str(trial.resolve()), "thread": thread.lower()}
    if binding_path.exists() and read_json(binding_path) != binding:
        raise RuntimeError("state directory is bound to a different trial/chat or is unreadable")
    if not binding_path.exists():
        write_json(binding_path, binding)
    status = read_json(trial / "run_status.json")
    watch_path, delivery_path = state_dir / "watch_status.json", state_dir / "delivery.json"
    watch = read_json(watch_path) or {"schema": 1, "parent_mismatch_count": 0}
    watch.update(watcher_pid=os.getpid(), running=True)
    delivery = read_json(delivery_path) or {"schema": 1, "events": {}}
    events = delivery.setdefault("events", {})
    for record in events.values():
        if record.get("phase") == "reserved":
            record.update(phase="uncertain", reason="watcher_restarted_after_reservation")
    if not status:
        watch.update(last_poll_utc=now(), trial_status="unreadable")
        write_json(watch_path, watch)
        write_json(delivery_path, delivery)
        return None
    if status.get("status") == "running":
        pid = status.get("parent_pid")
        captured = watch.get("parent_created_utc")
        if captured:
            ident = parent_identity(pid, captured) if pid == watch.get("parent_pid") else "mismatch"
        else:
            stamp = parent_identity(pid, "capture")
            ident = stamp if stamp in {"missing", "unknown", "mismatch"} else "match"
            if ident == "match":
                watch.update(parent_pid=pid, parent_created_utc=stamp)
        count = watch.get("parent_mismatch_count", 0)
        watch["parent_mismatch_count"] = count + 1 if ident in {"missing", "mismatch"} else 0
    else:
        watch["parent_mismatch_count"] = 0
    event = event_for(status, trial, watch["parent_mismatch_count"])
    watch.update(last_poll_utc=now(), trial_status=status.get("status"), parent_identity=locals().get("ident", "not_running"))
    write_json(watch_path, watch)
    write_json(delivery_path, delivery)
    if not event or event in events:
        return event
    record = {"phase": "reserved", "reserved_at_utc": now()}
    events[event] = record
    write_json(delivery_path, delivery)
    command = [str(codex), "queue", "--thread", thread, "--message", prompt_for(event, trial)]
    runner = runner or subprocess.run
    try:
        result = runner(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                        timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        stdout, stderr = result.stdout or "", result.stderr or ""
        match = QUEUED.search(stdout)
        confirmed = bool(result.returncode == 0 and match and match.group(2).lower() == thread.lower())
        record.update(phase="queued" if confirmed else "uncertain", finished_at_utc=now(),
                      returncode=result.returncode, stdout_tail=stdout[-400:], stderr_tail=stderr[-400:])
        if confirmed:
            record["message_id"] = match.group(1)
        elif not match:
            record["reason"] = "queue_confirmation_missing_or_unrecognized"
        else:
            record["reason"] = "queue_confirmation_thread_mismatch"
    except Exception as exc:
        record.update(phase="uncertain", finished_at_utc=now(), reason=type(exc).__name__)
    write_json(delivery_path, delivery)
    return event

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--thread", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--codex", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true", help="inspect once (may queue an event)")
    args = parser.parse_args()
    if not args.thread or not re.fullmatch(UUID, args.thread):
        parser.error("--thread or CODEX_THREAD_ID must be a UUID")
    args.codex = args.codex or Path(shutil.which("codex") or "")
    if not args.codex.is_file():
        parser.error("--codex must point to the installed Codex CLI executable")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock = lock_state(args.state_dir / "watcher.lock")
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        while True:
            event = tick(args.trial.resolve(), args.thread, args.codex, args.state_dir)
            if event:
                print(f"event={event}; see {args.state_dir / 'delivery.json'}", flush=True)
            if args.once or event in {"pipeline_finished", "pipeline_failed", "parent_crashed"}:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        state = read_json(args.state_dir / "watch_status.json") or {}
        state.update(running=False, stopped_utc=now())
        write_json(args.state_dir / "watch_status.json", state)
        lock.close()

if __name__ == "__main__":
    main()
