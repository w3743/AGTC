"""Stop only the previously verified PI-v6 viewer before replacing its backend."""
import json
import os
from pathlib import Path
import signal
import time

pid = 198426
proc = Path(f"/proc/{pid}")
if not proc.exists():
    print(json.dumps({"pid": pid, "status": "already_exited"}))
else:
    args = (proc / "cmdline").read_bytes().decode().split("\0")
    expected = ["-m", "isaac_field", "viewer"]
    if (args[1:4] != expected
            or "results/isaac_field_viewer/side_pi_20261020/viewer_8192_replay" not in args
            or args[-3:-1] != ["--port", "8769"]):
        raise RuntimeError(f"PID {pid} is not the verified old viewer; refusing signal: {args}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while proc.exists() and time.monotonic() < deadline:
        try:
            if "State:\tZ" in (proc / "status").read_text():
                break
        except FileNotFoundError:
            break
        time.sleep(0.1)
    else:
        if proc.exists():
            raise RuntimeError("Verified viewer did not exit within 10 seconds; no stronger signal sent")
    print(json.dumps({"pid": pid, "status": "terminated_verified_old_viewer", "args": args}, ensure_ascii=True))
