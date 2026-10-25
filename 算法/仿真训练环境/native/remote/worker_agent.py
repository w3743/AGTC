"""LAN worker agent: run native trials on a second machine over HTTP (no admin needed).

Start on the GPU machine (Linux or Windows + WSL):

    python native/remote/worker_agent.py --host 0.0.0.0 --port 8790 \
        --token <random> --repo /path/to/AGTC

Endpoints (all require header `X-Auth-Token`):
    POST /start   {"tag": "pilot02", "sim_dt": 0.001, "decimation": 50,
                   "live_state_file": "/tmp/live.json", "extra_args": []}
    GET  /status
    GET  /metrics?lines=50
    POST /stop
    GET  /live_state          # raw live-state JSON for the local Viser viewer
    GET  /files?trial=...&name=metrics.jsonl

The Lead machine polls `/live_state` and writes it to a local file; the existing
`scripts/view_training_state_web.py --language zh` consumes that file.

Only stdlib is used; no root, no firewall change beyond allowing the chosen port.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE_LOCK = threading.Lock()
STATE = {
    "process": None,
    "pid": None,
    "tag": "",
    "trial_dir": None,
    "log_path": None,
    "started_at": None,
    "live_state_file": None,
    "finished_at": None,
    "returncode": None,
}
TOKEN = ""
NATIVE_DIR = Path(".")


def _json_response(handler, code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_token(handler) -> bool:
    return handler.headers.get("X-Auth-Token", "") == TOKEN


def _latest_trial() -> Path | None:
    candidates = sorted(NATIVE_DIR.glob("runs/native_v1_1_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return [line.rstrip("\n") for line in stream.readlines()[-lines:]]


def _start(payload: dict) -> dict:
    with STATE_LOCK:
        if STATE["process"] is not None and STATE["process"].poll() is None:
            return {"ok": False, "error": "a trial is already running", "pid": STATE["pid"]}
        tag = str(payload.get("tag") or f"lan_{int(time.time())}")
        env = dict(os.environ)
        env["SIM_DT"] = str(payload.get("sim_dt", 0.001))
        env["DECIMATION"] = str(payload.get("decimation", 50))
        args = ["bash", "train/run_native_v1_trial.sh", tag]
        live = payload.get("live_state_file")
        if live:
            args += ["--live-state-file", str(live), "--live-state-interval",
                     str(payload.get("live_state_interval", 5))]
        args += [str(value) for value in payload.get("extra_args", [])]
        log_dir = NATIVE_DIR / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"worker_{tag}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(args, cwd=str(NATIVE_DIR), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        STATE.update(process=process, pid=process.pid, tag=tag, log_path=str(log_path),
                     started_at=time.time(), live_state_file=live, finished_at=None,
                     returncode=None, trial_dir=None)
        return {"ok": True, "pid": process.pid, "tag": tag, "log": str(log_path)}


def _stop() -> dict:
    with STATE_LOCK:
        process = STATE["process"]
        if process is None or process.poll() is not None:
            return {"ok": True, "already_stopped": True}
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        STATE["finished_at"] = time.time()
        STATE["returncode"] = process.returncode
        return {"ok": True, "returncode": process.returncode}


def _status() -> dict:
    with STATE_LOCK:
        process = STATE["process"]
        running = process is not None and process.poll() is None
        if process is not None and not running and STATE["finished_at"] is None:
            STATE["finished_at"] = time.time()
            STATE["returncode"] = process.returncode
        trial = Path(STATE["trial_dir"]) if STATE["trial_dir"] else _latest_trial()
        metrics = trial / "metrics.jsonl" if trial else None
        return {
            "ok": True,
            "running": running,
            "pid": STATE["pid"],
            "tag": STATE["tag"],
            "started_at": STATE["started_at"],
            "finished_at": STATE["finished_at"],
            "returncode": STATE["returncode"],
            "trial_dir": str(trial) if trial else None,
            "log": STATE["log_path"],
            "metrics_tail": _tail(metrics, 5) if metrics else [],
            "failure": json.loads((trial / "failure.json").read_text(encoding="utf-8"))
            if trial and (trial / "failure.json").is_file() else None,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "AGTCWorker/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _guard(self) -> bool:
        if _read_token(self):
            return True
        _json_response(self, 403, {"ok": False, "error": "bad or missing X-Auth-Token"})
        return False

    def do_GET(self):
        if not self._guard():
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/status":
            _json_response(self, 200, _status())
        elif parsed.path == "/metrics":
            query = urllib.parse.parse_qs(parsed.query)
            lines = int(query.get("lines", ["50"])[0])
            trial = _latest_trial()
            _json_response(self, 200, {"ok": True, "trial": str(trial) if trial else None,
                                       "lines": _tail(trial / "metrics.jsonl", lines) if trial else []})
        elif parsed.path == "/live_state":
            path = STATE.get("live_state_file")
            if not path or not Path(path).is_file():
                _json_response(self, 404, {"ok": False, "error": "no live state"})
                return
            body = Path(path).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/files":
            query = urllib.parse.parse_qs(parsed.query)
            trial = _latest_trial()
            name = query.get("name", ["metrics.jsonl"])[0]
            if not trial or "/" in name or ".." in name:
                _json_response(self, 400, {"ok": False, "error": "bad request"})
                return
            path = trial / name
            if not path.is_file():
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        if not self._guard():
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/start":
            _json_response(self, 200, _start(payload))
        elif self.path == "/stop":
            _json_response(self, 200, _stop())
        else:
            _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})


def main() -> int:
    global TOKEN, NATIVE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--token", default=os.environ.get("AGTC_WORKER_TOKEN", ""))
    parser.add_argument("--repo", default=os.environ.get("AGTC_REPO", ""),
                        help="path to AGTC repo; native dir is <repo>/算法/仿真训练环境/native")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or AGTC_WORKER_TOKEN is required")
    TOKEN = args.token
    if args.repo:
        NATIVE_DIR = Path(args.repo) / "算法" / "仿真训练环境" / "native"
    else:
        NATIVE_DIR = Path(__file__).resolve().parents[1]
    if not (NATIVE_DIR / "train" / "run_native_v1_trial.sh").is_file():
        parser.error(f"native dir does not look like the AGTC native stack: {NATIVE_DIR}")
    print(f"worker listening on http://{args.host}:{args.port}  native={NATIVE_DIR}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
