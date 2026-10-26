"""Poll a LAN worker's /live_state and write it locally for the existing Viser viewer.

Run on the Lead machine while the GPU machine runs `worker_agent.py`:

    python native/remote/pull_live_state.py \
        --url http://192.168.1.50:8790 --token <random> \
        --out C:/pioneer/wsl-native/live/native_live_state.json

Then keep `scripts/view_training_state_web.py --language zh --state-file <out>`
running locally; the viewer shows the remote training in real time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="http://<worker-ip>:8790")
    parser.add_argument("--token", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(args.url.rstrip("/") + "/live_state",
                                     headers={"X-Auth-Token": args.token})
    print(f"polling {args.url}/live_state -> {out}", flush=True)
    while True:
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                body = response.read()
            if body:
                temporary = out.with_name(out.name + ".tmp")
                temporary.write_bytes(body)
                os.replace(temporary, out)
        except Exception as exc:  # keep polling; the worker may be starting
            print(f"poll error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
