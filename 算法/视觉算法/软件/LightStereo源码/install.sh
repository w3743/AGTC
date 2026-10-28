#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
[ "$(id -u)" = 0 ] || { echo 'Run as root'; exit 1; }
[ "$ROOT" = /opt/rk-lightstereo ] || { echo 'Extract package to /opt/rk-lightstereo first'; exit 1; }
export PYTHONPATH="$ROOT/runtime/vendor"
python3 -B "$ROOT/verify.py" --system
sh "$ROOT/run.sh" self-test --runs 3
old_active=$(systemctl is-active rk-stereo.service 2>/dev/null || true)
cat > /etc/systemd/system/rk-lightstereo.service <<'UNIT'
[Unit]
Description=LightStereo RK3588 stereo depth
After=NetworkManager.service
Conflicts=rk-stereo.service
[Service]
Type=simple
WorkingDirectory=/opt/rk-lightstereo
ExecStart=/bin/sh /opt/rk-lightstereo/run.sh serve
Restart=on-failure
RestartSec=3
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
if systemctl start rk-lightstereo.service && python3 - <<'PY'
import json, time
from urllib.request import urlopen
deadline = time.monotonic()+25
while time.monotonic() < deadline:
    try:
        with urlopen('http://127.0.0.1:8088/depth.jpg', timeout=2) as r:
            image = r.read()
        with urlopen('http://127.0.0.1:8088/status', timeout=2) as r:
            status = json.load(r)
        if (image[:2] == b'\xff\xd8' and status.get('mode') == 'calibrated'
                and status.get('algorithm') == 'LightStereo FP16 NPU'
                and status.get('age_ms') is not None and status['age_ms'] < 2000):
            print('LightStereo HTTP and depth stream ready')
            break
    except Exception:
        pass
    time.sleep(.4)
else:
    raise SystemExit('LightStereo depth stream did not become ready')
PY
then
    echo 'Service startup checked'
else
    systemctl stop rk-lightstereo.service || true
    [ "$old_active" != active ] || systemctl start rk-stereo.service
    echo 'LightStereo start failed; prior running service restored'
    exit 1
fi
systemctl enable rk-lightstereo.service
if systemctl cat rk-stereo.service >/dev/null 2>&1; then
    systemctl disable rk-stereo.service
fi
echo 'LightStereo is active on port 8088. Existing camera captures and old service files are preserved.'
