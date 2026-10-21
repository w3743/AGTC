"""Read-only browser snapshots from the *same* environment used for training.

HTTP threads only read serialized caches. They never call or modify a Gym env.
Publishing is wall-clock throttled so accelerated rollouts do not flood the UI.
"""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time


WEB = Path(__file__).resolve().parents[1] / 'field_sim' / 'web'


class _MonitorHTTPServer(ThreadingHTTPServer):
    # Unix SO_REUSEADDR permits restart after TIME_WAIT but not two listeners.
    # Windows requires exclusive binding to prevent two viewers sharing a port.
    allow_reuse_address = os.name != 'nt'
    allow_reuse_port = False

    def server_bind(self):
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class TrainingMonitor:
    def __init__(self, port=8768):
        self.lock = threading.Lock()
        self.state_bytes = b'{"backend":"gazebo","read_only":true,"error":"Waiting for the first training observation"}'
        self.scene_bytes = b'{}'
        self._scene_key = None
        self._scene_serial = 0
        self._last_publish = 0.0
        self._intervals = deque(maxlen=100)
        self._rate_phase = None
        self.publish_ms = deque(maxlen=200)
        self.port = port
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def respond(self, data, content_type='application/json; charset=utf-8', status=200):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path in ('/api/state', '/api/scene'):
                    with monitor.lock:
                        data = monitor.state_bytes if self.path == '/api/state' else monitor.scene_bytes
                    self.respond(data)
                    return
                static = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/index.html': ('index.html', 'text/html; charset=utf-8'),
                          '/web/first_person.js': ('first_person.js', 'text/javascript; charset=utf-8'),
                          '/web/vendor/three.module.min.js': ('vendor/three.module.min.js', 'text/javascript; charset=utf-8'),
                          '/web/vendor/three.core.min.js': ('vendor/three.core.min.js', 'text/javascript; charset=utf-8')}
                if self.path in static:
                    filename, kind = static[self.path]
                    self.respond((WEB / filename).read_bytes(), kind)
                else:
                    self.respond(b'{"error":"not found"}', status=404)

            def do_POST(self):
                self.respond(b'{"error":"Training monitor is read-only"}', status=403)

        self.server = _MonitorHTTPServer(('127.0.0.1', int(port)), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._closed = False
        print(f'Training monitor: http://127.0.0.1:{port}/ (same environment, read-only)', flush=True)

    def publish(self, env, phase='training', steps=0, total=None, force=False,
                policy_ms=None, step_ms=None):
        now = time.perf_counter()
        key = (id(env), id(getattr(env, 'scene', None)), getattr(env, 'scene_id', None))
        changed = key != self._scene_key
        if not force and not changed and now-self._last_publish < .2:
            return
        begin = time.perf_counter()
        if changed:
            self._intervals.clear()
            scene = env.scene_payload()
            self._scene_serial += 1
            scene['scene_id'] = self._scene_serial
            scene_data = json.dumps(scene, ensure_ascii=False, allow_nan=False).encode('utf8')
        else:
            scene_data = None
        state = env.state_payload()
        state['scene_id'] = self._scene_serial
        state.update(backend=state.get('metrics', {}).get('backend', 'lightweight'),
                     read_only=True, controller='policy', speed=1.0,
                     paused=phase in ('optimizing', 'finished'), error='',
                     policy_label='training rollout; task competence not established',
                     training={'phase': phase, 'steps': int(steps), 'total': total,
                               'same_environment': True, 'monitor_only': True})
        if (self._rate_phase != phase or
                (self._intervals and int(steps) < self._intervals[-1][1])):
            self._intervals.clear()
        self._rate_phase = phase
        if phase == 'smoke':
            state['controller'] = 'fixed'
            state['policy_label'] = 'fixed actions; sensor and control self-check'
        self._intervals.append((now, int(steps)))
        first, last = self._intervals[0], self._intervals[-1]
        hz = max(0, last[1]-first[1]) / max(last[0]-first[0], 1e-9)
        state['performance'] = {'loop_hz': hz, 'step_ms': step_ms,
                                'policy_ms': policy_ms}
        state_data = json.dumps(state, ensure_ascii=False, allow_nan=False).encode('utf8')
        with self.lock:
            self.state_bytes = state_data
            if scene_data is not None:
                self.scene_bytes = scene_data
        self._scene_key = key
        self._last_publish = now
        self.publish_ms.append((time.perf_counter()-begin)*1000)

    def close(self):
        if not self._closed:
            self._closed = True
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=3)
