"""Local-only browser viewer with a single clocked environment owner."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import numpy as np

from .env import Config, FieldEnv


class Application:
    def __init__(self, cfg: Config, mode='search', seed=7, checkpoint=None,
                 *, env_factory=None, backend='lightweight', initially_paused=False,
                 initial_controller=None):
        self.env_factory = env_factory or FieldEnv
        self.backend = backend
        self.env = self.env_factory(cfg, mode)
        self.seed = seed
        try:
            self.obs, _ = self.env.reset(seed=seed)
        except BaseException:
            self.env.close()
            raise
        self.controller = 'manual' if initially_paused else 'demo'
        self.paused, self.speed = bool(initially_paused), 1.0
        self.action = np.zeros(4, np.float32)
        self.action_at = 0.0
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.times = deque(maxlen=200)
        self.policy_times = deque(maxlen=200)
        self.loop_times = deque(maxlen=200)
        self.runner = None
        self._demo = None
        self.checkpoint = checkpoint
        self.scene_cache = self.env.scene_payload()
        self.error = ''
        if initial_controller is not None:
            try:
                self.set_controller(initial_controller)
            except BaseException:
                self.env.close()
                raise

    def set_controller(self, name):
        if name not in ('manual', 'demo', 'policy'):
            raise ValueError('unknown controller')
        if name in ('demo', 'policy'):
            # Torch/SB3 are optional for the basic viewer, imported on demand.
            from .controllers import DemoController
            if name == 'demo' and self._demo is None:
                self._demo = DemoController()
            if name == 'policy' and self.runner is None:
                if not self.checkpoint:
                    raise ValueError('No checkpoint loaded. Start server with --checkpoint; demo is not a trained policy.')
                from .policy import PolicyRunner
                self.runner = PolicyRunner(self.env, self.checkpoint)
            if name == 'policy' and self.controller != name:
                self.runner.reset()
        self.controller = name

    def reset_environment(self, data):
        """Build and install a reset environment only after all inputs validate."""
        new_seed = int(data.get('seed', self.seed))
        new_config = asdict(self.env.cfg)
        if 'plants' in data:
            count = int(data['plants'])
            if not 0 <= count <= 3000:
                raise ValueError('plants out of range')
            new_config['plants'] = count
        for name in ('terrain_amplitude', 'terrain_wavelength_min', 'terrain_wavelength_max'):
            if name not in data:
                continue
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{name} must be a finite number')
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
            new_config[name] = value

        replacement = self.env_factory(Config(**new_config), data.get('mode', self.env.mode))
        try:
            new_obs, _ = replacement.reset(seed=new_seed)
        except BaseException:
            replacement.close()
            raise
        replacement.scene_id = self.env.scene_id + 1
        # Commit only after config, environment, and scene creation all succeed.
        previous = self.env
        self.env, self.obs, self.seed = replacement, new_obs, new_seed
        previous.close()
        self.error = ''
        self.scene_cache = self.env.scene_payload()
        self.times.clear()
        self.policy_times.clear()
        self.loop_times.clear()
        if self.runner:
            self.runner.reset()

    def load_checkpoint(self, checkpoint):
        """Install a saved policy without rebuilding a paused Gazebo world."""
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != '.zip':
            raise ValueError('checkpoint must be an existing .zip file')
        from .policy import PolicyRunner
        candidate = PolicyRunner(self.env, path)
        self.runner, self.checkpoint = candidate, path
        self.controller, self.paused = 'policy', True
        self.error = ''
        self.times.clear()
        self.policy_times.clear()
        self.loop_times.clear()

    def run(self):
        next_step = time.perf_counter()
        while not self.stop.is_set():
            with self.lock:
                if self.paused or self.env.done:
                    next_step = time.perf_counter()
                elif time.perf_counter() >= next_step:
                    t0 = time.perf_counter()
                    try:
                        if self.controller == 'demo':
                            self.set_controller('demo')
                            action = self._demo.action(self.env)
                        elif self.controller == 'policy':
                            action = self.runner.action(self.obs)
                        else:
                            action = self.action if time.perf_counter() - self.action_at < .5 else np.zeros(4)
                        t1 = time.perf_counter()
                        self.obs, _, _, _, _ = self.env.step(action)
                    except Exception as exc:
                        self.error = str(exc)
                        self.paused = True
                        continue
                    t2 = time.perf_counter()
                    self.policy_times.append((t1 - t0) * 1000)
                    self.times.append((t2 - t1) * 1000)
                    self.loop_times.append(t2)
                    interval = self.env.cfg.dt / self.speed
                    next_step = max(next_step + interval, t2 - interval)
            self.stop.wait(.001)

    def state(self):
        state = self.env.state_payload()
        hz = (len(self.loop_times) - 1) / max(1e-6, self.loop_times[-1] - self.loop_times[0]) if len(self.loop_times) > 1 else 0
        if self.paused or self.env.done or not self.loop_times or time.perf_counter() - self.loop_times[-1] > 1:
            hz = 0
        state.update(backend=self.backend, seed=self.seed, controller=self.controller, paused=self.paused, speed=self.speed, error=self.error,
                     policy_label=self.runner.label if self.controller == 'policy' and self.runner else 'rule demo (not trained)' if self.controller == 'demo' else 'manual',
                     performance={'loop_hz': hz, 'step_ms': float(np.mean(self.times)) if self.times else None,
                                  'policy_ms': float(np.mean(self.policy_times)) if self.policy_times else None})
        if self.runner is not None:
            state['policy_checkpoint'] = str(self.checkpoint)
            state['policy_training_steps'] = int(self.runner.model.num_timesteps)
        return state


def serve(cfg, *, mode='search', seed=7, port=8765, checkpoint=None,
          env_factory=None, backend='lightweight', initially_paused=False,
          initial_controller=None):
    app = Application(cfg, mode, seed, checkpoint, env_factory=env_factory,
                      backend=backend, initially_paused=initially_paused,
                      initial_controller=initial_controller)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            static_files = {'/': ('index.html', 'text/html; charset=utf-8'),
                            '/index.html': ('index.html', 'text/html; charset=utf-8'),
                            '/web/first_person.js': ('first_person.js', 'text/javascript; charset=utf-8'),
                            '/web/vendor/three.module.min.js': ('vendor/three.module.min.js', 'text/javascript; charset=utf-8'),
                            '/web/vendor/three.core.min.js': ('vendor/three.core.min.js', 'text/javascript; charset=utf-8')}
            if self.path in static_files:
                filename, content_type = static_files[self.path]
                data = (Path(__file__).parent / 'web' / filename).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path in ('/api/state', '/api/scene'):
                with app.lock:
                    data = app.state() if self.path == '/api/state' else app.scene_cache
                self.send_json(data)
            else:
                self.send_json({'error': 'not found'}, 404)

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 65536:
                    raise ValueError('request too large')
                data = json.loads(self.rfile.read(length) or b'{}')
                with app.lock:
                    if self.path == '/api/control':
                        a = np.asarray(data['action'], np.float32)
                        if a.shape != (4,) or not np.all(np.isfinite(a)):
                            raise ValueError('four finite controls required')
                        app.action = np.clip(a, -1, 1)
                        app.action_at = time.perf_counter()
                    elif self.path == '/api/reset':
                        app.reset_environment(data)
                    elif self.path == '/api/goal':
                        app.env.set_goal(data['goal'])
                        app.obs = app.env._observation()
                    elif self.path == '/api/settings':
                        if 'controller' in data:
                            app.set_controller(data['controller'])
                        if 'paused' in data:
                            app.paused = bool(data['paused'])
                        if 'speed' in data:
                            speed = float(data['speed'])
                            if not np.isfinite(speed) or not .1 <= speed <= 20:
                                raise ValueError('speed must be 0.1..20')
                            app.speed = speed
                        if 'mode' in data:
                            if data['mode'] not in ('search', 'goto'):
                                raise ValueError('unknown mode')
                            app.env.mode = data['mode']
                            app.env.done, app.env.reason = False, ''
                            app.env._hold = 0
                            app.obs = app.env._observation()
                    elif self.path == '/api/checkpoint':
                        app.load_checkpoint(data['checkpoint'])
                    else:
                        self.send_json({'error': 'not found'}, 404)
                        return
                    self.send_json({'ok': True})
            except (ValueError, TypeError, KeyError, ImportError, OverflowError) as e:
                self.send_json({'error': str(e)}, 400)
            except (RuntimeError, OSError, TimeoutError) as e:
                self.send_json({'error': str(e)}, 503)

    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except BaseException:
        app.env.close()
        raise
    worker = threading.Thread(target=app.run, daemon=True)
    worker.start()
    print(f'Field viewer [{backend}]: http://127.0.0.1:{port}  | Ctrl+C to stop', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        worker.join(timeout=5)
        app.env.close()
        server.server_close()
