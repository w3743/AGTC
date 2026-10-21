"""Demand-driven NPU depth preview, independent of the fast depth worker."""
from collections import deque
from pathlib import Path
import threading
import time
import cv2
from lightstereo_depth import MetricDepth


class NpuLive:
    def __init__(self, frame_provider, stopped):
        self.frame_provider, self.stopped = frame_provider, stopped
        self.condition = threading.Condition()
        self.until = 0
        self.sequence = 0
        self.jpeg = None
        self.updated = None
        self.state = dict(mode='idle', message='按需启用 NPU 精细深度', fps=0)
        self.thread = threading.Thread(target=self.run, name='npu-depth-preview', daemon=True)
        self.thread.start()

    def request(self):
        with self.condition:
            self.until = time.monotonic() + 5

    def snapshot(self):
        with self.condition:
            age = time.monotonic() - self.updated if self.updated is not None else None
            return dict(self.state, sequence=self.sequence,
                        age_ms=round(age*1000) if age is not None else None,
                        active=time.monotonic() < self.until)

    def run(self):
        engine = None
        times = deque(maxlen=20)
        last_stamp = None
        try:
            while not self.stopped.is_set():
                if time.monotonic() >= self.until:
                    if engine is not None:
                        engine.close()
                        engine = None
                    with self.condition:
                        self.state.update(mode='idle', fps=0)
                        self.jpeg = None
                    times.clear()
                    self.stopped.wait(.2)
                    continue
                try:
                    if engine is None:
                        with self.condition:
                            self.state.update(mode='loading', message='正在加载隔离 RKNN 会话')
                        engine = MetricDepth(Path(__file__).with_name('calibration.npz'), input_size=(640,360))
                    pair = self.frame_provider()
                    if pair is None or pair[0][0] == last_stamp:
                        self.stopped.wait(.02)
                        continue
                    last_stamp = pair[0][0]
                    # Captures own their immutable arrays, so inference can keep
                    # this snapshot while the capture thread publishes newer ones.
                    color, depth, details = engine.compute(pair[0][1], pair[1][1])
                    ok, encoded = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY,80])
                    if not ok:
                        raise RuntimeError('NPU preview JPEG failed')
                    now = time.monotonic()
                    times.append(now)
                    with self.condition:
                        self.sequence += 1
                        self.jpeg = encoded.tobytes()
                        self.updated = now
                        self.state = dict(details, mode='running', message='NPU 正反向一致性深度',
                                          fps=round((len(times)-1)/(times[-1]-times[0]),2)
                                          if len(times)>1 else 0,
                                          source_pts_ns=int(pair[0][0]))
                        self.condition.notify_all()
                except Exception as exc:
                    if engine is not None:
                        engine.close()
                        engine = None
                    with self.condition:
                        self.state.update(mode='error', message=str(exc), fps=0)
                        self.jpeg = None
                        self.condition.notify_all()
                    self.stopped.wait(1)
        finally:
            if engine is not None:
                engine.close()

    def close(self):
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=5)
