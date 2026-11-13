"""Synthetic threaded capture -> real loopback HTTP -> depth probe integration."""
import concurrent.futures
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
from probe_depth import DepthProbeCollector
import depth_wire


class VisionServiceTest(unittest.TestCase):
    def test_real_http_and_capture_fault(self):
        exe = os.environ.get("AGTC_SERVICE_FIXTURE")
        if not exe:
            self.skipTest("Set AGTC_SERVICE_FIXTURE to the built test executable")
        with tempfile.TemporaryDirectory() as temp:
            calibration = pathlib.Path(temp) / "calibration.yml"
            source = HERE.parent / "app" / "calibration.npz"
            subprocess.run([sys.executable, "-B", str(HERE / "export_calibration.py"),
                            str(source), str(calibration)], check=True)
            process = subprocess.Popen([exe, str(calibration)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, bufsize=1)
            lines = queue.Queue()
            def read_lines():
                for line in process.stdout:
                    lines.put(line.strip())
            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            try:
                port = int(lines.get(timeout=15))
                base = f"http://127.0.0.1:{port}"
                def fetch(path="/depth.bin", headers=None):
                    request = urllib.request.Request(base + path, headers=headers or {})
                    try:
                        response = urllib.request.urlopen(request, timeout=5)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        return response.status, response.headers.items(), response.read()
                self.assertEqual(fetch("/depth.bin?sequence=bad")[0], 400)
                if os.environ.get("AGTC_EXPECT_PREVIEW") == "1":
                    self.assertEqual(fetch("/depth.jpg?scale=auto")[0], 400)
                    self.assertEqual(fetch("/depth.jpg")[0], 400)
                    for _ in range(10):
                        status, headers, image = fetch("/stereo.jpg")
                        if status == 200:
                            break
                        time.sleep(.05)
                    self.assertEqual(status, 200)
                    fields = dict(headers)
                    self.assertEqual(fields["Content-Type"], "image/jpeg")
                    self.assertEqual(fields["X-Stereo-Exposure-Sync"], "unverified")
                    self.assertEqual(int(fields["X-Stereo-Pair-Skew-Ns"]), abs(
                        int(fields["X-Stereo-Left-Pts-Ns"]) - int(fields["X-Stereo-Right-Pts-Ns"])))
                    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
                    self.assertEqual(decoded.shape, (360, 1280))
                    self.assertLessEqual(abs(int(decoded[180, 320]) - 42), 2)
                    self.assertLessEqual(abs(int(decoded[180, 960]) - 99), 2)
                    self.assertEqual(fetch("/stereo.jpg", {"Range": "bytes=0-3"})[0], 400)
                time.sleep(.3)
                process.stdin.write("state\n"); process.stdin.flush()
                self.assertEqual(lines.get(timeout=5), "empty")
                start = time.perf_counter_ns()
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    responses = list(executor.map(lambda _: fetch(), range(16)))
                collector = DepthProbeCollector()
                try:
                    for status, headers, body in responses:
                        self.assertEqual(status, 200)
                        row = collector.ingest_response(status=status, headers=headers, body=body,
                            host_send_monotonic_ns=start, host_receive_monotonic_ns=time.perf_counter_ns())
                        self.assertEqual(row["validation_errors"], [])
                finally:
                    collector.close_archive()
                if os.environ.get("AGTC_EXPECT_PREVIEW") == "1":
                    for _ in range(5):
                        status, headers, image = fetch("/depth.jpg?scale=fixed")
                        if status == 200:
                            break
                        time.sleep(.05)
                    self.assertEqual(status, 200)
                    fields = dict(headers)
                    self.assertEqual(fields["Content-Type"], "image/jpeg")
                    meta = depth_wire.parse_depth_metadata(fields["X-Depth-Metadata"])
                    self.assertEqual(int(fields["X-Depth-Sequence"]), meta["seq"])
                    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(decoded.shape, (240, 131, 3))
                    self.assertEqual(fetch("/depth.jpg?scale=auto")[0], 400)
                self.assertEqual(fetch("/depth.bin?sequence=bad")[0], 400)
                self.assertEqual(fetch("/depth.bin", {"Range": "bytes=0-3"})[0], 400)
                self.assertEqual(fetch("/not-found")[0], 404)
                time.sleep(5.3)
                process.stdin.write("state\n"); process.stdin.flush()
                idle_sequence = int(lines.get(timeout=5))
                time.sleep(.25)
                process.stdin.write("state\n"); process.stdin.flush()
                self.assertEqual(int(lines.get(timeout=5)), idle_sequence)
                process.stdin.write("pause\n"); process.stdin.flush()
                self.assertEqual(lines.get(timeout=8), "paused")
                time.sleep(1.7)
                self.assertEqual(fetch()[0], 503)
                if os.environ.get("AGTC_EXPECT_PREVIEW") == "1":
                    self.assertEqual(fetch("/stereo.jpg")[0], 503)
                process.stdin.write("fault\n"); process.stdin.flush()
                self.assertEqual(lines.get(timeout=8), "faulted")
                self.assertEqual(fetch()[0], 503)
                if os.environ.get("AGTC_EXPECT_PREVIEW") == "1":
                    self.assertEqual(fetch("/depth.jpg?scale=fixed")[0], 503)
                    self.assertEqual(fetch("/stereo.jpg")[0], 503)
                process.stdin.write("stop\n"); process.stdin.flush()
                self.assertEqual(process.wait(timeout=8), 0)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                process.stdin.close(); process.stdout.close(); process.stderr.close()


if __name__ == "__main__":
    unittest.main()
