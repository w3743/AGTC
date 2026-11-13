"""Loopback HTTP integration against the existing depth probe; no device access."""
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
from probe_depth import DepthProbeCollector


class HttpContractTest(unittest.TestCase):
    def test_native_http_over_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = pathlib.Path(tmp) / ("http_fixture.exe" if sys.platform == "win32" else "http_fixture")
            sources = ("depth_encode.cpp", "depth_filter.cpp", "depth_store.cpp", "depth_producer.cpp", "depth_request.cpp",
                       "depth_http.cpp", "test_http_fixture.cpp")
            command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                       *(str(HERE / source) for source in sources), "-o", str(exe)]
            if sys.platform == "win32":
                command.append("-lws2_32")
            subprocess.run(command, check=True)
            process = subprocess.Popen([str(exe)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                port = int(process.stdout.readline())
                base = f"http://127.0.0.1:{port}"
                collector = DepthProbeCollector()
                try:
                    def fetch(path):
                        try:
                            return urllib.request.urlopen(base + path, timeout=5)
                        except urllib.error.HTTPError as exc:
                            return exc
                    start = time.perf_counter_ns()
                    with fetch("/depth.bin") as response:
                        body = response.read()
                        row = collector.ingest_response(status=response.status, headers=response.headers.items(), body=body,
                            host_send_monotonic_ns=start, host_receive_monotonic_ns=time.perf_counter_ns())
                        self.assertEqual(response.status, 200)
                        self.assertEqual(row["validation_errors"], [])
                        self.assertEqual(len(body), 131 * 240 * 2)
                    with fetch("/depth.bin?sequence=%31") as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.read(), body)
                    for path, expected in (("/depth.bin?sequence=1&sequence=1", 400),
                                           ("/depth.bin?sequence=%", 400),
                                           ("/depth.bin?sequence=2", 409),
                                           ("/other", 404),
                                           ("/depth.bin?sequence=%0d%0a", 400),
                                           ("/depth.bin?sequence=", 400)):
                        with fetch(path) as response:
                            self.assertEqual(response.status, expected, path)
                            self.assertFalse(json.loads(response.read())["ok"])
                    self.assertEqual(process.wait(timeout=5), 0)
                finally:
                    collector.close_archive()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                process.stdout.close()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
