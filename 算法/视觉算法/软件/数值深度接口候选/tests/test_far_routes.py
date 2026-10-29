"""Removed far-depth views cannot activate any secondary producer."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from test_depth_producer_http import fake_live, server


class _HandlerCaptured(Exception):
    pass


class FarRouteTests(unittest.TestCase):
    def setUp(self):
        self.live = fake_live()
        captured = []

        def capture_handler(address, handler):
            captured.append(handler)
            raise _HandlerCaptured()

        args = SimpleNamespace(left="unused-left", right="unused-right",
                               host="127.0.0.1", port=0)
        with patch.object(server, "Live", return_value=self.live), \
                patch.object(server, "ThreadingHTTPServer", side_effect=capture_handler):
            with self.assertRaises(_HandlerCaptured):
                server.serve(args)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), captured[0])
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=lambda: self.httpd.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())

    def request(self, path, method="GET", body=None):
        connection = HTTPConnection(*self.httpd.server_address, timeout=3)
        try:
            connection.request(method, path, body=body)
            response = connection.getresponse()
            payload = response.read()
            return response.status, dict(response.getheaders()), payload
        finally:
            connection.close()

    def test_far_page_redirects_to_the_single_depth_view(self):
        status, headers, body = self.request("/far")
        self.assertEqual(status, 302)
        self.assertEqual(headers["Location"], "/?view=depth")
        self.assertEqual(body, b"")

    def test_hardware_monitor_does_not_start_depth(self):
        self.live.depth_until = 0.0
        self.live.hardware = lambda: {"npu": {"load_percent": [0, 0, 0]}}
        status, _, body = self.request("/hardware/status")
        self.assertEqual(status, 200)
        self.assertIn(b'"npu"', body)
        self.assertEqual(self.live.depth_until, 0.0)

    def test_far_and_npu_routes_are_gone_and_do_not_start_depth(self):
        self.live.depth_until = 0.0
        for path in ("/far.jpg", "/far/status", "/far/point", "/npu.jpg"):
            status, _, body = self.request(path)
            self.assertEqual(status, 410, path)
            self.assertIn("已移除".encode(), body)
        status, _, body = self.request("/far/point", method="POST", body=b"{}")
        self.assertEqual(status, 410)
        self.assertIn("已移除".encode(), body)
        self.assertEqual(self.live.depth_until, 0.0)
        self.assertIsNone(self.live.measure_input)


if __name__ == "__main__":
    unittest.main()
