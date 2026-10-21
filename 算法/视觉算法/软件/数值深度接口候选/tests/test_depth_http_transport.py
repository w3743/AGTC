"""Exercise the candidate's real HTTP handler with a camera-free live fixture."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from test_depth_producer_http import fake_live, metadata, depth_wire, server


class _HandlerCaptured(Exception):
    pass


class DepthHTTPTransportTests(unittest.TestCase):
    def setUp(self):
        self.live = fake_live()
        self.record = depth_wire.build_depth_record(
            bytes(131 * 240 * 2), metadata(seq=7))
        self.live.measure_history.append(self.record.history_entry())
        captured = []

        def capture_handler(address, handler):
            captured.append(handler)
            # Stop before serve() installs signals or starts any runtime loop.
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
            body = response.read()
            headers = dict(response.getheaders())
            self.assertEqual(int(headers["Content-Length"]), len(body))
            self.assertEqual(headers["Cache-Control"], "no-store")
            return response.status, headers, body
        finally:
            connection.close()

    def test_real_handler_sends_one_complete_record_and_exact_length(self):
        for path in ("/depth.bin", "/depth.bin?sequence=7"):
            status, headers, payload = self.request(path)
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "application/octet-stream")
            self.assertEqual(len(payload), 62880)
            self.assertEqual(payload, self.record.payload)
            self.assertEqual(headers["X-Depth-Metadata"], self.record.metadata_ascii)
            self.assertEqual(depth_wire.parse_depth_metadata(
                headers["X-Depth-Metadata"])["publish_mono_ns"], 2**53 + 7)
        self.assertGreater(self.live.depth_until, 0)

    def test_real_route_rejects_bad_sequence_and_unavailable_states(self):
        for query in ("", "abc", "-1", "1&sequence=2"):
            self.assertEqual(self.request("/depth.bin?sequence=" + query)[0], 400)
        self.assertEqual(self.request("/depth.bin?sequence=8")[0], 409)
        self.live.calibration_active = True
        self.assertEqual(self.request("/depth.bin")[0], 409)
        self.live.calibration_active = False
        self.live.depth_wire_geometry = None
        self.assertEqual(self.request("/depth.bin")[0], 503)
        self.live.depth_wire_geometry = object()
        self.live.measure_history.clear()
        self.assertEqual(self.request("/depth.bin")[0], 503)

    def test_old_pages_redirect_to_depth_view(self):
        for path in ("/measure", "/pointcloud", "/far"):
            status, headers, body = self.request(path)
            self.assertEqual(status, 302)
            self.assertEqual(headers["Location"], "/?view=depth")
            self.assertEqual(body, b"")

    def test_old_4k_pages_redirect_to_raw_view(self):
        for path in ("/stereo4k", "/measure4k"):
            status, headers, body = self.request(path)
            self.assertEqual(status, 302)
            self.assertEqual(headers["Location"], "/?view=raw")
            self.assertEqual(body, b"")

    def test_4k_stream_endpoints_are_gone_without_activating_depth(self):
        self.live.depth_until = 0.0
        for path in ("/video/0.mp4", "/video/1.mp4"):
            status, _, body = self.request(path)
            self.assertEqual(status, 410, path)
            self.assertIn("已移除".encode(), body)
        self.assertEqual(self.live.depth_until, 0.0)

    def test_removed_feature_routes_return_410_without_activating_depth(self):
        self.live.depth_until = 0.0
        for path in ("/measure.jpg", "/pointcloud.bin", "/range4k", "/far/point"):
            status, _, body = self.request(path)
            self.assertEqual(status, 410)
            self.assertIn("已移除".encode(), body)
        for path in ("/measure-point", "/measure-point4k"):
            status, _, body = self.request(path, method="POST", body=b"{}")
            self.assertEqual(status, 410)
            self.assertIn("已移除".encode(), body)
        self.assertEqual(self.live.depth_until, 0.0)
        self.assertIsNone(self.live.measure_input)
        self.assertIsNone(self.live.pointcloud_input)


if __name__ == "__main__":
    unittest.main()
