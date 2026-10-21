"""Exercise the candidate's real HTTP handler with a camera-free live fixture."""
import gzip
from contextlib import redirect_stderr
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from test_depth_producer_http import fake_live, metadata, depth_wire, server
import probe_depth


class _HandlerCaptured(Exception):
    pass


class DepthHTTPTransportTests(unittest.TestCase):
    def setUp(self):
        self.live = fake_live()
        self.live.hardware = lambda: {"sampled_monotonic": time.monotonic(),
                                      "sample_age_ms": 0.0}
        current_metadata = metadata(seq=7)
        current_metadata['publish_mono_ns'] = time.monotonic_ns()
        self.record = depth_wire.build_depth_record(
            bytes(131 * 240 * 2), current_metadata)
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
        # Keep the large integer wire-precision assertion independent of the
        # host uptime used for the latest-cache freshness check.
        self.record = depth_wire.build_depth_record(bytes(62_880), metadata(seq=7))
        self.live.measure_history.clear()
        self.live.measure_history.append(self.record.history_entry())
        for path in ("/depth.bin", "/depth.bin?sequence=7"):
            with patch.object(server.time, 'monotonic_ns', return_value=2**53 + 1_000_000):
                status, headers, payload = self.request(path)
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "application/octet-stream")
            self.assertEqual(len(payload), 62880)
            self.assertEqual(payload, self.record.payload)
            self.assertEqual(headers["X-Depth-Metadata"], self.record.metadata_ascii)
            self.assertEqual(depth_wire.parse_depth_metadata(
                headers["X-Depth-Metadata"])["publish_mono_ns"], 2**53 + 7)
        self.assertGreater(self.live.depth_until, 0)

    def test_latest_waits_for_fresh_record_while_explicit_history_stays_immutable(self):
        old_metadata = metadata(seq=7)
        old_metadata['publish_mono_ns'] = time.monotonic_ns() - 10_000_000_000
        old_record = depth_wire.build_depth_record(bytes(62_880), old_metadata)
        self.live.measure_history.clear()
        self.live.measure_history.append(old_record.history_entry())

        def publish_after_wakeup():
            time.sleep(.04)
            fresh_metadata = metadata(seq=8)
            fresh_metadata['publish_mono_ns'] = time.monotonic_ns()
            fresh_record = depth_wire.build_depth_record(bytes(62_880), fresh_metadata)
            with self.live.condition:
                self.live.measure_history.append(fresh_record.history_entry())
                self.live.condition.notify_all()

        publisher = threading.Thread(target=publish_after_wakeup)
        publisher.start()
        try:
            status, headers, body = self.request('/depth.bin')
        finally:
            publisher.join(2)
        self.assertFalse(publisher.is_alive())
        self.assertEqual(status, 200)
        self.assertEqual(depth_wire.parse_depth_metadata(headers['X-Depth-Metadata'])['seq'], 8)
        status, headers, body = self.request('/depth.bin?sequence=7')
        self.assertEqual(status, 200)
        self.assertEqual(body, old_record.payload)
        self.assertEqual(headers['X-Depth-Metadata'], old_record.metadata_ascii)

    def test_probe_measures_sub_tick_rtt_when_the_platform_uptime_clock_is_coarse(self):
        # Windows Python 3.11 monotonic_ns uses GetTickCount64 (15.625 ms),
        # so a complete real HTTP exchange may fit inside one uptime tick.
        connection = HTTPConnection(*self.httpd.server_address, timeout=3)
        try:
            with patch.object(time, 'monotonic_ns', return_value=self.record.metadata['publish_mono_ns']):
                status, _, _, sent, received, complete, error = probe_depth._exchange(connection, '/depth.bin')
        finally:
            connection.close()
        self.assertEqual(status, 200)
        self.assertTrue(complete)
        self.assertIsNone(error)
        self.assertGreater(received, sent)

    def test_invalid_probe_configuration_cannot_truncate_original_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'original.json'
            archive = Path(directory) / 'original.frames.bin.gz'
            output.write_bytes(b'original report')
            archive.write_bytes(b'original raw archive')
            cases = [(0, .005, 'http://localhost'),
                     (float('nan'), .005, 'http://localhost'),
                     (float('inf'), .005, 'http://localhost'),
                     (1e308, .005, 'http://localhost'),
                     (1, float('nan'), 'http://localhost'),
                     (1, -.001, 'http://localhost'),
                     (1, .005, 'ftp://localhost'),
                     (1, .005, 'http://localhost:bad')]
            for duration, interval, url in cases:
                with self.subTest(duration=duration, interval=interval, url=url):
                    with self.assertRaises(ValueError):
                        probe_depth.run_probe(url, duration, interval, output, overwrite=True)
                    self.assertEqual(output.read_bytes(), b'original report')
                    self.assertEqual(archive.read_bytes(), b'original raw archive')

    def test_probe_rejects_url_userinfo_before_network_or_output_changes(self):
        urls = (
            'https://synthetic-user:synthetic-pass@example.invalid',
            'https://synthetic-user@example.invalid',
            'https://@example.invalid',
            'https://synthetic%2Duser@example.invalid',
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'original.json'
            archive = Path(directory) / 'original.frames.bin.gz'
            output.write_bytes(b'original report')
            archive.write_bytes(b'original raw archive')

            with patch.object(probe_depth, '_connection_for') as connection_factory, \
                    patch.object(probe_depth, '_exchange') as exchange, \
                    patch.object(probe_depth, 'DepthProbeCollector') as collector_factory:
                for url in urls:
                    with self.subTest(url=url):
                        with self.assertRaisesRegex(ValueError, 'user information'):
                            probe_depth.run_probe(
                                url, 1, .005, output, overwrite=True)
                        self.assertEqual(output.read_bytes(), b'original report')
                        self.assertEqual(archive.read_bytes(), b'original raw archive')

                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = probe_depth.main([
                        '--url', urls[0], '--output', str(output), '--overwrite'])

            self.assertEqual(exit_code, 2)
            self.assertNotIn('synthetic-user', stderr.getvalue())
            self.assertNotIn('synthetic-pass', stderr.getvalue())
            self.assertNotIn(urls[0], stderr.getvalue())
            self.assertEqual(output.read_bytes(), b'original report')
            self.assertEqual(archive.read_bytes(), b'original raw archive')
            connection_factory.assert_not_called()
            exchange.assert_not_called()
            collector_factory.assert_not_called()

    def test_latest_stale_timeout_is_warming_and_recovers_on_next_publish(self):
        stale = metadata(seq=7)
        stale['publish_mono_ns'] = time.monotonic_ns() - 10_000_000_000
        self.live.measure_history.clear()
        self.live.measure_history.append(depth_wire.build_depth_record(bytes(62_880), stale).history_entry())
        status, _, body = self.request('/depth.bin')
        self.assertEqual(status, 503)
        self.assertIn(b'"warming"', body)
        fresh = metadata(seq=8)
        fresh['publish_mono_ns'] = time.monotonic_ns()
        with self.live.condition:
            self.live.measure_history.append(depth_wire.build_depth_record(bytes(62_880), fresh).history_entry())
            self.live.condition.notify_all()
        status, headers, _ = self.request('/depth.bin')
        self.assertEqual(status, 200)
        self.assertEqual(depth_wire.parse_depth_metadata(headers['X-Depth-Metadata'])['seq'], 8)

    def test_probe_archives_one_frame_and_counts_an_exact_same_sequence_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.frames.bin.gz"
            collector = probe_depth.DepthProbeCollector(archive)
            for _ in range(2):
                status, headers, body = self.request("/depth.bin")
                tick = 10_000_000 + len(collector.requests) * 1_000_000
                row = collector.ingest_response(
                    status=status, headers=headers, body=body,
                    host_send_monotonic_ns=tick,
                    host_receive_monotonic_ns=tick + 100_000)
                self.assertEqual(row["metadata_ascii"], self.record.metadata_ascii)
            collector.close_archive()

            self.assertEqual(collector.unique_frames, 1)
            self.assertEqual(collector.duplicate_count, 1)
            self.assertEqual(collector.bad_response_count, 0)
            self.assertEqual([row["archive_frame_index"] for row in collector.requests], [0, 0])
            with gzip.open(archive, "rb") as stored:
                self.assertEqual(stored.read(), self.record.payload)

    def test_probe_cli_flow_records_clock_brackets_and_compact_archive_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture-report.json"
            report, exit_code = probe_depth.run_probe(
                "http://127.0.0.1:%d" % self.httpd.server_address[1],
                duration_s=0.025, interval_s=0.005, output_path=output)

            self.assertEqual(exit_code, 0)
            self.assertGreater(report["summary"]["http_request_count"], 0)
            self.assertEqual(report["summary"]["unique_frames"], 1)
            self.assertEqual(len(report["hardware_status"]), 2)
            bounds = report["clock_diagnostics"]["board_minus_host_offset_bounds_ns"]
            self.assertLessEqual(bounds["lower_ns"], bounds["upper_ns"])
            self.assertFalse(report["summary"]["clock_mapping_validated"])
            self.assertFalse(report["summary"]["legacy_ready"])
            self.assertEqual(report["requests"][0]["metadata_ascii"], self.record.metadata_ascii)
            self.assertTrue(all(row["metadata_ascii"] is None
                                for row in report["requests"] if row["is_duplicate"]))
            archive = output.with_name("fixture-report.frames.bin.gz")
            self.assertTrue(archive.is_file())
            with gzip.open(archive, "rb") as stored:
                self.assertEqual(stored.read(), self.record.payload)

    def test_probe_rejects_corrupt_payload_size_and_illegal_nonzero_depth(self):
        collector = probe_depth.DepthProbeCollector()
        original = server.serve_depth_bin
        corruptions = (bytes(62_879), bytes((0xC8, 0x00)) + bytes(62_878))
        for body in corruptions:
            with patch.object(server, "serve_depth_bin",
                              side_effect=lambda handler, live, query, body=body:
                              handler.send(body, "application/octet-stream",
                                           headers={"X-Depth-Metadata": self.record.metadata_ascii})):
                status, headers, payload = self.request("/depth.bin")
            tick = 20_000_000 + len(collector.requests) * 1_000_000
            collector.ingest_response(
                status=status, headers=headers, body=payload,
                host_send_monotonic_ns=tick,
                host_receive_monotonic_ns=tick + 100_000)
        collector.close_archive()

        self.assertIs(server.serve_depth_bin, original)
        self.assertEqual(collector.bad_response_count, 2)
        self.assertEqual(collector.unique_frames, 0)
        self.assertTrue(any("exactly 62880" in error["message"]
                            for error in collector.errors))
        self.assertTrue(any("300..5000" in error["message"]
                            for error in collector.errors))

    def test_probe_counts_consumer_gaps_and_records_sequence_and_clock_rollbacks(self):
        collector = probe_depth.DepthProbeCollector()
        records = []
        for seq, pts in ((7, 100), (9, 300), (8, 200)):
            row = metadata(seq=seq, left_pts=pts)
            row["publish_mono_ns"] = 10_000 + seq
            record = depth_wire.build_depth_record(bytes(62_880), row)
            self.live.measure_history.append(record.history_entry())
            records.append(record)
        final_row = metadata(seq=10, left_pts=400)
        final_row["publish_mono_ns"] = 10_010
        final_record = depth_wire.build_depth_record(bytes(62_880), final_row)
        self.live.measure_history.append(final_record.history_entry())
        records.append(final_record)
        for record in records:
            status, headers, body = self.request("/depth.bin?sequence=" + str(record.seq))
            tick = 30_000_000 + len(collector.requests) * 1_000_000
            collector.ingest_response(
                status=status, headers=headers, body=body,
                host_send_monotonic_ns=tick,
                host_receive_monotonic_ns=tick + 100_000)
        collector.close_archive()

        self.assertEqual(collector.consumer_unobserved_sequence_count, 1)
        self.assertEqual(collector.sequence_gap_events, 1)
        self.assertEqual(collector.requests[1]["consumer_unobserved_sequence_count_before"], 1)
        codes = {error["code"] for error in collector.errors}
        self.assertIn("sequence_rollback", codes)
        self.assertIn("source_pts_rollback", codes)
        self.assertIn("publish_time_rollback", codes)

    def test_probe_detects_a_previously_seen_frame_returning_after_a_newer_frame(self):
        collector = probe_depth.DepthProbeCollector()
        for seq in (7, 8, 7, 9):
            if seq != 7 or not any(entry[0] == 7 for entry in self.live.measure_history):
                record = depth_wire.build_depth_record(bytes(62_880), metadata(seq=seq))
                self.live.measure_history.append(record.history_entry())
            status, headers, body = self.request("/depth.bin?sequence=" + str(seq))
            tick = 35_000_000 + len(collector.requests) * 1_000_000
            collector.ingest_response(status=status, headers=headers, body=body,
                                      host_send_monotonic_ns=tick,
                                      host_receive_monotonic_ns=tick + 100_000)
        collector.close_archive()

        self.assertEqual(collector.duplicate_count, 1)
        self.assertEqual(collector.sequence_gap_events, 0)
        self.assertEqual(collector.consumer_unobserved_sequence_count, 0)
        self.assertEqual(collector.requests[2]["frame_key"], ["test-session", 4, 7])
        self.assertTrue(collector.requests[2]["is_duplicate"])
        self.assertTrue(any(error["code"] == "sequence_rollback"
                            for error in collector.requests[2]["temporal_errors"]))

    def test_probe_keeps_negative_arrival_to_publish_values_and_records_them(self):
        row = metadata(seq=10, left_pts=100)
        row["publish_mono_ns"] = 10_000
        row["left_arrival_mono_ns"] = 11_000
        row["right_arrival_mono_ns"] = 12_000
        record = depth_wire.build_depth_record(bytes(62_880), row)
        self.live.measure_history.append(record.history_entry())
        status, headers, body = self.request("/depth.bin?sequence=10")
        collector = probe_depth.DepthProbeCollector()
        collector.ingest_response(status=status, headers=headers, body=body,
                                  host_send_monotonic_ns=50_000_000,
                                  host_receive_monotonic_ns=50_100_000)
        collector.close_archive()

        observed = collector.requests[0]
        self.assertEqual(observed["left_arrival_to_publish_ms"], -0.001)
        self.assertEqual(observed["right_arrival_to_publish_ms"], -0.002)
        self.assertIn("publish_before_left_arrival",
                      {error["code"] for error in observed["temporal_errors"]})
        self.assertIn("publish_before_right_arrival",
                      {error["code"] for error in observed["temporal_errors"]})

    def test_probe_requires_exact_raw_metadata_header_for_a_repeated_key(self):
        collector = probe_depth.DepthProbeCollector()
        status, headers, body = self.request("/depth.bin")
        tick = 40_000_000
        collector.ingest_response(status=status, headers=headers, body=body,
                                  host_send_monotonic_ns=tick,
                                  host_receive_monotonic_ns=tick + 100_000)
        changed_header = self.record.metadata_ascii.replace(",", ", ", 1)
        with patch.object(server, "serve_depth_bin",
                          side_effect=lambda handler, live, query:
                          handler.send(self.record.payload, "application/octet-stream",
                                       headers={"X-Depth-Metadata": changed_header})):
            status, headers, body = self.request("/depth.bin")
        collector.ingest_response(status=status, headers=headers, body=body,
                                  host_send_monotonic_ns=tick + 1_000_000,
                                  host_receive_monotonic_ns=tick + 1_100_000)
        collector.close_archive()

        self.assertEqual(collector.duplicate_mismatch_count, 1)
        self.assertEqual(collector.bad_response_count, 1)
        self.assertEqual(collector.requests[0]["metadata_ascii_sha256"],
                         probe_depth._sha256(self.record.metadata_ascii.encode("ascii")))
        self.assertNotEqual(collector.requests[0]["metadata_ascii_sha256"],
                            collector.requests[1]["metadata_ascii_sha256"])

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
