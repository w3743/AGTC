import os
from collections import deque
from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

import depth_wire
import server


def metadata(seq=1, left_pts=100):
    return dict(
        schema=depth_wire.SCHEMA,
        session_id="test-session",
        pipeline_epoch=4,
        seq=seq,
        width=depth_wire.SOURCE_WIDTH,
        height=depth_wire.SOURCE_HEIGHT,
        encoding="16UC1",
        byte_order="little",
        unit="mm",
        convention="camera_z",
        frame_id="left_rectified_optical",
        K=[147.487193831, 0, 65.766880989, 0, 147.487193831, 121.086791039, 0, 0, 1],
        calibration_id="fixture-calibration",
        extrinsics_id=None,
        preprocessing_id="fixture-fast-profile",
        min_mm=300,
        max_mm=5000,
        left_pts_ns=left_pts,
        right_pts_ns=left_pts + 1,
        publish_mono_ns=2**53 + seq,
        source_mono_ns=None,
        clock_mapping_id=None,
        clock_quality=None,
        monotonic_clock_id="test-boot:CLOCK_MONOTONIC",
        source_time_reference="left_pts",
        left_arrival_mono_ns=300,
        right_arrival_mono_ns=301,
    )


class FakeMetric:
    depth_factor = 1.0
    disparity_offset = 0.0


def fake_live():
    live = SimpleNamespace()
    live.publish_lock = threading.Lock()
    live.condition = threading.Condition()
    live.calibration_active = False
    live.calibration_generation = 3
    live.pipeline_epoch = 4
    live.metric = FakeMetric()
    live.seq = 0
    live.jpeg = None
    live.jpeg_fixed = None
    live.measure_history = deque(maxlen=180)
    live.measure_input = None
    live.pointcloud_input = None
    live.state = {}
    live.updated = None
    live.capture_running = True
    live.depth_wire_geometry = object()
    live.depth_wire_error = None
    live.depth_until = 0.0
    live.stopped = threading.Event()
    live.frames = [deque(maxlen=6), deque(maxlen=6)]
    live.capture_pipeline_stop = threading.Event()
    return live


class FakeHandler:
    def __init__(self):
        self.result = None

    def json_response(self, value, status=200):
        self.result = (status, value, None, None)
        return self.result

    def send(self, body, kind, headers=None):
        self.result = (200, None, body, (kind, headers or {}))
        return self.result


class DepthProducerTests(unittest.TestCase):
    def test_little_endian_range_and_source_precision(self):
        depth = np.array([[.3, 1.2346, 5.0, np.nan],
                          [.2999999999, 5.000000001, np.inf, -np.inf]], dtype=np.float64)
        payload, width, height = depth_wire.encode_depth_payload(depth)
        self.assertEqual((width, height), (4, 2))
        self.assertEqual(np.frombuffer(payload, dtype="<u2").tolist(),
                         [300, 1235, 5000, 0, 0, 0, 0, 0])
        self.assertEqual(payload[:6], bytes((0x2C, 0x01, 0xD3, 0x04, 0x88, 0x13)))
        with self.assertRaises(ValueError):
            depth_wire.encode_depth_payload(depth[:, ::-1])

    def test_record_is_immutable_and_large_ns_stays_integer(self):
        source = np.full((depth_wire.SOURCE_HEIGHT, depth_wire.SOURCE_WIDTH), .875, np.float32)
        payload, _, _ = depth_wire.encode_depth_payload(source)
        source[:] = 4.5
        row = metadata(seq=9)
        record = depth_wire.build_depth_record(payload, row)
        self.assertEqual(record.millimetres[0, 0], 875)
        self.assertFalse(record.millimetres.flags.writeable)
        self.assertEqual(record.metadata["publish_mono_ns"], 2**53 + 9)
        with self.assertRaises(ValueError):
            record.millimetres[0, 0] = 100

    def test_fast_k_comes_from_source_calibration_transform(self):
        metric = SimpleNamespace(raw_width=240, raw_height=135, map_scale=240 / 3840,
                                 crop=2, width=131, height=240, input_size=(640, 360),
                                 scale=.375, identifier=None, RANGE_M=(.3, 5.0),
                                 algorithm="Calibrated SGBM CPU (240x131)",
                                 num_disparities=64,
                                 matcher=SimpleNamespace(getBlockSize=lambda: 5))
        geometry = depth_wire.fast_geometry_descriptor(metric, APP / "calibration.npz")
        np.testing.assert_allclose(geometry["K"],
            [147.487193831, 0, 65.766880989, 0, 147.487193831, 121.086791039, 0, 0, 1],
            rtol=0, atol=1e-8)
        self.assertIsNone(geometry["extrinsics_id"])

    def test_multiwriter_sequence_metadata_and_payload_commit_atomically(self):
        live = fake_live()
        image = np.zeros((8, 8, 3), np.uint8)
        left = np.zeros((8, 8), np.uint8)
        barrier = threading.Barrier(5)

        def writer(index):
            barrier.wait()
            depth = np.full((240, 131), .7 + index / 10, np.float32)
            server.Live.publish(
                live, image, depth=depth, rectified_left=left,
                worker_generation=3, worker_pipeline_epoch=4, worker_metric=live.metric,
                depth_metadata_context=metadata(left_pts=100 + index),
                source_pts_ns=100 + index, mode="test")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(1, 5)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(live.seq, 4)
        self.assertEqual([entry[0] for entry in live.measure_history], [1, 2, 3, 4])
        for entry in live.measure_history:
            record = entry[3]
            self.assertEqual(record.seq, entry[0])
            self.assertEqual(record.metadata["seq"], entry[0])
            self.assertEqual(record.metadata["left_pts_ns"], entry[2])
            self.assertEqual(int(record.millimetres[0, 0]), 700 + (entry[2] - 100) * 100)
        self.assertIsNone(live.pointcloud_input)
        self.assertIsNone(live.measure_input)

    def test_concurrent_http_reader_never_mixes_published_frames(self):
        live = fake_live()
        image = np.zeros((8, 8, 3), np.uint8)
        left = np.zeros((8, 8), np.uint8)
        expected_by_pts = {1: 800}
        server.Live.publish(
            live, image, depth=np.full((240, 131), .8, np.float32), rectified_left=left,
            worker_generation=3, worker_pipeline_epoch=4, worker_metric=live.metric,
            depth_metadata_context=metadata(left_pts=1), source_pts_ns=1)
        gate = threading.Barrier(3)
        writers_done = threading.Event()
        observed = []
        failures = []
        data_lock = threading.Lock()

        def writer(offset):
            gate.wait()
            for value in range(offset, offset + 12):
                z = .7 + value / 100.0
                pts = 1000 + value
                with data_lock:
                    expected_by_pts[pts] = int(round(z * 1000))
                server.Live.publish(
                    live, image, depth=np.full((240, 131), z, np.float32), rectified_left=left,
                    worker_generation=3, worker_pipeline_epoch=4, worker_metric=live.metric,
                    depth_metadata_context=metadata(left_pts=pts), source_pts_ns=pts)
                time.sleep(.002)

        def reader():
            gate.wait()
            while not writers_done.is_set():
                handler = FakeHandler()
                server.serve_depth_bin(handler, live, {}, wait_timeout=.02)
                status, _, body, response = handler.result
                if status != 200:
                    failures.append((status, handler.result))
                    continue
                metadata_row = depth_wire.parse_depth_metadata(response[1]["X-Depth-Metadata"])
                pts = metadata_row["left_pts_ns"]
                values = np.frombuffer(body, dtype="<u2")
                with data_lock:
                    expected = expected_by_pts.get(pts)
                if expected is None or metadata_row["seq"] < 1 or not np.all(values == expected):
                    failures.append((metadata_row, int(values[0]) if len(values) else None, expected))
                else:
                    observed.append((metadata_row["seq"], pts, int(values[0])))
                time.sleep(.0005)

        reader_thread = threading.Thread(target=reader)
        writer_threads = [threading.Thread(target=writer, args=(offset,)) for offset in (1, 101)]
        reader_thread.start()
        for thread in writer_threads:
            thread.start()
        for thread in writer_threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        writers_done.set()
        reader_thread.join(5)
        self.assertFalse(reader_thread.is_alive())
        self.assertGreaterEqual(len(observed), 5)
        self.assertEqual(failures, [])

    def test_stale_generation_or_epoch_work_is_discarded(self):
        live = fake_live()
        image = np.zeros((8, 8, 3), np.uint8)
        depth = np.full((240, 131), .8, np.float32)
        left = np.zeros((8, 8), np.uint8)
        for kwargs in ({"worker_generation": 2, "worker_pipeline_epoch": 4},
                       {"worker_generation": 3, "worker_pipeline_epoch": 3}):
            server.Live.publish(live, image, depth=depth, rectified_left=left,
                                worker_metric=live.metric,
                                depth_metadata_context=metadata(), **kwargs)
        self.assertEqual(live.seq, 0)
        self.assertFalse(live.measure_history)

    def test_epoch_transition_rejects_late_capture_sample(self):
        live = fake_live()
        old_token = live.capture_pipeline_stop
        frame = (10, np.zeros((2, 2), np.uint8), 1.0, 0.1)
        self.assertTrue(server.append_capture_frame(live, 0, old_token, frame))
        server.invalidate_capture_pipeline(live)
        self.assertFalse(live.capture_running)
        self.assertEqual([len(rows) for rows in live.frames], [0, 0])
        self.assertFalse(server.append_capture_frame(live, 0, old_token, frame))
        new_token = threading.Event()
        server.activate_capture_pipeline(live, new_token)
        self.assertFalse(server.append_capture_frame(live, 0, old_token, frame))
        self.assertTrue(server.append_capture_frame(live, 0, new_token, frame))

    def test_http_statuses_return_same_immutable_record_without_refresh(self):
        live = fake_live()
        payload = bytes(depth_wire.SOURCE_WIDTH * depth_wire.SOURCE_HEIGHT * 2)
        record = depth_wire.build_depth_record(payload, metadata())
        live.measure_history.append(record.history_entry())
        before = record.metadata["publish_mono_ns"]
        handler = FakeHandler()
        server.serve_depth_bin(handler, live, {"sequence": ["bad"]})
        self.assertEqual(handler.result[0], 400)
        server.serve_depth_bin(handler, live, {"sequence": ["999"]})
        self.assertEqual(handler.result[0], 409)
        server.serve_depth_bin(handler, live, {})
        status, body_obj, body, headers = handler.result
        self.assertEqual(status, 200)
        self.assertEqual(body, payload)
        self.assertEqual(headers[0], "application/octet-stream")
        decoded = depth_wire.parse_depth_metadata(headers[1]["X-Depth-Metadata"])
        self.assertEqual(decoded["seq"], record.seq)
        self.assertEqual(decoded["publish_mono_ns"], before)
        live.measure_history.clear()
        server.serve_depth_bin(handler, live, {})
        self.assertEqual(handler.result[0], 503)


if __name__ == "__main__":
    unittest.main()
