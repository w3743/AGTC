import json
from pathlib import Path
import sys
import threading
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from hardware_video import HardwareVideo  # noqa: E402
from native_roi import NativeROIDepth, NativeDepthRecord, _GrayFrame, _pts_bucket  # noqa: E402


class FakeGst:
    def __init__(self):
        self.MapFlags = SimpleNamespace(READ=1)
        self.Format = FakeFormat()
        self.CLOCK_TIME_NONE = (1 << 64) - 1
        self.SECOND = 1_000_000_000


class FakeFormat:
    def __init__(self):
        self.access_threads = []

    @property
    def TIME(self):
        self.access_threads.append(threading.current_thread().name)
        return "TIME"


class FakeVideo:
    @staticmethod
    def buffer_get_video_meta(buffer):
        return SimpleNamespace(stride=[3840], offset=[0])


class FakeVideoInfo:
    def __init__(self):
        self.stride = [3840]
        self.offset = [0]
        self.caps = None

    def from_caps(self, caps):
        self.caps = caps
        return True


FakeVideo.VideoInfo = FakeVideoInfo


class FakeVideoWithoutMeta:
    VideoInfo = FakeVideoInfo

    @staticmethod
    def buffer_get_video_meta(buffer):
        return None


class FakeCapsStructure:
    def get_value(self, name):
        return {"width": 3840, "height": 2160}[name]

    def get_string(self, name):
        return "NV12" if name == "format" else None


class FakeCaps:
    def get_size(self):
        return 1

    def get_structure(self, index):
        return FakeCapsStructure()


class FakeBuffer:
    def __init__(self):
        self.map_info = SimpleNamespace(data=bytearray(3840 * 2160))
        self.unmapped = False

    def map(self, flags):
        return flags == 1, self.map_info

    def unmap(self, map_info):
        self.unmapped = True


class FakeSample:
    def __init__(self):
        self.caps = FakeCaps()
        self.buffer = FakeBuffer()

    def get_caps(self):
        return self.caps

    def get_buffer(self):
        return self.buffer


class FakePtsBuffer:
    def __init__(self, pts):
        self.pts = pts


class FakeSegment:
    def __init__(self, running_offset=0):
        self.running_offset = running_offset
        self.format = None

    def to_running_time(self, format_value, pts):
        self.format = format_value
        return pts - self.running_offset


class FakePtsSample:
    def __init__(self, pts, running_offset=0):
        self.buffer = FakePtsBuffer(pts)
        self.segment = FakeSegment(running_offset)

    def get_buffer(self):
        return self.buffer

    def get_segment(self):
        return self.segment


def write_calibration(path):
    k = np.array([[2000.0, 0.0, 1920.0],
                  [0.0, 2000.0, 1080.0],
                  [0.0, 0.0, 1.0]])
    k2 = np.array([[1985.0, 0.0, 1914.0],
                   [0.0, 2012.0, 1087.0],
                   [0.0, 0.0, 1.0]])
    p1 = np.column_stack((k, np.zeros(3)))
    p2 = p1.copy()
    p2[1, 3] = -200000.0
    np.savez(path, K1=k, D1=np.zeros((1, 5)), R1=np.eye(3), P1=p1,
             K2=k2, D2=np.array([[0.01, -0.02, 0.0002, -0.0001, 0.003]]),
             R2=np.eye(3), P2=p2,
             image_size=np.array([3840, 2160]))


class NativeROITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.calibration_path = Path(self.temp.name) / "calibration.npz"
        write_calibration(self.calibration_path)

    def tearDown(self):
        self.temp.cleanup()

    def make_native(self, canvas_wh=(640, 320), near_guard=None, rate=10):
        return NativeROIDepth(FakeGst(), FakeVideo(), self.calibration_path,
                              canvas_wh=canvas_wh, near_guard=near_guard, rate=rate)

    def test_compact_rectification_maps_and_upright_k_match_centered_crop(self):
        native = self.make_native()
        self.assertEqual(native.raw_roi_xywh, (1760, 760, 320, 640))
        self.assertEqual(native._maps_x[0].shape, (640, 320))
        self.assertEqual(native._maps_y[0].shape, (640, 320))
        self.assertEqual(native._maps_x[1].shape, (640, 320))
        self.assertEqual(native._maps_y[1].shape, (640, 320))
        self.assertEqual(native._valid_canvases[0].shape, (320, 640))
        self.assertEqual(native._valid_canvases[1].shape, (320, 640))
        self.assertTrue(native._valid_canvases[0].all())
        self.assertTrue(native._valid_canvases[1].all())
        self.assertFalse(np.array_equal(native._maps_x[0], native._maps_x[1]))
        self.assertFalse(np.array_equal(native._maps_y[0], native._maps_y[1]))
        np.testing.assert_allclose(native.K_canvas,
                                   [[2000.0, 0.0, 320.0],
                                    [0.0, 2000.0, 159.0],
                                    [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(native._matcher.fB_px_m, 200.0)
        self.assertEqual(native._matcher.disparity_offset_px, 0.0)
        self.assertEqual(native._roi_xywh, (64, 0, 576, 320))
        self.assertLess(native._maps_x[0].nbytes + native._maps_y[0].nbytes, 2_000_000)
        self.assertLess(native._maps_x[1].nbytes + native._maps_y[1].nbytes, 2_000_000)

        # With rectified pinhole maps, a constant luma frame stays constant;
        # this exercises the crop, remap, CCW orientation, and mask path.
        gray, valid, elapsed_ms = native._rectify_luma(np.full((2160, 3840), 73, np.uint8))
        self.assertEqual(gray.shape, (320, 640))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertTrue(np.all(gray == 73))
        self.assertTrue(valid.all())
        self.assertGreaterEqual(elapsed_ms, 0.0)
        source_pattern = ((np.arange(3840, dtype=np.uint16)[None, :] * 3
                           + np.arange(2160, dtype=np.uint16)[:, None] * 5) % 251).astype(np.uint8)
        left_rectified, _, _ = native._rectify_luma(source_pattern, eye=0)
        right_rectified, _, _ = native._rectify_luma(source_pattern, eye=1)
        self.assertFalse(np.array_equal(left_rectified, right_rectified))

    def test_offline_construction_without_gst_remains_available(self):
        native = NativeROIDepth(None, None, self.calibration_path, canvas_wh=(128, 64))

        gray, valid, _ = native._rectify_luma(np.full((2160, 3840), 73, np.uint8), eye=0)

        self.assertEqual(gray.shape, (64, 128))
        self.assertTrue(valid.all())
        with self.assertRaisesRegex(RuntimeError, "GStreamer bindings"):
            native.start(object(), pipeline_epoch=1)

    def test_sample_layout_falls_back_to_video_info_from_caps(self):
        native = NativeROIDepth(FakeGst(), FakeVideoWithoutMeta(), self.calibration_path,
                                canvas_wh=(128, 64))
        sample = FakeSample()

        gray, valid, _ = native._sample_to_canvas(sample, 0)

        self.assertEqual(gray.shape, (64, 128))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertTrue(valid.all())
        self.assertTrue(sample.buffer.unmapped)

    def test_running_pts_uses_main_thread_cached_gst_time_enum(self):
        gst = FakeGst()
        native = NativeROIDepth(gst, FakeVideo(), self.calibration_path,
                                canvas_wh=(128, 64))
        sample = FakePtsSample(1_234_567_890, running_offset=34_567_890)
        result = []
        worker = threading.Thread(target=lambda: result.append(native._sample_running_pts(sample)),
                                  name="fake-gst-capture-worker")

        worker.start()
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [1_200_000_000])
        self.assertEqual(sample.segment.format, "TIME")
        self.assertEqual(gst.Format.access_threads, [threading.main_thread().name])

    def test_records_are_immutable_history_is_bounded_and_metadata_is_snapshot(self):
        native = self.make_native(canvas_wh=(128, 64))
        native._pipeline_epoch = 7
        native._request_until = 1e20
        shape = (64, 64)
        base_depth = np.full(shape, 8.0, dtype=np.float64)
        base_depth[0, 0] = np.nan
        base_depth[0, 1] = 50.00000001
        base_depth[0, 2] = 4.99999999
        base_depth[0, 3] = 60.0
        base_valid = np.isfinite(base_depth)
        result = SimpleNamespace(
            depth_z_m=base_depth,
            valid=base_valid,
            K_roi=np.eye(3),
            roi_to_source=np.array([[1, 0, 64], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            compute_geometry={"matcher_input_shape_hw": [64, 128]},
            timings_ms={"matcher_compute_ms": 1.0},
        )
        left = _GrayFrame(100, 1000, np.zeros((64, 128), np.uint8),
                          np.ones((64, 128), bool), 2.0)
        right = _GrayFrame(101, 1001, np.zeros((64, 128), np.uint8),
                           np.ones((64, 128), bool), 2.5)
        for _ in range(13):
            native._publish(left, right, 1, result, generation=0)
        native._publish(left, right, 1, result, generation=-1)

        self.assertEqual(len(native._history), 12)
        record = native.get_record()
        self.assertIsInstance(record, NativeDepthRecord)
        self.assertEqual(record.sequence, 13)
        self.assertEqual(record.depth_z_m.shape, shape)
        self.assertFalse(record.valid[0, 0])
        self.assertFalse(record.valid[0, 1])
        self.assertFalse(record.valid[0, 2])
        self.assertFalse(record.valid[0, 3])
        self.assertTrue(np.isnan(record.depth_z_m[0, 0]))
        self.assertTrue(np.isnan(record.depth_z_m[0, 1]))
        self.assertTrue(np.isnan(record.depth_z_m[0, 2]))
        self.assertTrue(np.isnan(record.depth_z_m[0, 3]))
        self.assertGreater(len(record.preview_jpeg), 100)
        for array in (record.depth_z_m, record.valid):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)

        metadata = record.metadata()
        self.assertEqual(metadata["seq"], record.sequence)
        self.assertEqual(metadata["pipeline_epoch"], 7)
        self.assertEqual(metadata["left_pts_ns"], 100)
        self.assertEqual(metadata["right_pts_ns"], 101)
        self.assertEqual(metadata["clock_quality"], "unready")
        self.assertIsNone(metadata["source_mono_ns"])
        self.assertIsNone(metadata["extrinsics_id"])
        self.assertIsNone(metadata["physical_accuracy_claim_m"])
        self.assertEqual(metadata["canvas_origin_upright_xy"], [1016, 1888])
        metadata["seq"] = -1
        self.assertEqual(json.loads(record.metadata_json)["seq"], 13)
        self.assertIsNone(native.get_record(1))
        self.assertEqual(native.get_record(2).sequence, 2)

    def test_near_guard_filters_before_preview_and_retains_detached_metadata(self):
        keep = np.ones((64, 64), dtype=bool)
        keep[:, :8] = False
        callback_metadata = {"status": "applied", "available": np.bool_(True),
                             "near_sequence": np.int64(42)}
        observed = {}

        def guard(left_pts, right_pts, epoch, k_roi, shape):
            self.assertFalse(native._condition._is_owned())
            observed.update(left_pts=left_pts, right_pts=right_pts, epoch=epoch,
                            K=k_roi, shape=shape)
            return keep, callback_metadata

        native = self.make_native(canvas_wh=(128, 64), near_guard=guard)
        native._pipeline_epoch = 9
        native._request_until = 1e20
        depth = np.full((64, 64), 12.0, dtype=np.float64)
        valid = np.ones((64, 64), dtype=bool)
        result = SimpleNamespace(
            depth_z_m=depth,
            valid=valid,
            K_roi=np.array([[2000.0, 0.0, 4.0], [0.0, 2000.0, 12.0], [0.0, 0.0, 1.0]]),
            roi_to_source=np.array([[1, 0, 64], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            compute_geometry={},
            timings_ms={},
        )
        left = _GrayFrame(111, 1001, np.zeros((64, 128), np.uint8),
                          np.ones((64, 128), bool), 1.0)
        right = _GrayFrame(119, 1009, np.zeros((64, 128), np.uint8),
                           np.ones((64, 128), bool), 1.2)
        native._publish(left, right, 8, result, generation=0)

        record = native.get_record()
        self.assertEqual(observed["left_pts"], 111)
        self.assertEqual(observed["right_pts"], 119)
        self.assertEqual(observed["epoch"], 9)
        self.assertEqual(observed["shape"], (64, 64))
        np.testing.assert_array_equal(observed["K"], result.K_roi)
        self.assertFalse(record.valid[:, :8].any())
        self.assertTrue(record.valid[:, 8:].all())
        self.assertTrue(np.isnan(record.depth_z_m[:, :8]).all())
        self.assertTrue(np.all(record.depth_z_m[:, 8:] == 12.0))
        metadata = record.metadata()
        self.assertEqual(metadata["near_guard"], "applied")
        self.assertEqual(metadata["near_guard_verification"], "mask_applied")
        self.assertEqual(metadata["near_guard_metadata"]["near_sequence"], 42)
        self.assertEqual(metadata["left_pts_ns"], 111)
        self.assertEqual(metadata["right_pts_ns"], 119)
        keep[:, :] = True
        callback_metadata["near_sequence"] = 999
        self.assertFalse(record.valid[:, :8].any())
        self.assertEqual(record.metadata()["near_guard_metadata"]["near_sequence"], 42)

    def test_near_guard_error_fails_closed_for_the_whole_frame(self):
        def failing_guard(*args):
            raise RuntimeError("near depth history unavailable")

        native = self.make_native(canvas_wh=(128, 64), near_guard=failing_guard)
        native._pipeline_epoch = 3
        native._request_until = 1e20
        result = SimpleNamespace(
            depth_z_m=np.full((64, 64), 10.0, dtype=np.float32),
            valid=np.ones((64, 64), dtype=bool),
            K_roi=np.eye(3),
            roi_to_source=np.eye(3),
            compute_geometry={},
            timings_ms={},
        )
        left = _GrayFrame(10, 100, np.zeros((64, 128), np.uint8), np.ones((64, 128), bool), 1.0)
        right = _GrayFrame(11, 101, np.zeros((64, 128), np.uint8), np.ones((64, 128), bool), 1.0)
        native._publish(left, right, 1, result, generation=0)
        record = native.get_record()
        self.assertFalse(record.valid.any())
        self.assertTrue(np.isnan(record.depth_z_m).all())
        metadata = record.metadata()
        self.assertEqual(metadata["near_guard"], "error")
        self.assertEqual(metadata["near_guard_verification"], "unverified")
        self.assertEqual(metadata["near_guard_metadata"]["error_type"], "RuntimeError")
        self.assertIn("history unavailable", metadata["near_guard_metadata"]["error"])

    def test_missing_near_guard_is_explicitly_unavailable(self):
        native = self.make_native(canvas_wh=(128, 64))
        native._pipeline_epoch = 3
        native._request_until = 1e20
        result = SimpleNamespace(
            depth_z_m=np.full((64, 64), 10.0, dtype=np.float32),
            valid=np.ones((64, 64), dtype=bool),
            K_roi=np.eye(3),
            roi_to_source=np.eye(3),
            compute_geometry={},
            timings_ms={},
        )
        left = _GrayFrame(10, 100, np.zeros((64, 128), np.uint8), np.ones((64, 128), bool), 1.0)
        right = _GrayFrame(11, 101, np.zeros((64, 128), np.uint8), np.ones((64, 128), bool), 1.0)
        native._publish(left, right, 1, result, generation=0)
        record = native.get_record()
        self.assertTrue(record.valid.all())
        self.assertEqual(record.metadata()["near_guard"], "not_available")
        self.assertEqual(record.metadata()["near_guard_verification"], "unverified")
        self.assertEqual(record.metadata()["near_guard_metadata"]["status"], "not_available")

    def test_guard_callback_can_report_missing_near_history_as_unverified(self):
        def no_near_record(*args):
            return np.zeros((64, 64), dtype=bool), {
                "status": "unavailable", "available": False, "reason": "no within-40ms near frame"
            }

        native = self.make_native(canvas_wh=(128, 64), near_guard=no_near_record)
        native._pipeline_epoch = 3
        native._request_until = 1e20
        result = SimpleNamespace(
            depth_z_m=np.full((64, 64), 10.0, dtype=np.float32),
            valid=np.ones((64, 64), dtype=bool),
            K_roi=np.eye(3), roi_to_source=np.eye(3), compute_geometry={}, timings_ms={},
        )
        left = _GrayFrame(10, 100, np.zeros((64, 128), np.uint8), np.ones((64, 128), bool), 1.0)
        right = _GrayFrame(11, 101, np.zeros((64, 128), np.uint8), np.ones((64, 128), bool), 1.0)
        native._publish(left, right, 1, result, generation=0)
        metadata = native.get_record().metadata()
        self.assertFalse(native.get_record().valid.any())
        self.assertEqual(metadata["near_guard"], "unverified")
        self.assertEqual(metadata["near_guard_verification"], "unverified")
        self.assertEqual(metadata["near_guard_metadata"]["reason"], "no within-40ms near frame")

    def test_pts_pair_selector_uses_latest_unused_within_tolerance(self):
        native = self.make_native(canvas_wh=(128, 64))
        fresh_ns = time.monotonic_ns()

        def frame(pts, arrival=fresh_ns):
            return _GrayFrame(pts, arrival, np.zeros((64, 128), np.uint8),
                              np.ones((64, 128), bool), 1.0)

        stale = fresh_ns - 1_000_000_000
        native._queues[0].extend([frame(100, stale), frame(200), frame(300)])
        native._queues[1].extend([frame(104, stale), frame(195), frame(306)])
        pair = native._select_pair_locked()
        self.assertEqual((pair[0].pts_ns, pair[1].pts_ns, pair[2]), (300, 306, 6))
        self.assertIsNone(native._select_pair_locked())

    def test_raw_pts_buckets_limit_each_eye_to_one_remap_per_interval(self):
        native = self.make_native(canvas_wh=(128, 64), rate=10)
        native._request_until = 1e20
        span = 100_000_000
        pts = 50 * span + 20_000_000
        self.assertEqual(_pts_bucket(pts, 10), 50)
        self.assertEqual(native._claim_remap_bucket(0, pts, 0, 0), 50)
        self.assertIsNone(native._claim_remap_bucket(0, pts + 30_000_000, 0, 0))
        self.assertEqual(native._claim_remap_bucket(1, pts + 50_000_000, 0, 0), 50)
        self.assertEqual(native._claim_remap_bucket(0, pts + span, 0, 0), 51)
        self.assertIsNone(native._claim_remap_bucket(0, pts + 1_000_000, 0, 0))
        self.assertIsNone(native._claim_remap_bucket(0, pts + 2 * span, 1, 0))
        with self.assertRaises(ValueError):
            _pts_bucket(-1, 10)

    def test_capture_mapping_reads_one_luma_plane_and_unmaps_buffer(self):
        native = self.make_native(canvas_wh=(128, 64))
        sample = FakeSample()
        gray, valid, _ = native._sample_to_canvas(sample, eye=1)
        self.assertEqual(gray.shape, (64, 128))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertTrue(valid.all())
        self.assertTrue(sample.buffer.unmapped)

    def test_request_extends_demand_without_sharing_external_stop_state(self):
        native = self.make_native(canvas_wh=(128, 64))
        first = native.request()["request_until_mono"]
        second = native.request()["request_until_mono"]
        self.assertGreaterEqual(second, first)
        self.assertFalse(native._stop_event.is_set())
        native._last_remap_bucket = [12, 12]
        native.close()
        self.assertTrue(native._stop_event.is_set())
        self.assertEqual(native._last_remap_bucket, [None, None])


class HardwareVideoRawBranchTests(unittest.TestCase):
    def test_default_branch_remains_legacy_and_raw_hook_is_optional(self):
        video = HardwareVideo(FakeGst(), bitrate=5_000_000)
        old_branch = (
            "v4l2src device=/dev/video0 io-mode=dmabuf ! "
            "video/x-raw,format=NV12,width=3840,height=2160,framerate=30/1 ! "
            "queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            "mpph264enc bps=5000000 gop=15 level=51 max-pending=4 max-reenc=0 "
            "header-mode=each-idr ! h264parse name=encoded0 ! "
            "video/x-h264,stream-format=avc,alignment=au ! "
            "mp4mux fragment-duration=500 streamable=true trak-timescale=90000 ! "
            "appsink name=mp4_0 max-buffers=32 drop=false sync=false"
        )
        self.assertEqual(video.branch("/dev/video0", 0), old_branch)
        hooked = video.branch("/dev/video0", 1, raw_rate=10)
        self.assertIn("tee name=native1", hooked)
        self.assertIn("native1. ! queue max-size-buffers=4", hooked)
        self.assertIn("mpph264enc bps=5000000", hooked)
        self.assertIn("native1. ! queue max-size-buffers=1", hooked)
        self.assertIn("leaky=downstream ! appsink name=far1", hooked)
        self.assertIn("appsink name=far1 max-buffers=1 drop=true sync=false", hooked)
        self.assertIn("framerate=30/1", hooked)
        self.assertNotIn("videorate", hooked)
        self.assertNotIn("video/x-raw,framerate=10/1", hooked)
        with self.assertRaises(ValueError):
            video.branch("/dev/video0", 1, raw_rate=0)
        with self.assertRaises(ValueError):
            video.branch("/dev/video0", 1, raw_rate=10.0)


if __name__ == "__main__":
    unittest.main()
