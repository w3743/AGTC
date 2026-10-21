import sys
import unittest
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from depth_adapter import (  # noqa: E402
    DepthAdapter,
    DepthAdapterError,
    MonotonicClockMapping,
    OutOfOrderFrameError,
    SourceStateEvidence,
    decode_frame,
)
from depth_wire import build_depth_record, parse_depth_metadata  # noqa: E402


SOURCE_K = [147.487194, 0.0, 65.766881,
            0.0, 147.487194, 121.086791,
            0.0, 0.0, 1.0]


def make_adapter(k=SOURCE_K):
    return DepthAdapter(
        expected_calibration_id="cal-fixture",
        expected_preprocessing_id="prep-fixture",
        expected_extrinsics_id="ext-fixture",
        expected_source_k=k,
    )


def make_metadata(**updates):
    values = {
        "schema": "depth-z-u16le-v1",
        "session_id": "session-fixture",
        "pipeline_epoch": 1,
        "seq": 1,
        "width": 131,
        "height": 240,
        "encoding": "16UC1",
        "byte_order": "little",
        "unit": "mm",
        "convention": "camera_z",
        "frame_id": "left_rectified_optical",
        "K": list(SOURCE_K),
        "calibration_id": "cal-fixture",
        "extrinsics_id": "ext-fixture",
        "preprocessing_id": "prep-fixture",
        "min_mm": 300,
        "max_mm": 5000,
        "left_pts_ns": 1_000_000_000,
        "right_pts_ns": 1_000_000_000,
        "publish_mono_ns": 1_100_000_000,
        "source_mono_ns": 1_000_000_000,
        "clock_mapping_id": "pts-map-fixture",
        "clock_quality": "verified",
        "monotonic_clock_id": "mono-A",
        "source_time_reference": "left_pts",
    }
    values.update(updates)
    return values


def make_payload(fill=1000):
    image = np.full((240, 131), fill, dtype="<u2", order="C")
    return image, image.tobytes(order="C")


def state_for(metadata):
    return SourceStateEvidence(
        source_mono_ns=metadata["source_mono_ns"],
        clock_id=metadata["monotonic_clock_id"],
        pipeline_epoch=metadata["pipeline_epoch"],
        pose_available=True,
        measured_gimbal_available=True,
        aligned_to_source_time=True,
    )


def decode(adapter, metadata, payload, **kwargs):
    source_ns = metadata.get("source_mono_ns")
    consume_ns = (source_ns + 250_000_000
                  if isinstance(source_ns, int) and not isinstance(source_ns, bool)
                  else 1_250_000_000)
    args = dict(
        consume_mono_ns=consume_ns,
        consume_clock_id=metadata.get("monotonic_clock_id") or "mono-A",
        consume_pipeline_epoch=metadata["pipeline_epoch"],
        source_state=state_for(metadata) if metadata.get("source_mono_ns") is not None
        and metadata.get("monotonic_clock_id") else None,
    )
    args.update(kwargs)
    return decode_frame(metadata, payload, adapter=adapter, **args)


class DepthAdapterTests(unittest.TestCase):
    def test_range_endpoints_and_invalid_values_are_not_clipped(self):
        image, payload = make_payload(1000)
        image[0, :6] = np.array([0, 299, 300, 1000, 5000, 5001], dtype="<u2")
        payload = image.tobytes(order="C")
        result = decode(make_adapter(), make_metadata(), payload)

        np.testing.assert_array_equal(
            result.source_valid[0, :6], [False, False, True, True, True, False]
        )
        self.assertTrue(result.payload_parseable)
        self.assertEqual(result.debug_tensor.dtype, np.float32)
        self.assertEqual(result.debug_tensor.shape, (1, 160, 88))
        self.assertTrue(np.isfinite(result.debug_tensor).all())
        self.assertTrue(np.all(result.debug_tensor >= 0.0))
        self.assertTrue(np.all(result.debug_tensor <= 1.0))
        self.assertAlmostEqual(float(result.debug_tensor[0, 80, 44]), 0.2)

    def test_mask_is_applied_before_normalization(self):
        image, payload = make_payload(1000)
        mask = np.ones((240, 131), dtype=np.bool_)
        mask[121, 66] = False
        result = decode(make_adapter(), make_metadata(), payload, source_mask=mask)
        self.assertFalse(result.source_valid[121, 66])
        self.assertFalse(result.target_valid[80, 44])
        self.assertEqual(float(result.debug_tensor[0, 80, 44]), 0.0)

    def test_row_major_little_endian_marker_and_producer_record_roundtrip(self):
        image, payload = make_payload(1000)
        image[7, 19] = 0x1234
        payload = image.tobytes(order="C")
        self.assertEqual(payload[(7 * 131 + 19) * 2:(7 * 131 + 19) * 2 + 2], b"\x34\x12")

        # Exercise the producer's immutable record/ASCII-JSON boundary, then
        # pass its decoded dict through the adapter's documented dict API.
        record = build_depth_record(payload, make_metadata())
        parsed_header = parse_depth_metadata(record.metadata_ascii)
        result = decode(make_adapter(), parsed_header, record.payload)
        self.assertEqual(int(result.source_mm[7, 19]), 0x1234)
        self.assertEqual(result.metadata["seq"], record.seq)

    def test_rejects_wire_profile_length_and_layout_mismatches(self):
        _, payload = make_payload()
        bad_cases = [
            ("schema", "depth-z-u16le-v999"),
            ("encoding", "32FC1"),
            ("byte_order", "big"),
            ("width", 240),
            ("row_stride_bytes", 264),
            ("shape", [131, 240]),
        ]
        for key, value in bad_cases:
            with self.subTest(key=key):
                with self.assertRaises((DepthAdapterError, ValueError)):
                    decode(make_adapter(), make_metadata(**{key: value}), payload)
        with self.assertRaises((DepthAdapterError, ValueError)):
            decode(make_adapter(), make_metadata(), payload[:-2])
        with self.assertRaises((DepthAdapterError, ValueError)):
            decode(make_adapter(), make_metadata(), payload + b"\0\0")

    def test_rejects_unsupported_or_mismatched_intrinsics(self):
        _, payload = make_payload()
        for invalid_focal in (np.nan, np.inf, 0.0):
            with self.subTest(invalid_focal=invalid_focal):
                invalid_k = list(SOURCE_K)
                invalid_k[0] = invalid_focal
                with self.assertRaises((DepthAdapterError, ValueError)):
                    decode(make_adapter(), make_metadata(K=invalid_k), payload)

        skewed = list(SOURCE_K)
        skewed[1] = 0.01
        with self.assertRaises((DepthAdapterError, ValueError)):
            decode(make_adapter(skewed), make_metadata(K=skewed), payload)

        bool_k = list(SOURCE_K)
        bool_k[0] = True
        with self.assertRaises((DepthAdapterError, ValueError)):
            decode(make_adapter(), make_metadata(K=bool_k), payload)

        changed = list(SOURCE_K)
        changed[2] += 0.25
        result = decode(make_adapter(), make_metadata(K=changed), payload)
        self.assertFalse(result.legacy_ready)
        self.assertIn("K_mismatch", result.readiness_missing)
        self.assertIsNone(result.legacy_tensor)

    def test_nearest_geometry_preserves_same_depth_mask_and_crop_edges(self):
        image, payload = make_payload(1000)
        image[121, 66] = 300
        image[121, 65] = 5000
        payload = image.tobytes(order="C")
        result = decode(make_adapter(), make_metadata(), payload)
        self.assertAlmostEqual(float(result.target_z_m[80, 44]), 0.3)
        self.assertTrue(result.target_valid[80, 44])
        # The final target row maps beyond the 240-row source after nearest
        # sampling. It stays invalid; no border fill or neighbor substitution.
        self.assertFalse(result.target_valid[159, 44])
        self.assertEqual(float(result.debug_tensor[0, 159, 44]), 0.0)

    def test_nearest_ties_round_half_up_and_honor_source_mask(self):
        target_f = 98.3247958874
        tie_k = [target_f, 0.0, 0.5,
                 0.0, target_f, 0.5,
                 0.0, 0.0, 1.0]
        image, payload = make_payload(300)
        image[1, 1] = 5000
        payload = image.tobytes(order="C")
        result = decode(make_adapter(tie_k), make_metadata(K=tie_k), payload)
        self.assertAlmostEqual(float(result.target_z_m[80, 44]), 5.0)

        mask = np.ones((240, 131), dtype=np.bool_)
        mask[1, 1] = False
        masked = decode(make_adapter(tie_k), make_metadata(K=tie_k), payload,
                        source_mask=mask)
        self.assertFalse(masked.target_valid[80, 44])
        self.assertEqual(float(masked.debug_tensor[0, 80, 44]), 0.0)

    def test_age_uses_source_time_not_publish_time(self):
        _, payload = make_payload()
        metadata = make_metadata()
        result = decode(
            make_adapter(), metadata, payload,
            consume_mono_ns=1_250_000_000,
            consume_clock_id="mono-A",
        )
        self.assertAlmostEqual(result.age, 0.25)
        self.assertAlmostEqual(result.age_ratio, 0.125)
        self.assertTrue(result.legacy_ready)
        self.assertIsNotNone(result.legacy_tensor)

    def test_unknown_extrinsics_clock_or_left_pts_are_parseable_but_unready(self):
        _, payload = make_payload()
        cases = [
            make_metadata(extrinsics_id=None),
            make_metadata(source_mono_ns=None, clock_mapping_id=None, clock_quality=None),
            make_metadata(left_pts_ns=None),
            make_metadata(monotonic_clock_id=None),
        ]
        for metadata in cases:
            with self.subTest(metadata={key: metadata[key] for key in
                                        ("extrinsics_id", "source_mono_ns", "left_pts_ns",
                                         "monotonic_clock_id")}):
                result = decode(make_adapter(), metadata, payload)
                self.assertTrue(result.payload_parseable)
                self.assertFalse(result.legacy_ready)
                self.assertIsNone(result.legacy_tensor)

    def test_unknown_registry_and_source_state_are_not_ready(self):
        _, payload = make_payload()
        metadata = make_metadata()
        result = decode(DepthAdapter(), metadata, payload)
        self.assertFalse(result.legacy_ready)
        self.assertIn("expected_calibration_id_unregistered", result.readiness_missing)
        result = decode(make_adapter(), metadata, payload, source_state=None)
        self.assertFalse(result.legacy_ready)
        self.assertIn("source_time_pose_and_measured_gimbal_history", result.readiness_missing)

    def test_cross_host_requires_stable_validated_clock_mapping(self):
        _, payload = make_payload()
        metadata = make_metadata(monotonic_clock_id="mono-A")
        adapter = make_adapter()
        state = state_for(metadata)
        without_map = adapter.decode_frame(
            metadata, payload, consume_mono_ns=1_250_000_500,
            consume_clock_id="mono-B", consume_pipeline_epoch=1,
            source_state=state,
        )
        self.assertFalse(without_map.legacy_ready)
        self.assertIn("validated_producer_to_consumer_clock_mapping",
                      without_map.readiness_missing)

        mapping = MonotonicClockMapping("mono-A", "mono-B", 500, 1000,
                                        "clock-map-1", validated=True)
        adapter = make_adapter()
        with_map = adapter.decode_frame(
            metadata, payload, consume_mono_ns=1_250_000_500,
            consume_clock_id="mono-B", consume_pipeline_epoch=1,
            producer_to_consumer=mapping, source_state=state,
        )
        self.assertAlmostEqual(with_map.age, 0.25)
        self.assertEqual(with_map.age_error_bound_ns, 1000)
        self.assertTrue(with_map.legacy_ready)

        regression_adapter = make_adapter()
        regression_adapter.decode_frame(
            metadata, payload, consume_mono_ns=1_250_000_500,
            consume_clock_id="mono-B", consume_pipeline_epoch=1,
            producer_to_consumer=mapping, source_state=state,
        )
        regressed_source = make_metadata(seq=2, source_mono_ns=999_999_999,
                                         left_pts_ns=999_999_999)
        regressed = regression_adapter.decode_frame(
            regressed_source, payload, consume_mono_ns=1_250_000_600,
            consume_clock_id="mono-B", consume_pipeline_epoch=1,
            producer_to_consumer=mapping,
            source_state=state_for(regressed_source),
        )
        self.assertFalse(regressed.legacy_ready)
        self.assertIn("source_mono_clock_regressed", regressed.readiness_missing)

        changed = make_metadata(seq=2, source_mono_ns=1_100_000_000,
                                left_pts_ns=1_100_000_000)
        changed_state = state_for(changed)
        changed_mapping = MonotonicClockMapping("mono-A", "mono-B", 700, 1000,
                                                "clock-map-1", validated=True)
        drift = adapter.decode_frame(
            changed, payload, consume_mono_ns=1_350_000_700,
            consume_clock_id="mono-B", consume_pipeline_epoch=1,
            producer_to_consumer=changed_mapping, source_state=changed_state,
        )
        self.assertFalse(drift.legacy_ready)
        self.assertIn("consumer_clock_mapping_changed_without_epoch",
                      drift.readiness_missing)

    def test_duplicate_old_sequence_epoch_reset_and_clock_regressions(self):
        _, payload = make_payload()
        adapter = make_adapter()
        first = make_metadata(seq=10)
        first_result = decode(adapter, first, payload,
                              consume_mono_ns=1_250_000_000)
        self.assertTrue(first_result.new_frame)

        duplicate = decode(adapter, first, payload,
                           consume_mono_ns=1_260_000_000)
        self.assertFalse(duplicate.new_frame)
        with self.assertRaises(OutOfOrderFrameError):
            decode(adapter, make_metadata(seq=9), payload,
                   consume_mono_ns=1_270_000_000)

        duplicate_regressed = decode(adapter, first, payload,
                                     consume_mono_ns=1_240_000_000)
        self.assertFalse(duplicate_regressed.new_frame)
        self.assertIn("consume_mono_clock_regressed",
                      duplicate_regressed.readiness_missing)

        next_epoch = make_metadata(seq=1, pipeline_epoch=2,
                                   source_mono_ns=10, left_pts_ns=10,
                                   publish_mono_ns=20)
        reset = decode(adapter, next_epoch, payload, consume_mono_ns=300_000_010)
        self.assertTrue(reset.epoch_changed)
        self.assertTrue(reset.new_frame)

        regressed_source = make_metadata(seq=2, pipeline_epoch=2,
                                         source_mono_ns=9, left_pts_ns=9)
        regressed = decode(adapter, regressed_source, payload,
                           consume_mono_ns=300_000_020)
        self.assertIn("source_mono_clock_regressed", regressed.readiness_missing)

    def test_future_source_time_is_not_clipped_to_zero_age(self):
        _, payload = make_payload()
        metadata = make_metadata(source_mono_ns=2_000_000_000)
        result = decode(make_adapter(), metadata, payload,
                        consume_mono_ns=1_900_000_000)
        self.assertIsNone(result.age)
        self.assertFalse(result.legacy_ready)
        self.assertIn("source_time_in_future", result.readiness_missing)

    def test_future_rejection_does_not_commit_sequence_or_source_clock(self):
        _, payload = make_payload()
        # A rejected future frame must not poison either the accepted sequence
        # high-water mark or source-clock high-water mark. Exercise each failure
        # independently with a fresh adapter.
        cases = ((2, 3, "source-clock"), (100, 2, "sequence"))
        for future_seq, recovery_seq, guard in cases:
            with self.subTest(guard=guard):
                adapter = make_adapter()
                first = make_metadata(
                    seq=1,
                    source_mono_ns=1_000_000_000,
                    left_pts_ns=1_000_000_000,
                    right_pts_ns=1_000_000_000,
                    publish_mono_ns=1_050_000_000,
                )
                accepted = decode(
                    adapter, first, payload, consume_mono_ns=1_250_000_000
                )
                self.assertTrue(accepted.legacy_ready)
                self.assertTrue(accepted.new_frame)
                self.assertAlmostEqual(accepted.age, 0.25)
                self.assertIsNotNone(accepted.legacy_tensor)

                future = make_metadata(
                    seq=future_seq,
                    source_mono_ns=2_000_000_000,
                    left_pts_ns=2_000_000_000,
                    right_pts_ns=2_000_000_000,
                    publish_mono_ns=2_050_000_000,
                )
                rejected = decode(
                    adapter, future, payload, consume_mono_ns=1_900_000_000
                )
                self.assertFalse(rejected.legacy_ready)
                self.assertIsNone(rejected.legacy_tensor)
                self.assertIn("source_time_in_future", rejected.readiness_missing)

                recovery = make_metadata(
                    seq=recovery_seq,
                    source_mono_ns=1_950_000_000,
                    left_pts_ns=1_950_000_000,
                    right_pts_ns=1_950_000_000,
                    publish_mono_ns=2_000_000_000,
                )
                restored = decode(
                    adapter, recovery, payload, consume_mono_ns=2_100_000_000
                )
                self.assertTrue(restored.legacy_ready)
                self.assertTrue(restored.new_frame)
                self.assertAlmostEqual(restored.age, 0.15)
                self.assertIsNotNone(restored.legacy_tensor)


if __name__ == "__main__":
    unittest.main()
