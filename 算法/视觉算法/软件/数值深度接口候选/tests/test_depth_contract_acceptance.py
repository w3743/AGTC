"""Small offline black-box checks across the depth codec, wire, and adapter.

All timestamps/calibration IDs here are explicit synthetic fixtures. Passing
these tests does not establish real-camera exposure timing or field readiness.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import numpy as np

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(TESTS))

import depth_wire  # noqa: E402
import test_depth_adapter as adapter_fixtures  # noqa: E402
from depth_adapter import DepthAdapter  # noqa: E402


def _source_indices_for_target_row(u: int, v: int = 80) -> tuple[int, int]:
    """Independent same-optical-axis K mapping for this explicit fixture."""
    fx_s, cx_s = adapter_fixtures.SOURCE_K[0], adapter_fixtures.SOURCE_K[2]
    fy_s, cy_s = adapter_fixtures.SOURCE_K[4], adapter_fixtures.SOURCE_K[5]
    fx_t = fy_t = 98.3247958874
    cx_t, cy_t = 44.0, 80.0
    xi = math.floor((fx_s / fx_t) * (u - cx_t) + cx_s + 0.5)
    yi = math.floor((fy_s / fy_t) * (v - cy_t) + cy_s + 0.5)
    return yi, xi


class DepthContractAcceptanceTests(unittest.TestCase):
    def test_codec_and_wire_boundaries_reach_the_exact_legacy_pixels(self):
        # This four-pixel vector is a codec fixture, not a complete wire frame.
        codec_input = np.array([[0.0, 0.3], [1.0, 5.0]], dtype=np.float64)
        codec_bytes, codec_width, codec_height = depth_wire.encode_depth_payload(codec_input)
        self.assertEqual((codec_width, codec_height), (2, 2))
        self.assertEqual(
            codec_bytes,
            bytes((0x00, 0x00, 0x2C, 0x01, 0xE8, 0x03, 0x88, 0x13)),
        )

        source_ns = (1 << 53) + 37
        sample_locations = [_source_indices_for_target_row(u) for u in range(7)]
        metric_depth = np.full((240, 131), 1.0, dtype=np.float64, order="C")
        metric_values = np.array(
            [0.0, 0.299, 0.3, 1.0, 5.0, 5.001, 65.535], dtype=np.float64
        )
        for (yi, xi), value in zip(sample_locations, metric_values):
            metric_depth[yi, xi] = value

        payload, width, height = depth_wire.encode_depth_payload(metric_depth)
        self.assertEqual((width, height), (131, 240))
        metadata = adapter_fixtures.make_metadata(
            seq=101,
            left_pts_ns=source_ns,
            right_pts_ns=source_ns + 1_000_000,
            source_mono_ns=source_ns,
            publish_mono_ns=source_ns + 100_000_000,
        )
        record = depth_wire.build_depth_record(payload, metadata)
        parsed_metadata = depth_wire.parse_depth_metadata(record.metadata_ascii)
        adapter = adapter_fixtures.make_adapter(k=adapter_fixtures.SOURCE_K)
        result = adapter.decode_frame(
            parsed_metadata,
            record.payload,
            consume_mono_ns=source_ns + 250_000_000,
            consume_clock_id=parsed_metadata["monotonic_clock_id"],
            consume_pipeline_epoch=parsed_metadata["pipeline_epoch"],
            source_state=adapter_fixtures.state_for(parsed_metadata),
        )

        self.assertTrue(result.payload_parseable)
        self.assertTrue(result.legacy_ready, result.readiness_missing)
        self.assertTrue(result.new_frame)
        self.assertEqual(result.legacy_tensor.shape, (1, 160, 88))
        self.assertEqual(result.legacy_tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(result.legacy_tensor).all())
        self.assertAlmostEqual(result.age, 0.25)
        self.assertAlmostEqual(result.age_ratio, 0.125)
        np.testing.assert_array_equal(
            result.target_valid[80, :7],
            [False, False, True, True, True, False, False],
        )
        np.testing.assert_allclose(
            result.legacy_tensor[0, 80, :7],
            [0.0, 0.0, 0.06, 0.2, 1.0, 0.0, 0.0],
            rtol=0,
            atol=1e-7,
        )

        # Defensive consumer fixture: wire bytes can contain out-of-profile
        # integer values even though the producer codec would emit zero for them.
        raw_values = np.array([0, 299, 300, 1000, 5000, 5001, 65535], dtype="<u2")
        raw_mm = np.full((240, 131), 1000, dtype="<u2", order="C")
        for (yi, xi), value in zip(sample_locations, raw_values):
            raw_mm[yi, xi] = value
        raw_payload = raw_mm.tobytes(order="C")
        raw_metadata = adapter_fixtures.make_metadata(
            seq=102,
            left_pts_ns=source_ns,
            right_pts_ns=source_ns + 1_000_000,
            source_mono_ns=source_ns,
            publish_mono_ns=source_ns + 100_000_000,
        )
        raw_record = depth_wire.build_depth_record(raw_payload, raw_metadata)
        parsed_raw_metadata = depth_wire.parse_depth_metadata(raw_record.metadata_ascii)
        source_mask = np.ones((240, 131), dtype=np.bool_)
        source_mask[sample_locations[3]] = False
        raw_result = adapter_fixtures.decode(
            adapter_fixtures.make_adapter(k=adapter_fixtures.SOURCE_K),
            parsed_raw_metadata,
            raw_record.payload,
            source_mask=source_mask,
        )

        np.testing.assert_array_equal(
            raw_result.source_mm[[p[0] for p in sample_locations],
                                 [p[1] for p in sample_locations]],
            raw_values,
        )
        np.testing.assert_array_equal(
            raw_result.source_valid[[p[0] for p in sample_locations],
                                    [p[1] for p in sample_locations]],
            [False, False, True, False, True, False, False],
        )
        np.testing.assert_array_equal(
            raw_result.target_valid[80, :7],
            [False, False, True, False, True, False, False],
        )
        np.testing.assert_allclose(
            raw_result.legacy_tensor[0, 80, :7],
            [0.0, 0.0, 0.06, 0.0, 1.0, 0.0, 0.0],
            rtol=0,
            atol=1e-7,
        )
        self.assertTrue(np.isfinite(raw_result.legacy_tensor).all())

    def test_live_publish_http_to_adapter_keeps_unknown_time_and_extrinsics_unready(self):
        # Keep camera/server dependencies local: pure consumer acceptance can run
        # in the training CPU venv without importing OpenCV/GStreamer fixtures.
        import test_depth_producer_http as producer_fixtures

        live = producer_fixtures.fake_live()
        metadata_context = producer_fixtures.metadata(seq=77, left_pts=1_234_567_890)
        metadata_context.update(
            calibration_id="cal-fixture",
            preprocessing_id="prep-fixture",
            # These tempting values must be discarded by the production publisher.
            source_mono_ns=(1 << 53) + 9,
            clock_mapping_id="unverified-fixture-map",
            clock_quality="verified",
        )
        depth = np.full((240, 131), 1.0, dtype=np.float64, order="C")
        depth[0, :5] = np.array([np.nan, 0.0, 0.299, 5.001, 65.535], dtype=np.float64)
        color = np.zeros((8, 8, 3), dtype=np.uint8)
        rectified_left = np.zeros((240, 131), dtype=np.uint8)

        producer_fixtures.server.Live.publish(
            live,
            color,
            depth=depth,
            rectified_left=rectified_left,
            worker_generation=3,
            worker_pipeline_epoch=4,
            worker_metric=live.metric,
            depth_metadata_context=metadata_context,
            source_pts_ns=metadata_context["left_pts_ns"],
            mode="offline-depth-contract-fixture",
        )
        self.assertEqual(live.seq, 1)
        self.assertEqual(len(live.measure_history), 1)
        record = live.measure_history[-1][3]

        first_handler = producer_fixtures.FakeHandler()
        producer_fixtures.server.serve_depth_bin(
            first_handler, live, {"sequence": [str(record.seq)]}, wait_timeout=0.0
        )
        status, _, first_body, (content_type, first_headers) = first_handler.result
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "application/octet-stream")
        self.assertEqual(len(first_body), 62_880)
        first_metadata_ascii = first_headers["X-Depth-Metadata"]

        live.state.update(
            seq=999,
            K=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            source_mono_ns=(1 << 53) + 123,
            left_pts_ns=9_999_999_999,
        )
        second_handler = producer_fixtures.FakeHandler()
        producer_fixtures.server.serve_depth_bin(
            second_handler, live, {"sequence": [str(record.seq)]}, wait_timeout=0.0
        )
        second_status, _, second_body, (_, second_headers) = second_handler.result
        self.assertEqual(second_status, 200)
        self.assertEqual(second_body, record.payload)
        self.assertEqual(second_body, first_body)
        self.assertEqual(second_headers["X-Depth-Metadata"], record.metadata_ascii)
        self.assertEqual(second_headers["X-Depth-Metadata"], first_metadata_ascii)

        published_metadata = depth_wire.parse_depth_metadata(first_metadata_ascii)
        self.assertEqual(published_metadata["seq"], record.seq)
        self.assertEqual(published_metadata["left_pts_ns"], metadata_context["left_pts_ns"])
        self.assertIsNone(published_metadata["source_mono_ns"])
        self.assertIsNone(published_metadata["clock_mapping_id"])
        self.assertIsNone(published_metadata["clock_quality"])
        self.assertIsNone(published_metadata["extrinsics_id"])
        published_mm = depth_wire.decode_depth_payload(first_body, first_metadata_ascii)
        np.testing.assert_array_equal(published_mm[0, :5], np.zeros(5, dtype=np.uint16))

        # The registration is a named synthetic fixture, not an assertion that
        # the candidate's unknown extrinsics or PTS-to-monotonic mapping exist.
        consumer = adapter_fixtures.make_adapter(k=published_metadata["K"])
        adapted = consumer.decode_frame(
            published_metadata,
            first_body,
            consume_mono_ns=published_metadata["publish_mono_ns"] + 1,
            consume_clock_id=published_metadata["monotonic_clock_id"],
            consume_pipeline_epoch=published_metadata["pipeline_epoch"],
            # Deliberately no SourceStateEvidence for this producer record.
        )
        self.assertTrue(adapted.payload_parseable)
        self.assertFalse(adapted.legacy_ready)
        self.assertIsNone(adapted.legacy_tensor)
        self.assertIsNone(adapted.age)
        self.assertIsNone(adapted.age_ratio)
        self.assertIn("source_mono_ns", adapted.readiness_missing)
        self.assertIn("extrinsics_id_mismatch_or_unknown", adapted.readiness_missing)


import importlib
import importlib.util


_LASSC_BRIDGE = None


def _load_lassc_bridge():
    """Load the real bridge under a stable test-only module name."""
    global _LASSC_BRIDGE
    if _LASSC_BRIDGE is None:
        script = (Path(__file__).resolve().parents[4]
                  / "仿真训练环境" / "scripts" / "lassc_consumer.py")
        if not script.is_file():
            raise FileNotFoundError(script)
        name = "_depth_contract_lassc_consumer_acceptance"
        spec = importlib.util.spec_from_file_location(name, script)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load LASSC bridge from {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _LASSC_BRIDGE = module
    return _LASSC_BRIDGE


def _lassc_fixture():
    bridge = _load_lassc_bridge()
    adapter_module, _wire_module = bridge.load_depth_modules()
    adapter = adapter_module.DepthAdapter(
        expected_calibration_id="cal-fixture",
        expected_preprocessing_id="prep-fixture",
        expected_extrinsics_id="ext-fixture",
        expected_source_k=adapter_fixtures.SOURCE_K,
    )
    consumer = bridge.LASSCConsumer(
        adapter,
        cfg=bridge.Config(size=12, plants=0, terrain_enabled=False,
                          validate_scene=False, max_time=60),
        max_depth_age_s=0.5,
        max_other_age_s=0.5,
        runner=None,
    )
    feedback = bridge.FeedbackSnapshot(
        feedback=np.zeros(37, dtype=np.float32),
        sample_time=10.0,
        estimated_heading_rad=0.0,
        estimated_roll_rad=0.0,
        estimated_pitch_rad=0.0,
        estimated_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        estimated_gimbal_q_rad=np.zeros(2, dtype=np.float32),
        pose_valid=True,
    )
    sensor_pose = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    context = bridge.ObservationContext(
        episode_time=10.0,
        estimated_xy=(1.0, 1.0),
        goal_xy=(2.0, 2.0),
        previous_action=(0.0, 0.0, 0.0, 0.0),
        feedback=feedback,
        lidar={
            "ranges": np.ones(400, dtype=np.float32),
            "valid": np.ones(400, dtype=np.bool_),
            "capture": 9.9,
            "estimated_pose": sensor_pose.copy(),
            "new_frame": True,
        },
        camera={
            "detections": [],
            "capture": 9.9,
            "estimated_pose": sensor_pose.copy(),
            "new_frame": True,
        },
    )
    task_module = importlib.import_module("isaac_field.task")
    return bridge, adapter_module, consumer, context, task_module.STATE_FIELDS


def _lassc_metadata(seq: int, source_ns: int, *, epoch: int = 1,
                    session: str = "session-fixture", clock_id: str = "mono-A",
                    mapping_id: str = "pts-map-fixture") -> dict:
    return adapter_fixtures.make_metadata(
        seq=seq,
        session_id=session,
        pipeline_epoch=epoch,
        left_pts_ns=source_ns,
        right_pts_ns=source_ns + 25_000_000,
        source_mono_ns=source_ns,
        publish_mono_ns=source_ns + 100_000_000,
        monotonic_clock_id=clock_id,
        clock_mapping_id=mapping_id,
        clock_quality="verified",
    )


def _lassc_pose(bridge, metadata):
    return bridge.SourcePose(
        session_id=metadata["session_id"],
        pipeline_epoch=metadata["pipeline_epoch"],
        clock_id=metadata["monotonic_clock_id"],
        source_mono_ns=metadata["source_mono_ns"],
        pose=(1.0, 1.0, 0.0, 0.0, 0.0),
        measured_gimbal=True,
    )


def _transaction_state(consumer):
    return (
        consumer.accepted,
        consumer.stream,
        frozenset(consumer.retired_streams),
        consumer.last_identity,
        consumer.last_digest,
        id(consumer.adapter),
    )


class LASSCDepthAcceptanceTests(unittest.TestCase):
    """Exercise the production bridge and FieldTask packing with CPU fixtures."""

    @staticmethod
    def _frame_payload():
        return adapter_fixtures.make_payload(fill=1000)[1]

    def _consume(self, bridge, consumer, context, metadata, payload, consume_ns,
                 *, consume_clock_id="mono-A", source_pose=True,
                 producer_to_consumer=None):
        return consumer.consume(
            metadata,
            payload,
            context=context,
            source_pose=_lassc_pose(bridge, metadata) if source_pose else None,
            consume_mono_ns=consume_ns,
            consume_clock_id=consume_clock_id,
            producer_to_consumer=producer_to_consumer,
        )

    @staticmethod
    def _assert_observation_depth(test, result, state_fields, age_ratio, new_frame):
        observation = result.observation
        test.assertEqual(observation["depth"].shape, (1, 160, 88))
        test.assertEqual(observation["depth"].dtype, np.float32)
        test.assertAlmostEqual(float(observation["depth"][0, 80, 44]), 0.2, places=6)
        state = observation["state"]
        age_index = state_fields.index("depth_age_ratio")
        new_index = state_fields.index("depth_new_frame")
        test.assertAlmostEqual(float(state[age_index]), age_ratio, places=6)
        test.assertEqual(float(state[new_index]), float(new_frame))

    def test_lassc_future_rejection_does_not_commit_and_next_frame_packs_real_depth_state(self):
        bridge, _adapter_module, consumer, context, state_fields = _lassc_fixture()
        payload = self._frame_payload()
        first = _lassc_metadata(seq=1, source_ns=1_000_000_000)

        accepted = self._consume(
            bridge, consumer, context, first, payload, 1_250_000_000
        )
        self.assertTrue(accepted.ready, accepted.reasons)
        self.assertTrue(accepted.new_frame)
        self.assertFalse(accepted.reset_recurrent)
        self.assertAlmostEqual(accepted.age_s, 0.25)
        self._assert_observation_depth(self, accepted, state_fields, 0.125, True)
        committed = _transaction_state(consumer)

        future = _lassc_metadata(seq=100, source_ns=2_000_000_000)
        rejected = self._consume(
            bridge, consumer, context, future, payload, 1_900_000_000
        )
        self.assertFalse(rejected.ready)
        self.assertIn("source_time_in_future", rejected.reasons)
        self.assertEqual(_transaction_state(consumer), committed)

        following = _lassc_metadata(seq=2, source_ns=1_950_000_000)
        recovered = self._consume(
            bridge, consumer, context, following, payload, 2_100_000_000
        )
        self.assertTrue(recovered.ready, recovered.reasons)
        self.assertTrue(recovered.new_frame)
        self.assertFalse(recovered.reset_recurrent)
        self.assertAlmostEqual(recovered.age_s, 0.15)
        self._assert_observation_depth(self, recovered, state_fields, 0.075, True)

    def test_lassc_stale_age_and_clock_uncertainty_reject_without_commit(self):
        bridge, adapter_module, consumer, context, _state_fields = _lassc_fixture()
        payload = self._frame_payload()
        original_adapter_id = id(consumer.adapter)
        initial = _transaction_state(consumer)

        stale = _lassc_metadata(seq=1, source_ns=1_000_000_000)
        stale_result = self._consume(
            bridge, consumer, context, stale, payload, 5_000_000_000
        )
        self.assertFalse(stale_result.ready)
        self.assertIn("depth_stale", stale_result.reasons)
        self.assertAlmostEqual(stale_result.age_s, 4.0)
        self.assertEqual(_transaction_state(consumer), initial)
        self.assertEqual(id(consumer.adapter), original_adapter_id)

        producer_clock = "producer-mono-A"
        consumer_clock = "consumer-host-mono-B"
        map_id = "fixture-producer-to-consumer"
        cross_host = _lassc_metadata(
            seq=2,
            source_ns=1_000_000_000,
            clock_id=producer_clock,
            mapping_id=map_id,
        )
        unmapped = self._consume(
            bridge, consumer, context, cross_host, payload, 1_250_000_000,
            consume_clock_id=consumer_clock,
        )
        self.assertFalse(unmapped.ready)
        self.assertIn("validated_producer_to_consumer_clock_mapping", unmapped.reasons)
        self.assertEqual(_transaction_state(consumer), initial)

        mapping = adapter_module.MonotonicClockMapping(
            source_clock_id=producer_clock,
            target_clock_id=consumer_clock,
            offset_ns=0,
            error_bound_ns=5_000_000_000,
            mapping_id=map_id,
            validated=True,
        )
        uncertain = self._consume(
            bridge,
            consumer,
            context,
            cross_host,
            payload,
            1_250_000_000,
            consume_clock_id=consumer_clock,
            producer_to_consumer=mapping,
        )
        self.assertFalse(uncertain.ready)
        self.assertIn("depth_stale", uncertain.reasons)
        self.assertAlmostEqual(uncertain.age_s, 0.25)
        self.assertEqual(_transaction_state(consumer), initial)
        self.assertEqual(id(consumer.adapter), original_adapter_id)

        recovered = self._consume(
            bridge, consumer, context, stale, payload, 1_250_000_000
        )
        self.assertTrue(recovered.ready, recovered.reasons)
        self.assertEqual(consumer.accepted, 1)

    def test_lassc_epoch_switch_retires_prior_stream_and_rejects_late_epoch(self):
        bridge, _adapter_module, consumer, context, _state_fields = _lassc_fixture()
        payload = self._frame_payload()
        old = _lassc_metadata(seq=10, source_ns=1_000_000_000, epoch=1)
        first = self._consume(bridge, consumer, context, old, payload, 1_250_000_000)
        self.assertTrue(first.ready, first.reasons)
        self.assertFalse(first.reset_recurrent)

        current = _lassc_metadata(seq=1, source_ns=1_300_000_000, epoch=2)
        switched = self._consume(
            bridge, consumer, context, current, payload, 1_550_000_000
        )
        self.assertTrue(switched.ready, switched.reasons)
        self.assertTrue(switched.reset_recurrent)
        self.assertTrue(switched.new_frame)
        committed_current = _transaction_state(consumer)

        late_old = _lassc_metadata(seq=11, source_ns=1_500_000_000, epoch=1)
        retired = self._consume(
            bridge, consumer, context, late_old, payload, 1_750_000_000
        )
        self.assertFalse(retired.ready)
        self.assertIn("retired_session_or_epoch", retired.reasons)
        self.assertEqual(_transaction_state(consumer), committed_current)

        # An epoch older than the watermark is stale even if this consumer
        # never accepted that particular old epoch before.
        unseen_old = _lassc_metadata(seq=99, source_ns=1_500_000_000, epoch=0)
        unseen_result = self._consume(
            bridge, consumer, context, unseen_old, payload, 1_750_000_000
        )
        self.assertFalse(unseen_result.ready)
        self.assertIn("retired_session_or_epoch", unseen_result.reasons)
        self.assertEqual(_transaction_state(consumer), committed_current)

        next_current = _lassc_metadata(seq=2, source_ns=1_400_000_000, epoch=2)
        continued = self._consume(
            bridge, consumer, context, next_current, payload, 1_650_000_000
        )
        self.assertTrue(continued.ready, continued.reasons)
        self.assertFalse(continued.reset_recurrent)
        self.assertTrue(continued.new_frame)
        self.assertEqual(consumer.accepted, 3)

    def test_lassc_missing_pose_retry_duplicate_conflict_and_reuse(self):
        bridge, _adapter_module, consumer, context, state_fields = _lassc_fixture()
        metadata = _lassc_metadata(seq=1, source_ns=1_000_000_000)
        payload = self._frame_payload()
        initial = _transaction_state(consumer)

        missing_pose = self._consume(
            bridge, consumer, context, metadata, payload, 1_250_000_000,
            source_pose=False,
        )
        self.assertFalse(missing_pose.ready)
        self.assertIn("source_time_pose_and_measured_gimbal_history", missing_pose.reasons)
        self.assertEqual(_transaction_state(consumer), initial)

        first = self._consume(
            bridge, consumer, context, metadata, payload, 1_250_000_000
        )
        self.assertTrue(first.ready, first.reasons)
        self.assertTrue(first.new_frame)
        self.assertEqual(consumer.accepted, 1)
        committed = _transaction_state(consumer)

        changed_image, _ = adapter_fixtures.make_payload(fill=1000)
        target_y, target_x = _source_indices_for_target_row(44, v=80)
        changed_image[target_y, target_x] = 1500
        conflicting = self._consume(
            bridge,
            consumer,
            context,
            metadata,
            changed_image.tobytes(order="C"),
            1_300_000_000,
        )
        self.assertFalse(conflicting.ready)
        self.assertTrue(any("conflicting duplicate depth record" in reason
                            for reason in conflicting.reasons))
        self.assertEqual(_transaction_state(consumer), committed)

        repeated = self._consume(
            bridge, consumer, context, metadata, payload, 1_400_000_000
        )
        self.assertTrue(repeated.ready, repeated.reasons)
        self.assertFalse(repeated.new_frame)
        self.assertFalse(repeated.reset_recurrent)
        self.assertEqual(consumer.accepted, 2)
        self._assert_observation_depth(self, repeated, state_fields, 0.2, False)


if __name__ == "__main__":
    unittest.main()
