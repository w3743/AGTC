"""Offline consumer adapter for the depth-z-u16le-v1 candidate contract.

This module does not open a camera or depend on a policy runtime. Callers must
register the calibration, preprocessing, and same-optical-frame extrinsics they
have actually approved before a legacy policy tensor can be marked ready.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Support both ``import app.depth_adapter`` and a direct app/ PYTHONPATH.
    from .depth_wire import validate_metadata, decode_depth_payload
except ImportError:  # pragma: no cover - exercised by direct-module imports
    from depth_wire import validate_metadata, decode_depth_payload


SCHEMA = "depth-z-u16le-v1"
SOURCE_WIDTH = 131
SOURCE_HEIGHT = 240
MIN_MM = 300
MAX_MM = 5000
LEGACY_WIDTH = 88
LEGACY_HEIGHT = 160
DEFAULT_FRAME_ID = "left_rectified_optical"
TARGET_K = np.array(
    [[98.3247958874, 0.0, 44.0],
     [0.0, 98.3247958874, 80.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_USABLE_CLOCK_QUALITIES = frozenset(
    {"good", "ok", "valid", "verified", "usable", "locked", "high"}
)
_TRACKING_STATE_FIELDS = (
    "_stream_key",
    "_last_seq",
    "_last_clock_mapping_id",
    "_mapping_changed_without_epoch",
    "_last_monotonic_clock_id",
    "_source_clock_changed_without_epoch",
    "_last_source_mono_ns",
    "_last_consume_clock_id",
    "_last_consume_mono_ns",
    "_last_consumer_mapping_fingerprint",
    "_has_consumer_mapping_fingerprint",
    "_consumer_mapping_changed_without_epoch",
    "_consume_clock_changed_without_epoch",
)


class DepthAdapterError(ValueError):
    """The frame is malformed or violates the fixed depth wire profile."""


class OutOfOrderFrameError(DepthAdapterError):
    """A lower sequence number arrived within the same session and epoch."""


@dataclass(frozen=True)
class MonotonicClockMapping:
    """Caller-validated producer-to-consumer monotonic clock transform.

    ``offset_ns`` means consumer_time = producer_time + offset_ns. Setting
    ``validated`` is an explicit assertion by the caller; the adapter cannot
    establish cross-host clock synchronization on its own.
    """

    source_clock_id: str
    target_clock_id: str
    offset_ns: int
    error_bound_ns: int
    mapping_id: str
    validated: bool = False

    def __post_init__(self) -> None:
        if not self.source_clock_id or not self.target_clock_id or not self.mapping_id:
            raise ValueError("clock mapping IDs must be non-empty")
        if isinstance(self.offset_ns, bool) or not isinstance(self.offset_ns, int):
            raise ValueError("clock mapping offset_ns must be an integer")
        if (isinstance(self.error_bound_ns, bool)
                or not isinstance(self.error_bound_ns, int)
                or self.error_bound_ns < 0):
            raise ValueError("clock mapping error_bound_ns must be a non-negative integer")


@dataclass(frozen=True)
class SourceStateEvidence:
    """Evidence that pose and measured gimbal history match a source frame."""

    source_mono_ns: int
    clock_id: str
    pipeline_epoch: int
    pose_available: bool
    measured_gimbal_available: bool
    aligned_to_source_time: bool

    def __post_init__(self) -> None:
        if not _is_nonnegative_int(self.source_mono_ns):
            raise ValueError("source-state source_mono_ns must be a non-negative integer")
        if not isinstance(self.clock_id, str) or not self.clock_id:
            raise ValueError("source-state clock_id must be a non-empty string")
        if not _is_nonnegative_int(self.pipeline_epoch):
            raise ValueError("source-state pipeline_epoch must be a non-negative integer")
        for field in (self.pose_available, self.measured_gimbal_available,
                      self.aligned_to_source_time):
            if not isinstance(field, bool):
                raise ValueError("source-state availability/alignment flags must be bool")


@dataclass(frozen=True)
class AdaptedDepthFrame:
    """One validated source frame and its optional legacy debug projection."""

    metadata: Mapping[str, Any]
    payload_parseable: bool
    source_mm: np.ndarray
    source_valid: np.ndarray
    target_valid: Optional[np.ndarray]
    target_z_m: Optional[np.ndarray]
    debug_tensor: Optional[np.ndarray]
    legacy_ready: bool
    readiness_missing: Tuple[str, ...]
    new_frame: bool
    epoch_changed: bool
    age: Optional[float]
    age_ratio: Optional[float]
    age_error_bound_ns: Optional[int]

    @property
    def valid(self) -> np.ndarray:
        """Validity mask for the legacy grid when available, else source grid."""
        return self.target_valid if self.target_valid is not None else self.source_valid

    @property
    def legacy_tensor(self) -> Optional[np.ndarray]:
        """Tensor safe to hand to the old policy only when ready and fresh-new."""
        if self.legacy_ready and self.new_frame:
            return self.debug_tensor
        return None


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validated_k(value: Any, label: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        raw = value.tolist()
    else:
        raw = value
    try:
        if isinstance(raw, (list, tuple)):
            flattened = np.asarray(raw, dtype=object).reshape(-1).tolist()
            if any(isinstance(item, bool) or not isinstance(item, Real)
                   for item in flattened):
                raise ValueError("K entries must be real numbers, not booleans or strings")
        candidate = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DepthAdapterError(f"{label} must contain nine finite numbers or null") from exc
    if candidate.shape == (9,):
        candidate = candidate.reshape((3, 3))
    if candidate.shape != (3, 3) or not np.isfinite(candidate).all():
        raise DepthAdapterError(f"{label} must be a finite 3x3 pinhole matrix or null")
    if (candidate[0, 0] <= 0 or candidate[1, 1] <= 0
            or abs(candidate[0, 1]) > 1e-9 or abs(candidate[1, 0]) > 1e-9
            or not np.allclose(candidate[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)):
        raise DepthAdapterError(f"{label} is not a supported pinhole intrinsic matrix")
    return candidate


def _check_optional_layout(metadata: Mapping[str, Any]) -> None:
    """Reject explicit layout/padding declarations outside the wire contract."""
    width = metadata["width"]
    expected_stride = width * 2
    for name in ("row_stride_bytes", "stride_bytes", "step"):
        if name in metadata:
            value = metadata[name]
            if not _is_nonnegative_int(value) or value != expected_stride:
                raise DepthAdapterError(f"{name} must equal width*2; row padding is forbidden")
    for name in ("row_major", "c_contiguous"):
        if name in metadata and metadata[name] is not True:
            raise DepthAdapterError(f"{name} must be true for row-major wire data")
    for name in ("layout", "array_layout", "memory_layout"):
        if name in metadata and metadata[name] not in ("row-major", "row_major", "C"):
            raise DepthAdapterError(f"{name} must declare C-order row-major data")
    if "shape" in metadata:
        value = metadata["shape"]
        if not isinstance(value, (list, tuple)) or list(value) != [SOURCE_HEIGHT, SOURCE_WIDTH]:
            raise DepthAdapterError("shape must be [height,width] for the fixed profile")
    if "dtype" in metadata and metadata["dtype"] not in ("uint16", "u16", "<u2"):
        raise DepthAdapterError("dtype must declare uint16 little-endian data")


def _same_identifier(value: Any, expected: Optional[str]) -> bool:
    return bool(expected) and isinstance(value, str) and value == expected


def _nearest_source_indices(
    source_k: np.ndarray, target_k: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target pixel centers by K and nearest-sample (half-up ties)."""
    vv, uu = np.indices((LEGACY_HEIGHT, LEGACY_WIDTH), dtype=np.float64)
    target_pixels = np.stack(
        (uu.reshape(-1), vv.reshape(-1), np.ones(uu.size, dtype=np.float64)), axis=0
    )
    rays = np.linalg.solve(target_k, target_pixels)
    source_pixels = source_k @ rays
    x = source_pixels[0] / source_pixels[2]
    y = source_pixels[1] / source_pixels[2]
    finite = np.isfinite(x) & np.isfinite(y)
    # Fixed rule: floor(pixel + 0.5), so exact half ties go toward +infinity.
    xi = np.full(x.shape, -1, dtype=np.int64)
    yi = np.full(y.shape, -1, dtype=np.int64)
    xi[finite] = np.floor(x[finite] + 0.5).astype(np.int64)
    yi[finite] = np.floor(y[finite] + 0.5).astype(np.int64)
    in_bounds = (finite & (xi >= 0) & (xi < SOURCE_WIDTH)
                 & (yi >= 0) & (yi < SOURCE_HEIGHT))
    xi = np.where(in_bounds, xi, 0).reshape((LEGACY_HEIGHT, LEGACY_WIDTH))
    yi = np.where(in_bounds, yi, 0).reshape((LEGACY_HEIGHT, LEGACY_WIDTH))
    return yi, xi, in_bounds.reshape((LEGACY_HEIGHT, LEGACY_WIDTH))


class DepthAdapter:
    """Stateful offline adapter with explicit geometry and clock registries.

    The constructor's expected IDs and source K are consumer-side registrations,
    not values copied from the incoming frame. Defaults intentionally leave the
    adapter unregistered and unable to emit a policy-ready tensor.
    """

    def __init__(
        self,
        *,
        expected_calibration_id: Optional[str] = None,
        expected_preprocessing_id: Optional[str] = None,
        expected_extrinsics_id: Optional[str] = None,
        expected_source_k: Optional[Sequence[float]] = None,
        target_k: Optional[Sequence[float]] = None,
        expected_frame_id: Optional[str] = DEFAULT_FRAME_ID,
        usable_clock_qualities: Sequence[str] = tuple(_USABLE_CLOCK_QUALITIES),
    ) -> None:
        self.expected_calibration_id = expected_calibration_id
        self.expected_preprocessing_id = expected_preprocessing_id
        self.expected_extrinsics_id = expected_extrinsics_id
        self.expected_source_k = _validated_k(expected_source_k, "expected_source_k")
        self.target_k = _validated_k(
            TARGET_K if target_k is None else target_k, "target_k"
        )
        assert self.target_k is not None
        self.target_k.setflags(write=False)
        self.expected_frame_id = expected_frame_id
        self.usable_clock_qualities = frozenset(
            str(value).strip().lower() for value in usable_clock_qualities if str(value).strip()
        )
        self._stream_key: Optional[Tuple[str, Any]] = None
        self._last_seq: Optional[int] = None
        self._last_clock_mapping_id: Optional[str] = None
        self._mapping_changed_without_epoch = False
        self._last_monotonic_clock_id: Optional[str] = None
        self._source_clock_changed_without_epoch = False
        self._last_source_mono_ns: Optional[int] = None
        self._last_consume_clock_id: Optional[str] = None
        self._last_consume_mono_ns: Optional[int] = None
        self._last_consumer_mapping_fingerprint: Optional[Tuple[Any, ...]] = None
        self._has_consumer_mapping_fingerprint = False
        self._consumer_mapping_changed_without_epoch = False
        self._consume_clock_changed_without_epoch = False

    def _snapshot_tracking_state(self) -> dict[str, Any]:
        """Capture stream high-water marks before tentative frame validation."""
        return {name: getattr(self, name) for name in _TRACKING_STATE_FIELDS}

    def _restore_tracking_state(self, snapshot: Mapping[str, Any]) -> None:
        """Roll back sequence, epoch, and clock state for a rejected frame."""
        for name in _TRACKING_STATE_FIELDS:
            setattr(self, name, snapshot[name])

    def decode_frame(
        self,
        metadata: dict,
        payload: bytes,
        *,
        source_mask: Optional[np.ndarray] = None,
        consume_mono_ns: Optional[int] = None,
        consume_clock_id: Optional[str] = None,
        consume_pipeline_epoch: Any = None,
        producer_to_consumer: Optional[MonotonicClockMapping] = None,
        source_state: Optional[SourceStateEvidence] = None,
    ) -> AdaptedDepthFrame:
        """Decode one atomic record and derive debug tensor/readiness.

        ``source_state`` must identify the same source timestamp, clock domain,
        and pipeline epoch and attest that pose plus measured-gimbal history are
        aligned to that time. It defaults to unknown/unready.
        """
        if not isinstance(metadata, dict):
            raise DepthAdapterError("metadata must be a dict")
        try:
            # Validate a dict at the adapter API boundary. The lower-level wire
            # decoder accepts ASCII JSON, so serialize the validated immutable
            # snapshot rather than pretending the two input types are identical.
            _validated_k(metadata.get("K"), "K")  # catch bool/string coercion
            checked = validate_metadata(metadata, expected_shape=(SOURCE_HEIGHT, SOURCE_WIDTH))
            _check_optional_layout(checked)
            if not isinstance(payload, bytes):
                raise DepthAdapterError("payload must be immutable bytes")
            metadata_ascii = json.dumps(
                checked, ensure_ascii=True, allow_nan=False, separators=(",", ":")
            )
            metadata_ascii.encode("ascii", errors="strict")
            source_mm = decode_depth_payload(payload, metadata_ascii)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, DepthAdapterError):
                raise
            raise DepthAdapterError(str(exc)) from exc

        # Metadata parsers preserve Python ints exactly; this extra validation
        # keeps the consumer contract clear if a codec implementation changes.
        for name in ("seq", "pipeline_epoch", "left_pts_ns", "right_pts_ns",
                     "publish_mono_ns", "source_mono_ns"):
            value = checked.get(name)
            if value is not None and not _is_nonnegative_int(value):
                raise DepthAdapterError(f"{name} must be an exact non-negative JSON integer or null")
        if checked["width"] != SOURCE_WIDTH or checked["height"] != SOURCE_HEIGHT:
            raise DepthAdapterError("unexpected source profile dimensions")
        if checked["min_mm"] != MIN_MM or checked["max_mm"] != MAX_MM:
            raise DepthAdapterError("unexpected depth profile range")

        source_k = _validated_k(checked["K"], "K")
        mask = None
        if source_mask is not None:
            mask = np.asarray(source_mask)
            if mask.dtype != np.bool_ or mask.shape != (SOURCE_HEIGHT, SOURCE_WIDTH):
                raise DepthAdapterError("source_mask must be bool with the same HxW shape as depth")

        source_valid = ((source_mm != 0)
                        & (source_mm >= MIN_MM)
                        & (source_mm <= MAX_MM))
        if mask is not None:
            source_valid &= mask
        source_valid.setflags(write=False)

        tracking_before = self._snapshot_tracking_state()
        committed = False
        try:
            new_frame, epoch_changed = self._observe_sequence(checked)

            missing = self._geometry_missing(checked, source_k)
            target_valid = None
            target_z_m = None
            debug_tensor = None
            if not missing:
                assert source_k is not None
                yi, xi, in_bounds = _nearest_source_indices(source_k, self.target_k)
                target_valid = in_bounds & source_valid[yi, xi]
                target_z_m = np.zeros((LEGACY_HEIGHT, LEGACY_WIDTH), dtype=np.float32)
                target_z_m[target_valid] = (
                    source_mm[yi[target_valid], xi[target_valid]].astype(np.float32) * 0.001
                )
                debug_tensor = np.zeros((1, LEGACY_HEIGHT, LEGACY_WIDTH), dtype=np.float32)
                debug_tensor[0, target_valid] = target_z_m[target_valid] / 5.0
                target_valid.setflags(write=False)
                target_z_m.setflags(write=False)
                debug_tensor.setflags(write=False)

            age, age_ratio, error_bound, time_missing = self._age(
                checked,
                consume_mono_ns=consume_mono_ns,
                consume_clock_id=consume_clock_id,
                consume_pipeline_epoch=consume_pipeline_epoch,
                producer_to_consumer=producer_to_consumer,
                new_frame=new_frame,
                epoch_changed=epoch_changed,
            )
            missing.extend(time_missing)
            if not self._source_state_matches(source_state, checked):
                missing.append("source_time_pose_and_measured_gimbal_history")

            # Preserve stable order while avoiding duplicate readiness reasons.
            readiness_missing = tuple(dict.fromkeys(missing))
            result = AdaptedDepthFrame(
                metadata=dict(checked),
                payload_parseable=True,
                source_mm=source_mm,
                source_valid=source_valid,
                target_valid=target_valid,
                target_z_m=target_z_m,
                debug_tensor=debug_tensor,
                legacy_ready=not readiness_missing,
                readiness_missing=readiness_missing,
                new_frame=new_frame,
                epoch_changed=epoch_changed,
                age=age,
                age_ratio=age_ratio,
                age_error_bound_ns=error_bound,
            )
            committed = result.legacy_ready
            return result
        finally:
            # Sequence/epoch/mapping high-water marks advance only after the
            # frame passes the complete geometry, timing, and source-state gate.
            if not committed:
                self._restore_tracking_state(tracking_before)

    @staticmethod
    def _source_state_matches(
        evidence: Optional[SourceStateEvidence], metadata: Mapping[str, Any]
    ) -> bool:
        if not isinstance(evidence, SourceStateEvidence):
            return False
        return bool(
            evidence.pose_available
            and evidence.measured_gimbal_available
            and evidence.aligned_to_source_time
            and metadata.get("source_mono_ns") is not None
            and evidence.source_mono_ns == metadata.get("source_mono_ns")
            and evidence.clock_id == metadata.get("monotonic_clock_id")
            and evidence.pipeline_epoch == metadata.get("pipeline_epoch")
        )

    def _observe_sequence(self, metadata: Mapping[str, Any]) -> Tuple[bool, bool]:
        key = (metadata["session_id"], metadata["pipeline_epoch"])
        epoch_changed = key != self._stream_key
        seq = metadata["seq"]
        # Reject a stale frame before mutating the stream's clock/mapping state.
        if (not epoch_changed and self._last_seq is not None
                and seq < self._last_seq):
            raise OutOfOrderFrameError(
                f"out-of-order sequence {seq} is older than {self._last_seq} in this epoch"
            )
        if epoch_changed:
            self._stream_key = key
            self._last_seq = None
            self._last_clock_mapping_id = metadata.get("clock_mapping_id")
            self._mapping_changed_without_epoch = False
            self._last_monotonic_clock_id = metadata.get("monotonic_clock_id")
            self._source_clock_changed_without_epoch = False
            self._last_source_mono_ns = None
            self._last_consume_clock_id = None
            self._last_consume_mono_ns = None
            self._last_consumer_mapping_fingerprint = None
            self._has_consumer_mapping_fingerprint = False
            self._consumer_mapping_changed_without_epoch = False
            self._consume_clock_changed_without_epoch = False

        mapping_id = metadata.get("clock_mapping_id")
        if not epoch_changed and mapping_id != self._last_clock_mapping_id:
            self._mapping_changed_without_epoch = True
            self._last_clock_mapping_id = mapping_id
        source_clock_id = metadata.get("monotonic_clock_id")
        if not epoch_changed and source_clock_id != self._last_monotonic_clock_id:
            self._source_clock_changed_without_epoch = True
            self._last_monotonic_clock_id = source_clock_id

        if self._last_seq is not None and seq == self._last_seq:
            return False, epoch_changed
        self._last_seq = seq
        return True, epoch_changed

    def _geometry_missing(
        self, metadata: Mapping[str, Any], source_k: Optional[np.ndarray]
    ) -> list[str]:
        missing: list[str] = []
        if source_k is None:
            missing.append("K")
        elif self.expected_source_k is None:
            missing.append("expected_source_K_unregistered")
        elif not np.allclose(source_k, self.expected_source_k, rtol=0.0, atol=1e-6):
            missing.append("K_mismatch")

        for field, expected in (
            ("calibration_id", self.expected_calibration_id),
            ("preprocessing_id", self.expected_preprocessing_id),
            ("extrinsics_id", self.expected_extrinsics_id),
        ):
            if not expected:
                missing.append(f"expected_{field}_unregistered")
            elif not _same_identifier(metadata.get(field), expected):
                missing.append(f"{field}_mismatch_or_unknown")

        if not self.expected_frame_id:
            missing.append("expected_frame_id_unregistered")
        elif metadata.get("frame_id") != self.expected_frame_id:
            missing.append("frame_id_mismatch")
        return missing

    def _age(
        self,
        metadata: Mapping[str, Any],
        *,
        consume_mono_ns: Optional[int],
        consume_clock_id: Optional[str],
        consume_pipeline_epoch: Any,
        producer_to_consumer: Optional[MonotonicClockMapping],
        new_frame: bool,
        epoch_changed: bool,
    ) -> Tuple[Optional[float], Optional[float], Optional[int], list[str]]:
        missing: list[str] = []
        producer_clock_id = metadata.get("monotonic_clock_id")
        source_ns = metadata.get("source_mono_ns")
        mapping_id = metadata.get("clock_mapping_id")
        quality = metadata.get("clock_quality")
        if metadata.get("left_pts_ns") is None:
            missing.append("left_pts_ns_for_source_time")
        if source_ns is None:
            missing.append("source_mono_ns")
        if not mapping_id:
            missing.append("clock_mapping_id")
        if not isinstance(quality, str) or quality.strip().lower() not in self.usable_clock_qualities:
            missing.append("clock_quality_unusable_or_unknown")
        if not producer_clock_id:
            missing.append("monotonic_clock_id")
        if consume_mono_ns is None or not _is_nonnegative_int(consume_mono_ns):
            missing.append("consume_mono_ns")
        if not isinstance(consume_clock_id, str) or not consume_clock_id:
            missing.append("consume_clock_id")
        if consume_pipeline_epoch is None:
            missing.append("consume_pipeline_epoch")
        elif consume_pipeline_epoch != metadata["pipeline_epoch"]:
            missing.append("pipeline_epoch_mismatch")
        if self._mapping_changed_without_epoch:
            missing.append("clock_mapping_changed_without_epoch")
        if self._source_clock_changed_without_epoch:
            missing.append("monotonic_clock_changed_without_epoch")
        if self._consumer_mapping_changed_without_epoch:
            missing.append("consumer_clock_mapping_changed_without_epoch")
        if self._consume_clock_changed_without_epoch:
            missing.append("consume_clock_changed_without_epoch")

        if isinstance(producer_to_consumer, MonotonicClockMapping):
            mapping_fingerprint = (
                producer_to_consumer.source_clock_id,
                producer_to_consumer.target_clock_id,
                producer_to_consumer.offset_ns,
                producer_to_consumer.error_bound_ns,
                producer_to_consumer.mapping_id,
                producer_to_consumer.validated,
            )
        else:
            mapping_fingerprint = ("none" if producer_to_consumer is None else "invalid",)
        if not self._has_consumer_mapping_fingerprint:
            self._last_consumer_mapping_fingerprint = mapping_fingerprint
            self._has_consumer_mapping_fingerprint = True
        elif mapping_fingerprint != self._last_consumer_mapping_fingerprint:
            self._consumer_mapping_changed_without_epoch = True
            self._last_consumer_mapping_fingerprint = mapping_fingerprint
        if self._consumer_mapping_changed_without_epoch:
            missing.append("consumer_clock_mapping_changed_without_epoch")

        if source_ns is not None and new_frame and not epoch_changed:
            if (self._last_source_mono_ns is not None
                    and source_ns < self._last_source_mono_ns):
                missing.append("source_mono_clock_regressed")
            else:
                self._last_source_mono_ns = source_ns
        elif source_ns is not None and new_frame and epoch_changed:
            self._last_source_mono_ns = source_ns

        if (consume_mono_ns is not None and _is_nonnegative_int(consume_mono_ns)
                and isinstance(consume_clock_id, str) and consume_clock_id):
            if (self._last_consume_clock_id is not None
                    and consume_clock_id != self._last_consume_clock_id):
                self._consume_clock_changed_without_epoch = True
            if (not epoch_changed and consume_clock_id == self._last_consume_clock_id
                    and self._last_consume_mono_ns is not None
                    and consume_mono_ns < self._last_consume_mono_ns):
                missing.append("consume_mono_clock_regressed")
            else:
                self._last_consume_clock_id = consume_clock_id
                self._last_consume_mono_ns = consume_mono_ns
        if self._consume_clock_changed_without_epoch:
            missing.append("consume_clock_changed_without_epoch")

        error_bound: Optional[int] = None
        mapped_source_ns: Optional[int] = None
        if not missing:
            assert source_ns is not None and consume_mono_ns is not None
            assert isinstance(consume_clock_id, str) and producer_clock_id is not None
            if producer_clock_id == consume_clock_id:
                mapped_source_ns = source_ns
                error_bound = 0
            else:
                mapping = producer_to_consumer
                if not isinstance(mapping, MonotonicClockMapping):
                    missing.append("validated_producer_to_consumer_clock_mapping")
                elif not mapping.validated:
                    missing.append("producer_to_consumer_clock_mapping_unvalidated")
                elif (mapping.source_clock_id != producer_clock_id
                      or mapping.target_clock_id != consume_clock_id):
                    missing.append("producer_to_consumer_clock_ids_mismatch")
                else:
                    mapped_source_ns = source_ns + mapping.offset_ns
                    error_bound = mapping.error_bound_ns

        age: Optional[float] = None
        age_ratio: Optional[float] = None
        if not missing and mapped_source_ns is not None:
            assert consume_mono_ns is not None
            delta_ns = consume_mono_ns - mapped_source_ns
            if delta_ns < 0:
                missing.append("source_time_in_future")
            else:
                age = delta_ns / 1_000_000_000.0
                age_ratio = min(age / 2.0, 1.0)
        if missing:
            return None, None, error_bound, missing
        return age, age_ratio, error_bound, missing


_DEFAULT_ADAPTER = DepthAdapter()


def decode_frame(
    metadata: dict,
    payload: bytes,
    *,
    adapter: Optional[DepthAdapter] = None,
    source_mask: Optional[np.ndarray] = None,
    consume_mono_ns: Optional[int] = None,
    consume_clock_id: Optional[str] = None,
    consume_pipeline_epoch: Any = None,
    producer_to_consumer: Optional[MonotonicClockMapping] = None,
    source_state: Optional[SourceStateEvidence] = None,
) -> AdaptedDepthFrame:
    """Convenience entry point; pass one stateful adapter per consumer stream."""
    active = _DEFAULT_ADAPTER if adapter is None else adapter
    return active.decode_frame(
        metadata,
        payload,
        source_mask=source_mask,
        consume_mono_ns=consume_mono_ns,
        consume_clock_id=consume_clock_id,
        consume_pipeline_epoch=consume_pipeline_epoch,
        producer_to_consumer=producer_to_consumer,
        source_state=source_state,
    )

