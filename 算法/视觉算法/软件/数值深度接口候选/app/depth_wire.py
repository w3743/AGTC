"""Depth-Z immutable wire codec and producer geometry helpers.

This module is deliberately independent of camera/GStreamer and policy code.
The semantic source of truth is 算法/视觉算法/深度接口契约.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "depth-z-u16le-v1"
SOURCE_WIDTH = 131
SOURCE_HEIGHT = 240
MIN_MM = 300
MAX_MM = 5000
CONVENTION = "camera_z"
FRAME_ID = "left_rectified_optical"
REQUIRED_METADATA = (
    "schema", "session_id", "pipeline_epoch", "seq", "width", "height",
    "encoding", "byte_order", "unit", "convention", "frame_id", "K",
    "calibration_id", "extrinsics_id", "preprocessing_id", "min_mm", "max_mm",
    "left_pts_ns", "right_pts_ns", "publish_mono_ns", "source_mono_ns",
    "clock_mapping_id", "clock_quality", "monotonic_clock_id",
    "source_time_reference",
)
OPTIONAL_NS_FIELDS = ("left_arrival_mono_ns", "right_arrival_mono_ns")
NS_FIELDS = ("left_pts_ns", "right_pts_ns", "publish_mono_ns", "source_mono_ns") + OPTIONAL_NS_FIELDS


@dataclass(frozen=True)
class DepthRecord:
    """Immutable wire snapshot; metadata and payload are stored together."""

    seq: int
    width: int
    height: int
    payload: bytes
    metadata_ascii: str

    @property
    def metadata(self) -> Dict[str, Any]:
        # Return a fresh mapping so callers cannot mutate the cached record.
        return json.loads(self.metadata_ascii)

    @property
    def millimetres(self) -> np.ndarray:
        view = np.frombuffer(self.payload, dtype=np.dtype("<u2")).reshape((self.height, self.width))
        view.setflags(write=False)
        return view

    def history_entry(self) -> Tuple[int, np.ndarray, Optional[int], "DepthRecord"]:
        """Keep the existing (seq, mm, left_pts) prefix for measure_point/range_frame."""
        return self.seq, self.millimetres, self.metadata["left_pts_ns"], self


def encode_depth_payload(depth_m: np.ndarray,
                         min_mm: int = MIN_MM,
                         max_mm: int = MAX_MM) -> Tuple[bytes, int, int]:
    """Encode axis-Z metres as C-order uint16 little-endian millimetres.

    Non-C-contiguous input is rejected instead of silently turning a transposed
    or strided source into a plausible but geometrically mislabelled raster.
    """
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth must be a non-empty HxW array")
    if not depth.flags.c_contiguous:
        raise ValueError("depth raster must be C-contiguous row-major")
    if depth.dtype.kind not in "fiu":
        raise ValueError("depth raster must have a numeric dtype")
    if min_mm != MIN_MM or max_mm != MAX_MM:
        raise ValueError("depth-z-u16le-v1 range is fixed at 300..5000 mm")
    # Test limits against source precision before conversion; float32 rounding
    # must not turn a just-outside float64 value into a valid endpoint sample.
    values = np.asarray(depth)
    millimetres = np.zeros(values.shape, dtype=np.dtype("<u2"))
    valid = np.isfinite(values) & (values >= min_mm / 1000.0) & (values <= max_mm / 1000.0)
    if np.any(valid):
        rounded = np.rint(values[valid].astype(np.float64) * 1000.0)
        millimetres[valid] = np.clip(rounded, 1, max_mm).astype(np.dtype("<u2"))
    height, width = millimetres.shape
    return millimetres.tobytes(order="C"), width, height


def _strict_json_loads(value: str) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError("non-finite JSON number: " + token)
    return json.loads(value, parse_constant=reject_constant)


def validate_metadata(metadata: Mapping[str, Any],
                      expected_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """Validate required field structure and fixed wire encoding; do not claim readiness."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object")
    result = dict(metadata)
    missing = [name for name in REQUIRED_METADATA if name not in result]
    if missing:
        raise ValueError("metadata missing required fields: " + ", ".join(missing))
    if result["schema"] != SCHEMA:
        raise ValueError("unsupported depth schema")
    if result["encoding"] != "16UC1" or result["byte_order"] != "little" or result["unit"] != "mm":
        raise ValueError("unsupported encoding, byte order, or unit")
    if result["convention"] != CONVENTION:
        raise ValueError("unsupported coordinate/depth convention")
    if not isinstance(result["frame_id"], str) or not result["frame_id"]:
        raise ValueError("frame_id must be a non-empty string")
    if result["source_time_reference"] != "left_pts":
        raise ValueError("source_time_reference must be left_pts")
    if result["min_mm"] != MIN_MM or result["max_mm"] != MAX_MM:
        raise ValueError("unexpected profile range")
    for key in ("seq", "pipeline_epoch", "width", "height"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(key + " must be a non-negative integer")
    if result["seq"] < 1 or result["width"] < 1 or result["height"] < 1:
        raise ValueError("seq and image dimensions must be positive")
    if expected_shape is not None and (result["height"], result["width"]) != tuple(expected_shape):
        raise ValueError("depth dimensions do not match the configured profile")
    for key in NS_FIELDS:
        value = result.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(key + " must be a non-negative integer or null")
    if not isinstance(result["session_id"], str) or not result["session_id"]:
        raise ValueError("session_id must identify its producer session")
    clock_id = result["monotonic_clock_id"]
    if clock_id is not None and (not isinstance(clock_id, str) or not clock_id):
        raise ValueError("monotonic_clock_id must be a non-empty string or null")
    for key in ("calibration_id", "extrinsics_id", "preprocessing_id", "clock_mapping_id", "clock_quality"):
        if result[key] is not None and not isinstance(result[key], str):
            raise ValueError(key + " must be a string or null")
    if result["K"] is not None:
        values = np.asarray(result["K"], dtype=np.float64)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("K must be nine finite row-major values or null")
        if values[0] <= 0 or values[4] <= 0 or not np.allclose(values[6:9], [0, 0, 1], atol=1e-9):
            raise ValueError("K is not a valid pinhole intrinsic matrix")
        result["K"] = [float(x) for x in values]
    if "row_stride_bytes" in result and result["row_stride_bytes"] != result["width"] * 2:
        raise ValueError("row padding is not supported")
    return result


def build_depth_record(payload: bytes,
                       metadata: Mapping[str, Any],
                       expected_shape: Optional[Tuple[int, int]] = (SOURCE_HEIGHT, SOURCE_WIDTH)) -> DepthRecord:
    if not isinstance(payload, bytes):
        raise ValueError("payload must be immutable bytes")
    checked = validate_metadata(metadata, expected_shape=expected_shape)
    width, height = checked["width"], checked["height"]
    expected_bytes = width * height * 2
    if len(payload) != expected_bytes:
        raise ValueError("payload byte length does not equal width*height*2")
    header = json.dumps(checked, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    # The HTTP header value is intentionally ASCII even when additional metadata is added.
    header.encode("ascii", errors="strict")
    return DepthRecord(checked["seq"], width, height, payload, header)


def parse_depth_metadata(value: Any,
                         expected_shape: Optional[Tuple[int, int]] = (SOURCE_HEIGHT, SOURCE_WIDTH)) -> Dict[str, Any]:
    if isinstance(value, bytes):
        text = value.decode("ascii", errors="strict")
    elif isinstance(value, str):
        value.encode("ascii", errors="strict")
        text = value
    else:
        raise ValueError("metadata must be ASCII JSON")
    parsed = _strict_json_loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object")
    return validate_metadata(parsed, expected_shape=expected_shape)


def decode_depth_payload(payload: bytes,
                         metadata: Any,
                         expected_shape: Optional[Tuple[int, int]] = (SOURCE_HEIGHT, SOURCE_WIDTH)) -> np.ndarray:
    checked = parse_depth_metadata(metadata, expected_shape=expected_shape)
    if not isinstance(payload, bytes):
        raise ValueError("payload must be bytes; typed arrays need an explicit wire encoding")
    expected_bytes = checked["width"] * checked["height"] * 2
    if len(payload) != expected_bytes:
        raise ValueError("payload length mismatch; padding/truncation is forbidden")
    result = np.frombuffer(payload, dtype=np.dtype("<u2"))
    result = result.reshape((checked["height"], checked["width"]))
    result.setflags(write=False)
    return result


def fast_geometry_descriptor(metric: Any, calibration_path: Path) -> Dict[str, Any]:
    """Derive exact fast-path K from the engine's effective A@P1, crop and CCW rotation."""
    required = ("raw_width", "raw_height", "map_scale", "crop", "width", "height", "input_size", "scale")
    if any(not hasattr(metric, name) for name in required):
        raise ValueError("active depth backend does not expose the fast SGBM geometry")
    raw_width = int(metric.raw_width)
    raw_height = int(metric.raw_height)
    crop = int(metric.crop)
    width = int(metric.width)
    height = int(metric.height)
    if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
        raise ValueError("depth wire profile is available only for the calibrated 131x240 fast raster")
    depth_range = tuple(float(value) for value in metric.RANGE_M)
    if depth_range != (MIN_MM / 1000.0, MAX_MM / 1000.0):
        raise ValueError("active fast backend range disagrees with the fixed 0.3..5.0 m wire profile")
    if width != raw_height - 2 * crop or height != raw_width:
        raise ValueError("fast backend dimensions disagree with rotate/crop geometry")
    scale = raw_width / 3840.0
    if not math.isclose(float(metric.map_scale), scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fast backend map scale is inconsistent with calibration resize")
    path = Path(calibration_path)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    calibration_id = file_sha[:12]
    engine_id = getattr(metric, "identifier", None)
    if engine_id is not None and str(engine_id) != calibration_id:
        raise ValueError("active metric and calibration source do not identify the same calibration")
    with np.load(str(path), allow_pickle=False) as source:
        if "P1" not in source.files:
            raise ValueError("source calibration has no P1 projection")
        p1 = np.asarray(source["P1"], dtype=np.float64)
        if p1.shape != (3, 4) or not np.isfinite(p1).all():
            raise ValueError("source calibration P1 is invalid")
        resize = np.array([[scale, 0.0, (scale - 1.0) / 2.0],
                           [0.0, scale, (scale - 1.0) / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        scaled_p1 = resize @ p1
        # CCW rotation maps (x,y) to (y, raw_width-1-x); crop removes
        # `crop` columns on each side after rotation. Camera axes rotate too,
        # keeping positive fx/fy and camera-axis Z.
        k = np.array([[scaled_p1[1, 1], 0.0, scaled_p1[1, 2] - crop],
                      [0.0, scaled_p1[0, 0], raw_width - 1.0 - scaled_p1[0, 2]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("derived fast K is invalid")
    preprocess = {
        "backend": "fast-sgbm",
        "algorithm": str(getattr(metric, "algorithm", "Calibrated SGBM CPU")),
        "source_module_sha256": hashlib.sha256(
            Path(__file__).with_name("fast_depth.py").read_bytes()).hexdigest(),
        "num_disparities": int(metric.num_disparities),
        "block_size": int(metric.matcher.getBlockSize()),
        "input_size": [int(metric.input_size[0]), int(metric.input_size[1])],
        "scale": float(metric.scale),
        "raw_width": raw_width,
        "raw_height": raw_height,
        "crop_each_side": crop,
        "rotation": "counterclockwise-90",
        "resize": "area",
        "rectify": "linear",
        "range_mm": [MIN_MM, MAX_MM],
    }
    preprocessing_id = "fast-preprocess-" + hashlib.sha256(
        json.dumps(preprocess, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()[:16]
    return {
        "width": width,
        "height": height,
        "K": tuple(float(x) for x in k.reshape(-1)),
        "calibration_id": calibration_id,
        # Stereo rectification does not bind the camera to the robot body or
        # pan/tilt axes, so this candidate must not claim an extrinsics ID.
        "extrinsics_id": None,
        "preprocessing_id": preprocessing_id,
        "frame_id": FRAME_ID,
    }


def parse_requested_sequence(query: Mapping[str, Sequence[str]]) -> Optional[int]:
    values = query.get("sequence")
    if values is None:
        return None
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].isascii():
        raise ValueError("sequence must be one decimal integer")
    value = values[0]
    if not value or not value.isdigit() or len(value) > 19:
        raise ValueError("sequence must be one decimal integer")
    sequence = int(value, 10)
    if sequence < 1 or sequence > (1 << 63) - 1:
        raise ValueError("sequence is outside the supported range")
    return sequence


def select_depth_record(history: Iterable[Sequence[Any]], sequence: Optional[int] = None
                        ) -> Tuple[Optional[DepthRecord], int, str]:
    """Resolve an immutable history entry to (record, HTTP status, reason)."""
    entries = list(history)
    if not entries:
        return None, 503, "no depth frame is available"
    if sequence is None:
        record = entries[-1][3] if len(entries[-1]) > 3 else None
        if isinstance(record, DepthRecord):
            return record, 200, "ok"
        return None, 503, "latest history entry has no wire record"
    for entry in reversed(entries):
        if entry[0] == sequence:
            record = entry[3] if len(entry) > 3 else None
            if isinstance(record, DepthRecord):
                return record, 200, "ok"
            return None, 409, "requested sequence has no wire record"
    return None, 409, "requested sequence is expired or unavailable"
