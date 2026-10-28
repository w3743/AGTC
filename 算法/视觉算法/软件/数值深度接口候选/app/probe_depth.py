"""Continuously sample and preserve the candidate numeric depth HTTP wire."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np

import depth_wire


PAYLOAD_BYTES = 62_880
EXPECTED_SHAPE = (240, 131)
AGE_ROUNDING_ERROR_NS = 100_000  # /hardware/status rounds sample_age_ms to 0.1 ms.
REPORT_SCHEMA = "depth-http-probe-v1"


def _sha256(data: Optional[bytes]) -> Optional[str]:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _header_bytes(value: str) -> bytes:
    # http.client maps each wire header byte to ISO-8859-1; preserve those bytes
    # for the exact-header digest even when an invalid non-ASCII value arrives.
    return value.encode("latin-1", errors="strict")


def _all_headers(headers: Any) -> Dict[str, List[str]]:
    items = headers.items() if isinstance(headers, Mapping) else headers
    result: Dict[str, List[str]] = {}
    for name, value in items:
        result.setdefault(str(name).casefold(), []).append(str(value))
    return result


def _one_header(headers: Dict[str, List[str]], name: str) -> Optional[str]:
    values = headers.get(name.casefold(), [])
    return values[0] if len(values) == 1 else None


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min_ms": None, "p50_ms": None,
                "p95_ms": None, "max_ms": None}
    ordered = [float(value) for value in values]
    return {
        "count": len(ordered),
        "min_ms": min(ordered),
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "max_ms": max(ordered),
    }


class DepthProbeCollector:
    """Validate response rows, identify exact repeats, and archive unique frames."""

    def __init__(self, archive_path: Optional[Path] = None):
        self.requests: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.external_errors: List[Dict[str, Any]] = []
        self.archive_path = Path(archive_path) if archive_path is not None else None
        self.archive_complete = archive_path is not None
        self._archive_file = None
        self._archive_gzip = None
        self._spool = tempfile.TemporaryFile(mode="w+b")
        self._seen: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._last_epoch_by_session: Dict[str, int] = {}
        self._last_seq_by_stream: Dict[Tuple[str, int], int] = {}
        self._last_pts_by_stream: Dict[Tuple[str, int, str], int] = {}
        self._last_publish_by_session: Dict[str, int] = {}
        self.unique_frames = 0
        self.duplicate_count = 0
        self.duplicate_mismatch_count = 0
        self.bad_response_count = 0
        self.http_error_count = 0
        self.transport_error_count = 0
        self.sequence_gap_events = 0
        self.consumer_unobserved_sequence_count = 0
        self.temporal_regression_count = 0
        if self.archive_path is not None:
            try:
                self.archive_path.parent.mkdir(parents=True, exist_ok=True)
                self._archive_file = self.archive_path.open("wb")
                self._archive_gzip = gzip.GzipFile(
                    fileobj=self._archive_file, mode="wb", compresslevel=1, mtime=0)
            except OSError as exc:
                self.archive_complete = False
                self._add_external_error("archive_open_error", str(exc))

    def _add_external_error(self, code: str, message: str, **details: Any) -> None:
        row = {"code": code, "message": message}
        row.update(details)
        self.external_errors.append(row)
        self.errors.append(row.copy())

    def add_external_error(self, code: str, message: str, **details: Any) -> None:
        self._add_external_error(code, message, **details)

    def ingest_transport_error(self, *, host_send_monotonic_ns: int,
                               host_receive_monotonic_ns: int,
                               message: str, partial_body: Optional[bytes] = None,
                               status: Optional[int] = None,
                               headers: Any = ()) -> Dict[str, Any]:
        if status is not None:
            row = self.ingest_response(
                status=status, headers=headers, body=partial_body or b"",
                host_send_monotonic_ns=host_send_monotonic_ns,
                host_receive_monotonic_ns=host_receive_monotonic_ns,
                response_complete=False)
            row["transport_error"] = message
            self.transport_error_count += 1
            self._record_row_error(row, "transport_error", message)
            return row
        row = self._empty_request_row(host_send_monotonic_ns, host_receive_monotonic_ns)
        row.update({
            "response_complete": False,
            "payload_length": len(partial_body) if partial_body is not None else None,
            "payload_sha256": _sha256(partial_body),
            "transport_error": message,
        })
        self.transport_error_count += 1
        self._record_row_error(row, "transport_error", message)
        self.requests.append(row)
        return row

    def _empty_request_row(self, send_ns: int, receive_ns: int) -> Dict[str, Any]:
        return {
            "request_index": len(self.requests),
            "status": None,
            "response_complete": True,
            "host_send_monotonic_ns": int(send_ns),
            "host_receive_monotonic_ns": int(receive_ns),
            "rtt_ms": max(0, int(receive_ns) - int(send_ns)) / 1_000_000.0,
            "content_type": None,
            "content_length_header": None,
            "cache_control": None,
            "payload_length": None,
            "payload_sha256": None,
            "metadata_ascii": None,
            "metadata_ascii_sha256": None,
            "metadata": None,
            "frame_key": None,
            "archive_frame_index": None,
            "is_duplicate": False,
            "validation_errors": [],
            "temporal_errors": [],
            "consumer_unobserved_sequence_count_before": 0,
            "left_arrival_to_publish_ms": None,
            "right_arrival_to_publish_ms": None,
            "left_minus_right_pts_skew_ms": None,
            "estimated_publish_to_receive_ms_bounds": None,
            "source_to_publish_ms": None,
        }

    def _record_row_error(self, row: Dict[str, Any], code: str,
                          message: str, **details: Any) -> None:
        error = {"code": code, "message": message,
                 "request_index": row["request_index"]}
        error.update(details)
        self.errors.append(error)

    def _archive_payload(self, payload: bytes) -> Optional[int]:
        spool_index = self.unique_frames
        self._spool.seek(spool_index * PAYLOAD_BYTES)
        self._spool.write(payload)
        archive_index: Optional[int] = None
        if self._archive_gzip is not None and self.archive_complete:
            try:
                self._archive_gzip.write(payload)
                archive_index = spool_index
            except OSError as exc:
                self.archive_complete = False
                self._add_external_error("archive_write_error", str(exc),
                                         archive_frame_index=spool_index)
                try:
                    self._archive_gzip.close()
                except OSError:
                    pass
                self._archive_gzip = None
        return archive_index

    def _same_stored_payload(self, spool_index: int, payload: bytes) -> bool:
        self._spool.seek(spool_index * PAYLOAD_BYTES)
        return self._spool.read(PAYLOAD_BYTES) == payload

    def _temporal_checks(self, row: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        session = metadata["session_id"]
        epoch = metadata["pipeline_epoch"]
        seq = metadata["seq"]
        stream = (session, epoch)
        prior_epoch = self._last_epoch_by_session.get(session)
        if prior_epoch is not None and epoch < prior_epoch:
            self._record_temporal_error(row, "pipeline_epoch_rollback",
                                        "pipeline_epoch moved backwards",
                                        previous=prior_epoch, current=epoch)
        self._last_epoch_by_session[session] = max(epoch, prior_epoch or epoch)

        prior_seq = self._last_seq_by_stream.get(stream)
        if prior_seq is not None:
            if seq < prior_seq:
                self._record_temporal_error(row, "sequence_rollback",
                                            "seq moved backwards within a session and epoch",
                                            previous=prior_seq, current=seq)
            elif seq > prior_seq + 1:
                missing = seq - prior_seq - 1
                row["consumer_unobserved_sequence_count_before"] = missing
                self.consumer_unobserved_sequence_count += missing
                self.sequence_gap_events += 1
            self._last_seq_by_stream[stream] = max(prior_seq, seq)
        else:
            self._last_seq_by_stream[stream] = seq

        for field in ("left_pts_ns", "right_pts_ns"):
            value = metadata.get(field)
            if value is None:
                continue
            key = (session, epoch, field)
            prior = self._last_pts_by_stream.get(key)
            if prior is not None and value < prior:
                self._record_temporal_error(row, "source_pts_rollback",
                                            field + " moved backwards",
                                            field=field, previous=prior, current=value)
            self._last_pts_by_stream[key] = max(prior or value, value)

        publish = metadata.get("publish_mono_ns")
        if publish is not None:
            prior_publish = self._last_publish_by_session.get(session)
            if prior_publish is not None and publish < prior_publish:
                self._record_temporal_error(row, "publish_time_rollback",
                                            "publish_mono_ns moved backwards",
                                            previous=prior_publish, current=publish)
            self._last_publish_by_session[session] = max(prior_publish or publish, publish)

    def _record_temporal_error(self, row: Dict[str, Any], code: str, message: str,
                               **details: Any) -> None:
        event = {"code": code, "message": message}
        event.update(details)
        row["temporal_errors"].append(event)
        self.temporal_regression_count += 1
        self._record_row_error(row, code, message, **details)

    def ingest_response(self, *, status: int, headers: Any, body: bytes,
                        host_send_monotonic_ns: int,
                        host_receive_monotonic_ns: int,
                        response_complete: bool = True) -> Dict[str, Any]:
        row = self._empty_request_row(host_send_monotonic_ns,
                                      host_receive_monotonic_ns)
        row["status"] = int(status)
        row["response_complete"] = bool(response_complete)
        row["payload_length"] = len(body)
        row["payload_sha256"] = _sha256(body)
        all_headers = _all_headers(headers)
        metadata_values = all_headers.get("x-depth-metadata", [])
        metadata_ascii = metadata_values[0] if metadata_values else None
        row["metadata_ascii"] = metadata_ascii
        try:
            row["metadata_ascii_sha256"] = (
                hashlib.sha256(_header_bytes(metadata_ascii)).hexdigest()
                if metadata_ascii is not None else None)
        except UnicodeEncodeError:
            row["metadata_ascii_sha256"] = None

        row["content_type"] = _one_header(all_headers, "content-type")
        row["content_length_header"] = _one_header(all_headers, "content-length")
        row["cache_control"] = _one_header(all_headers, "cache-control")

        if status != 200:
            self.http_error_count += 1
            self._record_row_error(row, "http_status_error",
                                   "depth endpoint returned a non-200 status",
                                   status=int(status))
            self.requests.append(row)
            return row

        validation_errors: List[str] = row["validation_errors"]
        if not response_complete:
            validation_errors.append("HTTP response body was incomplete")
        if len(all_headers.get("content-type", [])) != 1:
            validation_errors.append("Content-Type header must occur exactly once")
        elif (row["content_type"] or "").strip().casefold() != "application/octet-stream":
            validation_errors.append("Content-Type must be application/octet-stream")
        if len(all_headers.get("cache-control", [])) != 1:
            validation_errors.append("Cache-Control header must occur exactly once")
        elif (row["cache_control"] or "").strip().casefold() != "no-store":
            validation_errors.append("Cache-Control must be no-store")
        if len(all_headers.get("content-length", [])) != 1:
            validation_errors.append("Content-Length header must occur exactly once")
        else:
            length_value = row["content_length_header"] or ""
            if not re.fullmatch(r"[0-9]+", length_value.strip()):
                validation_errors.append("Content-Length must be a decimal integer")
            elif int(length_value.strip()) != len(body):
                validation_errors.append("Content-Length does not equal the received body length")
        if len(metadata_values) != 1:
            validation_errors.append("X-Depth-Metadata header must occur exactly once")
        elif metadata_ascii is not None and not metadata_ascii.isascii():
            validation_errors.append("X-Depth-Metadata must contain ASCII only")
        if len(body) != PAYLOAD_BYTES:
            validation_errors.append("payload length must be exactly 62880 bytes")

        metadata: Optional[Dict[str, Any]] = None
        if len(metadata_values) == 1 and metadata_ascii is not None:
            try:
                metadata = depth_wire.parse_depth_metadata(
                    metadata_ascii, expected_shape=EXPECTED_SHAPE)
                row["metadata"] = metadata
                row["frame_key"] = [metadata["session_id"],
                                    metadata["pipeline_epoch"], metadata["seq"]]
            except (ValueError, TypeError, UnicodeError) as exc:
                validation_errors.append("invalid depth metadata: " + str(exc))

        if metadata is not None:
            try:
                decoded = depth_wire.decode_depth_payload(
                    body, metadata_ascii, expected_shape=EXPECTED_SHAPE)
                if not bool(np.all((decoded == 0) |
                                   ((decoded >= depth_wire.MIN_MM) &
                                    (decoded <= depth_wire.MAX_MM)))):
                    validation_errors.append("depth samples must be 0 or within 300..5000 mm")
            except (ValueError, TypeError, UnicodeError) as exc:
                validation_errors.append("invalid depth payload: " + str(exc))

        key = None
        if metadata is not None:
            key = (metadata["session_id"], metadata["pipeline_epoch"], metadata["seq"])
        stored = self._seen.get(key) if key is not None else None
        if stored is not None:
            row["is_duplicate"] = True
            row["archive_frame_index"] = stored["archive_frame_index"]
            same_header = metadata_ascii == stored["metadata_ascii"]
            same_payload = self._same_stored_payload(stored["spool_index"], body)
            row["duplicate_bytes_match"] = same_header and same_payload
            if same_header and same_payload:
                self.duplicate_count += 1
            else:
                self.duplicate_mismatch_count += 1
                validation_errors.append(
                    "repeated frame key has different raw metadata header or payload bytes")
            if metadata is not None:
                self._temporal_checks(row, metadata)

        if validation_errors:
            self.bad_response_count += 1
            for message in validation_errors:
                self._record_row_error(row, "bad_response", message)

        if stored is not None:
            self.requests.append(row)
            return row

        if validation_errors or metadata is None or key is None:
            self.requests.append(row)
            return row

        self._temporal_checks(row, metadata)
        publish = metadata.get("publish_mono_ns")
        left_arrival = metadata.get("left_arrival_mono_ns")
        right_arrival = metadata.get("right_arrival_mono_ns")
        left_pts = metadata.get("left_pts_ns")
        right_pts = metadata.get("right_pts_ns")
        if publish is not None and left_arrival is not None:
            row["left_arrival_to_publish_ms"] = (publish - left_arrival) / 1_000_000.0
            if publish < left_arrival:
                self._record_temporal_error(
                    row, "publish_before_left_arrival",
                    "publish_mono_ns precedes left_arrival_mono_ns",
                    publish_mono_ns=publish, left_arrival_mono_ns=left_arrival)
        if publish is not None and right_arrival is not None:
            row["right_arrival_to_publish_ms"] = (publish - right_arrival) / 1_000_000.0
            if publish < right_arrival:
                self._record_temporal_error(
                    row, "publish_before_right_arrival",
                    "publish_mono_ns precedes right_arrival_mono_ns",
                    publish_mono_ns=publish, right_arrival_mono_ns=right_arrival)
        if left_pts is not None and right_pts is not None:
            row["left_minus_right_pts_skew_ms"] = (left_pts - right_pts) / 1_000_000.0

        archive_index = self._archive_payload(body)
        row["archive_frame_index"] = archive_index
        spool_index = self.unique_frames
        self._seen[key] = {
            "metadata_ascii": metadata_ascii,
            "spool_index": spool_index,
            "archive_frame_index": archive_index,
        }
        self.unique_frames += 1
        self.requests.append(row)
        return row

    def close_archive(self) -> None:
        if self._archive_gzip is not None:
            try:
                self._archive_gzip.close()
            except OSError as exc:
                self.archive_complete = False
                self._add_external_error("archive_close_error", str(exc))
            self._archive_gzip = None
        if self._archive_file is not None:
            try:
                self._archive_file.close()
            except OSError as exc:
                self.archive_complete = False
                self._add_external_error("archive_close_error", str(exc))
            self._archive_file = None
        try:
            self._spool.close()
        except OSError as exc:
            self._add_external_error("spool_close_error", str(exc))

    def _aggregate_clock_bounds(self, hardware_status: Sequence[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        valid = [row["board_minus_host_offset_bounds_ns"] for row in hardware_status
                 if row.get("board_minus_host_offset_bounds_ns") is not None]
        if not valid:
            return None
        return {
            "lower_ns": min(item["lower_ns"] for item in valid),
            "upper_ns": max(item["upper_ns"] for item in valid),
        }

    def build_report(self, *, url: str, duration_s: float, interval_s: float,
                     output_path: Path, hardware_status: Sequence[Dict[str, Any]],
                     started_monotonic_ns: Optional[int],
                     finished_monotonic_ns: Optional[int], interrupted: bool = False) -> Dict[str, Any]:
        offset_bounds = self._aggregate_clock_bounds(hardware_status)
        publish_age_lower: List[float] = []
        publish_age_upper: List[float] = []
        rtts = [row["rtt_ms"] for row in self.requests if row.get("rtt_ms") is not None]
        left_arrival_values: List[float] = []
        right_arrival_values: List[float] = []
        pts_skew_values: List[float] = []
        abs_pts_skew_values: List[float] = []
        source_publish_values: List[float] = []
        unique_rows = [row for row in self.requests
                       if row.get("frame_key") is not None and not row.get("is_duplicate")
                       and not row.get("validation_errors")]
        for row in self.requests:
            metadata = row.get("metadata")
            if not metadata or row.get("validation_errors"):
                continue
            if row.get("left_arrival_to_publish_ms") is not None:
                left_arrival_values.append(row["left_arrival_to_publish_ms"])
            if row.get("right_arrival_to_publish_ms") is not None:
                right_arrival_values.append(row["right_arrival_to_publish_ms"])
            if row.get("left_minus_right_pts_skew_ms") is not None:
                pts_skew_values.append(row["left_minus_right_pts_skew_ms"])
                abs_pts_skew_values.append(abs(row["left_minus_right_pts_skew_ms"]))
            if row.get("source_to_publish_ms") is not None:
                source_publish_values.append(row["source_to_publish_ms"])
        if offset_bounds is not None:
            for row in unique_rows:
                publish = row["metadata"].get("publish_mono_ns")
                host_receive = row.get("host_receive_monotonic_ns")
                if publish is None or host_receive is None:
                    continue
                lower = (host_receive + offset_bounds["lower_ns"] - publish) / 1_000_000.0
                upper = (host_receive + offset_bounds["upper_ns"] - publish) / 1_000_000.0
                row["estimated_publish_to_receive_ms_bounds"] = {
                    "lower_ms": lower, "upper_ms": upper,
                }
                publish_age_lower.append(lower)
                publish_age_upper.append(upper)

        if started_monotonic_ns is not None and finished_monotonic_ns is not None:
            sample_duration = max(0, finished_monotonic_ns - started_monotonic_ns) / 1e9
        else:
            sample_duration = 0.0
        unique_publish_values = [row["metadata"].get("publish_mono_ns")
                                 for row in unique_rows
                                 if row["metadata"].get("publish_mono_ns") is not None]
        unique_host_values = [row.get("host_receive_monotonic_ns") for row in unique_rows
                              if row.get("host_receive_monotonic_ns") is not None]
        publish_span = (unique_publish_values[-1] - unique_publish_values[0]
                        if len(unique_publish_values) > 1 else None)
        host_span = (unique_host_values[-1] - unique_host_values[0]
                     if len(unique_host_values) > 1 else None)
        request_count = len(self.requests)
        archive_name = self.archive_path.name if self.archive_path is not None else None
        summary = {
            "requested_duration_s": duration_s,
            "sample_duration_s": sample_duration,
            "sampling_interval_s": interval_s,
            "http_request_count": request_count,
            "http_request_rate_hz": (request_count / sample_duration
                                     if sample_duration > 0 else None),
            "unique_frames": self.unique_frames,
            "unique_frame_rate_by_publish_mono_hz": (
                (len(unique_publish_values) - 1) / (publish_span / 1e9)
                if len(unique_publish_values) > 1 and publish_span is not None
                and publish_span > 0 else None),
            "unique_frame_rate_by_host_observation_hz": (
                (self.unique_frames - 1) / (host_span / 1e9)
                if host_span is not None and host_span > 0 else None),
            "first_last_publish_span_ms": (publish_span / 1e6
                                            if publish_span is not None else None),
            "first_last_host_observation_span_ms": (host_span / 1e6
                                                    if host_span is not None else None),
            "rtt_ms": _stats(rtts),
            "bad_response_count": self.bad_response_count,
            "http_error_count": self.http_error_count,
            "transport_error_count": self.transport_error_count,
            "duplicate_count": self.duplicate_count,
            "duplicate_mismatch_count": self.duplicate_mismatch_count,
            "sequence_gap_events": self.sequence_gap_events,
            "consumer_unobserved_sequence_count": self.consumer_unobserved_sequence_count,
            "sequence_gap_accounting_note": (
                "Sequence gaps count unobserved numbers at each forward jump (later arrivals do not subtract); they may result from polling skips "
                "or publications without a wire frame, and do not prove camera or transport loss."),
            "temporal_regression_count": self.temporal_regression_count,
            "left_arrival_to_publish_ms": _stats(left_arrival_values),
            "right_arrival_to_publish_ms": _stats(right_arrival_values),
            "left_minus_right_pts_skew_ms": _stats(pts_skew_values),
            "absolute_left_right_pts_skew_ms": _stats(abs_pts_skew_values),
            "source_to_publish_ms": _stats(source_publish_values),
            "source_to_publish_status": "unknown_without_verified_source_clock_mapping",
            "source_to_publish_unknown_count": sum(
                1 for row in unique_rows if row.get("source_to_publish_ms") is None),
            "estimated_publish_to_receive_ms_bounds": {
                "lower_bound_ms": _stats(publish_age_lower),
                "upper_bound_ms": _stats(publish_age_upper),
            },
            "scope_excludes_exposure_age": True,
            "clock_mapping_validated": False,
            "legacy_ready": False,
            "archive_path": archive_name,
            "archive_frame_index_base": 0,
            "archive_complete": bool(self.archive_complete),
            "uncompressed_bytes": self.unique_frames * PAYLOAD_BYTES,
            "error_count": len(self.errors),
            "interrupted": bool(interrupted),
        }
        request_rows = []
        for source_row in self.requests:
            row = dict(source_row)
            if row.get("is_duplicate") and row.get("duplicate_bytes_match"):
                row["metadata_ascii"] = None
                row["metadata"] = None
                row["metadata_reference_frame_index"] = row.get("archive_frame_index")
            request_rows.append(row)
        return {
            "schema": REPORT_SCHEMA,
            "url": url,
            "output": output_path.name,
            "probe": {
                "duration_s": duration_s,
                "interval_s": interval_s,
                "payload_bytes_expected": PAYLOAD_BYTES,
                "depth_range_mm": [depth_wire.MIN_MM, depth_wire.MAX_MM],
                "zero_means_invalid_depth": True,
                "source_age": None,
                "source_age_status": "unknown_without_verified_source_clock_mapping",
                "legacy_ready": False,
            },
            "clock_diagnostics": {
                "host_clock": "time.perf_counter_ns (monotonic high-resolution host clock)",
                "host_clock_implementation": time.get_clock_info('perf_counter').implementation,
                "host_clock_resolution_s": time.get_clock_info('perf_counter').resolution,
                "board_clock_source": "/hardware/status.sampled_monotonic",
                "sample_age_rounding_error_ns": AGE_ROUNDING_ERROR_NS,
                "board_minus_host_offset_bounds_ns": offset_bounds,
                "fixed_offset_assumption_over_sampling_window": True,
                "clock_mapping_validated": False,
                "scope_excludes_exposure_age": True,
            },
            "hardware_status": list(hardware_status),
            "requests": request_rows,
            "summary": summary,
            "errors": self.errors,
        }


def _connection_for(parsed: Any, timeout_s: float):
    host = parsed.hostname
    if parsed.scheme == "http":
        return http.client.HTTPConnection(host, parsed.port or 80, timeout=timeout_s)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(host, parsed.port or 443, timeout=timeout_s)
    raise ValueError("URL scheme must be http or https")


def _request_target(parsed: Any, suffix: str) -> str:
    prefix = parsed.path.rstrip("/")
    target = (prefix + suffix) if prefix else suffix
    return target or "/"


def _exchange(connection: Any, target: str) -> Tuple[Optional[int], List[Tuple[str, str]], bytes,
                                                       int, int, bool, Optional[str]]:
    sent_ns = time.perf_counter_ns()
    response = None
    headers: List[Tuple[str, str]] = []
    body = b""
    try:
        connection.request("GET", target, headers={"Accept": "application/octet-stream, application/json"})
        response = connection.getresponse()
        headers = response.getheaders()
        body = response.read()
        received_ns = time.perf_counter_ns()
        return response.status, headers, body, sent_ns, received_ns, True, None
    except http.client.IncompleteRead as exc:
        received_ns = time.perf_counter_ns()
        partial = exc.partial if isinstance(exc.partial, bytes) else b""
        status = response.status if response is not None else None
        return status, headers, partial, sent_ns, received_ns, False, str(exc)
    except (OSError, http.client.HTTPException) as exc:
        received_ns = time.perf_counter_ns()
        status = response.status if response is not None else None
        return status, headers, body, sent_ns, received_ns, False, str(exc)


def _hardware_status(connection: Any, target: str,
                     collector: DepthProbeCollector) -> Dict[str, Any]:
    status, headers, body, sent_ns, received_ns, complete, transport_error = _exchange(connection, target)
    item: Dict[str, Any] = {
        "status": status,
        "host_send_monotonic_ns": sent_ns,
        "host_receive_monotonic_ns": received_ns,
        "rtt_ms": max(0, received_ns - sent_ns) / 1_000_000.0,
        "sampled_monotonic": None,
        "sample_age_ms": None,
        "board_minus_host_offset_bounds_ns": None,
        "error": None,
    }
    if transport_error is not None:
        item["error"] = "hardware status transport error: " + transport_error
        collector.add_external_error("hardware_status_transport_error", item["error"])
        return item
    if status != 200:
        item["error"] = "hardware status returned HTTP " + str(status)
        collector.add_external_error("hardware_status_http_error", item["error"], status=status)
        return item
    if not complete:
        item["error"] = "hardware status response body was incomplete"
        collector.add_external_error("hardware_status_incomplete", item["error"])
        return item
    try:
        payload = json.loads(body.decode("utf-8"))
        sampled = payload.get("sampled_monotonic") if isinstance(payload, dict) else None
        age_ms = payload.get("sample_age_ms") if isinstance(payload, dict) else None
        if (isinstance(sampled, bool) or not isinstance(sampled, (int, float))
                or not math.isfinite(float(sampled)) or float(sampled) < 0):
            raise ValueError("sampled_monotonic must be a finite non-negative number")
        if (isinstance(age_ms, bool) or not isinstance(age_ms, (int, float))
                or not math.isfinite(float(age_ms)) or float(age_ms) < 0):
            raise ValueError("sample_age_ms must be a finite non-negative number")
        item["sampled_monotonic"] = float(sampled)
        item["sample_age_ms"] = float(age_ms)
        board_at_request_ns = int(round(float(sampled) * 1e9 + float(age_ms) * 1e6))
        item["board_minus_host_offset_bounds_ns"] = {
            "lower_ns": board_at_request_ns - received_ns - AGE_ROUNDING_ERROR_NS,
            "upper_ns": board_at_request_ns - sent_ns + AGE_ROUNDING_ERROR_NS,
        }
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        item["error"] = "invalid hardware status clock fields: " + str(exc)
        collector.add_external_error("hardware_status_invalid", item["error"])
    return item


def _write_report(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=True, allow_nan=False,
                                        separators=(",", ":")) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def run_probe(url: str, duration_s: float, interval_s: float,
              output_path: Path, overwrite: bool = False) -> Tuple[Dict[str, Any], int]:
    # Reject bad configuration before opening/truncating any original evidence,
    # including when an explicit overwrite was requested.
    if duration_s <= 0 or not math.isfinite(duration_s) or not math.isfinite(duration_s * 1e9):
        raise ValueError("--duration must be a finite value greater than 0")
    if interval_s <= 0 or not math.isfinite(interval_s) or not math.isfinite(interval_s * 1e9):
        raise ValueError("--interval must be a finite value greater than 0")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("--url must be an absolute http or https URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--url must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("--url must not contain a query or fragment")
    parsed.port  # Validate malformed/out-of-range ports before opening files.
    stem = output_path.with_suffix("")
    archive_path = stem.with_name(stem.name + ".frames.bin.gz")
    if not overwrite:
        existing = [path for path in (output_path, archive_path,
                                      output_path.with_name(output_path.name + ".tmp"))
                    if path.exists()]
        if existing:
            raise FileExistsError("probe output already exists: " + ", ".join(map(str, existing))
                                  + " (pass --overwrite to replace it)")
    collector = DepthProbeCollector(archive_path)
    hardware_rows: List[Dict[str, Any]] = []
    started_ns = None
    finished_ns = None
    interrupted = False
    connection = None
    try:
        connection = _connection_for(parsed, timeout_s=5.0)
        hardware_target = _request_target(parsed, "/hardware/status")
        depth_target = _request_target(parsed, "/depth.bin")
        hardware_rows.append(_hardware_status(connection, hardware_target, collector))

        started_ns = time.perf_counter_ns()
        deadline_ns = started_ns + int(duration_s * 1e9)
        next_request_ns = started_ns
        while time.perf_counter_ns() < deadline_ns:
            delay_ns = next_request_ns - time.perf_counter_ns()
            if delay_ns > 0:
                time.sleep(delay_ns / 1e9)
            status, headers, body, sent_ns, received_ns, complete, error = _exchange(
                connection, depth_target)
            if error is not None:
                collector.ingest_transport_error(
                    host_send_monotonic_ns=sent_ns,
                    host_receive_monotonic_ns=received_ns,
                    message=error, partial_body=body,
                    status=status, headers=headers)
                try:
                    connection.close()
                except OSError:
                    pass
                connection = _connection_for(parsed, timeout_s=5.0)
            else:
                collector.ingest_response(
                    status=status, headers=headers, body=body,
                    host_send_monotonic_ns=sent_ns,
                    host_receive_monotonic_ns=received_ns,
                    response_complete=complete)
            next_request_ns = max(next_request_ns + int(interval_s * 1e9),
                                  time.perf_counter_ns())
        finished_ns = time.perf_counter_ns()
    except KeyboardInterrupt:
        interrupted = True
        finished_ns = time.perf_counter_ns()
        collector.add_external_error("interrupted", "sampling interrupted by user")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        finished_ns = time.perf_counter_ns()
        collector.add_external_error("probe_setup_error", str(exc))
    finally:
        if connection is not None:
            # Reuse the client; an HTTP/1.0 server may reconnect per response.
            try:
                if parsed is not None:
                    hardware_rows.append(_hardware_status(
                        connection, _request_target(parsed, "/hardware/status"), collector))
            except Exception as exc:
                collector.add_external_error("hardware_status_error", str(exc))
            try:
                connection.close()
            except OSError:
                pass
        collector.close_archive()
    report = collector.build_report(
        url=url, duration_s=duration_s, interval_s=interval_s,
        output_path=output_path, hardware_status=hardware_rows,
        started_monotonic_ns=started_ns, finished_monotonic_ns=finished_ns,
        interrupted=interrupted)
    if collector.unique_frames == 0:
        collector.add_external_error("no_valid_frames", "no valid numeric depth frames were received")
        report["errors"] = collector.errors
        report["summary"]["error_count"] = len(collector.errors)
    try:
        _write_report(output_path, report)
    except OSError as exc:
        print("could not write probe report: " + str(exc), file=sys.stderr)
        return report, 2
    return report, (130 if interrupted else 1 if report["errors"] else 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_output = Path(__file__).resolve().parents[1] / "evidence" / "depth-http-probe.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8877")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="sampling duration in seconds (default: 60)")
    parser.add_argument("--interval", type=float, default=0.005,
                        help="minimum seconds between HTTP requests (default: 0.005)")
    parser.add_argument("--output", type=Path, default=default_output,
                        help="report JSON path; unique payloads go beside it as *.frames.bin.gz")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing report or frame archive")
    args = parser.parse_args(argv)
    try:
        report, exit_code = run_probe(args.url, args.duration, args.interval,
                                      args.output, overwrite=args.overwrite)
    except (FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "report": str(args.output),
        "archive": report["summary"].get("archive_path"),
        "requests": report["summary"].get("http_request_count"),
        "unique_frames": report["summary"].get("unique_frames"),
        "errors": report["summary"].get("error_count"),
    }, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
