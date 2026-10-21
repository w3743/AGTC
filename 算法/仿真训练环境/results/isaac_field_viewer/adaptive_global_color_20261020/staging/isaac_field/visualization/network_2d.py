"""Small localhost-only HTTP view of one immutable policy-network frame."""

from __future__ import annotations

import base64
import copy
from collections.abc import Mapping
from dataclasses import dataclass
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .activation_maps import COLOR_MAP_NAME, SPECTRUM_STOPS, DETECTION_LABELS, STATE_LABELS, activation_image


_HTML_PATH = Path(__file__).with_name("network_2d.html")
_ACTIVE_CLIENT_SECONDS = 3.0
_MAX_CONTROL_BODY = 4096
_CONTROL_COMMANDS = {
    "run": "network_run",
    "pause": "network_pause",
    "reset": "reset",
    "preview": "network_preview",
}
_ACTION_FIELDS = ("前进", "差速转向", "云台偏航速率", "云台俯仰速率")


def _numpy_copy(value: Any) -> np.ndarray:
    """Copy a numeric value to CPU NumPy on the caller thread."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    """Make a detached, read-only copy of JSON-like values and arrays."""
    if isinstance(value, np.ndarray):
        return _numpy_copy(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if callable(getattr(value, "detach", None)):
        return _numpy_copy(value)
    return copy.deepcopy(value)


def _json_value(value: Any) -> Any:
    """Convert copied state to strict JSON values without emitting NaN tokens."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finite_stat(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _finite_max_abs(values: Any) -> float:
    """Return the largest finite absolute value, or zero when none are usable."""
    if values is None:
        return 0.0
    try:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return 0.0
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return 0.0
        return float(np.max(np.abs(finite)))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _usable_scale(observed_max_abs: float) -> float:
    return observed_max_abs if math.isfinite(observed_max_abs) and observed_max_abs > 0.0 else 1.0


def _time_number(value: Any) -> float | None:
    """Return a finite numeric sequence/time value for rollback detection."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _fields_for(key: str) -> tuple[str, ...]:
    if key == "input.state":
        return STATE_LABELS
    if key == "input.detections":
        return DETECTION_LABELS
    if key == "action.applied":
        return _ACTION_FIELDS
    return ()


@dataclass(frozen=True, slots=True)
class _Frame:
    revision: int
    metadata: Mapping[str, Any]
    layers: tuple[Mapping[str, Any], ...]
    state: Any
    scale: float = 1.0
    observed_max_abs: float = 0.0


class _NetworkHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _RequestHandler(BaseHTTPRequestHandler):
    server: _NetworkHTTPServer

    @property
    def owner(self) -> Network2DServer:
        return self.server.owner  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: Any) -> None:
        """Keep routine HTTP polling out of the Isaac viewer console."""

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/":
            self.owner._touch_client()
            try:
                document = self.owner._html_path.read_text(encoding="utf-8")
            except OSError:
                self._error(404, "2D viewer page is unavailable")
                return
            document = document.replace("__VIEWER_URL__", html.escape(self.owner.viewer_url, quote=True))
            self._send(200, document.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path != "/api/frame":
            self._error(404, "unknown path")
            return

        self.owner._touch_client()
        try:
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            payload = self.owner._make_frame_response(query)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        self._send_json(200, payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path != "/api/control":
            self._error(404, "unknown path")
            return
        origin = self.headers.get("Origin")
        if origin and not self.owner._same_origin(origin, self.headers.get("Host", "")):
            self._error(400, "cross-origin control requests are not allowed")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        if length < 0 or length > _MAX_CONTROL_BODY:
            self._error(400, "control request is too large")
            return
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "invalid JSON body")
            return
        if not isinstance(request, dict):
            self._error(400, "JSON body must be an object")
            return
        command = request.get("command")
        mapped = _CONTROL_COMMANDS.get(command) if isinstance(command, str) else None
        if mapped is None:
            self._error(400, "unsupported control command")
            return
        try:
            self.owner._enqueue(mapped)
        except Exception:
            self._error(400, "control request could not be queued")
            return
        self._send_json(200, {"ok": True, "command": command})


class Network2DServer:
    """Serve cached policy tensors over loopback without touching the simulator."""

    def __init__(self, port: int, enqueue: Callable[[str], None], viewer_url: str):
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer from 0 through 65535")
        self._enqueue = enqueue
        self.viewer_url = str(viewer_url)
        self._html_path = _HTML_PATH
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._last_get = 0.0
        self._revision = 0
        self._frame: _Frame | None = None
        self._parameters: Mapping[str, Mapping[str, np.ndarray]] = MappingProxyType({})
        self._parameter_max_abs = 0.0
        self._adaptive_scale = 1.0
        self._has_frame_token = False
        self._last_frame_token: tuple[str, str] | None = None
        self._last_sequence_number: float | None = None
        self._last_sim_time_number: float | None = None
        self._image_cache: dict[tuple[int, str, str, str, float], str] = {}

        httpd = _NetworkHTTPServer(("127.0.0.1", port), _RequestHandler)
        httpd.owner = self  # type: ignore[attr-defined]
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}/"
        self._thread = threading.Thread(target=httpd.serve_forever, name=f"network-2d-http-{self.port}", daemon=True)
        self._thread.start()

    def _touch_client(self) -> None:
        with self._lock:
            self._last_get = time.monotonic()

    def has_clients(self) -> bool:
        with self._lock:
            return time.monotonic() - self._last_get < _ACTIVE_CLIENT_SECONDS

    def bind(self, inspector: Any) -> None:
        """Cache copied NumPy parameters on the caller (simulation-owner) thread."""
        parameters: dict[str, Mapping[str, np.ndarray]] = {}
        parameter_max_abs = 0.0
        if inspector is not None:
            for key in inspector.layers:
                raw = inspector.parameters(key)
                if not isinstance(raw, Mapping):
                    continue
                copied = {str(name): _numpy_copy(value) for name, value in raw.items()}
                parameters[str(key)] = MappingProxyType(copied)
                parameter_max_abs = max(
                    [parameter_max_abs, *(_finite_max_abs(value) for value in copied.values())]
                )
        with self._lock:
            self._parameters = MappingProxyType(parameters)
            self._parameter_max_abs = parameter_max_abs
            self._adaptive_scale = _usable_scale(parameter_max_abs)
            self._has_frame_token = False
            self._last_frame_token = None
            self._last_sequence_number = None
            self._last_sim_time_number = None
            self._frame = None
            self._revision += 1
        with self._cache_lock:
            self._image_cache.clear()

    def publish(self, snapshot: Mapping[str, Any] | None, layers: list[dict] | tuple[dict, ...], state: Any = None) -> None:
        """Publish detached, read-only values; safe to consume from HTTP threads."""
        metadata_source = snapshot or {}
        metadata = MappingProxyType(
            {
                name: _freeze(metadata_source.get(name))
                for name in ("sequence", "sim_time", "output_time", "mode", "critic", "errors")
            }
        )
        copied_layers: list[Mapping[str, Any]] = []
        for layer in layers or ():
            values = layer.get("values")
            record = {
                "key": str(layer.get("key", "")),
                "title": str(layer.get("title", layer.get("key", ""))),
                "description": str(layer.get("description", "")),
                "branch": str(layer.get("branch", "")),
                "column": _freeze(layer.get("column", 0.0)),
                "parents": tuple(str(parent) for parent in (layer.get("parents") or ())),
                "summary": bool(layer.get("summary", False)),
                "source": str(layer.get("source", "未知来源")),
                "values": None if values is None else _numpy_copy(values),
            }
            copied_layers.append(MappingProxyType(record))
        frozen_state = _freeze(state)
        activation_max_abs = max(
            (_finite_max_abs(layer["values"]) for layer in copied_layers),
            default=0.0,
        )
        sequence = metadata.get("sequence")
        sim_time = metadata.get("sim_time")
        sequence_number = _time_number(sequence)
        sim_time_number = _time_number(sim_time)
        frame_token = (repr(_json_value(sequence)), repr(_json_value(sim_time)))
        with self._lock:
            observed_max_abs = max(self._parameter_max_abs, activation_max_abs)
            target_scale = _usable_scale(observed_max_abs)
            reset_range = snapshot is None
            if self._has_frame_token:
                reset_range = reset_range or (
                    sequence_number is not None
                    and self._last_sequence_number is not None
                    and sequence_number < self._last_sequence_number
                )
                reset_range = reset_range or (
                    sim_time_number is not None
                    and self._last_sim_time_number is not None
                    and sim_time_number < self._last_sim_time_number
                )

            if reset_range or not self._has_frame_token:
                scale = target_scale
            elif target_scale > self._adaptive_scale:
                scale = target_scale
            elif frame_token != self._last_frame_token:
                scale = max(target_scale, self._adaptive_scale * 0.95)
            else:
                scale = self._adaptive_scale

            self._adaptive_scale = scale
            self._has_frame_token = snapshot is not None
            self._last_frame_token = frame_token if snapshot is not None else None
            self._last_sequence_number = sequence_number if snapshot is not None else None
            self._last_sim_time_number = sim_time_number if snapshot is not None else None
            self._revision += 1
            self._frame = _Frame(
                self._revision,
                metadata,
                tuple(copied_layers),
                frozen_state,
                scale,
                observed_max_abs,
            )
        with self._cache_lock:
            self._image_cache.clear()

    def _image_data(self, frame: _Frame, key: str, tensor: str, values: np.ndarray | None, scale: float) -> str | None:
        if values is None or values.size == 0:
            return None
        sequence_token = repr(_json_value(frame.metadata.get("sequence")))
        cache_key = (frame.revision, sequence_token, key, tensor, scale)
        with self._cache_lock:
            cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached

        rgb, _indices, _limit = activation_image(values, scale=scale)
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        with self._cache_lock:
            self._image_cache[cache_key] = data_url
        return data_url

    @staticmethod
    def _query_value(query: Mapping[str, list[str]], name: str, default: str) -> str:
        values = query.get(name)
        if not values:
            return default
        if len(values) != 1:
            raise ValueError(f"{name} must be specified once")
        return values[0]

    @staticmethod
    def _read_stats(values: np.ndarray | None) -> tuple[list[int] | None, int | None, float | None, float | None, float | None, float | None, int | None]:
        if values is None:
            return None, None, None, None, None, None, None
        shape = [int(axis) for axis in values.shape]
        size = int(values.size)
        if size == 0:
            return shape, size, None, None, None, None, 0
        numeric = np.asarray(values)
        finite = numeric[np.isfinite(numeric)]
        nonfinite = int(size - finite.size)
        if finite.size == 0:
            return shape, size, None, None, None, None, nonfinite
        return (
            shape,
            size,
            _finite_stat(float(finite.min())),
            _finite_stat(float(finite.max())),
            _finite_stat(float(finite.mean())),
            _finite_stat(float(finite.std())),
            nonfinite,
        )

    @staticmethod
    def _same_origin(origin: str, host: str) -> bool:
        try:
            parsed = urlsplit(origin)
            return (
                parsed.scheme.lower() == "http"
                and bool(parsed.netloc)
                and parsed.path in ("", "/")
                and not parsed.query
                and not parsed.fragment
                and parsed.username is None
                and parsed.password is None
                and parsed.netloc.casefold() == host.strip().casefold()
            )
        except ValueError:
            return False

    def _make_frame_response(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        detail_text = self._query_value(query, "detail", "0")
        if detail_text not in ("0", "1"):
            raise ValueError("detail must be 0 or 1")
        try:
            index = int(self._query_value(query, "index", "0"))
        except ValueError as exc:
            raise ValueError("index must be a non-negative integer") from exc
        if index < 0 or index > 2_147_483_647:
            raise ValueError("index is out of range")
        scale_mode = self._query_value(query, "scale", "auto")
        if scale_mode != "auto":
            raise ValueError("scale must be auto")

        with self._lock:
            frame = self._frame
            parameters = self._parameters
            parameter_max_abs = self._parameter_max_abs
        if frame is None:
            frame = _Frame(
                0,
                MappingProxyType({}),
                (),
                None,
                _usable_scale(parameter_max_abs),
                parameter_max_abs,
            )
        scale = frame.scale
        layers = frame.layers
        layer_by_key = {str(layer["key"]): layer for layer in layers}
        default_key = "features_extractor" if "features_extractor" in layer_by_key else next(iter(layer_by_key), "")
        selected = self._query_value(query, "selected", default_key)
        if selected and selected not in layer_by_key:
            raise ValueError("selected layer is unknown")
        selected_layer = layer_by_key.get(selected)
        parameter_values = parameters.get(selected, MappingProxyType({}))
        parameter_names = sorted(parameter_values)
        tensor = self._query_value(query, "tensor", "激活值")
        if tensor.startswith("参数："):
            tensor = tensor.removeprefix("参数：")
        if tensor == "激活值":
            detail_values = None if selected_layer is None else selected_layer["values"]
        else:
            detail_values = parameter_values.get(tensor)
            if detail_values is None:
                raise ValueError("tensor parameter is unavailable for selected layer")
        if detail_values is not None and index >= detail_values.size:
            raise ValueError("index is outside the selected tensor")

        visible_layers = [
            layer for layer in layers
            if detail_text == "1" or bool(layer["summary"]) or str(layer["key"]).startswith("input.")
        ]
        nodes = []
        for layer in visible_layers:
            values = layer["values"]
            nodes.append(
                {
                    "key": layer["key"],
                    "title": layer["title"],
                    "description": layer["description"],
                    "branch": layer["branch"],
                    "column": _json_value(layer["column"]),
                    "parents": list(layer["parents"]),
                    "summary": layer["summary"],
                    "source": layer["source"],
                    "shape": None if values is None else [int(axis) for axis in values.shape],
                    "image": self._image_data(frame, str(layer["key"]), "激活值", values, scale),
                }
            )

        stats = self._read_stats(detail_values)
        shape, size, minimum, maximum, mean, std, nonfinite = stats
        coordinate: list[int] | None = None
        element: Any = None
        shown_index: int | None = None
        if detail_values is not None and detail_values.size:
            shown_index = index
            coordinate = [int(axis) for axis in np.unravel_index(index, detail_values.shape)] if detail_values.shape else []
            raw_element = float(detail_values.reshape(-1)[index])
            element = _json_value(raw_element)
        elif detail_values is not None:
            shown_index = None

        detail_info = {
            "key": None if selected_layer is None else selected_layer["key"],
            "title": None if selected_layer is None else selected_layer["title"],
            "shape": shape,
            "size": size,
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "nonfinite": nonfinite,
            "index": shown_index,
            "coordinate": coordinate,
            "value": element,
            "parameters": parameter_names,
            "image": self._image_data(frame, selected, tensor, detail_values, scale),
            "description": None if selected_layer is None else selected_layer["description"],
            "source": None if selected_layer is None else selected_layer["source"],
            "fields": [] if selected_layer is None else list(_fields_for(str(selected_layer["key"]))),
        }
        metadata = frame.metadata
        errors_value = _json_value(metadata.get("errors"))
        if errors_value is None:
            errors_value = []
        elif isinstance(errors_value, str):
            errors_value = [errors_value]
        return {
            "sequence": _json_value(metadata.get("sequence")),
            "sim_time": _json_value(metadata.get("sim_time")),
            "output_time": _json_value(metadata.get("output_time")),
            "mode": _json_value(metadata.get("mode")),
            "critic": _json_value(metadata.get("critic")),
            "errors": errors_value,
            "state": _json_value(frame.state),
            "legend": {
                "name": COLOR_MAP_NAME,
                "colors": _json_value(SPECTRUM_STOPS),
                "minimum": -scale,
                "zero": 0.0,
                "maximum": scale,
                "mode": "adaptive-global",
                "scale": scale,
                "observed_max_abs": frame.observed_max_abs,
            },
            "nodes": nodes,
            "detail": detail_info,
        }

    def close(self) -> None:
        """Stop the HTTP thread and release the loopback socket; safe to call twice."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)


__all__ = ["Network2DServer"]
