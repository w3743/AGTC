"""Local web server for the 20-neuron online next-digit predictor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit
import uuid

import torch

from model import (
    CONTEXT_LENGTH,
    INPUT_COUNT,
    LEARNING_RATE,
    NEURON_COUNT,
    TRAINING_UPDATES_PER_PRESS,
    PlasticRNN,
)


DEFAULT_DATA_PARENT = Path("C:/pioneer/local-neuron-runs")
MAX_REQUEST_BYTES = 4096
UNIFORM_PROBABILITIES = [1.0 / INPUT_COUNT] * INPUT_COUNT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def create_unique_run_root(parent: Path) -> Path:
    """Create a unique run directory under a caller-selected parent."""

    parent = parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = parent / f"keypress20_{stamp}_{uuid.uuid4().hex[:8]}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


class KeypressService:
    """Owns one run and serializes all session changes behind a single lock."""

    def __init__(self, data_parent: Path = DEFAULT_DATA_PARENT, seed: int = 0) -> None:
        # Tiny matrix operations are faster and more predictable with one CPU
        # worker than with a large BLAS thread pool.
        torch.set_num_threads(1)
        self.seed = int(seed)
        self.data_parent = Path(data_parent)
        self.run_root = create_unique_run_root(self.data_parent)
        self._lock = threading.RLock()
        self._session_number = 0
        self.source_sha256 = self._capture_source_snapshot()
        self._start_session()

    def _config(self) -> dict[str, Any]:
        return {
            "input_ports": INPUT_COUNT,
            "neuron_count": NEURON_COUNT,
            "recurrent_edges": NEURON_COUNT * NEURON_COUNT,
            "context_length": CONTEXT_LENGTH,
            "training_updates_per_press": TRAINING_UPDATES_PER_PRESS,
            "learning_rate": LEARNING_RATE,
            "parameter_count": INPUT_COUNT * NEURON_COUNT
            + NEURON_COUNT * NEURON_COUNT
            + NEURON_COUNT
            + 3,
            "training_window": "latest observed digits; adjacent next-digit pairs",
            "output_ports": "first 10 activities times one shared positive scale",
            "torch_num_threads": torch.get_num_threads(),
        }

    def _capture_source_snapshot(self) -> dict[str, str | None]:
        """Copy the exact app sources and built UI into this run's data folder."""

        project_root = Path(__file__).resolve().parent
        snapshot_root = self.run_root / "source_snapshot"
        snapshot_root.mkdir(parents=True, exist_ok=False)
        source_hashes: dict[str, str | None] = {}
        copied_files: list[str] = []

        for name in ("model.py", "server.py", "requirements.txt", "README.md", "start.ps1"):
            source = project_root / name
            if not source.is_file():
                source_hashes[name] = None
                continue
            raw = source.read_bytes()
            (snapshot_root / name).write_bytes(raw)
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
            copied_files.append(name)

        ui_root = project_root / "dist"
        if ui_root.is_dir():
            for source in sorted(ui_root.rglob("*")):
                if not source.is_file():
                    continue
                try:
                    relative = source.relative_to(ui_root)
                except ValueError:
                    continue
                raw = source.read_bytes()
                destination = snapshot_root / "dist" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                relative_name = str(Path("dist") / relative).replace("\\", "/")
                source_hashes[relative_name] = hashlib.sha256(raw).hexdigest()
                copied_files.append(relative_name)

        source_manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "source_sha256": source_hashes,
            "copied_files": copied_files,
        }
        (snapshot_root / "manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        run_manifest = {
            "schema_version": 1,
            "run_directory": self.run_root.name,
            "created_at": utc_now(),
            "config": self._config(),
            "config_sha256": hashlib.sha256(json_bytes(self._config())).hexdigest(),
            "source_snapshot": "source_snapshot/",
            "source_snapshot_manifest": "source_snapshot/manifest.json",
            "source_sha256": source_hashes,
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        }
        run_manifest_bytes = json.dumps(
            run_manifest, ensure_ascii=False, indent=2, allow_nan=False
        ).encode("utf-8") + b"\n"
        (self.run_root / "run_manifest.json").write_bytes(run_manifest_bytes)
        (snapshot_root / "run_manifest.json").write_bytes(run_manifest_bytes)
        return source_hashes

    def _new_manifest(self, session_id: str, model_seed: int) -> dict[str, Any]:
        config = self._config()
        config_digest = hashlib.sha256(json_bytes(config)).hexdigest()
        return {
            "schema_version": 1,
            "session_id": session_id,
            "created_at": utc_now(),
            "seed": model_seed,
            "config": config,
            "config_sha256": config_digest,
            "source_snapshot": "../source_snapshot/",
            "source_sha256": self.source_sha256,
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "initial_prediction": {
                "prediction_id": self.prediction_id,
                "step": 0,
                "prediction": 0,
                "probabilities": list(UNIFORM_PROBABILITIES),
                "prediction_source": "uniform_cold_start",
            },
        }

    def _start_session(self) -> None:
        self._session_number += 1
        self.session_id = str(uuid.uuid4())
        self.session_dir = self.run_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.session_dir / "events.jsonl"

        model_seed = self.seed + self._session_number - 1
        self.model_seed = model_seed
        self.model = PlasticRNN(seed=model_seed)
        self.step = 0
        self.correct = 0
        self.majority_correct = 0
        self.digit_counts = [0] * INPUT_COUNT
        self.history: list[int] = []
        self.last_result: dict[str, Any] | None = None
        self.training_seconds = 0.0
        self.probabilities = list(UNIFORM_PROBABILITIES)
        self.prediction = 0
        self.prediction_source = "uniform_cold_start"
        self.prediction_id = str(uuid.uuid4())
        self.neurons = [0.0] * NEURON_COUNT
        # Edges exist from session start. The matrix is initialized by the
        # model; row=destination and column=source.
        _, _, self.connections = self.model.predict_next([])
        manifest = self._new_manifest(self.session_id, model_seed)
        (self.session_dir / "session.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(line + "\n")
            target.flush()
            os.fsync(target.fileno())

    def _majority_guess(self) -> int:
        # max() returns the first value on a tie, so ties choose the lowest digit.
        return max(range(INPUT_COUNT), key=lambda digit: self.digit_counts[digit])

    def _metrics(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "total": self.step,
            "accuracy": (self.correct / self.step) if self.step else 0.0,
            "random_baseline": 1.0 / INPUT_COUNT,
            "majority_baseline": (self.majority_correct / self.step) if self.step else 0.0,
        }

    def _state_unlocked(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "step": self.step,
            "prediction_id": self.prediction_id,
            "probabilities": list(self.probabilities),
            "prediction": self.prediction,
            "prediction_source": self.prediction_source,
            "training_ms": self.training_seconds * 1000.0,
            "metrics": self._metrics(),
            "neurons": list(self.neurons),
            "connections": [list(row) for row in self.connections],
            "history": list(self.history[-30:]),
            "last_result": dict(self.last_result) if self.last_result is not None else None,
        }

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_unlocked()

    def press(self, digit: int, prediction_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Score a sealed forecast, log it, then append/train/predict atomically."""

        if type(digit) is not int or not 0 <= digit < INPUT_COUNT:
            raise ValueError("digit must be an integer from 0 to 9")
        if type(prediction_id) is not str or not prediction_id:
            raise ValueError("prediction_id must be a non-empty string")

        with self._lock:
            if prediction_id != self.prediction_id:
                raise StalePredictionError

            # Freeze and score the forecast produced before this request. This
            # object is complete and persisted before the new digit is observed
            # by the model or added to the training context.
            scored_prediction_id = self.prediction_id
            scored_step = self.step + 1
            scored_probabilities = list(self.probabilities)
            scored_choice = self.prediction
            correct = scored_choice == digit
            majority_guess = self._majority_guess()
            result = {
                "prediction_id": scored_prediction_id,
                "step": scored_step,
                "prediction": scored_choice,
                "actual_digit": digit,
                "probabilities": scored_probabilities,
                "prediction_source": self.prediction_source,
                "correct": correct,
                "score": 1 if correct else 0,
            }
            self._append_event(
                {
                    "event": "press_scored",
                    "session_id": self.session_id,
                    "recorded_at": utc_now(),
                    **result,
                    "majority_prediction": majority_guess,
                    "majority_correct": majority_guess == digit,
                }
            )

            self.step = scored_step
            self.correct += int(correct)
            self.majority_correct += int(majority_guess == digit)
            self.digit_counts[digit] += 1
            self.history.append(digit)
            if len(self.history) > CONTEXT_LENGTH:
                del self.history[: len(self.history) - CONTEXT_LENGTH]
            self.last_result = result

            started = time.perf_counter()
            training_loss = self.model.train_on_history(self.history)
            self.training_seconds = time.perf_counter() - started

            self.probabilities, self.neurons, self.connections = self.model.predict_next(self.history)
            self.prediction = max(range(INPUT_COUNT), key=self.probabilities.__getitem__)
            self.prediction_source = (
                "plastic_rnn" if self.model.optimizer_updates > 0 else "uniform_cold_start"
            )
            self.prediction_id = str(uuid.uuid4())
            self._append_event(
                {
                    "event": "next_prediction_ready",
                    "session_id": self.session_id,
                    "recorded_at": utc_now(),
                    "after_step": self.step,
                    "prediction_id": self.prediction_id,
                    "prediction": self.prediction,
                    "prediction_source": self.prediction_source,
                    "probabilities": list(self.probabilities),
                    "neurons": list(self.neurons),
                    "connections": [list(row) for row in self.connections],
                    "training_loss": training_loss,
                    "training_pairs": self.model.last_training_pairs,
                    "training_updates": self.model.optimizer_updates,
                    "training_seconds": self.training_seconds,
                    "training_ms": self.training_seconds * 1000.0,
                    "eta": float(self.model.eta.detach().item()),
                    "trace_decay": float(self.model.trace_decay.detach().item()),
                    "parameter_count": sum(
                        parameter.numel() for parameter in self.model.parameters()
                    ),
                }
            )
            self._write_checkpoint()
            return self._state_unlocked(), result

    def _write_checkpoint(self) -> None:
        """Atomically save the latest learned state after a press completes."""

        checkpoint = {
            "schema_version": 1,
            "session_id": self.session_id,
            "saved_at": utc_now(),
            "seed": self.model_seed,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "optimizer_updates": self.model.optimizer_updates,
            "last_training_loss": self.model.last_training_loss,
            "last_training_pairs": self.model.last_training_pairs,
            "history": list(self.history),
            "digit_counts": list(self.digit_counts),
            "correct": self.correct,
            "majority_correct": self.majority_correct,
            "sealed_next_prediction": {
                "prediction_id": self.prediction_id,
                "prediction": self.prediction,
                "prediction_source": self.prediction_source,
                "probabilities": list(self.probabilities),
            },
            "neurons": list(self.neurons),
            "connections": [list(row) for row in self.connections],
            "last_result": self.last_result,
        }
        target = self.session_dir / "latest_checkpoint.pt"
        temporary = self.session_dir / "latest_checkpoint.pt.tmp"
        with temporary.open("wb") as output:
            torch.save(checkpoint, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)

    def reset(self) -> dict[str, Any]:
        """Start a fresh session while leaving every prior session directory intact."""

        with self._lock:
            self._start_session()
            return self._state_unlocked()

    def export(self) -> dict[str, Any]:
        """Read all session manifests and JSONL events under this run root."""

        with self._lock:
            sessions: list[dict[str, Any]] = []
            for session_dir in sorted(self.run_root.iterdir(), key=lambda path: path.name):
                manifest_path = session_dir / "session.json"
                events_path = session_dir / "events.jsonl"
                if not session_dir.is_dir() or not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                events = []
                if events_path.is_file():
                    with events_path.open("r", encoding="utf-8") as source:
                        for line in source:
                            if line.strip():
                                events.append(json.loads(line))
                sessions.append({"session": manifest, "events": events})
            return {
                "exported_at": utc_now(),
                "run_directory": self.run_root.name,
                "active_state": self._state_unlocked(),
                "sessions": sessions,
            }


class StalePredictionError(Exception):
    pass


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], service: KeypressService, static_root: Path):
        self.service = service
        self.static_root = static_root.resolve()
        super().__init__(server_address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: LocalHTTPServer
    protocol_version = "HTTP/1.1"

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, value: Any, extra_headers: dict[str, str] | None = None) -> None:
        self._send_bytes(status, json_bytes(value), "application/json; charset=utf-8", extra_headers)

    def _send_error_json(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _read_json_body(self, allow_empty: bool = False) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                raise RequestBodyError(400, "invalid_request", "Chunked request bodies are not supported")
            if allow_empty:
                return {}
            self.close_connection = True
            raise RequestBodyError(411, "length_required", "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            self.close_connection = True
            raise RequestBodyError(400, "invalid_request", "Content-Length must be an integer") from exc
        if length < 0:
            self.close_connection = True
            raise RequestBodyError(400, "invalid_request", "Content-Length must be non-negative")
        if length > MAX_REQUEST_BYTES:
            self.close_connection = True
            raise RequestBodyError(413, "request_too_large", "Request body exceeds 4096 bytes")
        if length == 0:
            if allow_empty:
                return {}
            raise RequestBodyError(400, "invalid_request", "Request body is required")
        if self.headers.get_content_type() != "application/json":
            self.close_connection = True
            raise RequestBodyError(415, "unsupported_media_type", "Content-Type must be application/json")
        payload = self.rfile.read(length)
        if len(payload) != length:
            self.close_connection = True
            raise RequestBodyError(400, "invalid_request", "Request body ended before Content-Length bytes arrived")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestBodyError(400, "invalid_request", "Request body must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise RequestBodyError(400, "invalid_request", "Request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/api/state":
            self._send_json(200, self.server.service.get_state())
            return
        if path == "/api/export":
            payload = json_bytes(self.server.service.export())
            self._send_bytes(
                200,
                payload,
                "application/json; charset=utf-8",
                {"Content-Disposition": 'attachment; filename="keypress20-export.json"'},
            )
            return
        if path.startswith("/api/"):
            self._send_error_json(404, "not_found", "API endpoint not found")
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/api/press":
            try:
                payload = self._read_json_body()
                if set(payload) != {"digit", "prediction_id"}:
                    raise RequestBodyError(
                        400,
                        "invalid_request",
                        "Body must contain exactly digit and prediction_id",
                    )
                digit = payload["digit"]
                prediction_id = payload["prediction_id"]
                if type(digit) is not int or not 0 <= digit <= 9:
                    raise RequestBodyError(
                        400,
                        "invalid_request",
                        "digit must be an integer from 0 to 9",
                    )
                if type(prediction_id) is not str or not prediction_id:
                    raise RequestBodyError(
                        400,
                        "invalid_request",
                        "prediction_id must be a non-empty string",
                    )
                state, last_result = self.server.service.press(digit, prediction_id)
            except RequestBodyError as exc:
                self._send_error_json(exc.status, exc.code, exc.message)
                return
            except StalePredictionError:
                self._send_error_json(
                    409,
                    "stale_prediction",
                    "prediction_id is stale; reload state",
                )
                return
            except ValueError as exc:
                self._send_error_json(400, "invalid_request", str(exc))
                return
            self._send_json(200, {**state, "last_result": last_result})
            return

        if path == "/api/reset":
            try:
                body = self._read_json_body(allow_empty=True)
                if body:
                    raise RequestBodyError(400, "invalid_request", "Reset body must be empty or {}")
            except RequestBodyError as exc:
                self._send_error_json(exc.status, exc.code, exc.message)
                return
            self._send_json(200, self.server.service.reset())
            return

        if path.startswith("/api/"):
            self._send_error_json(404, "not_found", "API endpoint not found")
            return
        self._send_error_json(404, "not_found", "POST endpoint not found")

    def _serve_static(self, request_path: str) -> None:
        if request_path == "/":
            relative = "index.html"
        else:
            relative = unquote(request_path).lstrip("/")
        candidate = (self.server.static_root / relative).resolve()
        try:
            candidate.relative_to(self.server.static_root)
        except ValueError:
            self._send_error_json(404, "not_found", "File not found")
            return
        if not candidate.is_file():
            self._send_error_json(404, "not_found", "File not found")
            return
        try:
            body = candidate.read_bytes()
        except OSError:
            self._send_error_json(500, "file_read_failed", "Unable to read static file")
            return
        content_type, _ = mimetypes.guess_type(candidate.name)
        content_type = content_type or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._send_bytes(200, body, content_type)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep useful request diagnostics while avoiding noisy default timestamps.
        super().log_message(format, *args)


class RequestBodyError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local 20-neuron next-digit predictor")
    parser.add_argument("--host", choices=("localhost",), default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_PARENT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> None:
    args = parse_args()
    service = KeypressService(data_parent=args.data_root, seed=args.seed)
    static_root = Path(__file__).with_name("dist")
    server = LocalHTTPServer(("127.0.0.1", args.port), service, static_root)
    print(f"Listening on http://127.0.0.1:{args.port}/")
    print(f"Run data: {service.run_root}")
    print("Press Ctrl+C to stop; this does not change the saved JSONL session logs.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
