"""Local-only HTTP protocol checks for the keypress predictor service.

The test binds an ephemeral IPv4 loopback port and writes all service data under
KEYPRESS20_QA_OUTPUT when that environment variable is set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import torch

from model import INPUT_COUNT, NEURON_COUNT
from online_model import CHECKPOINT_FORMAT_VERSION
from server import KeypressService, LocalHTTPServer


SEED = 20261026
_NO_BODY = object()
OUTPUT_DIR = os.environ.get("KEYPRESS20_QA_OUTPUT")


class KeypressHTTPProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        if OUTPUT_DIR:
            cls.output_root = Path(OUTPUT_DIR)
            cls.output_root.mkdir(parents=True, exist_ok=True)
        else:
            cls.output_root = Path(tempfile.mkdtemp(prefix="keypress20-server-qa-"))

        cls.data_parent = cls.output_root / "server_runs"
        cls.data_parent.mkdir(parents=True, exist_ok=True)
        cls.service = KeypressService(data_parent=cls.data_parent, seed=SEED, mode="window")
        cls.server = LocalHTTPServer(
            ("127.0.0.1", 0),
            cls.service,
            Path(__file__).with_name("dist"),
        )
        cls.host, cls.port = cls.server.server_address
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5)

    def _request(
        self,
        path: str,
        method: str = "GET",
        body: object = _NO_BODY,
        content_type: str = "application/json",
    ) -> tuple[int, object, bytes]:
        headers: dict[str, str] = {}
        data: bytes | None
        if body is _NO_BODY:
            data = None
        elif isinstance(body, bytes):
            data = body
        else:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = content_type

        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def _json_request(self, *args: object, **kwargs: object) -> tuple[int, object, dict[str, object]]:
        status, headers, payload = self._request(*args, **kwargs)  # type: ignore[arg-type]
        decoded = json.loads(payload.decode("utf-8"))
        return status, headers, decoded

    def _assert_state_shape(self, state: dict[str, object]) -> None:
        self.assertIsInstance(state.get("session_id"), str)
        self.assertIsInstance(state.get("prediction_id"), str)
        probabilities = state["probabilities"]
        neurons = state["neurons"]
        connections = state["connections"]
        self.assertIsInstance(probabilities, list)
        self.assertIsInstance(neurons, list)
        self.assertIsInstance(connections, list)
        self.assertEqual(len(probabilities), INPUT_COUNT)
        self.assertEqual(len(neurons), NEURON_COUNT)
        self.assertEqual(len(connections), NEURON_COUNT)
        self.assertTrue(all(len(row) == NEURON_COUNT for row in connections))
        self.assertTrue(all(math.isfinite(float(value)) for value in probabilities))
        self.assertTrue(all(math.isfinite(float(value)) for value in neurons))
        self.assertTrue(
            all(math.isfinite(float(value)) for row in connections for value in row)
        )
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in probabilities))
        self.assertAlmostEqual(sum(float(value) for value in probabilities), 1.0, places=6)

    def test_http_session_protocol_logging_reset_and_export(self) -> None:
        self.assertEqual(self.host, "127.0.0.1")
        self.assertNotEqual(self.port, 8765)

        index_status, headers, page = self._request("/")
        self.assertEqual(index_status, 200)
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        page_text = page.decode("utf-8")
        self.assertIn("固定 20 个神经元", page_text)
        self.assertIn("id=\"digitPad\"", page_text)
        app_status, _, app_js = self._request("/app.js")
        self.assertEqual(app_status, 200)
        js_text = app_js.decode("utf-8")
        self.assertIn("/api/state", js_text)
        self.assertIn("/api/press", js_text)
        self.assertIn("prediction_id", js_text)
        self.assertIn('document.addEventListener("keydown"', js_text)

        status, state_headers, state_payload = self._json_request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state_headers.get("Cache-Control"), "no-store")
        initial = state_payload
        self._assert_state_shape(initial)
        self.assertEqual(initial["step"], 0)
        self.assertEqual(initial["prediction"], 0)
        self.assertEqual(initial["prediction_source"], "uniform_cold_start")
        self.assertEqual(initial["probabilities"], [0.1] * 10)
        session_id_1 = str(initial["session_id"])
        session_dir_1 = self.service.session_dir
        events_path_1 = session_dir_1 / "events.jsonl"
        manifest_1 = json.loads((session_dir_1 / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_1["session_id"], session_id_1)
        self.assertEqual(manifest_1["initial_prediction"]["prediction_id"], initial["prediction_id"])
        self.assertEqual(manifest_1["initial_prediction"]["probabilities"], [0.1] * 10)

        # Invalid types and malformed payloads must not consume the sealed token.
        invalid_values: list[object] = [True, False, 3.0, 3.5, "3", None, -1, 10]
        for invalid in invalid_values:
            with self.subTest(invalid_digit=repr(invalid)):
                status, _, error = self._json_request(
                    "/api/press",
                    "POST",
                    {"digit": invalid, "prediction_id": initial["prediction_id"]},
                )
                self.assertEqual(status, 400)
                self.assertEqual(error["error"]["code"], "invalid_request")
                unchanged_status, _, unchanged = self._json_request("/api/state")
                self.assertEqual(unchanged_status, 200)
                self.assertEqual(unchanged["step"], 0)
                self.assertEqual(unchanged["prediction_id"], initial["prediction_id"])
        status, _, error = self._json_request(
            "/api/press",
            "POST",
            json.dumps({"digit": 0, "prediction_id": initial["prediction_id"]}).encode("utf-8"),
            "text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(error["error"]["code"], "unsupported_media_type")
        status, _, error = self._json_request(
            "/api/press",
            "POST",
            {"digit": 0, "prediction_id": initial["prediction_id"], "extra": 1},
        )
        self.assertEqual(status, 400)
        self.assertEqual(error["error"]["code"], "invalid_request")
        self.assertFalse(events_path_1.exists())

        inputs = [index % 10 for index in range(40)]
        before_states: list[dict[str, object]] = []
        after_states: list[dict[str, object]] = []
        frozen_results: list[dict[str, object]] = []
        stale_retry_evidence: dict[str, object] | None = None

        for index, digit in enumerate(inputs):
            before_status, _, before = self._json_request("/api/state")
            self.assertEqual(before_status, 200)
            self._assert_state_shape(before)
            self.assertEqual(before["session_id"], session_id_1)
            self.assertEqual(before["step"], index)
            self.assertEqual(before["history"], inputs[max(0, index - 30) : index])
            before_states.append(before)

            status, _, response = self._json_request(
                "/api/press",
                "POST",
                {"digit": digit, "prediction_id": before["prediction_id"]},
            )
            self.assertEqual(status, 200)
            self._assert_state_shape(response)
            result = response["last_result"]
            self.assertEqual(result["prediction_id"], before["prediction_id"])
            self.assertEqual(result["prediction"], before["prediction"])
            self.assertEqual(result["probabilities"], before["probabilities"])
            self.assertEqual(result["prediction_source"], before["prediction_source"])
            self.assertEqual(result["actual_digit"], digit)
            self.assertEqual(result["correct"], result["prediction"] == digit)
            self.assertEqual(result["score"], int(result["correct"]))
            self.assertEqual(response["step"], index + 1)
            self.assertEqual(response["history"], inputs[max(0, index - 29) : index + 1])
            frozen_results.append(result)
            after_states.append(response)

            if index == 0:
                self.assertEqual(result["prediction"], 0)
                self.assertEqual(result["probabilities"], [0.1] * 10)
                self.assertEqual(response["prediction_source"], "uniform_cold_start")
                self.assertEqual(response["probabilities"], [0.1] * 10)
                self.assertEqual(self.service.model.optimizer_updates, 0)

                # A stale retry must have no training or logging side effects.
                old_events = events_path_1.read_bytes()
                checkpoint = session_dir_1 / "latest_checkpoint.pt"
                checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                updates = self.service.model.optimizer_updates
                status, _, error = self._json_request(
                    "/api/press",
                    "POST",
                    {"digit": digit, "prediction_id": before["prediction_id"]},
                )
                self.assertEqual(status, 409)
                self.assertEqual(error["error"]["code"], "stale_prediction")
                unchanged_status, _, unchanged = self._json_request("/api/state")
                self.assertEqual(unchanged_status, 200)
                self.assertEqual(unchanged["step"], 1)
                self.assertEqual(unchanged["prediction_id"], response["prediction_id"])
                self.assertEqual(self.service.model.optimizer_updates, updates)
                self.assertEqual(events_path_1.read_bytes(), old_events)
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_digest
                )
                stale_retry_evidence = {
                    "status": status,
                    "error_code": error["error"]["code"],
                    "step_after_retry": unchanged["step"],
                    "events_unchanged": events_path_1.read_bytes() == old_events,
                    "checkpoint_unchanged": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    == checkpoint_digest,
                    "optimizer_updates_unchanged": self.service.model.optimizer_updates == updates,
                }

            if index == 0:
                self.assertEqual(response["history"], [0])
                self.assertEqual(response["prediction_source"], "uniform_cold_start")
            if index == 1:
                # Training uses the two labels only after the second forecast was scored.
                self.assertEqual(before["prediction_source"], "uniform_cold_start")
                self.assertEqual(before["probabilities"], [0.1] * 10)
                self.assertEqual(response["prediction_source"], "plastic_rnn")
                self.assertEqual(self.service.model.optimizer_updates, 3)

        self.assertIsNotNone(stale_retry_evidence)
        self.assertEqual(after_states[-1]["history"], inputs[-30:])
        self.assertEqual(len(after_states[-1]["neurons"]), 20)
        self.assertEqual(len(after_states[-1]["connections"]), 20)

        raw_events_before_reset = events_path_1.read_bytes()
        decoded_events = [json.loads(line) for line in raw_events_before_reset.decode("utf-8").splitlines()]
        self.assertEqual(len(decoded_events), 2 * len(inputs))
        for pair_index in range(len(inputs)):
            scored = decoded_events[2 * pair_index]
            ready = decoded_events[2 * pair_index + 1]
            before = before_states[pair_index]
            after = after_states[pair_index]
            self.assertEqual(scored["event"], "press_scored")
            self.assertEqual(ready["event"], "next_prediction_ready")
            self.assertEqual(scored["session_id"], session_id_1)
            self.assertEqual(scored["prediction_id"], before["prediction_id"])
            self.assertEqual(scored["probabilities"], before["probabilities"])
            self.assertEqual(scored["prediction"], before["prediction"])
            self.assertEqual(scored["actual_digit"], inputs[pair_index])
            self.assertEqual(scored["step"], pair_index + 1)
            self.assertEqual(ready["after_step"], pair_index + 1)
            self.assertEqual(ready["prediction_id"], after["prediction_id"])
            self.assertEqual(ready["probabilities"], after["probabilities"])
            self.assertEqual(ready["neurons"], after["neurons"])
            self.assertEqual(ready["connections"], after["connections"])
            self.assertEqual(ready["training_updates"], (pair_index + 1 - 1) * 3)

        scored_events = [event for event in decoded_events if event["event"] == "press_scored"]
        replayed_correct = sum(
            int(event["prediction"] == event["actual_digit"]) for event in scored_events
        )
        replayed_accuracy = replayed_correct / len(scored_events)
        final_metrics = after_states[-1]["metrics"]
        self.assertEqual(replayed_correct, final_metrics["correct"])
        self.assertEqual(len(scored_events), final_metrics["total"])
        self.assertAlmostEqual(replayed_accuracy, final_metrics["accuracy"], places=12)
        self.assertTrue(all(event["correct"] == (event["prediction"] == event["actual_digit"]) for event in scored_events))

        # Rebuild the majority baseline from only labels preceding each prediction.
        counts = [0] * INPUT_COUNT
        majority_hits = 0
        for event in scored_events:
            expected_guess = max(range(INPUT_COUNT), key=lambda digit: counts[digit])
            self.assertEqual(event["majority_prediction"], expected_guess)
            majority_hits += int(expected_guess == event["actual_digit"])
            counts[event["actual_digit"]] += 1
        self.assertEqual(final_metrics["majority_baseline"], majority_hits / len(scored_events))

        status, export_headers, export_data = self._json_request("/api/export")
        self.assertEqual(status, 200)
        self.assertIn("keypress20-export.json", export_headers.get("Content-Disposition", ""))
        self.assertEqual(len(export_data["sessions"]), 1)
        self.assertEqual(export_data["sessions"][0]["session"]["session_id"], session_id_1)
        self.assertEqual(export_data["sessions"][0]["events"], decoded_events)

        old_log_digest = hashlib.sha256(raw_events_before_reset).hexdigest()
        status, _, reset_state = self._json_request("/api/reset", "POST", {})
        self.assertEqual(status, 200)
        self._assert_state_shape(reset_state)
        session_id_2 = str(reset_state["session_id"])
        self.assertNotEqual(session_id_1, session_id_2)
        self.assertEqual(reset_state["step"], 0)
        self.assertEqual(reset_state["history"], [])
        self.assertEqual(reset_state["prediction_source"], "uniform_cold_start")
        self.assertEqual(reset_state["probabilities"], [0.1] * 10)
        self.assertTrue(session_dir_1.is_dir())
        self.assertEqual(hashlib.sha256(events_path_1.read_bytes()).hexdigest(), old_log_digest)

        status, _, second_result = self._json_request(
            "/api/press",
            "POST",
            {"digit": 9, "prediction_id": reset_state["prediction_id"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(second_result["session_id"], session_id_2)
        self.assertEqual(second_result["last_result"]["prediction_source"], "uniform_cold_start")
        self.assertEqual(second_result["last_result"]["prediction"], 0)
        self.assertEqual(second_result["last_result"]["actual_digit"], 9)
        self.assertEqual(hashlib.sha256(events_path_1.read_bytes()).hexdigest(), old_log_digest)

        second_events_path = self.service.session_dir / "events.jsonl"
        second_events = [
            json.loads(line)
            for line in second_events_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(second_events), 2)
        self.assertTrue(all(event["session_id"] == session_id_2 for event in second_events))
        status, _, full_export = self._json_request("/api/export")
        self.assertEqual(status, 200)
        self.assertEqual(
            {item["session"]["session_id"] for item in full_export["sessions"]},
            {session_id_1, session_id_2},
        )
        exported_first = next(
            item for item in full_export["sessions"] if item["session"]["session_id"] == session_id_1
        )
        exported_second = next(
            item for item in full_export["sessions"] if item["session"]["session_id"] == session_id_2
        )
        self.assertEqual(exported_first["events"], decoded_events)
        self.assertEqual(exported_second["events"], second_events)

        source_manifest = json.loads(
            (self.service.run_root / "source_snapshot" / "manifest.json").read_text(encoding="utf-8")
        )
        snapshot_hashes_match = True
        for relative, expected_hash in source_manifest["source_sha256"].items():
            snapshot_path = self.service.run_root / "source_snapshot" / relative
            self.assertTrue(snapshot_path.is_file(), relative)
            actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, relative)

        if OUTPUT_DIR:
            report_path = Path(OUTPUT_DIR) / "server_summary.json"
            report = {
                "seed": SEED,
                "listen_host": self.host,
                "listen_port": self.port,
                "run_root": str(self.service.run_root),
                "sessions": [session_id_1, session_id_2],
                "session_1_input_digits": inputs,
                "session_1_press_events": len(scored_events),
                "session_1_jsonl_event_lines": len(decoded_events),
                "replayed_correct": replayed_correct,
                "replayed_accuracy": replayed_accuracy,
                "state_metrics": final_metrics,
                "context_limit_observed": len(after_states[-1]["history"]) == 30,
                "reset_preserved_old_log_sha256": old_log_digest,
                "stale_retry": stale_retry_evidence,
                "source_snapshot_hashes_match": snapshot_hashes_match,
                "event_order": "press_scored then next_prediction_ready for every key",
                "model_output_matrix": "20 neurons; 20x20 finite connections; row target, column source",
                "ui_static_routes": {
                    "index_html": index_status == 200,
                    "app_js": app_status == 200,
                    "api_state_calls": "/api/state" in js_text,
                    "api_press_calls": "/api/press" in js_text,
                    "keyboard_listener": 'document.addEventListener("keydown"' in js_text,
                },
                "invalid_digit_types_rejected": ["bool", "float", "string", "null", "out of range"],
                "all_data_local_under": str(self.output_root),
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )


class UOROHTTPProtocolTests(unittest.TestCase):
    """HTTP checks for persistent UORO mode, separate from the window baseline."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        if OUTPUT_DIR:
            cls.output_root = Path(OUTPUT_DIR)
            cls.output_root.mkdir(parents=True, exist_ok=True)
        else:
            cls.output_root = Path(tempfile.mkdtemp(prefix="keypress20-uoro-server-qa-"))

        cls.data_parent = cls.output_root / "uoro_server_runs"
        cls.data_parent.mkdir(parents=True, exist_ok=True)
        cls.service = KeypressService(
            data_parent=cls.data_parent,
            seed=SEED,
            mode="uoro",
            model_options={"rank": 4, "learning_rate": 0.003, "sparsity_strength": 0.02},
        )
        cls.server = LocalHTTPServer(
            ("127.0.0.1", 0),
            cls.service,
            Path(__file__).with_name("dist"),
        )
        cls.host, cls.port = cls.server.server_address
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5)

    def _request(
        self,
        path: str,
        method: str = "GET",
        body: object = _NO_BODY,
    ) -> tuple[int, object, bytes]:
        headers: dict[str, str] = {}
        data: bytes | None
        if body is _NO_BODY:
            data = None
        elif isinstance(body, bytes):
            data = body
        else:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def _json_request(self, *args: object, **kwargs: object) -> tuple[int, object, dict[str, object]]:
        status, headers, payload = self._request(*args, **kwargs)  # type: ignore[arg-type]
        return status, headers, json.loads(payload.decode("utf-8"))

    def _assert_input_topology(self, state: dict[str, object]) -> None:
        connections = state["input_connections"]
        self.assertEqual(len(connections), NEURON_COUNT)
        self.assertTrue(all(len(row) == INPUT_COUNT for row in connections))
        expected_counts = [
            sum(float(row[digit]) != 0.0 for row in connections)
            for digit in range(INPUT_COUNT)
        ]
        self.assertEqual(state["input_counts"], expected_counts)
        info = state["model_info"]
        self.assertEqual(info["input_weights"], connections)
        self.assertEqual(info["by_digit"], expected_counts)
        self.assertEqual(info["input_nonzero_total"], sum(expected_counts))

    def _assert_same_tree(self, actual: object, expected: object, path: str = "root") -> None:
        if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, path)
            self.assertIsInstance(expected, torch.Tensor, path)
            self.assertTrue(torch.equal(actual, expected), path)
        elif isinstance(actual, dict) or isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertIsInstance(expected, dict, path)
            self.assertEqual(set(actual), set(expected), path)
            for key in actual:
                self._assert_same_tree(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), path)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self._assert_same_tree(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(actual, expected, path)

    def test_uoro_press_order_stale_reset_and_complete_checkpoint(self) -> None:
        self.assertEqual(self.host, "127.0.0.1")
        self.assertNotEqual(self.port, 8765)

        status, _, initial = self._json_request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(initial["step"], 0)
        self.assertEqual(initial["training_mode"], "uoro")
        self.assertEqual(initial["model_info"]["mode"], "uoro")
        self.assertEqual(initial["model_info"]["uoro_rank"], 4)
        self.assertTrue(initial["model_info"]["exact_readout_step"])
        self.assertEqual(initial["model_info"]["observed_steps"], 0)
        self._assert_input_topology(initial)

        first_digit = 4
        status, _, after_first = self._json_request(
            "/api/press",
            "POST",
            {"digit": first_digit, "prediction_id": initial["prediction_id"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(after_first["last_result"]["prediction_id"], initial["prediction_id"])
        self.assertEqual(after_first["last_result"]["probabilities"], initial["probabilities"])
        self.assertEqual(after_first["step"], 1)
        self.assertEqual(after_first["model_info"]["observed_steps"], 1)
        self.assertEqual(after_first["model_info"]["optimizer_updates"], 0)
        self.assertEqual(after_first["input_counts"], initial["input_counts"])

        # A duplicate sealed token must leave the log, checkpoint, and model untouched.
        events_path = self.service.session_dir / "events.jsonl"
        checkpoint_path = self.service.session_dir / "latest_checkpoint.pt"
        old_events = events_path.read_bytes()
        old_checkpoint = checkpoint_path.read_bytes()
        stale_status, _, stale_error = self._json_request(
            "/api/press",
            "POST",
            {"digit": first_digit, "prediction_id": initial["prediction_id"]},
        )
        self.assertEqual(stale_status, 409)
        self.assertEqual(stale_error["error"]["code"], "stale_prediction")
        self.assertEqual(events_path.read_bytes(), old_events)
        self.assertEqual(checkpoint_path.read_bytes(), old_checkpoint)
        self.assertEqual(self.service.model.optimizer_updates, 0)

        inputs = [first_digit] + [index % INPUT_COUNT for index in range(1, 40)]
        states = [initial, after_first]
        for index, digit in enumerate(inputs[1:], start=1):
            before_status, _, before = self._json_request("/api/state")
            self.assertEqual(before_status, 200)
            self.assertEqual(before["step"], index)
            status, _, after = self._json_request(
                "/api/press",
                "POST",
                {"digit": digit, "prediction_id": before["prediction_id"]},
            )
            self.assertEqual(status, 200)
            result = after["last_result"]
            self.assertEqual(result["prediction_id"], before["prediction_id"])
            self.assertEqual(result["prediction"], before["prediction"])
            self.assertEqual(result["probabilities"], before["probabilities"])
            self.assertEqual(result["actual_digit"], digit)
            self.assertEqual(after["step"], index + 1)
            self.assertEqual(after["training_mode"], "uoro")
            self.assertEqual(after["model_info"]["observed_steps"], index + 1)
            self.assertEqual(after["model_info"]["optimizer_updates"], index)
            self.assertEqual(after["history"], inputs[max(0, index - 29) : index + 1])
            self._assert_input_topology(after)
            states.append(after)

        # The model has seen more than the old 32-token window; the checkpoint
        # preserves its complete runtime and is not reconstructed from history.
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(raw_checkpoint["training_mode"], "uoro")
        model_checkpoint = raw_checkpoint["model_checkpoint_state"]
        self.assertEqual(model_checkpoint["format_version"], CHECKPOINT_FORMAT_VERSION)
        self.assertIn("model", model_checkpoint)
        self.assertIn("optimizer", model_checkpoint)
        self.assertIn("runtime", model_checkpoint)
        runtime = model_checkpoint["runtime"]
        self.assertEqual(runtime["observed_steps"], 40)
        self.assertEqual(runtime["optimizer_updates"], 39)
        self.assertEqual(tuple(runtime["h"].shape), (NEURON_COUNT,))
        self.assertEqual(tuple(runtime["fast_edges"].shape), (NEURON_COUNT, NEURON_COUNT))
        self.assertEqual(tuple(runtime["uoro_a"].shape), (4, 420))
        self.assertEqual(tuple(runtime["uoro_b"].shape), (4, 623))
        self.assertIn("rng_state", runtime)
        self.assertTrue(model_checkpoint["config"]["exact_readout_step"])
        pending = runtime["pending_transition"]
        self.assertEqual(tuple(pending["base_state"].shape), (420,))
        self.assertEqual(tuple(pending["uoro_a"].shape), (4, 420))
        self.assertEqual(tuple(pending["uoro_b"].shape), (4, 623))
        self.assertEqual(tuple(pending["theta"].shape), (623,))
        self.assertIsInstance(pending["digit"], int)
        self._assert_same_tree(model_checkpoint, self.service.model.checkpoint_state())
        completed_optimizer_updates = self.service.model.optimizer_updates

        session_id = str(initial["session_id"])
        session_dir = self.service.session_dir
        raw_log = events_path.read_bytes()
        decoded = [json.loads(line) for line in raw_log.decode("utf-8").splitlines()]
        self.assertEqual(len(decoded), 2 * len(inputs))
        for index, digit in enumerate(inputs):
            scored, ready = decoded[2 * index : 2 * index + 2]
            before = states[index]
            after = states[index + 1]
            self.assertEqual(scored["event"], "press_scored")
            self.assertEqual(ready["event"], "next_prediction_ready")
            self.assertEqual(scored["actual_digit"], digit)
            self.assertEqual(scored["prediction_id"], before["prediction_id"])
            self.assertEqual(scored["probabilities"], before["probabilities"])
            self.assertEqual(ready["after_step"], index + 1)
            self.assertEqual(ready["training_mode"], "uoro")
            self.assertEqual(ready["model_info"]["observed_steps"], index + 1)
            self.assertEqual(ready["model_info"]["optimizer_updates"], index)
            self.assertEqual(ready["probabilities"], after["probabilities"])
        old_log_digest = hashlib.sha256(raw_log).hexdigest()

        snapshot_manifest = json.loads(
            (self.service.run_root / "source_snapshot" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("online_model.py", snapshot_manifest["source_sha256"])
        self.assertTrue((self.service.run_root / "source_snapshot" / "online_model.py").is_file())

        reset_status, _, reset_state = self._json_request("/api/reset", "POST", {})
        self.assertEqual(reset_status, 200)
        self.assertNotEqual(reset_state["session_id"], session_id)
        self.assertEqual(reset_state["step"], 0)
        self.assertEqual(reset_state["history"], [])
        self.assertEqual(reset_state["training_mode"], "uoro")
        self.assertEqual(reset_state["model_info"]["observed_steps"], 0)
        self.assertEqual(hashlib.sha256(events_path.read_bytes()).hexdigest(), old_log_digest)
        self.assertTrue(session_dir.is_dir())

        export_status, _, exported = self._json_request("/api/export")
        self.assertEqual(export_status, 200)
        exported_session = next(
            item for item in exported["sessions"] if item["session"]["session_id"] == session_id
        )
        self.assertEqual(exported_session["events"], decoded)

        if OUTPUT_DIR:
            report_path = Path(OUTPUT_DIR) / "uoro_server_summary.json"
            report_path.write_text(
                json.dumps(
                    {
                        "seed": SEED,
                        "mode": "uoro",
                        "listen_host": self.host,
                        "ephemeral_port": self.port,
                        "session_id": session_id,
                        "presses": len(inputs),
                        "optimizer_updates": completed_optimizer_updates,
                        "checkpoint_observed_steps": runtime["observed_steps"],
                        "reset_preserved_old_log_sha256": old_log_digest,
                        "event_order": "press_scored then next_prediction_ready per key",
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
