import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

if os.name != "nt":
    raise unittest.SkipTest("Windows completion watcher")

SCRIPT = Path(__file__).parents[1] / "scripts" / "watch_training_completion.py"
spec = importlib.util.spec_from_file_location("training_completion_watch", SCRIPT)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)
THREAD = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
MESSAGE = "123e4567-e89b-12d3-a456-426614174000"


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial, self.state = self.root / "trial", self.root / "state"
        self.trial.mkdir()
        self.calls = []

    def status(self, value):
        (self.trial / "run_status.json").write_text(json.dumps(value), encoding="utf-8")

    def sender(self, output=None):
        def run(command, **kwargs):
            self.calls.append((command, kwargs))
            text = output if output is not None else f"Queued message {MESSAGE} for thread {THREAD}\n"
            return subprocess.CompletedProcess(command, 0, text, "")
        return run

    def tick(self, runner=None):
        return watcher.tick(self.trial, THREAD, "fake-codex", self.state, runner or self.sender())

    def test_running_does_not_queue(self):
        self.status({"status": "running", "parent_pid": 1, "children": []})
        self.assertIsNone(self.tick())
        self.assertEqual(self.calls, [])

    def test_training_exit_and_passed_validation_queue_once(self):
        self.status({"status": "running", "children": [
            {"name": "resume_training_cli", "actual_exit_code": 0}]})
        (self.trial / "final_validation.json").write_text('{"passed":true}', encoding="utf-8")
        self.assertEqual(self.tick(), "train_finished")
        self.assertEqual(self.tick(), "train_finished")
        self.assertEqual(len(self.calls), 1)
        self.assertIn("--thread", self.calls[0][0])
        saved = json.loads((self.state / "delivery.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["events"]["train_finished"]["phase"], "queued")

    def test_completed_startup_queues_only_final_event_and_deduplicates(self):
        self.status({"status": "completed", "children": [
            {"name": "resume_training_cli", "actual_exit_code": 0},
            {"name": "evaluate_baseline", "actual_exit_code": 0},
            {"name": "evaluate_final", "actual_exit_code": 0}]})
        (self.trial / "comparison.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.tick(), "pipeline_finished")
        self.assertEqual(self.tick(), "pipeline_finished")
        self.assertEqual(len(self.calls), 1)
        events = json.loads((self.state / "delivery.json").read_text(encoding="utf-8"))["events"]
        self.assertEqual(set(events), {"pipeline_finished"})

    def test_failed_pipeline_queues_failure(self):
        self.status({"status": "failed", "children": []})
        self.assertEqual(self.tick(), "pipeline_failed")
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_json_waits_without_queue(self):
        (self.trial / "run_status.json").write_text("{", encoding="utf-8")
        self.assertIsNone(self.tick())
        self.assertEqual(self.calls, [])
        state = json.loads((self.state / "watch_status.json").read_text(encoding="utf-8"))
        self.assertEqual(state["trial_status"], "unreadable")

    def test_unconfirmed_queue_is_uncertain_and_not_retried(self):
        self.status({"status": "running", "children": [
            {"name": "resume_training_cli", "actual_exit_code": 0}]})
        (self.trial / "final_validation.json").write_text('{"passed":true}', encoding="utf-8")
        fake = self.sender("codex returned without a queue confirmation")
        self.tick(fake)
        self.tick(fake)
        self.assertEqual(len(self.calls), 1)
        saved = json.loads((self.state / "delivery.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["events"]["train_finished"]["phase"], "uncertain")

    def test_optional_export_failure_is_not_pipeline_failure(self):
        self.status({"status": "running", "children": [
            {"name": "optional_export", "optional": True, "actual_exit_code": 1}]})
        self.assertIsNone(self.tick())
        self.assertEqual(self.calls, [])

    def test_parent_crash_requires_two_misses_and_unknown_is_safe(self):
        self.status({"status": "running", "parent_pid": 42, "children": []})
        with patch.object(watcher, "parent_identity", return_value="unknown"):
            self.tick()
            self.tick()
        self.assertEqual(self.calls, [])
        with patch.object(watcher, "parent_identity", return_value="missing"):
            self.assertIsNone(self.tick())
            self.assertEqual(self.tick(), "parent_crashed")
        self.assertEqual(len(self.calls), 1)

    def test_both_real_evaluation_names_are_required(self):
        self.status({"status": "completed", "children": [
            {"name": "evaluate_baseline", "actual_exit_code": 0}]})
        (self.trial / "comparison.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(self.tick())
        self.assertEqual(self.calls, [])

    def test_training_waits_for_successful_final_audit(self):
        self.status({"status": "running", "children": [
            {"name": "resume_training_cli", "actual_exit_code": 0}]})
        self.assertIsNone(self.tick())
        (self.trial / "final_validation.json").write_text('{"status":"failed"}', encoding="utf-8")
        self.assertIsNone(self.tick())
        self.assertEqual(self.calls, [])

    def test_nonzero_queue_exit_remains_uncertain_even_with_ack(self):
        self.status({"status": "failed"})
        def failed(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, f"Queued message {MESSAGE} for thread {THREAD}", "error")
        self.tick(failed)
        saved = json.loads((self.state / "delivery.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["events"]["pipeline_failed"]["phase"], "uncertain")

    def test_state_cannot_be_reused_for_another_chat(self):
        self.status({"status": "running"})
        self.tick()
        with self.assertRaises(RuntimeError):
            watcher.tick(self.trial, MESSAGE, "fake-codex", self.state, self.sender())
        self.assertEqual(self.calls, [])

    def test_windows_process_creation_identity_is_exact(self):
        from datetime import datetime, timedelta
        stamp = watcher.parent_identity(os.getpid(), "capture")
        self.assertNotIn(stamp, {"unknown", "missing", "mismatch"})
        self.assertEqual(watcher.parent_identity(os.getpid(), stamp), "match")
        changed = (datetime.fromisoformat(stamp) + timedelta(seconds=1)).isoformat()
        self.assertEqual(watcher.parent_identity(os.getpid(), changed), "mismatch")

    def test_windows_lock_releases_after_holder_exits(self):
        path = self.root / "watcher.lock"
        first = watcher.lock_state(path)
        try:
            with self.assertRaises(RuntimeError):
                watcher.lock_state(path)
        finally:
            first.close()
        watcher.lock_state(path).close()


if __name__ == "__main__":
    unittest.main()
