from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import unittest


sys.dont_write_bytecode = True
TRAINING_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = (
    TRAINING_ROOT
    / "results/isaac_field_improvement/physx_pointmove_flat12_recover_20261024_a05/run_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("isaac_field_headless_recovery_pipeline_under_test", PIPELINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {PIPELINE_PATH}")
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class HeadlessRecoveryPipelineTests(unittest.TestCase):
    def test_resume_arithmetic_from_26112_transitions(self) -> None:
        plan = pipeline.resume_plan(iteration=48, transitions=26112, adam_step=1020, optimizer_entries=37)
        self.assertEqual(plan, {
            "remaining_iterations": 333,
            "expected_num_timesteps": 196608,
            "expected_final_payload_iter": 380,
            "expected_adam_step": 7680,
        })

    def test_resume_plan_rejects_partial_or_incomplete_optimizer_checkpoint(self) -> None:
        for transitions in (26111, 196608):
            with self.subTest(transitions=transitions), self.assertRaises(ValueError):
                pipeline.resume_plan(48, transitions, 1020, 37)
        with self.assertRaises(ValueError):
            pipeline.resume_plan(48, 26112, 1020, 0)

    def test_training_argv_uses_single_headless_cli_and_a05_live_state(self) -> None:
        plan = pipeline.resume_plan(48, 26112, 1020, 37)
        argv = pipeline.training_argv(pipeline.A04_CHECKPOINT, plan)
        self.assertEqual(argv[argv.index("--viz") + 1], "none")
        self.assertEqual(argv[argv.index("--max_iterations") + 1], "333")
        self.assertEqual(argv[argv.index("--resume_optimizer_state")], "--resume_optimizer_state")
        self.assertEqual(argv[argv.index("--run_timestamp") + 1], "training_recover_a05")
        self.assertEqual(Path(argv[argv.index("--live-state-file") + 1]), pipeline.LIVE_STATE_PATH.resolve())
        self.assertEqual(argv[-2:], ["env.task.size=12.0", "env.scene.env_spacing=22.0"])
        self.assertNotIn("--headless", argv)
        self.assertNotIn("--kit_args", argv)

    def test_headless_environment_keeps_observer_state_id_only_for_training(self) -> None:
        training = pipeline._headless_child_environment({"HEADLESS": "0", "LIVESTREAM": "1", "XR": "1"}, training=True)
        self.assertEqual((training["HEADLESS"], training["LIVESTREAM"], training["XR"]), ("1", "0", "0"))
        self.assertEqual(training["ISAAC_FIELD_LIVE_STATE_RUN_ID"], pipeline.TRIAL_DIR.name)
        evaluation = pipeline._headless_child_environment({"ISAAC_FIELD_LIVE_STATE_RUN_ID": "stale"}, training=False)
        self.assertNotIn("ISAAC_FIELD_LIVE_STATE_RUN_ID", evaluation)

    def test_web_observer_argv_uses_localhost_and_all_required_output_flags(self) -> None:
        argv = pipeline.live_observer_argv()
        self.assertEqual(argv[1], str((pipeline.TRAINING_ROOT / pipeline.VIEWER_SCRIPT).resolve()))
        self.assertEqual(argv[argv.index("--state-file") + 1], str(pipeline.LIVE_STATE_PATH.resolve()))
        self.assertEqual(argv[argv.index("--output") + 1], str((pipeline.TRIAL_DIR / "live_viewer").resolve()))
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], "8790")
        self.assertEqual(argv[argv.index("--expected-run-id") + 1], pipeline.TRIAL_DIR.name)

    def test_a04_a03_helpers_are_redirected_without_changing_a02_lineage(self) -> None:
        helper = pipeline.A04._helper()
        self.assertEqual(helper.TRIAL_DIR.resolve(), pipeline.TRIAL_DIR.resolve())
        self.assertEqual(pipeline.A03.TRIAL_DIR.resolve(), pipeline.TRIAL_DIR.resolve())
        self.assertEqual(pipeline.A03.A02_DIR.resolve(), pipeline.A02_DIR.resolve())
        self.assertEqual(pipeline.A04.A03_DIR.resolve(), pipeline.A04_DIR.resolve())
        self.assertNotEqual(helper.TRIAL_DIR.resolve(), pipeline.A04_DIR.resolve())

    def test_production_manifest_allows_only_the_expected_new_web_viewer(self) -> None:
        parent = {"files": [
            {"path": "isaac_field/env.py", "origin": "project", "sha256": "env"},
            {"path": "scripts/view_training_state.py", "origin": "project", "sha256": "old-viewer"},
            {"path": "results/old/run_pipeline.py", "origin": "project", "sha256": "artifact"},
        ]}
        current = {"files": [
            {"path": "isaac_field/env.py", "origin": "project", "sha256": "env"},
            {"path": "scripts/view_training_state.py", "origin": "project", "sha256": "old-viewer"},
            {"path": pipeline.VIEWER_SCRIPT, "origin": "project", "sha256": "new-viewer"},
            {"path": "source_snapshot/inherited/a04/protocol.json", "origin": "trial", "sha256": "pinned"},
        ]}
        result = pipeline.compare_production_manifests(parent, current)
        self.assertEqual(result["added_files"], [pipeline.VIEWER_SCRIPT])
        self.assertTrue(result["only_allowed_new_viewer"])

    def test_production_manifest_rejects_unexpected_source_drift(self) -> None:
        parent = {"files": [{"path": "isaac_field/env.py", "origin": "project", "sha256": "before"}]}
        for current in (
            {"files": [{"path": "isaac_field/env.py", "origin": "project", "sha256": "after"}]},
            {"files": [
                {"path": "isaac_field/env.py", "origin": "project", "sha256": "before"},
                {"path": "scripts/unrelated.py", "origin": "project", "sha256": "new"},
            ]},
            {"files": []},
        ):
            with self.subTest(current=current), self.assertRaises(RuntimeError):
                pipeline.compare_production_manifests(parent, current)

    @staticmethod
    def _process_records() -> list[dict[str, object]]:
        created = {
            "controller": "2026-10-24T08:48:40.299016Z",
            "resume_training_cli": "2026-10-24T08:48:43.623846Z",
            "training_native": "2026-10-24T08:48:43.677027Z",
            "watcher": "2026-10-24T08:52:50.182064Z",
            "viewer": None,
        }
        return [
            {"role": role, "pid": 26052 + index, "created_at_utc": created[role],
             "last_seen_at_utc": "2026-10-24T08:56:19Z", "state": "stopped"}
            for index, role in enumerate(pipeline.PROCESS_ROLES)
        ]

    def test_reboot_handoff_accepts_missing_original_pids_and_postboot_pid_reuse(self) -> None:
        boot = pipeline.parse_utc("2026-10-24T09:00:21.665056Z")
        records = self._process_records()
        reused_pid = records[1]["pid"]
        after_boot = datetime(2026, 10, 24, 9, 1, tzinfo=timezone.utc)
        outcomes = pipeline._verify_recovery_processes(
            records,
            boot,
            lambda pid: after_boot if pid == reused_pid else None,
        )
        self.assertEqual(len(outcomes), len(pipeline.PROCESS_ROLES))
        self.assertEqual(outcomes[0]["current_pid_state"], "missing")
        self.assertEqual(outcomes[1]["current_pid_state"], "pid_reused_after_boot")

    def test_reboot_handoff_blocks_the_same_original_pid_creation_identity(self) -> None:
        boot = pipeline.parse_utc("2026-10-24T09:00:21.665056Z")
        records = self._process_records()
        original = pipeline.parse_utc(records[0]["created_at_utc"])
        with self.assertRaisesRegex(RuntimeError, "original controller identity"):
            pipeline._verify_recovery_processes(records, boot, lambda pid: original if pid == records[0]["pid"] else None)

    def test_unknown_viewer_creation_time_still_requires_missing_or_postboot_pid(self) -> None:
        boot = pipeline.parse_utc("2026-10-24T09:00:21.665056Z")
        records = self._process_records()
        viewer = next(row for row in records if row["role"] == "viewer")
        before_boot = datetime(2026, 10, 24, 8, 59, tzinfo=timezone.utc)
        with self.assertRaisesRegex(RuntimeError, "pre-reboot creation time"):
            pipeline._verify_recovery_processes(records, boot, lambda pid: before_boot if pid == viewer["pid"] else None)


if __name__ == "__main__":
    unittest.main()
