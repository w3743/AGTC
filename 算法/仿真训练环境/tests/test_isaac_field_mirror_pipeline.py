from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = (
    TRAINING_ROOT
    / "results/isaac_field_improvement/physx_pointmove_flat12_mirror_20261024_a04/run_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("isaac_field_mirror_pipeline_under_test", PIPELINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {PIPELINE_PATH}")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class MirrorPipelineTests(unittest.TestCase):
    def test_resume_arithmetic_uses_cumulative_transitions_and_second_resume_iter_rule(self) -> None:
        plan = pipeline.resume_plan(iteration=48, transitions=25600, adam_step=1000, optimizer_entries=37)
        self.assertEqual(plan["remaining_iterations"], 334)
        self.assertEqual(plan["expected_num_timesteps"], 196608)
        self.assertEqual(plan["expected_final_payload_iter"], 381)
        self.assertEqual(plan["expected_adam_step"], 7680)

        # Iteration metadata changes the expected final payload iter only; it does not
        # replace cumulative transition arithmetic or filename-based calculations.
        shifted = pipeline.resume_plan(iteration=49, transitions=25600, adam_step=1000, optimizer_entries=37)
        self.assertEqual(shifted["remaining_iterations"], plan["remaining_iterations"])
        self.assertEqual(shifted["expected_adam_step"], plan["expected_adam_step"])
        self.assertEqual(shifted["expected_final_payload_iter"], 382)

    def test_resume_arithmetic_rejects_partial_or_complete_rollouts(self) -> None:
        for transitions in (25599, 196608):
            with self.subTest(transitions=transitions), self.assertRaises(ValueError):
                pipeline.resume_plan(48, transitions, 1000, 37)
        with self.assertRaises(ValueError):
            pipeline.resume_plan(48, 25600, 1000, 0)

    def test_training_command_is_headless_and_exports_to_this_trial(self) -> None:
        plan = pipeline.resume_plan(48, 25600, 1000, 37)
        argv = pipeline.training_argv(Path("a03/model_48.pt"), plan)
        self.assertIn("--viz", argv)
        self.assertEqual(argv[argv.index("--viz") + 1], "none")
        self.assertIn("--live-state-file", argv)
        self.assertEqual(
            Path(argv[argv.index("--live-state-file") + 1]),
            pipeline.LIVE_STATE_PATH.resolve(),
        )
        self.assertEqual(argv[argv.index("--max_iterations") + 1], "334")
        self.assertNotIn("--kit_args", argv)
        self.assertNotIn("--headless", argv)

    def test_helper_redirection_keeps_a03_and_a02_as_separate_inputs(self) -> None:
        self.assertNotEqual(pipeline.TRIAL_DIR, pipeline.A03_DIR)
        self.assertEqual(pipeline.A03.A02_DIR.resolve(), pipeline.A02_DIR.resolve())
        helper = pipeline._helper()
        self.assertEqual(helper.TRIAL_DIR.resolve(), pipeline.TRIAL_DIR.resolve())
        self.assertNotEqual(helper.TRIAL_DIR.resolve(), pipeline.A03_DIR.resolve())

    def test_parent_freeze_checks_snapshot_bytes_without_comparing_live_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary)
            snapshot = trial / "source_snapshot/isaac_field/env.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"frozen a03 source")
            manifest = {"files": [{"path": "isaac_field/env.py", "snapshot_path": snapshot.relative_to(trial).as_posix(),
                                   "sha256": pipeline.sha256_file(snapshot)}]}
            self.assertEqual(pipeline._snapshot_mismatches(trial, manifest), [])
            snapshot.write_bytes(b"modified frozen bytes")
            self.assertEqual(pipeline._snapshot_mismatches(trial, manifest), ["isaac_field/env.py"])

    def test_stopped_process_roles_accept_training_gui_or_legacy_static_gui(self) -> None:
        base = {
            role: {"state": "stopped", "verified": True, "pid": index + 1, "evidence_path": "evidence.json"}
            for index, role in enumerate(pipeline.CORE_PROCESS_ROLES)
        }
        for gui_role in ("training_gui", "static_gui"):
            records = {**base, gui_role: {"state": "stopped", "verified": True, "pid": 1, "evidence_path": "evidence.json"}}
            self.assertEqual(pipeline._process_roles({"old_processes": records})[-1], gui_role)

    def test_training_child_gets_live_state_run_id_and_headless_environment(self) -> None:
        env = pipeline._headless_child_environment({"HEADLESS": "0", "XR": "1"}, training=True)
        self.assertEqual(env["HEADLESS"], "1")
        self.assertEqual(env["LIVESTREAM"], "0")
        self.assertEqual(env["XR"], "0")
        self.assertEqual(env["ISAAC_FIELD_LIVE_STATE_RUN_ID"], pipeline.TRIAL_DIR.name)
        evaluation = pipeline._headless_child_environment({"ISAAC_FIELD_LIVE_STATE_RUN_ID": "stale"}, training=False)
        self.assertNotIn("ISAAC_FIELD_LIVE_STATE_RUN_ID", evaluation)

if __name__ == "__main__":
    unittest.main()
