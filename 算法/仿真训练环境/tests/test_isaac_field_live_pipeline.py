from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PIPELINE = (
    Path(__file__).resolve().parents[1]
    / "results/isaac_field_improvement/physx_pointmove_flat12_live_20261024_a03/run_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("isaac_field_live_pipeline_under_test", PIPELINE)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class LivePipelineTests(unittest.TestCase):
    def test_remaining_iterations_use_cumulative_transitions(self) -> None:
        self.assertEqual(pipeline.remaining_iterations(16896), 351)
        self.assertEqual(pipeline.remaining_iterations(25088), 335)
        self.assertEqual(pipeline.remaining_iterations(196096), 1)

    def test_remaining_iterations_reject_partial_or_complete_checkpoint(self) -> None:
        for transitions in (16895, 196608, 200000):
            with self.subTest(transitions=transitions), self.assertRaises(ValueError):
                pipeline.remaining_iterations(transitions)

    def test_training_argv_selects_kit_without_hiding_environments(self) -> None:
        argv = pipeline.training_argv(Path("source/model_32.pt"), 351)
        self.assertEqual(argv[argv.index("--max_iterations") + 1], "351")
        self.assertEqual(argv[argv.index("--viz") + 1], "kit")
        self.assertIn("--resume_optimizer_state", argv)
        self.assertIn("--kit_args=--/isaac/startup/create_new_stage=false", argv)
        self.assertNotIn("--headless", argv)
        self.assertNotIn("--max_visible_envs", argv)

    def test_training_process_gets_gui_environment_but_evaluation_does_not(self) -> None:
        base = {"PATH": "python"}
        gui = pipeline.apply_training_gui_environment(base, training=True)
        evaluation = pipeline.apply_training_gui_environment(base, training=False)
        self.assertEqual((gui["HEADLESS"], gui["LIVESTREAM"], gui["XR"]), ("0", "0", "0"))
        self.assertNotIn("HEADLESS", evaluation)

    def test_checkpoint_selector_uses_a02_source_directory(self) -> None:
        self.assertEqual(
            pipeline.a02_training_run_dir(),
            pipeline.A02_DIR / "logs/rsl_rl" / pipeline.A02_DIR.name / "training_a02",
        )

    def test_adam_step_requires_all_parameters_to_agree(self) -> None:
        self.assertEqual(
            pipeline._adam_step({"optimizer_state_dict": {"state": {
                "a": {"step": 660}, "b": {"step": 660},
            }}}),
            660,
        )
        with self.assertRaisesRegex(ValueError, "diverge"):
            pipeline._adam_step({"optimizer_state_dict": {"state": {
                "a": {"step": 660}, "b": {"step": 661},
            }}})

    def test_handoff_requires_verified_stopped_process_roles(self) -> None:
        records = {
            role: {"state": "stopped", "verified": True, "evidence_path": "evidence.json"}
            for role in pipeline.PROCESS_ROLES
        }
        with tempfile.TemporaryDirectory() as temporary:
            handoff_path = Path(temporary) / "handoff.json"
            handoff_path.write_text(
                '{"checkpoint": {}, "old_processes": '
                + json.dumps({key: {**value, "pid": 1000 + index} for index, (key, value) in enumerate(records.items())})
                + ', "old_run_status": {"status": "superseded", "superseded_by": '
                + json.dumps(str(pipeline.TRIAL_DIR.resolve())) + '}}',
                encoding="utf-8",
            )
            accepted = pipeline._require_stopped_handoff(handoff_path)
            self.assertEqual(accepted["old_run_status"]["status"], "superseded")
            records["watcher"]["verified"] = False
            handoff_path.write_text(
                '{"checkpoint": {}, "old_processes": '
                + json.dumps({key: {**value, "pid": 1000 + index} for index, (key, value) in enumerate(records.items())})
                + ', "old_run_status": {"status": "superseded", "superseded_by": '
                + json.dumps(pipeline.TRIAL_DIR.name) + '}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "watcher"):
                pipeline._require_stopped_handoff(handoff_path)

    def test_preflight_checks_real_pid_state_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "process_handoff.json"
            evidence.write_text("{}", encoding="utf-8")
            processes = {
                role: {"pid": 1000 + index, "state": "stopped", "verified": True,
                       "evidence_path": str(evidence)}
                for index, role in enumerate(pipeline.PROCESS_ROLES)
            }
            handoff = {"old_processes": processes}
            with mock.patch.object(pipeline, "_pid_running", return_value=False):
                pipeline._verify_process_evidence(handoff)
            with mock.patch.object(pipeline, "_pid_running", side_effect=[True] + [False] * 4):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    pipeline._verify_process_evidence(handoff)

    def test_checkpoint_selector_falls_back_from_incomplete_latest(self) -> None:
        class Helper:
            @staticmethod
            def _numeric_model_checkpoints(run_dir: Path):
                return [(31, run_dir / "model_31.pt"), (32, run_dir / "model_32.pt")]

            @staticmethod
            def cpu_load_checkpoint(path: Path):
                iteration = int(path.stem.split("_")[1])
                if iteration == 32:
                    raise ValueError("truncated checkpoint")
                transitions = 196608 if iteration == 382 else (iteration + 1) * 512
                step = 7680 if iteration == 382 else 640
                return ({
                    "iter": iteration,
                    "num_timesteps": transitions,
                    "actor_state_dict": {},
                    "critic_state_dict": {},
                    "optimizer_state_dict": {"state": {"p": {"step": step}}},
                    "field_policy_contract": {},
                }, {"device": "cpu", "finite_tensor_tree": True})

            @staticmethod
            def _assert_checkpoint_contract(contract, expected):
                return {"contract": contract}

            @staticmethod
            def expected_contract_requirements():
                return {}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for filename in ("model_31.pt", "model_32.pt"):
                (run_dir / filename).write_bytes(b"complete enough for stat")
            with mock.patch.object(pipeline, "_old_pipeline", return_value=Helper), mock.patch.object(pipeline.time, "sleep"):
                path, payload, details = pipeline.select_latest_complete_checkpoint(run_dir)
            self.assertEqual(path.name, "model_31.pt")
            self.assertEqual(payload["num_timesteps"], 16384)
            self.assertEqual(details["adam_step"], 640)

    def test_checkpoint_selector_accepts_runner_final_iteration_382_at_target(self) -> None:
        class Helper:
            @staticmethod
            def _numeric_model_checkpoints(run_dir: Path):
                return [(382, run_dir / "model_382.pt")]

            @staticmethod
            def cpu_load_checkpoint(path: Path):
                return ({
                    "iter": 382,
                    "num_timesteps": 196608,
                    "actor_state_dict": {},
                    "critic_state_dict": {},
                    "optimizer_state_dict": {"state": {"p": {"step": 7680}}},
                    "field_policy_contract": {},
                }, {"device": "cpu", "finite_tensor_tree": True})

            @staticmethod
            def _assert_checkpoint_contract(contract, expected):
                return {"contract": contract}

            @staticmethod
            def expected_contract_requirements():
                return {}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model_382.pt").write_bytes(b"final checkpoint")
            with mock.patch.object(pipeline, "_old_pipeline", return_value=Helper), mock.patch.object(pipeline.time, "sleep"):
                path, payload, details = pipeline.select_latest_complete_checkpoint(run_dir)
            self.assertEqual(path.name, "model_382.pt")
            self.assertEqual(payload["num_timesteps"], 196608)
            self.assertEqual(details["adam_step"], 7680)
            with self.assertRaises(ValueError):
                pipeline.remaining_iterations(payload["num_timesteps"])

    def test_reused_a02_zero_step_baseline_matches_current_policy_contract(self) -> None:
        helper = pipeline._old_pipeline()
        baseline_path = pipeline.A02_DIR / "logs/rsl_rl" / pipeline.A02_DIR.name / "baseline_a02/model_0.pt"
        payload, safe = helper.cpu_load_checkpoint(baseline_path)
        contract = helper._assert_checkpoint_contract(
            payload["field_policy_contract"], helper.expected_contract_requirements()
        )
        self.assertEqual(payload["num_timesteps"], 0)
        self.assertEqual(payload["iter"], 0)
        self.assertTrue(safe["finite_tensor_tree"])
        self.assertIsInstance(contract["contract_sha256"], str)

    def test_live_protocol_reuses_baseline_without_a_new_native_train_command(self) -> None:
        argv = pipeline.training_argv(Path("source/model_32.pt"), 351)
        self.assertEqual(argv[3], "resume")
        self.assertNotIn("train", argv[3:])


if __name__ == "__main__":
    unittest.main()
