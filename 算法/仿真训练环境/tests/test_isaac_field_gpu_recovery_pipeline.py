from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.dont_write_bytecode = True
TRAINING_ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = (
    TRAINING_ROOT
    / "results/isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_module("isaac_field_gpu_recovery_pipeline_under_test", TRIAL_DIR / "run_pipeline.py")
evaluation_adapter = _load_module(
    "isaac_field_gpu_evaluation_adapter_under_test", TRIAL_DIR / "evaluate_torch_batched.py",
)
first_checkpoint_verifier = _load_module(
    "isaac_field_gpu_first_checkpoint_verifier_under_test", TRIAL_DIR / "verify_first_checkpoint.py",
)


class GpuBatchedRecoveryPipelineTests(unittest.TestCase):
    def test_resume_arithmetic_from_audited_checkpoint(self) -> None:
        plan = pipeline.resume_plan(48, 26112, 1020, 37)
        self.assertEqual(plan, {
            "remaining_iterations": 333,
            "expected_num_timesteps": 196608,
            "expected_final_payload_iter": 380,
            "expected_adam_step": 7680,
        })

    def test_official_resume_cli_explicitly_selects_batched_backend(self) -> None:
        plan = pipeline.resume_plan(48, 26112, 1020, 37)
        argv = pipeline.training_argv(pipeline.A04_CHECKPOINT, plan)
        self.assertEqual(argv[:3], [str(pipeline.PYTHON_EXE), "-m", "isaac_field"])
        self.assertEqual(argv[3], "resume")
        self.assertEqual(argv[argv.index("--max_iterations") + 1], "333")
        self.assertEqual(argv[argv.index("--resume_optimizer_state")], "--resume_optimizer_state")
        self.assertEqual(argv[argv.index("--viz") + 1], "none")
        self.assertEqual(argv[argv.index("--live-state-file") + 1], str(pipeline.LIVE_STATE_PATH.resolve()))
        self.assertEqual(argv[-1], "env.task_backend=torch_batched")
        self.assertEqual(argv[-3:-1], ["env.task.size=12.0", "env.scene.env_spacing=22.0"])

    def test_both_paired_evaluations_use_the_trial_adapter_and_same_backend(self) -> None:
        baseline = pipeline.evaluation_argv(pipeline.A02_DIR / "baseline.pt", pipeline.TRIAL_DIR / "eval_baseline")
        final = pipeline.evaluation_argv(pipeline.A04_CHECKPOINT, pipeline.TRIAL_DIR / "eval_final")
        for argv in (baseline, final):
            self.assertEqual(argv[1], str(pipeline.GPU_EVALUATOR_PATH.resolve()))
            self.assertEqual(argv[argv.index("--task-backend") + 1], "torch_batched")
            self.assertEqual(argv[argv.index("--seed-start") + 1], "87010")
            self.assertEqual(argv[argv.index("--scenario-count") + 1], "16")
            self.assertEqual(argv[argv.index("--goal-angle-deg") + 1], "180.0")

    def test_live_state_run_id_is_only_set_for_training(self) -> None:
        training = pipeline._headless_child_environment(
            {"HEADLESS": "0", "LIVESTREAM": "1", "XR": "1"}, training=True,
        )
        self.assertEqual((training["HEADLESS"], training["LIVESTREAM"], training["XR"]), ("1", "0", "0"))
        self.assertEqual(training["ISAAC_FIELD_LIVE_STATE_RUN_ID"], pipeline.TRIAL_DIR.name)
        evaluation = pipeline._headless_child_environment(
            {"ISAAC_FIELD_LIVE_STATE_RUN_ID": "stale"}, training=False,
        )
        self.assertNotIn("ISAAC_FIELD_LIVE_STATE_RUN_ID", evaluation)

    def test_source_comparison_accepts_only_exact_pinned_deltas(self) -> None:
        parent_rows = [{
            "path": path,
            "origin": "project",
            "sha256": spec["parent_sha256"],
        } for path, spec in pipeline.AUTHORIZED_SOURCE_DELTAS.items() if spec["parent_sha256"]]
        current_rows = [{
            "path": path,
            "origin": "project",
            "sha256": spec["current_sha256"],
        } for path, spec in pipeline.AUTHORIZED_SOURCE_DELTAS.items()]
        parent_rows.append({"path": "isaac_field/feedback.py", "origin": "project", "sha256": "stable"})
        current_rows.append({"path": "isaac_field/feedback.py", "origin": "project", "sha256": "stable"})

        result = pipeline.compare_production_manifests({"files": parent_rows}, {"files": current_rows})

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["source_delta_allowlist_exact"])
        self.assertEqual({row["path"] for row in result["authorized_deltas"]},
                         set(pipeline.AUTHORIZED_SOURCE_DELTAS))
        self.assertTrue(all(row["reason"] for row in result["authorized_deltas"]))

    def test_source_comparison_rejects_unlisted_or_changed_allowed_sha(self) -> None:
        parent_rows = [{
            "path": path,
            "origin": "project",
            "sha256": spec["parent_sha256"],
        } for path, spec in pipeline.AUTHORIZED_SOURCE_DELTAS.items() if spec["parent_sha256"]]
        current_rows = [{
            "path": path,
            "origin": "project",
            "sha256": spec["current_sha256"],
        } for path, spec in pipeline.AUTHORIZED_SOURCE_DELTAS.items()]
        with self.assertRaisesRegex(RuntimeError, "exact a06 allowlist"):
            pipeline.compare_production_manifests(
                {"files": parent_rows},
                {"files": [*current_rows, {"path": "scripts/unrelated.py", "origin": "project", "sha256": "new"}]},
            )

        tampered = [dict(row) for row in current_rows]
        next(row for row in tampered if row["path"] == "isaac_field/env.py")["sha256"] = "wrong-sha"
        with self.assertRaisesRegex(RuntimeError, "pinned a06 allowlist"):
            pipeline.compare_production_manifests({"files": parent_rows}, {"files": tampered})

    def test_native_gpu_gate_evidence_matches_the_pinned_six_source_deltas(self) -> None:
        manifest = {"files": [{
            "path": path,
            "origin": "project",
            "sha256": spec["current_sha256"],
        } for path, spec in pipeline.AUTHORIZED_SOURCE_DELTAS.items()]}
        result = pipeline._verify_native_gpu_gate(manifest)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["steps"], 16)
        self.assertEqual(result["task_backend"], "torch_batched")
        self.assertEqual({row["path"] for row in result["source_sha256"]},
                         set(pipeline.AUTHORIZED_SOURCE_DELTAS))

    def test_trial_evaluator_forwards_backend_to_environment_factory(self) -> None:
        engine = types.SimpleNamespace(task_backend="torch_batched", close=lambda: None)
        env_module = types.ModuleType("isaac_field.env")
        env_module.SingleFieldAdapter = lambda selected: ("adapter", selected)
        runtime_module = types.ModuleType("isaac_field.runtime")
        received: dict[str, object] = {}

        def create_env(**kwargs):
            received.update(kwargs)
            return engine

        runtime_module.create_env = create_env
        with patch.object(evaluation_adapter.mastery, "_validate_rsl_task_contract", return_value=("task", "guides", "stability")), \
                patch.dict(sys.modules, {"isaac_field.env": env_module, "isaac_field.runtime": runtime_module}):
            created, adapter = evaluation_adapter._create_rsl_evaluation_adapter(
                {"contract": "test"}, 87010, task_backend="torch_batched",
            )
        self.assertIs(created, engine)
        self.assertEqual(adapter, ("adapter", engine))
        self.assertEqual(received["task_backend"], "torch_batched")
        self.assertEqual(received["num_envs"], 1)

    def test_evaluation_report_records_observed_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for name in ("evaluation.json", "run_manifest.json"):
                (output / name).write_text("{}\n", encoding="utf-8")
            evaluation_adapter._OBSERVED_BACKENDS[:] = ["torch_batched"]
            evaluation_adapter._record_backend(output, "torch_batched")
            for name in ("evaluation.json", "run_manifest.json"):
                result = json.loads((output / name).read_text(encoding="utf-8"))
                self.assertEqual(result["task_backend"], "torch_batched")
                self.assertTrue(result["task_backend_verified"])
        evaluation_adapter._OBSERVED_BACKENDS.clear()

    def test_first_checkpoint_cpu_gate_accepts_the_prepared_a06_protocol(self) -> None:
        result = first_checkpoint_verifier.check_wrapper_preflight(TRIAL_DIR, pipeline, optimizer_entries=37)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["resume_plan"]["remaining_iterations"], 333)
        self.assertEqual(result["training_argv"][-1], "env.task_backend=torch_batched")


if __name__ == "__main__":
    unittest.main()
