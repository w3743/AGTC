from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import benchmark_training as benchmark


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def _fixture_run(run_dir: Path, *, reset_count: int = 0, bad_channel: bool = False) -> None:
    run_dir.mkdir(parents=True)
    contract_sha = "a" * 64
    channels = {
        "lidar": {"mode": "real", "backend": "warp_kernel", "source": "raycast"},
        "depth": {"mode": "real", "backend": "analytic_warp", "source": "raycast", "verified_vs": "depth-ref"},
        "detections": {"mode": "real", "backend": "analytic_detector_warp", "source": "mask", "verified_vs": "det-ref"},
        "state": {"mode": "real", "source": "sim state"},
        "feedback": {"mode": "real", "backend": "vqf_reference", "source": "vqf", "verified_vs": "vqf-ref"},
    }
    if bad_channel:
        channels["depth"]["mode"] = "placeholder"
    contract = {
        "run_id": run_dir.name,
        "observation_contract_sha256": contract_sha,
        "physics_dt_s": 0.001,
        "decimation": 50,
        "physics_experimental": False,
        "task_backend": "fast_stop_v2",
        "capability_metrics_allowed": False,
        "channels": channels,
    }
    identity = {
        "banner": benchmark.BANNER,
        "seed": 0,
        "num_envs": 64,
        "num_steps": 32,
        "device": "cuda",
        "physics": "newton_xpbd",
        "lidar": "warp",
        "feedback": "vqf",
        "policy": {"policy_type": "lstm", "hidden_size": 512},
        "env": {"sim_dt_s": 0.001, "decimation": 50, "task_backend": "fast_stop_v2"},
    }
    summary = {
        "banner": benchmark.BANNER,
        "phase": "wsl_training_prep",
        "capability_metrics_allowed": False,
        "iterations": 7,
        "start_iteration": 0,
        "end_iteration": 7,
        "total_wall_s": 21.0,
        "num_envs": 64,
        "num_steps": 32,
        "device": "cuda",
        "physics": "newton_xpbd",
        "observation_contract_sha256": contract_sha,
        "final_checkpoint": str(run_dir / "latest.pt"),
    }
    rows = []
    for iteration in range(1, 8):
        rows.append({
            "iter": iteration,
            "wall_s": 3.0,
            "collection_s": 1.0,
            "learning_s": 2.0,
            "samples": 64 * 32,
            "samples_per_s": 64 * 32,
            "end_to_end_samples_per_s": (64 * 32) / 3.0,
            "ppo_loss": 0.25,
            "mean_reward": 0.5,
            "rss_mb": 100.0,
            "nonfinite_env_resets": reset_count,
            "nonfinite_actions": 0,
            "observation_contract_sha256": contract_sha,
        })
    _write_json(run_dir / "observation_contract.json", contract)
    _write_json(run_dir / "identity.json", identity)
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "latest.pt").write_bytes(b"fixture-checkpoint-bytes")
    launch = {
        "status": "success",
        "success": True,
        "warmup_iterations": 2,
        "source_identity_still_matches_at_finish": True,
        "source_identity": {
            "raw_source_bytes": [{"path": "train_wsl.py", "bytes": 3, "sha256": "b" * 64}],
            "git": {"commit": "deadbeef", "dirty": True, "status_porcelain_sha256": "c" * 64},
        },
        "execution": {"launcher_wall_s": 25.0, "returncode": 0},
    }
    _write_json(run_dir.parent / "benchmark_result.json", launch)


class BenchmarkTrainingTests(unittest.TestCase):
    def test_command_is_explicit_and_shell_free_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            command = benchmark.train_argv(Path(temp) / "run", 7, 9, 8)
        self.assertEqual(command[0], benchmark.sys.executable)
        self.assertIn("--task-backend", command)
        self.assertEqual(command[command.index("--task-backend") + 1], "fast_stop_v2")
        self.assertEqual(command[command.index("--nonfinite-limit") + 1], "0")
        self.assertEqual(command[command.index("--num-envs") + 1], "64")
        self.assertEqual(command[command.index("--num-steps") + 1], "32")
        self.assertEqual(command[command.index("--policy") + 1], "lstm")
        self.assertNotIn("--profile", command)

    def test_summarize_reports_measured_rate_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            _fixture_run(run)
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            report = benchmark.summarize_run(run)
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.assertTrue(report["success"], report["errors"])
        self.assertEqual(report["warmup_iterations"], 2)
        self.assertEqual(report["measured_iterations"], 5)
        self.assertEqual(report["throughput"]["measured_samples"], 5 * 64 * 32)
        self.assertAlmostEqual(report["throughput"]["collection_samples_per_s"], 64 * 32)
        self.assertAlmostEqual(report["throughput"]["ppo_update_samples_per_s"], (64 * 32) / 2.0)
        self.assertAlmostEqual(report["throughput"]["iteration_window_samples_per_s"], (64 * 32) / 3.0)
        self.assertEqual(report["throughput"]["launcher_wall_s"], 25.0)
        self.assertFalse(report["capability_claim"])
        self.assertEqual(before, after)

    def test_nonfinite_recovery_disqualifies_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            _fixture_run(run, reset_count=1)
            report = benchmark.summarize_run(run)
        self.assertFalse(report["success"])
        self.assertTrue(any("non-finite reset/action" in error for error in report["errors"]))

    def test_placeholder_channel_and_missing_checkpoint_cannot_succeed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            _fixture_run(run, bad_channel=True)
            (run / "latest.pt").unlink()
            report = benchmark.summarize_run(run)
        self.assertFalse(report["success"])
        self.assertTrue(any("channel depth is missing or not real" in error for error in report["errors"]))
        self.assertTrue(any("latest.pt" in error for error in report["errors"]))

    def test_json_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            _fixture_run(run)
            rows = benchmark._load_metrics(run / "metrics.jsonl")
            rows[4]["ppo_loss"] = float("nan")
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8"
            )
            report = benchmark.summarize_run(run)
        self.assertFalse(report["success"])
        self.assertTrue(any("invalid run artifact" in error for error in report["errors"]))

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "planned"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                returncode = benchmark.main(["--dry-run", "--out", str(output)])
            report = json.loads(stdout.getvalue())
        self.assertEqual(returncode, 0)
        self.assertFalse(output.exists())
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["warmup_iterations_in_child"], 2)
        self.assertEqual(report["measured_iterations_in_child"], 5)
        self.assertEqual(report["child_argv"][report["child_argv"].index("--iterations") + 1], "7")


if __name__ == "__main__":
    unittest.main()
