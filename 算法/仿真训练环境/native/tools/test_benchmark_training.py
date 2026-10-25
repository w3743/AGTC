from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import benchmark_training as bench


def _physics(nworld: int = 64) -> dict:
    return {
        "engine": "newton_xpbd", "device": "cuda:0", "newton": "1.7.0", "warp": "1.18.0",
        "mujoco": None, "nworld": nworld, "dt_s": 0.001, "decimation": 50,
        "policy_dt_s": 0.05, "dof": 24 * nworld, "joints": 19 * nworld,
        "bodies": 19 * nworld, "mimic_relations_per_world": 14,
        "effort_limit_nm": 2.0, "drive_kp": 0.04, "drive_ki": 0.1, "mu": 0.8,
        "max_track_speed_mps": 15.0 / 3.6, "drive_semantics": "velocity targets",
        "gravity_mps2": [0.0, 0.0, -9.81],
        "solver": {"kind": "newton_xpbd", "iterations": 10, "shape_ke": 2500.0, "shape_kd": 100.0},
        "contact": {"shape_mu": 0.8},
    }


def _fixture(run: Path, *, reset_count: int = 0, placeholder: bool = False,
             tamper_task: bool = False, child_returncode: int = 0) -> None:
    run.mkdir(parents=True)
    channels = {
        "lidar": {"shape": [1, 400], "mode": "real", "backend": "warp_kernel", "source": "raycast"},
        "depth": {"shape": [1, 160, 88], "mode": "real", "backend": "analytic_warp",
                  "source": "raycast", "verified_vs": "depth reference"},
        "detections": {"shape": [16, 8], "mode": "real", "backend": "analytic_detector_warp",
                       "source": "instance mask", "verified_vs": "detector reference"},
        "state": {"shape": [33], "mode": "real", "source": "sim state"},
        "feedback": {"shape": [37], "mode": "real", "backend": "vqf_reference",
                     "source": "VQF", "verified_vs": "feedback reference"},
    }
    if placeholder:
        channels["depth"]["mode"] = "placeholder"
    task = {"backend": "fast_stop_v2", "episode_length_s": 60.0,
            "reset_sampling": {"field_size_m": 100.0, "spawn_clearance_m": 0.01},
            "reward_termination": {"goal_tolerance": 0.4, "goal_hold": 0.3,
                                   "timeout_behavior": "terminal_failure"}}
    contract = {
        "run_id": run.name, "phase": "wsl_training_prep", "profile_enabled": False,
        "policy_dt_s": 0.05, "physics_dt_s": 0.001, "decimation": 50,
        "physics_experimental": False, "physics_note": "default 1 ms / decimation 50",
        "task_backend": "fast_stop_v2", "task": task, "physics": _physics(), "channels": channels,
        "action_components": ["left_track_speed", "right_track_speed"],
        "capability_metrics_allowed": False,
    }
    contract = bench.attach_contract_identity(contract)
    # Mutate after signing so the auditor has to detect tampered semantic fields.
    if tamper_task:
        contract["task"]["reward_termination"]["goal_hold"] = 0.5
    sha = contract["observation_contract_sha256"]
    identity = {
        "banner": bench.BANNER, "seed": 0, "num_envs": 64, "num_steps": 32,
        "device": "cuda", "physics": "newton_xpbd", "lidar": "warp", "feedback": "vqf",
        "task_backend": "fast_stop_v2", "policy": {"policy_type": "lstm", "hidden_size": 512},
        "identity_schema_version": bench.IDENTITY_SCHEMA_VERSION,
        "task_identity": contract["task_identity"], "task_identity_sha256": sha,
        "env": {"sim_dt_s": 0.001, "decimation": 50, "physics": contract["physics"],
                "task_backend": "fast_stop_v2", "identity_schema_version": bench.IDENTITY_SCHEMA_VERSION,
                "task_identity_sha256": sha},
    }
    summary = {"banner": bench.BANNER, "phase": "wsl_training_prep", "capability_metrics_allowed": False,
               "iterations": 7, "start_iteration": 0, "end_iteration": 7,
               "total_wall_s": 21.0, "num_envs": 64, "num_steps": 32, "device": "cuda",
               "physics": "newton_xpbd", "observation_contract_sha256": sha,
               "final_checkpoint": str(run / "latest.pt")}
    rows = []
    for iteration in range(1, 8):
        rows.append({"iter": iteration, "wall_s": 3.0, "collection_s": 1.0, "learning_s": 2.0,
                     "samples": 64 * 32, "samples_per_s": 64 * 32,
                     "end_to_end_samples_per_s": (64 * 32) / 3.0, "ppo_loss": 0.25,
                     "mean_reward": 0.5, "rss_mb": 100.0, "nonfinite_env_resets": reset_count,
                     "nonfinite_actions": 0, "observation_contract_sha256": sha})
    (run / "observation_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run / "latest.pt").write_bytes(b"fixture-checkpoint")
    launch = {
        "status": "success", "success": True, "warmup_iterations": 2, "child_iterations": 7,
        "source_identity_still_matches_at_finish": True,
        "source_identity": {"raw_source_bytes": [{"path": "train_wsl.py", "bytes": 1, "sha256": "b" * 64}],
                            "git": {"commit": "deadbeef", "dirty": True, "status_sha256": "c" * 64}},
        "execution": {"returncode": child_returncode, "launcher_wall_s": 25.0},
    }
    (run.parent / "benchmark_result.json").write_text(json.dumps(launch), encoding="utf-8")


class BenchmarkTrainingTests(unittest.TestCase):
    def test_command_is_explicit_and_nonfinite_limit_is_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            command = bench.train_argv(Path(temp) / "run", 7, 0, 8)
        self.assertEqual(command[command.index("--num-envs") + 1], "64")
        self.assertEqual(command[command.index("--num-steps") + 1], "32")
        self.assertEqual(command[command.index("--iterations") + 1], "7")
        self.assertEqual(command[command.index("--task-backend") + 1], "fast_stop_v2")
        self.assertEqual(command[command.index("--nonfinite-limit") + 1], "0")
        self.assertNotIn("--profile", command)

    def test_process_scan_matches_known_names_not_argument_text(self):
        note_command = ["python3", "/repo/native/tools/benchmark_training.py",
                        "--resource-contention-note", "Kit and isaac-sim are not active"]
        self.assertEqual(bench.process_labels(10, "python3", note_command), [])
        self.assertEqual(bench.process_labels(11, "python3",
                                             ["python3", "/repo/native/train/train_wsl.py"]), ["training"])
        self.assertEqual(bench.process_labels(14, "python3",
                                             ["python3", "-B", "-u", "/repo/native/train/train_wsl.py"]), ["training"])
        self.assertEqual(bench.process_labels(12, "llama-server",
                                             ["/usr/bin/llama-server.exe"]), ["llama"])
        self.assertEqual(bench.process_labels(13, "python3",
                                             ["python3", "/repo/native/tools/benchmark_training.py"],
                                             own_pid=13), [])

    def test_read_only_summary_uses_measured_rows_and_shared_contract_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root, run = Path(temp), Path(temp) / "run"
            _fixture(run)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = bench.summarize_run(run)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        self.assertTrue(result["success"], result["errors"])
        self.assertEqual(result["measured_iterations"], 5)
        self.assertEqual(result["throughput"]["measured_samples"], 5 * 64 * 32)
        self.assertAlmostEqual(result["throughput"]["collection_samples_per_s"], 64 * 32)
        self.assertAlmostEqual(result["throughput"]["ppo_update_samples_per_s"], 64 * 32 / 2)
        self.assertAlmostEqual(result["throughput"]["iteration_window_samples_per_s"], 64 * 32 / 3)
        self.assertFalse(result["capability_claim"])
        self.assertEqual(before, after)

    def test_task_contract_tampering_is_rejected_by_canonical_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, tamper_task=True)
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("shared canonical" in e or "canonical identity SHA" in e for e in result["errors"]))

    def test_nonfinite_reset_disqualifies_run(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, reset_count=1)
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("non-finite reset/action" in e for e in result["errors"]))

    def test_nonzero_child_exit_disqualifies_run(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, child_returncode=1)
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("child exit code" in e for e in result["errors"]))

    def test_placeholder_channel_cannot_succeed(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, placeholder=True)
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("channel depth" in e for e in result["errors"]))

    def test_missing_checkpoint_cannot_succeed(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run)
            (run / "latest.pt").unlink()
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("latest.pt" in e for e in result["errors"]))

    def test_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run)
            rows = bench._metrics(run / "metrics.jsonl")
            rows[4]["ppo_loss"] = float("nan")
            (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("invalid artifact" in e for e in result["errors"]))

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out, stdout = Path(temp) / "planned", io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bench.main(["--dry-run", "--out", str(out)])
            result = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["child_argv"][result["child_argv"].index("--iterations") + 1], "7")


if __name__ == "__main__":
    unittest.main()
