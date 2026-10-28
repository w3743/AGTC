from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_training as bench
from test_task_contract import newton_mujoco_warp_physics


def _physics(nworld: int = 64, engine: str = "newton_xpbd", *,
             graph_requested: bool = False, graph_enabled: bool = False,
             graph_capture_s: float = 0.0) -> dict:
    if engine == "newton_mujoco_warp":
        result = newton_mujoco_warp_physics(nworld)
        result["runtime_diagnostics"].update({
            "cuda_graph_requested": graph_requested,
            "cuda_graph_enabled": graph_enabled,
            "cuda_graph_capture_s": graph_capture_s,
        })
        return result
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
             tamper_task: bool = False, child_returncode: int = 0,
             physics_engine: str = "newton_xpbd", physics_experimental: bool = False,
             physics_cuda_graph_requested: bool = False,
             graph_enabled: bool | None = None) -> None:
    run.mkdir(parents=True)
    if graph_enabled is None:
        graph_enabled = physics_cuda_graph_requested
    graph_capture_s = 0.25 if graph_enabled else 0.0
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
        "physics_experimental": physics_experimental, "physics_note": "default 1 ms / decimation 50",
        "task_backend": "fast_stop_v2", "task": task,
        "physics": _physics(engine=physics_engine,
                            graph_requested=physics_cuda_graph_requested,
                            graph_enabled=graph_enabled, graph_capture_s=graph_capture_s),
        "channels": channels,
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
        "device": "cuda", "physics": physics_engine, "lidar": "warp", "feedback": "vqf",
        "physics_cuda_graph": physics_cuda_graph_requested,
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
               "physics": physics_engine, "observation_contract_sha256": sha,
               "physics_cuda_graph": physics_cuda_graph_requested,
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
        "status": "success", "success": True, "physics": physics_engine,
        "physics_cuda_graph_requested": physics_cuda_graph_requested,
        "target_profile": {**bench.EXPECTED, "physics": physics_engine,
                           "physics_cuda_graph": physics_cuda_graph_requested},
        "cuda_graph": {"requested": physics_cuda_graph_requested, "enabled": graph_enabled,
                       "mode": "cuda_graph" if graph_enabled else "eager",
                       "capture_s": graph_capture_s},
        "warmup_iterations": 2, "child_iterations": 7,
        "source_identity_still_matches_at_finish": True,
        "source_identity": {"raw_source_bytes": [{"path": "train_wsl.py", "bytes": 1, "sha256": "b" * 64}],
                            "git": {"commit": "deadbeef", "branch": "perf/max", "object_format": "sha1",
                                    "relevant_sources_dirty": False,
                                    "dirty_scope": "raw_source_bytes vs HEAD only; whole workspace dirty not measured",
                                    "modified_source_paths": [], "untracked_source_paths": [],
                                    "relevant_source_changes": [], "relevant_source_changes_sha256": "c" * 64}},
        "execution": {"command_argv": bench.train_argv(
                            run, 7, 0, 8, physics_engine, physics_cuda_graph_requested),
                      "returncode": child_returncode, "launcher_wall_s": 25.0},
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
        self.assertEqual(command[command.index("--physics") + 1], "newton_xpbd")
        self.assertNotIn("--profile", command)
        self.assertNotIn("--physics-cuda-graph", command)

    def test_command_can_select_newton_mujoco_warp(self):
        with tempfile.TemporaryDirectory() as temp:
            command = bench.train_argv(Path(temp) / "run", 7, 0, 8, "newton_mujoco_warp")
        self.assertEqual(command[command.index("--physics") + 1], "newton_mujoco_warp")
        self.assertNotIn("--physics-cuda-graph", command)

    def test_command_can_explicitly_request_cuda_graph_for_mujoco_warp(self):
        with tempfile.TemporaryDirectory() as temp:
            command = bench.train_argv(Path(temp) / "run", 7, 0, 8,
                                       "newton_mujoco_warp", True)
        self.assertIn("--physics-cuda-graph", command)

    def test_cuda_graph_request_rejects_other_physics_engines(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(bench.AuditError, "requires --physics newton_mujoco_warp"):
                bench.train_argv(Path(temp) / "run", 7, 0, 8, "newton_xpbd", True)
            out, stdout = Path(temp) / "planned", io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bench.main(["--dry-run", "--physics", "newton_xpbd",
                                   "--physics-cuda-graph", "--out", str(out)])
            result = json.loads(stdout.getvalue())
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())
        self.assertIn("requires --physics newton_mujoco_warp", result["error"])

    def test_mujoco_warp_version_is_recorded(self):
        with patch.object(bench.importlib.metadata, "version", side_effect=lambda name: name):
            recorded = bench.versions()
        self.assertEqual(recorded["mujoco-warp"], "mujoco-warp")

    def test_source_tree_comparison_uses_exact_raw_blob_bytes(self):
        sources = {
            "src/unchanged.py": b"print('same')\n",
            "src/modified.py": b"print('new')\n",
            "src/untracked.py": b"print('added')\n",
            "src/raw_lf.txt": b"first\nsecond\n",
        }
        head = {
            "src/unchanged.py": sources["src/unchanged.py"],
            "src/modified.py": b"print('old')\n",
            "src/raw_lf.txt": b"first\r\nsecond\r\n",
        }
        tree_output = b"".join(
            b"100644 blob " + bench._git_blob_sha(raw, "sha1").encode("ascii") + b"\t" +
            path.encode("utf-8") + b"\0" for path, raw in head.items()
        )
        tree = bench._parse_git_tree(tree_output)
        changes = bench._relevant_source_changes(sources, tree, "sha1")
        self.assertEqual({row["path"]: row["state"] for row in changes}, {
            "src/modified.py": "modified", "src/raw_lf.txt": "modified", "src/untracked.py": "untracked",
        })
        self.assertNotIn("src/unchanged.py", {row["path"] for row in changes})
        self.assertNotEqual(bench._git_blob_sha(sources["src/raw_lf.txt"], "sha1"),
                            bench._git_blob_sha(head["src/raw_lf.txt"], "sha1"))
        self.assertEqual(len(bench._git_blob_sha(b"raw bytes\n", "sha256")), 64)

    def test_source_identity_includes_each_native_protocol_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            train = Path(temp)
            for name in ("PROTOCOL_NATIVE_V1.1.md", "PROTOCOL_NATIVE_V2.md",
                         "PROTOCOL_NATIVE_V2.sha256", "OTHER_PROTOCOL.md"):
                (train / name).write_text("fixture", encoding="utf-8")
            included = bench._protocol_identity_paths(train)
        self.assertEqual({p.name for p in included}, {
            "PROTOCOL_NATIVE_V1.1.md", "PROTOCOL_NATIVE_V2.md", "PROTOCOL_NATIVE_V2.sha256",
        })

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

    def test_read_only_summary_accepts_newton_mujoco_warp_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp")
            result = bench.summarize_run(run)
        self.assertTrue(result["success"], result["errors"])
        self.assertEqual(result["physics"], "newton_mujoco_warp")
        self.assertEqual(result["cuda_graph"], {
            "requested": False, "enabled": False, "mode": "eager", "capture_s": 0.0,
        })

    def test_requested_cuda_graph_passes_only_when_contract_and_identity_enable_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root, run = Path(temp), Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp",
                     physics_cuda_graph_requested=True)
            result = bench.summarize_run(run)
            benchmark = bench.summarize_benchmark(root)
        self.assertTrue(result["success"], result["errors"])
        self.assertEqual(result["cuda_graph"], {
            "requested": True, "enabled": True, "mode": "cuda_graph", "capture_s": 0.25,
        })
        self.assertTrue(benchmark["success"], benchmark["errors"])
        self.assertEqual(benchmark["cuda_graph"]["mode"], "cuda_graph")
        self.assertEqual(benchmark["command_and_seed"]["physics_cuda_graph_requested"], True)

    def test_requested_cuda_graph_is_rejected_when_actual_graph_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp",
                     physics_cuda_graph_requested=True, graph_enabled=False)
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("enabled state differs from the request" in error
                            for error in result["errors"]))

    def test_unrequested_cuda_graph_is_rejected_when_actual_graph_is_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp",
                     physics_cuda_graph_requested=False, graph_enabled=True)
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("enabled state differs from the request" in error
                            for error in result["errors"]))

    def test_contract_and_identity_cuda_graph_diagnostics_must_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp",
                     physics_cuda_graph_requested=True)
            identity = json.loads((run / "identity.json").read_text(encoding="utf-8"))
            diagnostics = identity["env"]["physics"]["runtime_diagnostics"]
            diagnostics["cuda_graph_enabled"] = False
            diagnostics["cuda_graph_capture_s"] = 0.0
            (run / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("CUDA Graph enabled state" in error for error in result["errors"]))

    def test_runtime_cuda_graph_request_must_match_launcher_request(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp",
                     physics_cuda_graph_requested=True)
            contract = json.loads((run / "observation_contract.json").read_text(encoding="utf-8"))
            contract["physics"]["runtime_diagnostics"]["cuda_graph_requested"] = False
            (run / "observation_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("CUDA Graph request differs from the launcher request" in error
                            for error in result["errors"]))

    def test_identity_cuda_graph_request_must_match_launcher_request(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp",
                     physics_cuda_graph_requested=True)
            identity = json.loads((run / "identity.json").read_text(encoding="utf-8"))
            identity["physics_cuda_graph"] = False
            (run / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("identity CUDA Graph request differs" in error
                            for error in result["errors"]))

    def test_mujoco_launch_record_must_include_graph_request_and_argv(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp")
            launch_path = run.parent / "benchmark_result.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch.pop("physics_cuda_graph_requested")
            launch["target_profile"].pop("physics_cuda_graph")
            launch["execution"].pop("command_argv")
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("graph request" in error or "argv is missing" in error
                            for error in result["errors"]))

    def test_cuda_graph_runtime_flags_do_not_change_task_contract_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp")
            contract = json.loads((run / "observation_contract.json").read_text(encoding="utf-8"))
            before = bench.contract_identity_sha256(contract)
            diagnostics = contract["physics"]["runtime_diagnostics"]
            diagnostics["cuda_graph_requested"] = True
            diagnostics["cuda_graph_enabled"] = True
            diagnostics["cuda_graph_capture_s"] = 0.25
            after = bench.contract_identity_sha256(contract)
        self.assertEqual(before, after)

    def test_experimental_physics_run_cannot_pass_benchmark_gate(self):
        for engine in ("newton_xpbd", "newton_mujoco_warp"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as temp:
                run = Path(temp) / "run"
                _fixture(run, physics_engine=engine, physics_experimental=True)
                result = bench.summarize_run(run)
            self.assertFalse(result["success"])
            self.assertTrue(any("physics identity mismatch" in error for error in result["errors"]))

    def test_identity_engine_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            _fixture(run, physics_engine="newton_mujoco_warp")
            identity = json.loads((run / "identity.json").read_text(encoding="utf-8"))
            identity["physics"] = "newton_xpbd"
            (run / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            result = bench.summarize_run(run)
        self.assertFalse(result["success"])
        self.assertTrue(any("identity physics engine differs" in error for error in result["errors"]))

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
        self.assertFalse(result["physics_cuda_graph_requested"])
        self.assertFalse(result["target_profile"]["physics_cuda_graph"])

    def test_dry_run_selects_newton_mujoco_warp(self):
        with tempfile.TemporaryDirectory() as temp:
            out, stdout = Path(temp) / "planned", io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bench.main(["--dry-run", "--physics", "newton_mujoco_warp",
                                   "--physics-cuda-graph", "--out", str(out)])
            result = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())
        self.assertEqual(result["child_argv"][result["child_argv"].index("--physics") + 1],
                         "newton_mujoco_warp")
        self.assertIn("--physics-cuda-graph", result["child_argv"])
        self.assertTrue(result["physics_cuda_graph_requested"])
        self.assertTrue(result["target_profile"]["physics_cuda_graph"])


if __name__ == "__main__":
    unittest.main()
