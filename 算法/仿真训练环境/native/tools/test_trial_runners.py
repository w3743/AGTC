"""CPU-only standard-library tests for frozen native train/evaluation runners."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = NATIVE / "train"
if str(TRAIN_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRAIN_SOURCE))
from contract_identity import IDENTITY_SCHEMA_VERSION, attach_contract_identity  # noqa: E402

PROTOCOL_SHA = "7ff8d6fcba79defbbc907b09527b244fb6e27134ed6c43de13ed1bdc197f2998"
FAKE_PYTHON = r"""
import json
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("MOCK_CALL_LOG")
if log:
    with open(log, "a", encoding="utf-8") as stream:
        stream.write(" ".join(args[:3]) + "\n")

def option(flag, default=None):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return default

def seeds_from(value):
    first, last = value.split("-", 1)
    return list(range(int(first), int(last) + 1))

def emit_contract(out):
    sys.path.insert(0, os.getcwd())
    from contract_identity import IDENTITY_SCHEMA_VERSION, attach_contract_identity
    dt = float(os.environ.get("MOCK_CONTRACT_DT", option("--sim-dt", "0.001")))
    dec = int(os.environ.get("MOCK_CONTRACT_DECIMATION", option("--decimation", "50")))
    backend = os.environ.get("MOCK_TASK_BACKEND", option("--task-backend", "fast_stop_v2"))
    official = dt == 0.001 and dec == 50
    physics = {
        "engine": "newton_xpbd", "device": "cuda:0", "newton": "mock",
        "warp": "mock", "mujoco": None, "nworld": 64,
        "dt_s": dt, "decimation": dec, "policy_dt_s": dt * dec,
        "dof": 64 * 24, "joints": 64 * 19, "bodies": 64 * 19,
        "mimic_relations_per_world": 14, "effort_limit_nm": 2.0,
        "drive_kp": 10.0, "drive_ki": 0.0, "mu": 0.8,
        "drive_semantics": "mock velocity targets",
    }
    channels = {
        "lidar": {"shape": [1, 160], "mode": "real", "backend": "warp_kernel"},
        "depth": {"shape": [1, 88, 160], "mode": "real", "backend": "analytic_warp"},
        "detections": {"shape": [16, 8], "mode": "real", "backend": "analytic_detector_warp"},
        "state": {"shape": [33], "mode": "real"},
        "feedback": {"shape": [37], "mode": "real", "backend": "vqf_reference"},
    }
    task = {"backend": backend, "reward_termination": {"time_penalty": 0.01}}
    contract = {
        "task_backend": backend, "task": task, "physics": physics,
        "physics_dt_s": dt, "decimation": dec, "policy_dt_s": dt * dec,
        "physics_experimental": not official, "channels": channels,
        "profile_enabled": False,
    }
    attach_contract_identity(contract)
    identity = {
        "task_backend": backend, "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "task_identity": contract["task_identity"],
        "task_identity_sha256": contract["task_identity_sha256"],
        "physics": "newton_xpbd", "device": "cuda",
        "policy": {"policy_type": "lstm", "hidden_size": 512},
        "env": {
            "task_backend": backend, "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "task_identity_sha256": contract["task_identity_sha256"],
            "physics": physics, "sim_dt_s": dt, "decimation": dec,
            "experimental_physics": not official,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "observation_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (out / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (out / "latest.pt").write_bytes(b"mock checkpoint payload")

if args and Path(args[0]).name == "trial_checks.py":
    raise SystemExit(subprocess.call([sys.executable, args[0], *args[1:]]))
if args and Path(args[0]).name == "protocol_sha.py":
    print(json.dumps({"match": True}))
    raise SystemExit(0)
if len(args) >= 2 and args[0] == "-u" and Path(args[1]).name == "train_wsl.py":
    emit_contract(Path(option("--out")))
    for index in range(12):
        print(f"TRAIN_LINE_{index:02d}", flush=True)
    raise SystemExit(int(os.environ.get("MOCK_TRAIN_EXIT", "0")))
if len(args) >= 2 and args[0] == "-u" and Path(args[1]).name == "eval_native.py":
    checkpoint = Path(option("--checkpoint"))
    contract = json.loads((checkpoint.parent / "observation_contract.json").read_text(encoding="utf-8"))
    digest = os.environ.get("MOCK_EVAL_SHA", contract["task_identity_sha256"])
    backend = os.environ.get("MOCK_EVAL_BACKEND", option("--task-backend"))
    dt = float(os.environ.get("MOCK_EVAL_DT", option("--sim-dt", "0.001")))
    dec = int(os.environ.get("MOCK_EVAL_DECIMATION", option("--decimation", "50")))
    validation = os.environ.get("MOCK_EVAL_VALIDATION", "strict_match")
    seeds = seeds_from(option("--seeds"))
    reason = os.environ.get("MOCK_EVAL_REASON", "time_limit")
    row = {"seed": seeds[0], "success": False, "clean_success": False, "reason": reason,
           "steps": 1200, "final_distance": 4.2, "final_xy_speed": 0.0,
           "final_angular_speed": 0.0}
    if reason == "runtime_error":
        row["error"] = "RuntimeError: mocked evaluation exception"
        row["final_distance"] = float("nan")
    diagnostic_count = int(os.environ.get("MOCK_DIAG_COUNT", "0"))
    result = {
        "task_backend": backend, "task_identity_sha256": digest,
        "checkpoint_contract_validation": validation,
        "physics": {"dt_s": dt, "decimation": dec,
                    "experimental": not (dt == 0.001 and dec == 50)},
        "config": {
            "seeds": seeds, "max_steps": int(option("--max-steps", "1200")),
            "device": option("--device"), "physics": option("--physics"),
            "task_backend": option("--task-backend"), "lidar": option("--lidar"),
            "feedback": option("--feedback"), "depth": option("--depth"),
            "detections": option("--detections"),
            "detections_warp": "--detections-warp" in args,
            "sim_dt_s": float(option("--sim-dt", "0.001")),
            "decimation": int(option("--decimation", "50")),
        },
        "policy": {"policy_type": "lstm", "hidden_size": 512,
                   "task_identity_sha256": digest,
                   "checkpoint_contract_validation": validation},
        "rows": [dict(row, seed=seed) for seed in seeds],
        "env_diagnostics": {"count": diagnostic_count,
                            "events": [{"kind": "nonfinite"}] if diagnostic_count else [],
                            "last_mask": [], "sensor_frames": 0},
        "capability_metrics_allowed": False,
    }
    if os.environ.get("MOCK_EVAL_OMIT_ROW") == "1":
        result["rows"] = result["rows"][:-1]
    out = Path(option("--out"))
    if int(os.environ.get("MOCK_EVAL_EXIT", "0")) == 0:
        (out / "evaluation.json").write_text(json.dumps(result), encoding="utf-8")
    for index in range(12):
        print(f"EVAL_LINE_{index:02d}", flush=True)
    raise SystemExit(int(os.environ.get("MOCK_EVAL_EXIT", "0")))
print("MOCK_UNHANDLED " + repr(args), flush=True)
raise SystemExit(0)
"""


def contract_fixture(backend: str = "fast_stop_v2") -> tuple[dict, dict]:
    physics = {
        "engine": "newton_xpbd", "device": "cuda:0", "newton": "mock",
        "warp": "mock", "mujoco": None, "nworld": 64,
        "dt_s": 0.001, "decimation": 50, "policy_dt_s": 0.05,
        "dof": 64 * 24, "joints": 64 * 19, "bodies": 64 * 19,
        "mimic_relations_per_world": 14, "effort_limit_nm": 2.0,
        "drive_kp": 10.0, "drive_ki": 0.0, "mu": 0.8,
        "drive_semantics": "mock velocity targets",
    }
    task = {"backend": backend, "reward_termination": {"time_penalty": 0.01}}
    channels = {
        "lidar": {"mode": "real", "backend": "warp_kernel"},
        "depth": {"mode": "real", "backend": "analytic_warp"},
        "detections": {"mode": "real", "backend": "analytic_detector_warp"},
        "state": {"mode": "real"},
        "feedback": {"mode": "real", "backend": "vqf_reference"},
    }
    contract = {
        "task_backend": backend, "task": task, "physics": physics,
        "physics_dt_s": 0.001, "decimation": 50, "policy_dt_s": 0.05,
        "physics_experimental": False, "channels": channels,
    }
    attach_contract_identity(contract)
    identity = {
        "task_backend": backend, "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "task_identity": contract["task_identity"],
        "task_identity_sha256": contract["task_identity_sha256"],
        "physics": "newton_xpbd", "device": "cuda",
        "policy": {"policy_type": "lstm", "hidden_size": 512},
        "env": {
            "task_backend": backend, "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "task_identity_sha256": contract["task_identity_sha256"],
            "physics": physics, "sim_dt_s": 0.001, "decimation": 50,
            "experimental_physics": False,
        },
    }
    return contract, identity


class TrialRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wsl = None
        cls.wsl_distro = "Ubuntu-24.04-Arm64"
        cls.pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if os.name == "nt":
            cls.wsl = shutil.which("wsl.exe") or shutil.which("wsl")
            if cls.wsl:
                listed = subprocess.run([cls.wsl, "-l", "-q"], capture_output=True)
                text = (listed.stdout + listed.stderr).decode(errors="replace").replace("\x00", "")
                if cls.wsl_distro not in text:
                    cls.wsl = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agtc runner fixture with spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkout = self.root / "project checkout with spaces" / "native" / "train"
        self.checkout.mkdir(parents=True)
        self.caller = self.root / "arbitrary caller directory with spaces"
        self.caller.mkdir()
        self.fake_dir = self.root / "fake python runtime with spaces"
        self.fake_dir.mkdir()
        self.fake_python = self.fake_dir / "mock_python.py"
        fake_text = textwrap.dedent(FAKE_PYTHON).replace("\r\n", "\n").replace("\r", "\n")
        with self.fake_python.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("#!/usr/bin/env python3\n" + fake_text.lstrip("\n"))
        self.fake_python.chmod(0o755)
        for name in (
            "run_native_v1_trial.sh", "run_native_v1_trial.ps1",
            "run_eval_pilot_4seeds.sh", "run_eval_16seeds.sh",
            "trial_checks.py", "contract_identity.py", "protocol_sha.py",
            "PROTOCOL_NATIVE_V1.1.md", "PROTOCOL_NATIVE_V1.1.sha256",
            "PROTOCOL_NATIVE_V1.md", "PROTOCOL_NATIVE_V1.sha256",
        ):
            shutil.copy2(TRAIN_SOURCE / name, self.checkout / name)
        self.call_log = self.root / "mock calls.log"

    def _base_env(self, **values):
        env = os.environ.copy()
        env.pop("SIM_DT", None)
        env.pop("DECIMATION", None)
        env["MOCK_CALL_LOG"] = str(self.call_log)
        for name, value in values.items():
            env[name] = str(value)
        return env

    @staticmethod
    def _wsl_path(path: Path) -> str:
        path = path.resolve()
        if os.name != "nt":
            return path.as_posix()
        drive = path.drive[0].lower()
        tail = path.as_posix().split(":", 1)[1].lstrip("/")
        return f"/mnt/{drive}/{tail}"

    def _run_shell(self, script_name, args=(), extra_env=None):
        script = self.checkout / script_name
        env_values = {
            "PYTHON": self._wsl_path(self.fake_python),
            "MOCK_CALL_LOG": self._wsl_path(self.call_log),
        }
        if extra_env:
            env_values.update({key: str(value) for key, value in extra_env.items()})
        if os.name == "nt":
            if not self.wsl:
                self.skipTest("Ubuntu-24.04-Arm64 WSL distribution is unavailable")
            chmod = subprocess.run(
                [self.wsl, "-d", self.wsl_distro, "--exec", "chmod", "+x",
                 self._wsl_path(self.fake_python)],
                capture_output=True,
            )
            self.assertEqual(chmod.returncode, 0, chmod.stderr.decode(errors="replace"))
            command = [
                self.wsl, "-d", self.wsl_distro, "--cd", self._wsl_path(self.caller),
                "--exec", "env",
                *[f"{key}={value}" for key, value in env_values.items()],
                "bash", self._wsl_path(script),
                *[self._wsl_path(arg) if isinstance(arg, Path) else str(arg) for arg in args],
            ]
            return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        env = os.environ.copy()
        env.update(env_values)
        return subprocess.run(
            ["bash", str(script), *map(str, args)], cwd=self.caller, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

    def _make_formal_trial(self):
        trial = self.checkout / "runs" / ("formal fixture " + os.urandom(4).hex())
        trial.mkdir(parents=True)
        for name in (
            "PROTOCOL_NATIVE_V1.1.md", "PROTOCOL_NATIVE_V1.1.sha256",
            "PROTOCOL_NATIVE_V1.md", "PROTOCOL_NATIVE_V1.sha256",
        ):
            shutil.copy2(self.checkout / name, trial / name)
        contract, identity = contract_fixture()
        (trial / "observation_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (trial / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
        (trial / "latest.pt").write_bytes(b"mock checkpoint payload")
        manifest = {
            "protocol_version": "V1.1", "protocol_sha256": PROTOCOL_SHA,
            "phase": "formal_native_v1_trial", "exit_code": 0, "formal_eligible": True,
            "physics": {"sim_dt": 0.001, "decimation": 50, "experimental": False},
            "task_backend": "fast_stop_v2",
            "task_identity_sha256": contract["task_identity_sha256"],
            "checkpoints": ["latest.pt"],
        }
        (trial / "trial_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return trial / "latest.pt"

    def test_shell_train_from_other_cwd_keeps_full_log_and_formal_manifest(self):
        tag = "formal spaces " + os.urandom(3).hex()
        result = self._run_shell("run_native_v1_trial.sh", [tag])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        trial = next((self.checkout / "runs").glob(f"native_v1_1_{tag}_*"))
        self.assertEqual((trial / "train.log").read_text(encoding="utf-8").count("TRAIN_LINE_"), 12)
        manifest = json.loads((trial / "trial_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["exit_code"], 0)
        self.assertTrue(manifest["formal_eligible"])
        self.assertEqual(manifest["task_backend"], "fast_stop_v2")
        self.assertEqual(manifest["physics"], {"sim_dt": 0.001, "decimation": 50, "experimental": False})

    def test_shell_train_preserves_real_rc_and_rejects_physics_or_recipe_overrides(self):
        tag = "failed " + os.urandom(3).hex()
        failed = self._run_shell("run_native_v1_trial.sh", [tag], {"MOCK_TRAIN_EXIT": 7})
        self.assertEqual(failed.returncode, 7, failed.stdout + failed.stderr)
        trial = next((self.checkout / "runs").glob(f"native_v1_1_{tag}_*"))
        manifest = json.loads((trial / "trial_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["exit_code"], 7)
        self.assertFalse(manifest["formal_eligible"])
        self.assertEqual((trial / "train.log").read_text(encoding="utf-8").count("TRAIN_LINE_"), 12)

        bad_physics = self._run_shell("run_native_v1_trial.sh", ["bad physics"],
                                      {"SIM_DT": "0.005", "DECIMATION": "10"})
        self.assertEqual(bad_physics.returncode, 2, bad_physics.stdout + bad_physics.stderr)
        bad_override = self._run_shell("run_native_v1_trial.sh",
                                       ["bad override", "--iterations", "1"])
        self.assertEqual(bad_override.returncode, 2, bad_override.stdout + bad_override.stderr)

    def test_shell_train_gates_actual_written_contract(self):
        tag = "bad actual " + os.urandom(3).hex()
        result = self._run_shell("run_native_v1_trial.sh", [tag],
                                 {"MOCK_CONTRACT_DT": "0.005", "MOCK_CONTRACT_DECIMATION": "10"})
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        trial = next((self.checkout / "runs").glob(f"native_v1_1_{tag}_*"))
        manifest = json.loads((trial / "trial_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["formal_eligible"])
        self.assertTrue(manifest["physics"]["experimental"])
        self.assertTrue(any("dt_s is not 0.001" in error for error in manifest["validation_errors"]))

    def test_pilot_eval_checks_actual_contract_sha_and_allows_time_limit_rows(self):
        checkpoint = self._make_formal_trial()
        result = self._run_shell("run_eval_pilot_4seeds.sh",
                                 [checkpoint, "pilot output with spaces"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = self.caller / "pilot output with spaces"
        self.assertEqual((out / "eval.log").read_text(encoding="utf-8").count("EVAL_LINE_"), 12)
        verdict = json.loads((out / "sha_check.json").read_text(encoding="utf-8"))
        self.assertTrue(verdict["match"])
        self.assertEqual(verdict["seeds"], [87010, 87011, 87012, 87013])

    def test_eval_rejects_sha_runtime_and_nonfinite_diagnostics(self):
        checkpoint = self._make_formal_trial()
        mismatch = self._run_shell("run_eval_pilot_4seeds.sh",
                                   [checkpoint, "mismatch"],
                                   {"MOCK_EVAL_SHA": "different-sha"})
        self.assertEqual(mismatch.returncode, 4, mismatch.stdout + mismatch.stderr)
        self.assertFalse(json.loads((self.caller / "mismatch" / "sha_check.json").read_text())["match"])

        runtime = self._run_shell("run_eval_pilot_4seeds.sh",
                                  [checkpoint, "runtime row"],
                                  {"MOCK_EVAL_REASON": "runtime_error"})
        self.assertEqual(runtime.returncode, 4, runtime.stdout + runtime.stderr)
        reason = json.loads((self.caller / "runtime row" / "sha_check.json").read_text())
        self.assertTrue(any("runtime_error" in error for error in reason["validation_errors"]))
        self.assertTrue(any("non-finite" in error for error in reason["validation_errors"]))

        diagnostic = self._run_shell("run_eval_pilot_4seeds.sh",
                                     [checkpoint, "diagnostic"],
                                     {"MOCK_DIAG_COUNT": 1})
        self.assertEqual(diagnostic.returncode, 4, diagnostic.stdout + diagnostic.stderr)

    def test_eval_rejects_incomplete_seed_rows(self):
        checkpoint = self._make_formal_trial()
        result = self._run_shell("run_eval_pilot_4seeds.sh",
                                 [checkpoint, "missing seed row"],
                                 {"MOCK_EVAL_OMIT_ROW": "1"})
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)

    def test_eval_preserves_failure_rc_full_log_and_frozen_seeds(self):
        checkpoint = self._make_formal_trial()
        failed = self._run_shell("run_eval_pilot_4seeds.sh",
                                 [checkpoint, "failed eval"],
                                 {"MOCK_EVAL_EXIT": 7})
        self.assertEqual(failed.returncode, 7, failed.stdout + failed.stderr)
        self.assertEqual((self.caller / "failed eval" / "eval.log").read_text().count("EVAL_LINE_"), 12)

        wrong_seeds = self._run_shell("run_eval_16seeds.sh",
                                      [checkpoint, "wrong seeds", "87010-87015"])
        self.assertEqual(wrong_seeds.returncode, 2, wrong_seeds.stdout + wrong_seeds.stderr)

    def test_formal_16_seed_runner_uses_official_checkpoint_and_seeds(self):
        checkpoint = self._make_formal_trial()
        result = self._run_shell("run_eval_16seeds.sh", [checkpoint, "formal eval spaces"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verdict = json.loads((self.caller / "formal eval spaces" / "sha_check.json").read_text())
        self.assertTrue(verdict["match"])
        self.assertEqual(verdict["seeds"], list(range(87010, 87026)))

    def test_powershell_train_runner_keeps_log_and_exit_code(self):
        if os.name != "nt" or not self.pwsh:
            self.skipTest("PowerShell runner test requires Windows or pwsh")
        fake_cmd = self.fake_dir / "mock_python.cmd"
        fake_cmd.write_text(
            "@echo off\r\n"
            f'"{sys.executable}" "{self.fake_python}" %*\r\n'
            "exit /b %errorlevel%\r\n", encoding="utf-8",
        )
        tag = "powershell spaces " + os.urandom(3).hex()
        result = subprocess.run(
            [self.pwsh, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(self.checkout / "run_native_v1_trial.ps1"),
             "-Tag", tag, "-Python", str(fake_cmd)],
            cwd=self.caller, env=self._base_env(MOCK_TRAIN_EXIT=7),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        trial = next((self.checkout / "runs").glob(f"native_v1_1_{tag}_*"))
        self.assertEqual((trial / "train.log").read_text(encoding="utf-8-sig").count("TRAIN_LINE_"), 12)
        manifest = json.loads((trial / "trial_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["exit_code"], 7)
        self.assertFalse(manifest["formal_eligible"])

    def test_powershell_runner_rejects_experimental_physics_and_override(self):
        if os.name != "nt" or not self.pwsh:
            self.skipTest("PowerShell runner test requires Windows or pwsh")
        fake_cmd = self.fake_dir / "mock_python.cmd"
        fake_cmd.write_text(
            "@echo off\r\n"
            f'"{sys.executable}" "{self.fake_python}" %*\r\n'
            "exit /b %errorlevel%\r\n", encoding="utf-8",
        )
        script = self.checkout / "run_native_v1_trial.ps1"
        bad_physics = subprocess.run(
            [self.pwsh, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script), "-Tag", "bad physics", "-Python", str(fake_cmd),
             "-SimDt", "0.005", "-Decimation", "10"],
            cwd=self.caller, env=self._base_env(), capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(bad_physics.returncode, 2, bad_physics.stdout + bad_physics.stderr)
        q = lambda value: "'" + str(value).replace("'", "''") + "'"
        wrapper = self.root / "invoke runner with override.ps1"
        wrapper.write_text(
            f"& {q(script)} -Tag 'bad override' -Python {q(fake_cmd)} "
            "-ExtraArgs @('--iterations','1')\nexit $LASTEXITCODE\n",
            encoding="utf-8",
        )
        rejected = subprocess.run(
            [self.pwsh, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(wrapper)],
            cwd=self.caller, env=self._base_env(), capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(rejected.returncode, 2, rejected.stdout + rejected.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)