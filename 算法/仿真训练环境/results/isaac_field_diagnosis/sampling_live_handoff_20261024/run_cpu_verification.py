"""Run the established CPU regression set plus the handoff-specific checks."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIELD = ROOT.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = Path("C:/pioneer/isaac-training/logs/cpu_union_final_cli_20261024/command.txt")
    modules = previous.read_text(encoding="utf-8-sig").split("Modules: ", 1)[1].split()
    modules += ["tests/test_sampling_copy_cache.py", "tests/test_sampling_frozen_equivalence.py"]
    modules = list(dict.fromkeys(modules))
    if any(not (FIELD / path).is_file() for path in modules):
        raise FileNotFoundError("all regression modules must exist before verification")
    output = ROOT / "cpu_regression"
    output.mkdir(exist_ok=False)
    command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "--tb=short",
               f"--junitxml={output / 'junit.xml'}", *modules]
    environment = os.environ.copy()
    environment.update(ISAAC_FIELD_DISABLE_NATIVE_RUNTIME="1", CUDA_VISIBLE_DEVICES="-1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(FIELD),
                       ISAACLAB_ROOT=str(FIELD / "external/IsaacLab"),
                       ISAAC_FIELD_RUNTIME_DIR="C:/pioneer/isaac-training",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONIOENCODING="utf-8")
    tracked = [FIELD / "isaac_field/env.py", FIELD / "isaac_field/rsl_rl_cfg.py"]
    before = {str(path): digest(path) for path in tracked}
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "command.json").write_text(json.dumps({"argv": command, "cwd": str(FIELD),
        "native_runtime_disabled": True, "cuda_visible_devices": "-1"}, indent=2), encoding="utf-8")
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        result = subprocess.run(command, cwd=FIELD, env=environment, stdout=stdout, stderr=stderr)
    after = {str(path): digest(path) for path in tracked}
    report = {"started_at_utc": started, "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "actual_exit_code": result.returncode, "modules": modules,
              "source_sha256_before": before, "source_sha256_after": after,
              "source_unchanged_during_tests": before == after,
              "status": "passed" if result.returncode == 0 and before == after else "failed",
              "training_or_fps_benchmark_started": False}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "exit": result.returncode, "modules": len(modules)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
