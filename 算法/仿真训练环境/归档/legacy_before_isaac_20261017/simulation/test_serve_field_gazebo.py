"""Checkpoint selection must exclude unverified or proxy-backend runs."""
import json
import os
from pathlib import Path

from simulation.serve_field_gazebo import latest_checkpoint


def _run(root: Path, folder: str, name: str, *, backend: str,
         reload_passed: bool, mtime: int) -> Path:
    run = root / "results" / folder / name
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint.zip"
    checkpoint.write_bytes(b"test checkpoint")
    (run / "train_summary.json").write_text(json.dumps({
        "backend": backend,
        "checkpoint_reload_passed": reload_passed,
    }), encoding="utf-8")
    os.utime(checkpoint, (mtime, mtime))
    return checkpoint


def test_latest_checkpoint_uses_verified_gazebo_run_only(tmp_path):
    verified = _run(tmp_path, "gazebo_field_training", "verified",
                    backend="gazebo", reload_passed=True, mtime=100)
    _run(tmp_path, "gazebo_training", "newer_proxy",
         backend="field_sim", reload_passed=True, mtime=200)
    _run(tmp_path, "gazebo_training", "newer_unverified",
         backend="gazebo", reload_passed=False, mtime=300)

    assert latest_checkpoint(tmp_path) == verified
