"""Independently verify the saved zero-step minimal-input preview artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results/isaac_field_viewer/minimal_inputs_20261020"
RECORD = OUT_DIR / "untrained_minimal_preview.json"
CHECKPOINT = OUT_DIR / "untrained_minimal_preview.zip"
OLD_V5 = ROOT / "results/isaac_field_improvement/fresh_imu_drive_20261020/stage_04_turn90.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    with zipfile.ZipFile(CHECKPOINT) as archive:
        metadata = json.loads(archive.read("data"))
    old_sha = sha256(OLD_V5)
    guard = record["old_v5_checkpoint_encoder_guard"]

    assert record["schema"] == "minimal-input-zero-step-preview-v1"
    assert record["num_timesteps"] == metadata["num_timesteps"] == 0
    assert record["field_preview_only"] is metadata["field_preview_only"] is True
    assert record["environment_transitions"] == 0
    assert record["training_calls"] == {"learn": 0, "env_step": 0}
    assert metadata["field_encoder_config"]["schema"] == "depth64-lidar-single1d-minimal-v3"
    assert record["field_encoder_config"] == metadata["field_encoder_config"]
    assert record["observation_config"]["schema"] == "sensor-minimal-v6"
    assert record["observation_config"]["state_dim"] == 33
    assert record["observation_config"]["feedback_dim"] == 37
    assert record["observation_config"]["depth_channels"] == 1
    assert guard["guard"] == "_validate_checkpoint_encoder_metadata"
    assert guard["validation"]["result"] == "rejected_before_SB3_load"
    assert guard["sb3_model_load_attempted"] is False
    assert guard["sha256_before"] == guard["sha256_after"] == old_sha

    result = {
        "verified": True,
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "record_schema": record["schema"],
        "timesteps": metadata["num_timesteps"],
        "field_preview_only": metadata["field_preview_only"],
        "input_schema": record["observation_config"]["schema"],
        "state_dim": record["observation_config"]["state_dim"],
        "feedback_dim": record["observation_config"]["feedback_dim"],
        "old_v5_sha256_unchanged": old_sha,
        "old_v5_rejected_before_sb3_load": True,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
