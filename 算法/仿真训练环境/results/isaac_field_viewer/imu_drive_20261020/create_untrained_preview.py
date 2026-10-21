"""Create a zero-step direct-track policy checkpoint for UI inspection only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.action_contract import action_config
from isaac_field.env import make_env
from isaac_field.policy import make_model, validate_policy_spaces
from isaac_field.task import Config
from isaac_field.training import _sb3_wrapper


OUTPUT = ROOT / "results/isaac_field_viewer/imu_drive_20261020"
CHECKPOINT = OUTPUT / "untrained_imu_drive_preview.zip"
RECORD = OUTPUT / "untrained_imu_drive_preview.json"
LEGACY_CHECKPOINT = ROOT / (
    "results/isaac_field_improvement/fresh_depth_radar_20261019_v2/stage_04_turn90.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_legacy_action_guard() -> dict:
    from isaac_field.policy import _validate_checkpoint_action_metadata

    if not LEGACY_CHECKPOINT.is_file():
        raise FileNotFoundError(LEGACY_CHECKPOINT)
    before = sha256(LEGACY_CHECKPOINT)
    try:
        _validate_checkpoint_action_metadata(LEGACY_CHECKPOINT)
    except ValueError as exc:
        result = {"result": "rejected_before_SB3_load", "error": str(exc)}
    else:
        raise AssertionError("the legacy forward/differential checkpoint unexpectedly passed the action guard")
    after = sha256(LEGACY_CHECKPOINT)
    if before != after:
        raise AssertionError("legacy checkpoint changed during read-only action-contract validation")
    return {
        "path": LEGACY_CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256_before": before,
        "sha256_after": after,
        "validation": result,
        "sb3_model_load_attempted": False,
    }


def main() -> None:
    if CHECKPOINT.exists() or RECORD.exists():
        raise FileExistsError(f"refusing to overwrite preview artifacts: {CHECKPOINT} or {RECORD}")
    legacy_guard = verify_legacy_action_guard()
    cfg = Config(
        plants=0,
        terrain_enabled=False,
        max_time=20.0,
        damage_penalty=20.0,
        goal_progress_budget=5.0,
        damage_metric="stem-contact-v2",
    )
    env = make_env(
        cfg=cfg,
        mode="goto",
        num_envs=1,
        seed=69_001,
        auto_reset=False,
        goal_distance=(0.6, 1.2),
        goal_angle=math.radians(15.0),
        distance_cost=0.0,
    )
    wrapped = None
    try:
        live_action = action_config()
        if env.action_config != live_action:
            raise ValueError(f"environment action contract mismatch: {env.action_config!r}")
        wrapped = _sb3_wrapper(env)
        validate_policy_spaces(wrapped.observation_space, wrapped.action_space)
        model = make_model(
            wrapped,
            seed=69_001,
            n_steps=128,
            batch_size=128,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.995,
            log_std_init=-2.0,
            target_kl=0.03,
            device="cpu",
        )
        model.field_drive_config = dict(env.drive_config)
        model.field_observation_config = dict(env.observation_config)
        model.field_damage_config = dict(env.damage_config)
        model.field_action_config = live_action
        model.field_preview_only = True
        if int(model.num_timesteps) != 0:
            raise AssertionError(f"preview model was unexpectedly advanced: {model.num_timesteps}")
        model.save(str(CHECKPOINT))

        with zipfile.ZipFile(CHECKPOINT) as archive:
            checkpoint_data = json.loads(archive.read("data"))
        if checkpoint_data.get("field_action_config") != live_action:
            raise AssertionError("saved preview checkpoint is missing the action contract")
        if checkpoint_data.get("field_preview_only") is not True:
            raise AssertionError("saved preview checkpoint is missing field_preview_only=True")
        if int(checkpoint_data.get("num_timesteps", -1)) != 0:
            raise AssertionError("saved preview checkpoint does not have zero training steps")

        record = {
            "schema": "imu-drive-zero-step-preview-v1",
            "purpose": "UI preview only; random initialization, not trained and not a formal baseline",
            "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "num_timesteps": int(model.num_timesteps),
            "field_preview_only": True,
            "environment_transitions": 0,
            "training_calls": {"learn": 0, "env_step": 0},
            "model_parameters": {
                "seed": 69_001,
                "learning_rate": 3e-4,
                "gamma": 0.995,
                "log_std_init": -2.0,
                "n_steps": 128,
                "batch_size": 128,
                "n_epochs": 5,
                "target_kl": 0.03,
                "device": "cpu",
            },
            "field_action_config": model.field_action_config,
            "field_encoder_config": model.field_encoder_config,
            "drive_config": model.field_drive_config,
            "observation_config": model.field_observation_config,
            "damage_config": model.field_damage_config,
            "legacy_checkpoint_action_guard": legacy_guard,
        }
        RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == "__main__":
    main()

