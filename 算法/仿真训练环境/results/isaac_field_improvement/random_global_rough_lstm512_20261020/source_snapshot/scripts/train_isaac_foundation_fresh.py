"""Prepare and run a fresh, frozen RecurrentPPO foundation curriculum."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/fresh_coupled_tracks_20261020"
DEFAULT_RANDOM_ROUGH_OUTPUT = ROOT / "results/isaac_field_improvement/random_global_rough_20261020"
DEFAULT_RANDOM_ROUGH_COUPLED_GUIDES_OUTPUT = ROOT / (
    "results/isaac_field_improvement/random_global_rough_coupled_guides_20261020"
)
DEFAULT_LEGACY_OUTPUT = ROOT / "results/isaac_field_improvement/fresh_foundation_legacy_20261020"
SCHEMA = "isaac-field-fresh-foundation-v1"
ENCODER_SCHEMA = "depth64-lidar-single1d-minimal-v3"
INPUT_SCHEMA = "sensor-minimal-v6"
PROFILE_LEGACY_FOUNDATION = "legacy_foundation"
PROFILE_RANDOM_ROUGH = "random_rough"
PROFILES = (PROFILE_LEGACY_FOUNDATION, PROFILE_RANDOM_ROUGH)
TRAIN_SEED = 69_001
ENV_SEED = 80_003
RANDOM_ROUGH_TRAIN_SEED = 69_009
DEV_FIELD_SEEDS = tuple(range(82_001, 82_005))
DEV_FREE_SEEDS = tuple(range(87_001, 87_005))
RANDOM_ROUGH_TERRAIN_CONFIG = {
    "plants": 0, "terrain_enabled": True, "terrain_amplitude": 0.35,
    "terrain_wavelength_min": 4.0, "terrain_wavelength_max": 10.0,
    "rock_density": 0.0, "max_time": 60.0,
}
RANDOM_ROUGH_TIMESTEPS = 196_608
RANDOM_ROUGH_STAGE = {
    "name": "global_random", "goal_distance_m": None,
    "goal_angle_deg": 180.0, "timesteps": RANDOM_ROUGH_TIMESTEPS,
}
COUPLED_FRONT_GUIDE_MODEL = (
    "same_side_track_speed_coupled_front_guides_separate_proxy_not_continuous_track"
)

OBSERVATION_SHAPES = {
    "lidar": (1, 400), "depth": (1, 160, 88),
    "detections": (16, 8), "state": (33,), "feedback": (37,), "action": (4,),
}
STAGES = (
    {"name": "near", "goal_distance_m": [0.6, 1.2], "goal_angle_deg": 15.0, "timesteps": 32_768},
    {"name": "steer30", "goal_distance_m": [1.0, 2.0], "goal_angle_deg": 30.0, "timesteps": 32_768},
    {"name": "steer60", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0, "timesteps": 65_536},
    {"name": "turn90", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0, "timesteps": 65_536},
)
TRAINING = {
    "algorithm": "SB3-Contrib RecurrentPPO", "mode": "goto", "num_envs": 4,
    "training_seed": TRAIN_SEED,
    "stage_seeds": [TRAIN_SEED + i for i in range(len(STAGES))],
    "stages": [dict(stage) for stage in STAGES],
    "timesteps_total": sum(stage["timesteps"] for stage in STAGES),
    "goal_gain": 1.0, "n_steps": 128, "batch_size": 128, "n_epochs": 5,
    "learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95,
    "target_kl": 0.03, "log_std_init": -2.0, "use_sde": False,
    "checkpoint_interval_transitions": 8192,
    "initialization": "new random RecurrentPPO; no old checkpoint or weight transfer",
}
SOURCE_HELPERS = (
    "scripts/train_isaac_foundation.py",
    "scripts/improve_isaac_policy.py",
    "scripts/compare_isaac_exploration.py",
)
FIXED_SOURCES = ("scripts/isaaclab_python.sh", *SOURCE_HELPERS,
                 "scripts/train_isaac_foundation_fresh.py", "scripts/install_isaaclab.sh", "pyproject.toml",
                 "isaac_field/visualization/THIRD_PARTY_NOTICES.md",
                 "external/llm-viz/LICENSE", "external/llm-viz/src/utils/bezier.ts",
                 "external/llm-viz/src/llm/Sidebar.tsx", "external/llm-viz/src/llm/Sidebar.module.scss",
                 "external/llm-viz/src/llm/components/TocDiagram.tsx",
                 "external/llm-viz/src/llm/components/TocDiagram.module.scss")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_contract_equal(actual, expected) -> bool:
    """Compare JSON contracts exactly, including bool/int distinctions."""
    try:
        return json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def _protocol_sha(path: Path) -> str:
    return _sha256(path)


def _field_encoder_config() -> dict:
    from isaac_field import policy

    describe = getattr(policy, "_field_encoder_config", None)
    if not callable(describe):
        raise ValueError("policy does not expose its saved field_encoder_config")
    return dict(describe())


def _action_config() -> dict:
    from isaac_field.action_contract import action_config

    return dict(action_config())


def _live_action_config(env) -> dict:
    expected = _action_config()
    actual = getattr(env, "action_config", None)
    if actual != expected:
        raise ValueError(
            "live environment action_config is missing or differs from the frozen "
            f"direct-track action contract: {actual!r}"
        )
    return expected


def _live_environment_contract(env) -> dict:
    action = _live_action_config(env)
    drive = getattr(env, "drive_config", None)
    observation = getattr(env, "observation_config", None)
    if not isinstance(drive, dict) or drive.get("interface") != "independent_left_right_track_speed_coupled_pi_v7":
        raise ValueError(f"unexpected live drive interface: {drive!r}")
    coupling = drive.get("same_side_wheel_coupling")
    if not isinstance(coupling, dict) or coupling.get("enabled") is not True:
        raise ValueError(f"unexpected live same-side wheel coupling: {coupling!r}")
    physics_contact = drive.get("physics_contact")
    if not isinstance(physics_contact, dict):
        raise ValueError(f"unexpected live physics contact contract: {physics_contact!r}")
    expected_physics_contact = {
        "physics_dt_s": 0.001,
        "decimation": 50,
        "policy_dt_s": 0.05,
        "cone": "elliptic",
        "impratio": 1.0,
    }
    for key, expected in expected_physics_contact.items():
        observed = physics_contact.get(key)
        if isinstance(expected, (int, float)):
            if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                    or not math.isfinite(float(observed))
                    or not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(f"unexpected live physics contact contract: {physics_contact!r}")
        elif observed != expected:
            raise ValueError(f"unexpected live physics contact contract: {physics_contact!r}")
    if not isinstance(observation, dict) or observation.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"unexpected live observation contract: {observation!r}")
    return {
        "action_config": action,
        "drive_config": dict(drive),
        "observation_config": dict(observation),
    }


def _validate_rough_profile_environment(env, profile_physics_contract: dict) -> dict:
    live = _live_environment_contract(env)
    drive = live["drive_config"]
    guide_spec = getattr(env, "front_guide_spec", getattr(getattr(env, "cfg", None), "front_guide_spec", None))
    stability_config = getattr(
        env, "body_stability_config", getattr(getattr(env, "cfg", None), "body_stability_config", None)
    )
    if guide_spec is None or asdict(guide_spec) != profile_physics_contract["front_guide_spec"]:
        raise ValueError("live front-guide spec differs from the frozen rough profile")
    if (stability_config is None
            or asdict(stability_config) != profile_physics_contract["body_stability_config"]):
        raise ValueError("live body-stability config differs from the frozen rough profile")
    front = drive.get("front_guide")
    guide_coupling = profile_physics_contract["front_guide_drive_coupling"]
    if (drive.get("robot_asset_spec_sha256") != profile_physics_contract["robot_asset_spec_sha256"]
            or not isinstance(front, dict)
            or front.get("enabled") is not True
            or front.get("model") != COUPLED_FRONT_GUIDE_MODEL
            or front.get("wheels") != profile_physics_contract["front_guide_wheels"]
            or type(front.get("drive_motor_count")) is not int
            or front.get("drive_motor_count") != 0
            or isinstance(front.get("added_drive_torque_nm"), bool)
            or not isinstance(front.get("added_drive_torque_nm"), (int, float))
            or front.get("added_drive_torque_nm") != 0.0
            or front.get("rotation_coupling") != "same_side_track_speed"
            or not _json_contract_equal(front.get("coupling"), guide_coupling)):
        raise ValueError("live coupled front-guide geometry or drive metadata differs from the protocol")
    wheel_coupling = drive.get("same_side_wheel_coupling")
    expected_relation_counts = {
        "enabled": True,
        "relations_per_world": 14,
        "wheel_relations_per_world": 12,
        "guide_relations_per_world": 2,
        "front_guide_relation": guide_coupling["relation"],
    }
    if (not isinstance(wheel_coupling, dict)
            or any(wheel_coupling.get(key) != value
                   for key, value in expected_relation_counts.items())):
        raise ValueError("live same-side wheel/guide equality counts differ from the coupled profile")
    stability = drive.get("body_stability")
    if (not isinstance(stability, dict)
            or stability.get("enabled") is not True
            or stability.get("config") != profile_physics_contract["body_stability_config"]
            or stability.get("rate_source") != "root_link_com_angular_velocity_world_rotated_to_body"
            or stability.get("policy_observation_included") is not False
            or stability.get("hard_tilt_termination_rad") != 1.2):
        raise ValueError("live body-stability reward/termination metadata differs from the protocol")
    return live


def _validated_lstm_hidden_size(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("lstm_hidden_size must be a positive integer")
    return value


def _experiment_architecture(lstm_hidden_size: int = 128) -> dict:
    from isaac_field import policy

    lstm_hidden_size = _validated_lstm_hidden_size(lstm_hidden_size)
    return {
        "field_encoder_config": _field_encoder_config(),
        "depth_value_range": [0.0, 1.0],
        "depth_space_bounds": [-1.0, 1.0],
        "input_contract": INPUT_SCHEMA,
        "observation_shapes": {key: list(value) for key, value in policy.OBSERVATION_SHAPES.items()},
        "action_shape": list(policy.ACTION_SHAPE),
        "action_config": _action_config(),
        "goal_gain": 1.0,
        "lstm_hidden_size": lstm_hidden_size,
        "lstm_num_layers": 1,
        "architecture_is_searched": False,
    }


def _runtime_contract() -> dict:
    from isaac_field import policy
    from isaac_field.action_contract import ACTION_COMPONENTS, ACTION_SCHEMA

    actual_schema = getattr(policy, "ENCODER_SCHEMA", None)
    actual_shapes = getattr(policy, "OBSERVATION_SHAPES", None)
    expected_shapes = {key: tuple(value) for key, value in OBSERVATION_SHAPES.items() if key != "action"}
    if actual_schema != ENCODER_SCHEMA:
        raise ValueError(f"expected encoder {ENCODER_SCHEMA!r}, found {actual_schema!r}")
    if actual_shapes != expected_shapes or tuple(getattr(policy, "ACTION_SHAPE", ())) != (4,):
        raise ValueError(f"observation/action contract mismatch: {actual_shapes!r}")
    action = _action_config()
    expected_action = {
        "schema": "direct-track-speed-v1",
        "components": [
            "left_track_speed", "right_track_speed",
            "gimbal_yaw_rate", "gimbal_pitch_rate",
        ],
        "normalized_range": [-1.0, 1.0],
    }
    if ACTION_SCHEMA != expected_action["schema"] or list(ACTION_COMPONENTS) != expected_action["components"] or action != expected_action:
        raise ValueError(f"unexpected live action contract: {action!r}")
    config = _field_encoder_config()
    if (config.get("schema") != ENCODER_SCHEMA
            or config.get("features_dim") != 192
            or config.get("branch_dims") != {"lidar": 32, "depth": 64, "detections": 32, "state": 32, "feedback": 32}
            or config.get("depth_pool") != [8, 6]
            or config.get("radar_pool") != 50
            or TRAINING["goal_gain"] != 1.0):
        raise ValueError(f"unexpected live field_encoder_config: {config!r}")
    return {"encoder_schema": actual_schema, "field_encoder_config": config,
            "observation_shapes": {key: list(value) for key, value in actual_shapes.items()},
            "goal_gain": 1.0, "action_config": action}


def _config_dict() -> dict:
    from isaac_field.task import Config

    return asdict(Config(plants=0, terrain_enabled=False, max_time=20.0,
                         damage_penalty=20.0, goal_progress_budget=5.0,
                         damage_metric="stem-contact-v2"))


def _profile_config(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    config = _config_dict()
    if profile == PROFILE_RANDOM_ROUGH:
        config.update(RANDOM_ROUGH_TERRAIN_CONFIG)
    return config


def _profile_training(profile: str) -> dict:
    if profile == PROFILE_LEGACY_FOUNDATION:
        return json.loads(json.dumps(TRAINING))
    if profile == PROFILE_RANDOM_ROUGH:
        training = json.loads(json.dumps(TRAINING))
        training.update({
            "goal_sampling_profile": PROFILE_RANDOM_ROUGH,
            "training_seed": RANDOM_ROUGH_TRAIN_SEED,
            "stage_seeds": [RANDOM_ROUGH_TRAIN_SEED],
            "stages": [dict(RANDOM_ROUGH_STAGE)],
            "timesteps_total": RANDOM_ROUGH_TIMESTEPS,
        })
        return training
    raise ValueError(f"profile must be one of {PROFILES}")


def _profile_evaluation(profile: str) -> dict:
    common = {
        "environment_seed": ENV_SEED, "mode": "goto", "distance_cost": 0.0,
        "deterministic": True, "record_trajectories": True,
        "dataset_role": "development only; these four seeds do not establish generalization",
        "blind_seeds": {"range_label": "830xx", "used": False, "observed": False},
        "claim_limit": ("This is not an architecture ablation or blind validation."
                        if profile == PROFILE_LEGACY_FOUNDATION else
                        "Development rough-terrain training only; not blind validation or real-vehicle safety calibration."),
    }
    if profile == PROFILE_LEGACY_FOUNDATION:
        return {
            **common,
            "free_space": {"plants": 0, "seeds": list(DEV_FREE_SEEDS),
                           "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0},
            "field": {"plants": 16, "seeds": list(DEV_FIELD_SEEDS),
                      "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0},
        }
    if profile == PROFILE_RANDOM_ROUGH:
        return {
            **common,
            "goal_sampling_profile": PROFILE_RANDOM_ROUGH,
            "rough_a": {"plants": 0, "seeds": list(DEV_FREE_SEEDS),
                        "goal_distance_m": None, "goal_angle_deg": 180.0,
                        "environment_seed": 80_003},
            "rough_b": {"plants": 0, "seeds": list(DEV_FIELD_SEEDS),
                        "goal_distance_m": None, "goal_angle_deg": 180.0,
                        "environment_seed": 80_004},
        }
    raise ValueError(f"profile must be one of {PROFILES}")


def _evaluation_names(profile: str) -> tuple[str, str]:
    return ("rough_a", "rough_b") if profile == PROFILE_RANDOM_ROUGH else ("free_space", "field")


def _evaluation_settings(evaluation: dict, split: str) -> tuple[int, float, tuple[float, float] | None, int]:
    row = evaluation[split]
    distance = row.get("goal_distance_m")
    environment_seed = row.get("environment_seed", evaluation.get("environment_seed", ENV_SEED))
    return (int(row["plants"]), float(row.get("goal_angle_deg", 180.0)),
            None if distance is None else tuple(float(value) for value in distance),
            int(environment_seed))


def _outside_or_overturned_count(report: dict) -> int:
    return sum(row.get("reason") == "outside_or_overturned" for row in report.get("rows", []))


def _rough_termination_safety(before: dict, after: dict) -> dict:
    before_events = before["summary"]["termination_events"]
    after_events = after["summary"]["termination_events"]
    keys = ("boundary", "posture_failure_episodes", "outside_or_overturned_reason")
    counts = {
        key: {"baseline": int(before_events[key]), "final": int(after_events[key]),
              "not_increased": int(after_events[key]) <= int(before_events[key])}
        for key in keys
    }
    return {"events": counts, "not_increased": all(item["not_increased"] for item in counts.values())}


def _finite_json_tree(value) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            return False
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json_tree(child)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json_tree(child) for child in value)
    return False


def _add_rough_evaluation_summaries(report: dict) -> dict:
    rows = report.get("rows", [])
    if not rows or any(not isinstance(row.get("body_stability"), dict)
                       or not isinstance(row.get("termination_detail"), dict) for row in rows):
        raise ValueError("rough evaluation is missing per-episode posture or termination diagnostics")
    stability = [row["body_stability"] for row in rows]
    numeric_fields = (
        "sample_count", "duration_s", "max_abs_roll_rad", "max_abs_pitch_rad",
        "max_tilt_rad", "excess_tilt_duration_s", "roll_pitch_rate_sq_integral",
        "penalty_sum", "roll_pitch_angular_rate_rms_rad_s",
    )
    for index, item in enumerate(stability):
        for field in numeric_fields:
            value = item.get(field)
            try:
                finite = (not isinstance(value, bool)
                          and isinstance(value, (int, float))
                          and math.isfinite(float(value)))
            except (OverflowError, TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(
                    f"rough evaluation has missing/non-finite posture summary field {field} "
                    f"in episode row {index}"
                )

        if not _finite_json_tree(item):
            raise ValueError(f"rough evaluation has non-finite/unsupported posture summary data in row {index}")
    metrics = (
        "max_abs_roll_rad", "max_abs_pitch_rad", "max_tilt_rad",
        "roll_pitch_angular_rate_rms_rad_s",
    )
    observed = {}
    for metric in metrics:
        values = [float(item[metric]) for item in stability if metric in item]
        if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
            raise ValueError(f"rough evaluation has incomplete/non-finite posture metric {metric}")
        observed[metric] = {"max": max(values), "mean": sum(values) / len(values)}
    details = [row["termination_detail"] for row in rows]
    if any(not all(isinstance(detail.get(key), bool)
                   for key in ("boundary", "overtilt_roll", "overtilt_pitch")) for detail in details):
        raise ValueError("rough evaluation has incomplete posture termination details")
    result = dict(report)
    summary = dict(report.get("summary", {}))
    summary["body_stability"] = {
        "episodes_with_diagnostics": len(rows), "metrics": observed,
        "sampling": "per-episode policy-step summary; angular-rate RMS is an oscillation proxy, not substep extrema",
    }
    summary["termination_events"] = {
        "boundary": sum(detail["boundary"] for detail in details),
        "overtilt_roll": sum(detail["overtilt_roll"] for detail in details),
        "overtilt_pitch": sum(detail["overtilt_pitch"] for detail in details),
        "posture_failure_episodes": sum(
            detail["overtilt_roll"] or detail["overtilt_pitch"] for detail in details
        ),
        "outside_or_overturned_reason": _outside_or_overturned_count(report),
    }
    result["summary"] = summary
    return result


def _source_paths() -> tuple[str, ...]:
    package_files = sorted((ROOT / "isaac_field").rglob("*.py"))
    package_files += sorted((ROOT / "isaac_field").rglob("*.html"))
    paths = {path.relative_to(ROOT).as_posix() for path in package_files}
    paths.update(FIXED_SOURCES)
    missing = sorted(path for path in paths if not (ROOT / path).is_file())
    if missing:
        raise FileNotFoundError(f"fresh runner source files missing: {missing}")
    return tuple(sorted(paths))


def _source_manifest(output: Path, *, copy_files: bool) -> dict:
    rows = []
    for relative in _source_paths():
        source = ROOT / relative
        snapshot_path = f"source_snapshot/{relative}"
        digest = _sha256(source)
        if copy_files:
            target = output / snapshot_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rows.append({"path": relative, "snapshot_path": snapshot_path, "sha256": digest})
    return {"files": rows}


def _verify_sources(output: Path, manifest: dict) -> None:
    expected = set(_source_paths())
    rows = manifest.get("files", [])
    if {row.get("path") for row in rows} != expected:
        raise ValueError("frozen source manifest is incomplete or has unexpected files")
    for row in rows:
        live = ROOT / row["path"]
        snapshot = output / row["snapshot_path"]
        if not live.is_file() or not snapshot.is_file():
            raise FileNotFoundError(f"missing frozen source: {row['path']}")
        if _sha256(snapshot) != row["sha256"] or _sha256(live) != row["sha256"]:
            raise ValueError(f"source snapshot/live hash mismatch: {row['path']}")


def _package_versions() -> dict:
    names = ("torch", "stable-baselines3", "sb3-contrib", "gymnasium", "numpy", "scipy", "warp-lang", "vqf")
    versions = {name: None for name in names}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": versions}


def _isaac_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(ROOT / "external/IsaacLab"), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(
    output: Path,
    profile: str = PROFILE_LEGACY_FOUNDATION,
    lstm_hidden_size: int = 128,
) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse fresh experiment output: {output}")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    lstm_hidden_size = _validated_lstm_hidden_size(lstm_hidden_size)
    runtime_contract = _runtime_contract()
    config = _profile_config(profile)
    if config.get("plants") != 0:
        raise ValueError("foundation training requires plants=0")
    if (profile == PROFILE_LEGACY_FOUNDATION and config.get("terrain_enabled") is not False
            or profile == PROFILE_RANDOM_ROUGH and (
                config.get("terrain_enabled") is not True
                or config.get("terrain_amplitude") != 0.35
                or config.get("terrain_wavelength_min") != 4.0
                or config.get("terrain_wavelength_max") != 10.0
                or config.get("rock_density") != 0.0
            )):
        raise ValueError("foundation profile terrain configuration differs from its contract")
    expected_max_time = 20.0 if profile == PROFILE_LEGACY_FOUNDATION else 60.0
    if config.get("max_time") != expected_max_time or config.get("damage_penalty") != 20.0:
        raise ValueError("fresh foundation task configuration differs from the protocol")
    snapshot = _source_manifest(output, copy_files=False)
    output.mkdir(parents=True)
    snapshot = _source_manifest(output, copy_files=True)
    robot_geometry = _robot_geometry()
    protocol = {
        "schema": SCHEMA,
        "status": "prepared",
        "profile": profile,
        "architecture": _experiment_architecture(lstm_hidden_size),
        "action_config": _action_config(),
        "runtime_contract": runtime_contract,
        "config": config,
        "robot_geometry": robot_geometry,
        "profile_physics_contract": _profile_physics_contract(profile, config, robot_geometry),
        "training": _profile_training(profile),
        "initialization": {"checkpoint": None, "baseline_path": "untrained_baseline.zip",
                           "baseline_num_timesteps": 0, "weight_transfer": False},
        "evaluation": _profile_evaluation(profile),
        "runtime": {"versions": _package_versions(), "isaaclab_commit": _isaac_commit()},
        "source_snapshot": snapshot,
    }
    _json(output / "protocol.json", protocol)
    digest = _protocol_sha(output / "protocol.json")
    _json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": digest, "exit_code": 0,
        "num_timesteps": 0,
    })
    return {"output": str(output), "protocol_sha256": digest, "profile": profile,
            "source_files": len(snapshot["files"]),
            "timesteps_total": _profile_training(profile)["timesteps_total"]}


def preflight(output: Path) -> dict:
    output = output.resolve()
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --launch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    digest = _protocol_sha(protocol_path)
    if status.get("status") != "prepared" or status.get("protocol_sha256") != digest:
        raise ValueError("fresh trial is not an unchanged prepared experiment")
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported fresh protocol schema")
    profile = protocol.get("profile", PROFILE_LEGACY_FOUNDATION)
    if profile not in PROFILES:
        raise ValueError("prepared trial has an unknown training profile")
    architecture = protocol.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("prepared encoder, action, runtime, or training contract differs from this runner")
    try:
        frozen_lstm_hidden_size = _validated_lstm_hidden_size(architecture.get("lstm_hidden_size"))
    except ValueError as exc:
        raise ValueError("prepared encoder, action, runtime, or training contract differs from this runner") from exc
    if (architecture != _experiment_architecture(frozen_lstm_hidden_size)
            or protocol.get("runtime_contract") != _runtime_contract()
            or protocol.get("action_config") != _action_config()
            or protocol.get("training") != _profile_training(profile)):
        raise ValueError("prepared encoder, action, runtime, or training contract differs from this runner")
    if protocol.get("config") != _profile_config(profile):
        raise ValueError("prepared task config differs from the current Config")
    if protocol.get("evaluation") != _profile_evaluation(profile):
        raise ValueError("prepared evaluation conditions differ from the profile")
    geometry = json.loads(json.dumps(_robot_geometry(), ensure_ascii=False, allow_nan=False))
    if protocol.get("robot_geometry") != geometry:
        raise ValueError("prepared robot geometry differs from the current asset contract")
    if protocol.get("profile_physics_contract") != _profile_physics_contract(
            profile, _profile_config(profile), geometry):
        raise ValueError("prepared profile physics contract differs from the current configuration")
    _runtime_contract()
    _verify_sources(output, protocol.get("source_snapshot", {}))
    return protocol


def _robot_geometry() -> dict:
    from isaac_field.assets import ROBOT_SPEC, robot_asset_spec_sha256

    return {"source": "isaac_field.assets.ROBOT_SPEC", "spec": asdict(ROBOT_SPEC),
            "sha256": robot_asset_spec_sha256(ROBOT_SPEC)}


def _rough_profile_components():
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.body_stability import BodyStabilityConfig

    return FrontGuideSpec(), BodyStabilityConfig()


def _profile_physics_contract(profile: str, config: dict, robot_geometry: dict) -> dict | None:
    if profile != PROFILE_RANDOM_ROUGH:
        return None
    from isaac_field.assets import ROBOT_SPEC, robot_asset_spec_sha256

    front_guides, body_stability = _rough_profile_components()
    guide_coupled = front_guides.rotation_coupling == "same_side_track_speed"
    if not guide_coupled:
        raise ValueError("random rough profile requires guides coupled to same-side track speed")
    guide_ratio = ROBOT_SPEC.wheel_radius_m / front_guides.radius_m
    guide_coupling = {
        "enabled": True,
        "relation": "q_guide = (R_main/R_guide)*q_same_side_wheel0",
        "relations_per_world": 2,
        "leader_joint_by_side": {
            "left": "left_wheel_0_wheel_joint",
            "right": "right_wheel_0_wheel_joint",
        },
        "ratio_by_side": {"left": guide_ratio, "right": guide_ratio},
        "guide_actuator_count": 0,
        "added_pid_count": 0,
    }
    contract = {
        "front_guide_spec": asdict(front_guides),
        "front_guide_wheels": [asdict(wheel) for wheel in front_guides.wheel_specs(ROBOT_SPEC)],
        "front_guide_drive_coupling": guide_coupling,
        "same_side_wheel_coupling_counts": {
            "relations_per_world": 14,
            "wheel_relations_per_world": 12,
            "guide_relations_per_world": 2,
        },
        "robot_asset_spec_sha256": robot_asset_spec_sha256(ROBOT_SPEC, front_guides),
        "body_stability_config": asdict(body_stability),
        "terrain_config": {key: config[key] for key in RANDOM_ROUGH_TERRAIN_CONFIG},
        "action_config": _action_config(),
        "input_contract": _experiment_architecture()["input_contract"],
        "robot_geometry": robot_geometry,
        "interpretation": {
            "front_guides": (
                "one raised guide-wheel proxy per side, rotationally tied to the same-side wheel-0 track speed "
                "by the main/guide radius ratio; no guide actuator, PID, or added drive torque; "
                "separate proxy geometry, not a continuous track or vehicle calibration"
            ),
            "body_stability": (
                "per-policy-step dt times tilt_weight*max(0, acos(cos(roll)*cos(pitch))-threshold)^2 plus "
                "angular_rate_weight*(body_roll_rate^2+body_pitch_rate^2); coefficients are trial settings, "
                "not real-vehicle safety calibration"
            ),
            "no_added_observation_or_action": True,
            "overturn_termination_rad": 1.2,
        },
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def _checkpoint_steps(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read("data"))["num_timesteps"])


def _checkpoint_manifest(output: Path) -> list[dict]:
    expected_action_config = _action_config()
    rows = []
    for path in sorted(output.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            data = json.loads(archive.read("data"))
        checkpoint_action_config = data.get("field_action_config")
        if checkpoint_action_config != expected_action_config:
            raise ValueError(
                f"checkpoint action contract is missing or incompatible: {path}"
            )
        rows.append({"path": path.relative_to(output).as_posix(),
                     "sha256": _sha256(path), "num_timesteps": _checkpoint_steps(path),
                     "adam_optimizer_state_present": "policy.optimizer.pth" in names,
                     "action_config": checkpoint_action_config})
    _json(output / "checkpoint_manifest.json", {"checkpoints": rows})
    return rows


def _status(output: Path, runtime: dict, status: str = "running", **extra) -> None:
    _json(output / "run_status.json", {
        "status": status, "protocol_sha256": runtime.get("protocol_sha256"),
        "num_timesteps": int(runtime.get("num_timesteps", 0)),
        "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
    })


def _runtime_assets(env) -> list[dict]:
    """Hash the actual generated/cached USD files imported by this environment."""
    rows = []
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    for name in ("robot", "field", "corn"):
        asset = getattr(scene, name, None)
        filename = getattr(getattr(asset, "spawn", None), "usd_path", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"runtime USD asset is not a readable file: {path}")
        rows.append({"asset": name, "path": str(path), "sha256": _sha256(path),
                     "size_bytes": path.stat().st_size})
    return rows


def _evaluate(output: Path, cfg, checkpoint: Path, label: str, seeds: tuple[int, ...],
              plants: int, angle_deg: float, device: str,
              goal_distance: tuple[float, float] | None = (1.0, 3.0),
              environment_seed: int = ENV_SEED,
              front_guide_spec=None, body_stability_config=None,
              profile_physics_contract: dict | None = None) -> dict:
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.learning import evaluate_model
    from scripts.improve_isaac_policy import _add_safety_summary

    eval_cfg = replace(cfg, plants=plants)
    angle = math.radians(angle_deg)
    physical_kwargs = {}
    if front_guide_spec is not None:
        physical_kwargs["front_guide_spec"] = front_guide_spec
    if body_stability_config is not None:
        physical_kwargs["body_stability_config"] = body_stability_config
    engine = make_env(cfg=eval_cfg, mode="goto", num_envs=1, seed=environment_seed,
                      auto_reset=False, goal_distance=goal_distance,
                      goal_angle=angle, distance_cost=0.0, **physical_kwargs)
    adapter = SingleFieldAdapter(engine)
    try:
        if profile_physics_contract is None:
            _live_environment_contract(engine)
        else:
            _validate_rough_profile_environment(engine, profile_physics_contract)
        assets = _runtime_assets(engine)
        report = evaluate_model(adapter, checkpoint, seeds,
                                goal_distance=goal_distance, goal_angle=angle,
                                mode="goto", deterministic=True, device=device,
                                record_trajectories=True)
    finally:
        adapter.close()
    report = _add_safety_summary(report)
    if profile_physics_contract is not None:
        report = _add_rough_evaluation_summaries(report)
    report["runtime_assets"] = assets
    report["evaluation_config"] = {
        "environment_seed": int(environment_seed), "plants": int(plants),
        "goal_distance_m": None if goal_distance is None else list(goal_distance),
        "goal_angle_deg": float(angle_deg),
    }
    if profile_physics_contract is not None:
        terrain_config = {
            key: getattr(eval_cfg, key) for key in RANDOM_ROUGH_TERRAIN_CONFIG
        }
        if terrain_config != profile_physics_contract.get("terrain_config"):
            raise ValueError("rough evaluation terrain configuration differs from the frozen profile")
        report["evaluation_config"].update({
            "goal_sampling_profile": PROFILE_RANDOM_ROUGH,
            "terrain_config": terrain_config,
        })
    _json(output / f"evaluation_{label}.json", report)
    return report


def _paired_dev_conditions(
    reports: dict[str, dict], seeds: tuple[int, ...], *,
    require_rough_terrain_pair: bool = False,
) -> dict:
    from scripts.compare_isaac_exploration import _pair_initial_conditions

    result = _pair_initial_conditions(reports)
    expected = set(seeds)
    exact_rows = True
    for report in reports.values():
        rows = report.get("rows", [])
        observed = [row.get("seed") for row in rows]
        if len(rows) != len(seeds) or set(observed) != expected:
            exact_rows = False
            break
    evaluation_configs = [report.get("evaluation_config") for report in reports.values()]
    if require_rough_terrain_pair:
        rough_config_valid = bool(evaluation_configs) and all(
            isinstance(config, dict)
            and config.get("goal_sampling_profile") == PROFILE_RANDOM_ROUGH
            and config.get("terrain_config") == RANDOM_ROUGH_TERRAIN_CONFIG
            and isinstance(config.get("environment_seed"), int)
            and not isinstance(config.get("environment_seed"), bool)
            and int(config["environment_seed"]) >= 0
            for config in evaluation_configs
        )
        if not rough_config_valid:
            result.setdefault("mismatches", []).append({
                "seed": None, "variant": "evaluation_contract",
                "differences": ["missing_or_invalid_rough_terrain_config"],
            })
    if any(config is not None for config in evaluation_configs) or require_rough_terrain_pair:
        if any(config != evaluation_configs[0] for config in evaluation_configs[1:]):
            result.setdefault("mismatches", []).append({
                "seed": None, "variant": "evaluation_contract", "differences": ["evaluation_config"],
            })
        row_maps = {
            label: {row.get("seed"): row for row in report.get("rows", [])}
            for label, report in reports.items()
        }
        if require_rough_terrain_pair:
            expected_terrain_seed = (
                evaluation_configs[0].get("environment_seed")
                if isinstance(evaluation_configs[0], dict) else None
            )
            for seed in expected:
                observed = []
                for rows in row_maps.values():
                    row = rows.get(seed)
                    reset_info = row.get("reset_info") if isinstance(row, dict) else None
                    terrain_seed = reset_info.get("terrain_seed") if isinstance(reset_info, dict) else None
                    if (isinstance(terrain_seed, bool) or not isinstance(terrain_seed, int)
                            or terrain_seed < 0):
                        observed.append(None)
                    else:
                        observed.append(int(terrain_seed))
                differences = []
                if any(value is None for value in observed):
                    differences.append("terrain_seed_missing_or_invalid")
                elif len(set(observed)) != 1:
                    differences.append("terrain_seed_mismatch")
                elif (isinstance(expected_terrain_seed, int)
                      and observed[0] != expected_terrain_seed):
                    differences.append("terrain_seed_does_not_match_environment_seed")
                if differences:
                    result.setdefault("mismatches", []).append({
                        "seed": seed, "variant": "terrain_seed", "differences": differences,
                    })

            def field_asset(report: dict) -> tuple[str, str, int] | None:
                assets = report.get("runtime_assets")
                if not isinstance(assets, list):
                    return None
                fields = [item for item in assets
                          if isinstance(item, dict) and item.get("asset") == "field"]
                if len(fields) != 1:
                    return None
                item = fields[0]
                path, digest, size = item.get("path"), item.get("sha256"), item.get("size_bytes")
                if (not isinstance(path, str) or not path
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest.lower())
                        or isinstance(size, bool) or not isinstance(size, int) or size < 0):
                    return None
                return path, digest.lower(), int(size)

            field_assets = [field_asset(report) for report in reports.values()]
            if (any(item is None for item in field_assets)
                    or len(set(field_assets)) != 1):
                result.setdefault("mismatches", []).append({
                    "seed": None, "variant": "field_asset",
                    "differences": ["missing_or_mismatched_field_asset_path_hash_size"],
                })
        else:
            for seed in expected:
                terrain_seeds = {
                    rows.get(seed, {}).get("reset_info", {}).get("terrain_seed")
                    for rows in row_maps.values()
                }
                present = {value for value in terrain_seeds if value is not None}
                if len(present) > 1:
                    result.setdefault("mismatches", []).append({
                        "seed": seed, "variant": "terrain_seed", "differences": ["terrain_seed"],
                    })
    result["helper_valid_24_seed_rule"] = bool(result.get("valid"))
    result["expected_seed_count"] = len(seeds)
    result["valid"] = bool(
        not result.get("mismatches")
        and result.get("paired_seed_count") == len(seeds)
        and exact_rows
    )
    return result


def launch(output: Path, device: str = "cuda:0") -> dict:
    protocol = preflight(output)
    output = output.resolve()
    runtime = {"protocol_sha256": _protocol_sha(output / "protocol.json"), "num_timesteps": 0}
    _status(output, runtime, phase="initializing")
    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from isaac_field.env import make_env
        from isaac_field.learning import CurriculumStage, train_curriculum
        from isaac_field.task import Config
        from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment

        cfg = Config(**protocol["config"])
        profile = protocol.get("profile", PROFILE_LEGACY_FOUNDATION)
        training = protocol["training"]
        evaluation = protocol["evaluation"]
        front_guide_spec, body_stability_config = (
            _rough_profile_components() if profile == PROFILE_RANDOM_ROUGH else (None, None)
        )
        if profile == PROFILE_RANDOM_ROUGH:
            physical_contract = protocol["profile_physics_contract"]
            if (asdict(front_guide_spec) != physical_contract["front_guide_spec"]
                    or asdict(body_stability_config) != physical_contract["body_stability_config"]):
                raise ValueError("live rough-profile guide/stability config differs from the frozen protocol")
        stages = tuple(CurriculumStage(
            row["name"], "goto",
            None if row.get("goal_distance_m") is None else tuple(row["goal_distance_m"]),
            math.radians(row["goal_angle_deg"]),
        ) for row in training["stages"])
        latest = {"env": None}
        runtime_asset_records = []
        progress_by_stage = {}
        baseline_path = output / protocol["initialization"]["baseline_path"]

        def env_factory(**kwargs):
            _verify_sources(output, protocol["source_snapshot"])
            kwargs["num_envs"] = 4 if kwargs["auto_reset"] else 1
            if front_guide_spec is not None:
                kwargs["front_guide_spec"] = front_guide_spec
                kwargs["body_stability_config"] = body_stability_config
            env = make_env(**kwargs)
            try:
                if profile == PROFILE_RANDOM_ROUGH:
                    _validate_rough_profile_environment(env, protocol["profile_physics_contract"])
                else:
                    _live_environment_contract(env)
                runtime_asset_records.append({"environment_index": len(runtime_asset_records),
                    "seed": kwargs.get("seed"), "assets": _runtime_assets(env)})
                _json(output / "runtime_assets.json", {"environments": runtime_asset_records})
            except BaseException:
                env.close()
                raise
            latest["env"] = env
            return env

        def callback_factory(stage):
            index = next(i for i, item in enumerate(stages) if item.name == stage.name)
            stage_dir = output / f"stage_{index + 1}_{stage.name}"
            checkpoint_dir = stage_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _verify_sources(output, protocol["source_snapshot"])
            progress = _make_progress_callback(4, latest["env"], interval=8192)
            progress_by_stage[stage.name] = progress
            checkpoint = CheckpointCallback(save_freq=8192 // 4,
                                            save_path=str(checkpoint_dir),
                                            name_prefix=f"field_policy_{stage.name}", verbose=0)

            class InitialBaselineCallback(BaseCallback):
                def _on_training_start(self):
                    if baseline_path.exists():
                        raise FileExistsError(f"refusing to replace initial model: {baseline_path}")
                    if int(self.model.num_timesteps) != 0:
                        raise RuntimeError("new foundation model must start at zero timesteps")
                    extractor = self.model.policy.features_extractor
                    actor_lstm = getattr(self.model.policy, "lstm_actor", None)
                    critic_lstm = getattr(self.model.policy, "lstm_critic", None)
                    if int(getattr(extractor, "features_dim", -1)) != 192:
                        raise ValueError("live encoder fusion dimension is not 192")
                    if float(getattr(extractor, "goal_gain", math.nan)) != 1.0:
                        raise ValueError("live encoder goal_gain is not 1.0")
                    live_contract = (
                        _validate_rough_profile_environment(latest["env"], protocol["profile_physics_contract"])
                        if profile == PROFILE_RANDOM_ROUGH else _live_environment_contract(latest["env"])
                    )
                    expected_hidden_size = protocol["architecture"]["lstm_hidden_size"]
                    expected_num_layers = protocol["architecture"]["lstm_num_layers"]
                    lstm_modules = {"actor": actor_lstm, "critic": critic_lstm}
                    lstm_architecture = {}
                    for name, module in lstm_modules.items():
                        if module is None:
                            raise ValueError(f"live recurrent policy is missing its {name} LSTM")
                        actual_hidden_size = int(getattr(module, "hidden_size", -1))
                        actual_num_layers = int(getattr(module, "num_layers", -1))
                        if (actual_hidden_size != expected_hidden_size
                                or actual_num_layers != expected_num_layers):
                            raise ValueError(
                                f"live {name} LSTM architecture differs from the frozen protocol: "
                                f"hidden_size={actual_hidden_size}, num_layers={actual_num_layers}"
                            )
                        lstm_architecture[name] = {
                            "hidden_size": actual_hidden_size,
                            "num_layers": actual_num_layers,
                        }
                    from isaac_field.policy import _validate_model_action_contract

                    _validate_model_action_contract(self.model)
                    field_encoder_config = _field_encoder_config()
                    if getattr(self.model, "field_encoder_config", None) != field_encoder_config:
                        raise ValueError("new model field_encoder_config differs from the current policy")
                    self.model.save(str(baseline_path.with_suffix("")))
                    if _checkpoint_steps(baseline_path) != 0:
                        raise RuntimeError("untrained baseline checkpoint is not at zero steps")
                    from isaac_field.policy import _validate_checkpoint_action_metadata

                    _validate_checkpoint_action_metadata(baseline_path)
                    parameter_counts = {
                        "policy_total": sum(int(parameter.numel()) for parameter in self.model.policy.parameters()),
                        "actor_lstm": sum(int(parameter.numel()) for parameter in actor_lstm.parameters()),
                        "critic_lstm": sum(int(parameter.numel()) for parameter in critic_lstm.parameters()),
                    }
                    _json(output / "initialization.json", {
                        "checkpoint": baseline_path.name, "sha256": _sha256(baseline_path),
                        "num_timesteps": 0, "protocol_sha256": runtime["protocol_sha256"],
                        "field_encoder_config": field_encoder_config,
                        "field_action_config": self.model.field_action_config,
                        "protocol_action_config": protocol["action_config"],
                        "drive_config": live_contract["drive_config"],
                        "observation_config": live_contract["observation_config"],
                        "protocol_architecture": protocol["architecture"],
                        "policy_lstm_architecture": lstm_architecture,
                        "parameter_counts": parameter_counts,
                        "source_initialization": "new RecurrentPPO, no old checkpoint loaded",
                    })
                    _checkpoint_manifest(output)

                def _on_step(self):
                    return True

            class PersistStageProgress(BaseCallback):
                def _on_training_start(self):
                    self.stage_start = int(self.model.num_timesteps)
                    runtime["num_timesteps"] = self.stage_start
                    _status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)

                def _on_step(self):
                    runtime["num_timesteps"] = int(self.num_timesteps)
                    if (runtime["num_timesteps"] - self.stage_start) % 8192 == 0:
                        _checkpoint_manifest(output)
                        _status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)
                    return True

                def _on_training_end(self):
                    progress._refresh_episodes()
                    runtime["num_timesteps"] = int(self.model.num_timesteps)
                    _json(stage_dir / "progress.json", {
                        "stage": stage.name, "start_num_timesteps": self.stage_start,
                        "end_num_timesteps": runtime["num_timesteps"],
                        "aggregate_rows": progress.log_rows,
                        "last_completed_episodes": progress.recent_episodes[-10:],
                    })
                    _status(output, runtime, phase=f"completed_{stage.name}", stage=stage.name)

            callbacks = [checkpoint, progress, PersistStageProgress()]
            if index == 0:
                callbacks.insert(0, InitialBaselineCallback())
            return CallbackList(callbacks)

        curriculum = train_curriculum(
            cfg, output, checkpoint=None,
            timesteps_per_stage=tuple(row["timesteps"] for row in training["stages"]),
            training_seed=int(training["training_seed"]),
            heldout_seeds=tuple(evaluation[_evaluation_names(profile)[1]]["seeds"]),
            stages=stages, evaluate_before_after=False, goal_gain=1.0,
            device=device, n_epochs=5, batch_size=128,
            callback_factory=callback_factory, env_factory=env_factory,
            lstm_hidden_size=protocol["architecture"]["lstm_hidden_size"],
        )
        runtime["num_timesteps"] = int(curriculum.summary["stages"][-1]["total_timesteps"])
        final_checkpoint = curriculum.checkpoint
        if runtime["num_timesteps"] != training["timesteps_total"]:
            raise RuntimeError(f"expected {training['timesteps_total']} steps, got {runtime['num_timesteps']}")
        if not baseline_path.is_file():
            raise RuntimeError("the untrained baseline was not saved by the first-stage callback")
        checkpoints = _checkpoint_manifest(output)
        paired_results, evaluation_reports = {}, []
        for split in _evaluation_names(profile):
            seeds = tuple(evaluation[split]["seeds"])
            plants, angle, distance, environment_seed = _evaluation_settings(evaluation, split)
            suffix = f"{seeds[0]}_{seeds[-1]}"
            report_name = {"free_space": "free", "field": "field"}.get(split, split)
            _status(output, runtime, phase=f"evaluating_dev_{split}")
            _verify_sources(output, protocol["source_snapshot"])
            baseline_label = f"{report_name}_baseline_{suffix}"
            baseline_report = _evaluate(
                output, cfg, baseline_path, baseline_label, seeds, plants, angle, device,
                goal_distance=distance, environment_seed=environment_seed,
                front_guide_spec=front_guide_spec,
                body_stability_config=body_stability_config,
                profile_physics_contract=protocol.get("profile_physics_contract"),
            )
            _verify_sources(output, protocol["source_snapshot"])
            final_label = f"{report_name}_final_{suffix}"
            final_report = _evaluate(
                output, cfg, final_checkpoint, final_label, seeds, plants, angle, device,
                goal_distance=distance, environment_seed=environment_seed,
                front_guide_spec=front_guide_spec,
                body_stability_config=body_stability_config,
                profile_physics_contract=protocol.get("profile_physics_contract"),
            )
            pairing = _paired_dev_conditions(
                {"baseline": baseline_report, "final": final_report}, seeds,
                require_rough_terrain_pair=profile == PROFILE_RANDOM_ROUGH,
            )
            assessment = (_safe_goto_assessment(baseline_report, final_report)
                          if pairing["valid"] else "invalid_initial_conditions")
            paired_results[split] = {"pairing": pairing, "assessment": assessment}
            if profile == PROFILE_RANDOM_ROUGH:
                termination_safety = _rough_termination_safety(baseline_report, final_report)
                paired_results[split]["termination_safety"] = termination_safety
                paired_results[split]["body_stability"] = {
                    "baseline": baseline_report["summary"]["body_stability"],
                    "final": final_report["summary"]["body_stability"],
                }
                if pairing["valid"] and not termination_safety["not_increased"]:
                    paired_results[split]["assessment"] = "termination_safety_regression"
            evaluation_reports.extend([
                f"evaluation_{baseline_label}.json", f"evaluation_{final_label}.json",
            ])
        if profile == PROFILE_LEGACY_FOUNDATION:
            summary = {
                "schema": "isaac-field-fresh-foundation-development-summary-v1",
                "action_config": protocol["action_config"],
                "checkpoint_initial": str(baseline_path), "checkpoint_final": str(final_checkpoint),
                "actual_final_num_timesteps": runtime["num_timesteps"],
                "checkpoint_count": len(checkpoints),
                "free_space_pairing": paired_results["free_space"]["pairing"],
                "free_space_assessment": paired_results["free_space"]["assessment"],
                "field_pairing": paired_results["field"]["pairing"],
                "field_assessment": paired_results["field"]["assessment"],
                "evaluation_reports": evaluation_reports,
                "claim_limit": "Development comparison only; not an architecture ablation, blind validation, or full-field navigation result.",
                "blind_830xx": "not evaluated or observed",
            }
        else:
            summary = {
                "schema": "isaac-field-fresh-foundation-development-summary-v1",
                "profile": profile, "action_config": protocol["action_config"],
                "checkpoint_initial": str(baseline_path), "checkpoint_final": str(final_checkpoint),
                "actual_final_num_timesteps": runtime["num_timesteps"],
                "checkpoint_count": len(checkpoints), "evaluation_sets": paired_results,
                "evaluation_reports": evaluation_reports,
                "safety_gate": (
                    "clean successes must increase while damage, collision, boundary, posture failure, and "
                    "outside_or_overturned events do not increase; four-seed development evidence only"
                ),
                "claim_limit": (
                    "Four-seed rough-terrain development comparison only; no absolute posture threshold is a "
                    "safety certification, and this is not blind or full-field validation."
                ),
                "blind_830xx": "not evaluated or observed",
            }
        _json(output / "comparison_summary.json", summary)
        _status(output, runtime, "completed", phase="completed",
                summary="comparison_summary.json", exit_code=0)
        return summary
    except KeyboardInterrupt as exc:
        _status(output, runtime, "interrupted", phase="interrupted",
                exception=f"{type(exc).__name__}: {exc}", exit_code=130)
        raise
    except BaseException as exc:
        _status(output, runtime, "failed", phase="failed",
                exception=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_RANDOM_ROUGH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lstm-hidden-size", type=int, default=None,
                        help="positive LSTM hidden width; valid only with --prepare (default: 128)")
    args = parser.parse_args(argv)
    if args.lstm_hidden_size is not None and not args.prepare:
        parser.error("--lstm-hidden-size is only valid with --prepare")
    try:
        if args.prepare:
            default_output = (DEFAULT_RANDOM_ROUGH_COUPLED_GUIDES_OUTPUT
                              if args.profile == PROFILE_RANDOM_ROUGH
                              else DEFAULT_LEGACY_OUTPUT)
            output = args.output or default_output
            lstm_hidden_size = 128 if args.lstm_hidden_size is None else args.lstm_hidden_size
            print(json.dumps(prepare(output, profile=args.profile, lstm_hidden_size=lstm_hidden_size),
                             ensure_ascii=False, indent=2))
        elif args.preflight:
            output = args.output or DEFAULT_RANDOM_ROUGH_COUPLED_GUIDES_OUTPUT
            protocol = preflight(output)
            print(json.dumps({"status": "ready", "profile": protocol.get("profile", PROFILE_LEGACY_FOUNDATION),
                              "protocol_sha256": _protocol_sha(output / "protocol.json"),
                              "source_files": len(protocol["source_snapshot"]["files"])},
                             ensure_ascii=False, indent=2))
        else:
            output = args.output or DEFAULT_RANDOM_ROUGH_COUPLED_GUIDES_OUTPUT
            print(json.dumps(launch(output, args.device), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
