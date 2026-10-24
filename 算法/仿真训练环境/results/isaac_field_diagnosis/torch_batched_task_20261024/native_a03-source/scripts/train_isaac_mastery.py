"""Freeze and run controlled, instrumented Isaac Sim RSL-RL experiments.

One Kit runtime is held across stages, mastery probes, and selection suites.
Training updates use Isaac Lab's official RSL-RL ``OnPolicyRunner``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_isaac_foundation_fresh as foundation
from isaac_field.feedback import (
    ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA, SUPPORTED_INPUT_SCHEMAS, feedback_config,
)
from isaac_field.curriculum_control import (
    CurriculumControlConfig, CurriculumController, ProbeSpec,
    compare_best_checkpoints,
)
from isaac_field.task import Config, state_fields_for_schema

SCHEMA = "isaac-field-mastery-experiment-v2"
CONTRACT_KEYS = (
    "action_config", "drive_config", "observation_config", "damage_config",
    "damage_metric", "distance_cost", "goal_distance", "goal_angle_rad",
    "mode", "deterministic", "evaluation_config",
)
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/isaacsim_physx_rsl_mastery_20261024"
PROFILE_LEGACY = "legacy"
PROFILE_FAST_STOP_512 = "fast_stop_512"
PROFILES = (PROFILE_LEGACY, PROFILE_FAST_STOP_512)
STAGES = tuple(
    {"name": name, "mode": "goto", "goal_distance": distance,
     "goal_angle": math.radians(angle), "plants": plants}
    for name, distance, angle, plants in (
        ("near", [0.6, 1.2], 15, 0),
        ("turn30", [0.6, 1.2], 30, 0),
        ("turn60", [0.6, 1.2], 60, 0),
        ("turn90", [0.6, 1.2], 90, 0),
        ("range", [1.0, 3.0], 90, 0),
        ("sparse2", [1.0, 3.0], 90, 2),
        ("field8", [1.0, 3.0], 90, 8),
        ("field16", [1.0, 3.0], 90, 16),
    )
)
FAST_STOP_512_STAGES = STAGES[:5]
FAST_STOP_512_ARCHITECTURE = {"lstm_hidden_size": 512, "n_lstm_layers": 1}


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _sources() -> tuple[str, ...]:
    return tuple(sorted(set(foundation._source_paths()) | {
        "scripts/train_isaac_mastery.py", "scripts/run_isaac_mastery.sh",
        "isaac_field/runtime.py", "isaac_field/env.py", "isaac_field/assets.py",
        "isaac_field/scene_assets.py", "isaac_field/physx_drive.py",
        "isaac_field/learning.py", "isaac_field/policy_contract.py",
        "isaac_field/rsl_rl_cfg.py", "isaac_field/rsl_rl_policy.py", "isaac_field/rsl_rl_inference.py",
    }))


def _profile_config(profile: str, input_schema: str = INPUT_SCHEMA) -> dict:
    if input_schema not in SUPPORTED_INPUT_SCHEMAS:
        raise ValueError(
            f"input_schema must be one of {SUPPORTED_INPUT_SCHEMAS}, got {input_schema!r}"
        )
    if profile == PROFILE_LEGACY:
        return asdict(Config(plants=0, terrain_enabled=False, max_time=60.0,
                             damage_penalty=20.0, goal_progress_budget=5.0,
                             damage_metric="stem-contact-v2", input_schema=input_schema))
    if profile != PROFILE_FAST_STOP_512:
        raise ValueError(f"profile must be one of {PROFILES}")
    values = foundation._profile_config(foundation.PROFILE_RANDOM_ROUGH_FAST_STOP)
    values.update(plants=0, terrain_enabled=False, rock_density=0.0,
                  input_schema=input_schema)
    return asdict(Config(**values))


def _policy_architecture(profile: str, input_schema: str) -> dict | None:
    """Describe the encoder identity only for the opt-in schema.

    Existing v6 protocols retain their historical architecture shape; absolute
    goal protocols bind the recurrent shape to the sensor/state schema.
    """
    if input_schema == INPUT_SCHEMA:
        return dict(FAST_STOP_512_ARCHITECTURE) if profile == PROFILE_FAST_STOP_512 else None
    from isaac_field.policy_contract import _encoder_schema_for_input

    architecture = (
        dict(FAST_STOP_512_ARCHITECTURE)
        if profile == PROFILE_FAST_STOP_512
        else {"lstm_hidden_size": 128, "n_lstm_layers": 1}
    )
    architecture.update(
        input_schema=input_schema,
        encoder_schema=_encoder_schema_for_input(input_schema),
        state_fields=list(state_fields_for_schema(input_schema)),
    )
    return architecture


def _observation_contract(input_schema: str, config: dict) -> dict:
    """Freeze the schema-defining projection of the runtime observation contract."""
    from isaac_field.feedback import FEEDBACK_DIM, STATE_DIM

    contract = {
        "schema": input_schema,
        "state_dim": STATE_DIM,
        "state_fields": list(state_fields_for_schema(input_schema)),
        "observation_shapes": {
            "lidar": [1, config["lidar_rays"]],
            "depth": [1, config["depth_height"], config["depth_width"]],
            "detections": [config["max_detections"], 8],
            "state": [STATE_DIM],
            "feedback": [FEEDBACK_DIM],
        },
        "feedback": feedback_config(input_schema=input_schema),
    }
    if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        contract["absolute_goal_coordinates"] = {
            "source": "task.goal",
            "coordinate_frame": "field_world_xy",
            "origin_xy_m": [0.0, 0.0],
            "unit": "m",
            "field_size_m": float(config["size"]),
            "normalization": "world_xy / field_size_m * 2 - 1",
            "state_clip": [-1.0, 1.0],
        }
    return contract


def _observation_contract_projection(observation_config: dict) -> dict:
    keys = ("schema", "state_dim", "state_fields", "observation_shapes", "feedback",
            "absolute_goal_coordinates")
    return {key: observation_config[key] for key in keys if key in observation_config}


def _complete_observation_config(value) -> bool:
    if not isinstance(value, dict):
        return False
    fields = value.get("state_fields")
    shapes = value.get("observation_shapes")
    return (
        value.get("state_dim") == 33
        and isinstance(fields, list)
        and len(fields) == 33
        and isinstance(shapes, dict)
        and shapes.get("state") == [33]
    )


def _stages_for_profile(profile: str) -> tuple[dict, ...]:
    if profile == PROFILE_LEGACY:
        return STAGES
    if profile == PROFILE_FAST_STOP_512:
        return FAST_STOP_512_STAGES
    raise ValueError(f"profile must be one of {PROFILES}")


def _fast_stop_transfer_probe(input_schema: str = INPUT_SCHEMA) -> dict:
    """Freeze the existing rough-profile development suites as transfer probes."""
    profile = foundation.PROFILE_RANDOM_ROUGH_FAST_STOP
    evaluation = foundation._profile_evaluation(profile)
    global_config = foundation._profile_config(profile)
    global_config["input_schema"] = input_schema
    members = []
    for label in ("rough_a", "rough_b"):
        row = evaluation[label]
        members.append({
            "name": label,
            "plants": int(row["plants"]),
            "goal_distance": row["goal_distance_m"],
            "goal_angle": math.radians(float(row["goal_angle_deg"])),
            "environment_seed": int(row["environment_seed"]),
            "seeds": [int(seed) for seed in row["seeds"]],
        })
    return {
        "purpose": "fixed_original_global_rough_transfer_development_only",
        "environment_config": asdict(Config(**global_config)),
        "members": members,
        "claim_limit": "Transfer development probe only; this does not establish full-field capability or held-out generalization.",
    }


def _validate_profile_protocol(protocol: dict) -> None:
    profile = protocol.get("profile", PROFILE_LEGACY)
    if profile not in PROFILES:
        raise ValueError(f"unsupported mastery profile: {profile!r}")
    config = protocol.get("config")
    if not isinstance(config, dict):
        raise ValueError("mastery protocol is missing its task config")
    input_schema = protocol.get("input_schema", config.get("input_schema", INPUT_SCHEMA))
    if input_schema not in SUPPORTED_INPUT_SCHEMAS:
        raise ValueError(f"unsupported mastery input_schema: {input_schema!r}")
    if config.get("input_schema", INPUT_SCHEMA) != input_schema:
        raise ValueError("protocol input_schema differs from the task config")
    if "input_schema" in protocol and protocol["input_schema"] != input_schema:
        raise ValueError("protocol input_schema identity is inconsistent")
    if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        expected_architecture = _policy_architecture(profile, input_schema)
        if protocol.get("policy_architecture") != expected_architecture:
            raise ValueError("absolute-goal policy_architecture does not match the input schema")
        expected_observation = _observation_contract(input_schema, config)
        if protocol.get("observation_contract") != expected_observation:
            raise ValueError("absolute-goal observation_contract does not match the input schema")
        if json.loads(json.dumps(config)) != json.loads(
            json.dumps(_profile_config(profile, input_schema))
        ):
            raise ValueError("absolute-goal task config differs from its selected profile")
    if profile == PROFILE_LEGACY:
        return
    if protocol.get("method") != "mastery":
        raise ValueError("fast_stop_512 requires the mastery promotion controller")
    expected_stages = json.loads(json.dumps(FAST_STOP_512_STAGES))
    expected_config = json.loads(json.dumps(_profile_config(profile, input_schema)))
    if json.loads(json.dumps(protocol.get("config"))) != expected_config:
        raise ValueError("fast_stop_512 task config differs from its frozen profile")
    if json.loads(json.dumps(protocol.get("stages"))) != expected_stages:
        raise ValueError("fast_stop_512 stage schedule differs from its frozen profile")
    architecture = protocol.get("policy_architecture")
    expected_architecture = _policy_architecture(profile, input_schema)
    if architecture != expected_architecture:
        raise ValueError("fast_stop_512 policy architecture must be LSTM 512x1")
    expected_hp = {"n_steps": 128, "batch_size": 128, "n_epochs": 5,
                   "learning_rate": 3e-4, "gamma": foundation.FAST_STOP_GAMMA,
                   "gae_lambda": 0.95, "log_std_init": -2.0, "target_kl": 0.03,
                   "lstm_hidden_size": 512}
    if protocol.get("hyperparameters") != expected_hp:
        raise ValueError("fast_stop_512 hyperparameters differ from the frozen profile")
    transfer = protocol.get("transfer_probe")
    expected_transfer = json.loads(json.dumps(_fast_stop_transfer_probe(input_schema)))
    if json.loads(json.dumps(transfer)) != expected_transfer:
        raise ValueError("fast_stop_512 global rough transfer probe differs from its frozen profile")
    expected_physics = foundation._profile_physics_contract(
        foundation.PROFILE_RANDOM_ROUGH_FAST_STOP,
        foundation._profile_config(foundation.PROFILE_RANDOM_ROUGH_FAST_STOP),
        foundation._robot_geometry(),
    )
    if (json.loads(json.dumps(protocol.get("profile_physics_contract")))
            != json.loads(json.dumps(expected_physics))):
        raise ValueError("fast_stop_512 guide/body-stability contract differs from the frozen rough profile")
    expected_names = [member["name"] for member in expected_transfer["members"]]
    expected_banks = [member["seeds"] for member in expected_transfer["members"]]
    expected_selection = {
        "stage_indices": [], "stage_names": expected_names, "episodes_per_stage": 4,
        "seed_banks": expected_banks, "every_probes": 4, "also_on_promotion": True,
        "criterion": "clean successes increase; damaged episodes, mean damage and collisions do not increase",
    }
    if protocol.get("selection") != expected_selection:
        raise ValueError("fast_stop_512 selection must use the fixed global rough transfer suites")
    expected_mastery = {
        "episodes_per_stage": 16, "min_clean_success_rate": 0.8,
        "required_consecutive_passes": 2,
        "probe_interval_timesteps": protocol.get("block_steps"),
        "max_damaged_episodes": 0, "max_mean_damage": 0.0,
        "max_total_collisions": 0, "active_blocks_per_replay": 3,
        "seed_banks": {stage["name"]: list(range(88_100 + i * 100, 88_116 + i * 100))
                       for i, stage in enumerate(expected_stages)},
    }
    if protocol.get("mastery") != expected_mastery:
        raise ValueError("fast_stop_512 mastery promotion and replay contract differs from its frozen profile")
    if protocol.get("claim_limit") != expected_transfer["claim_limit"]:
        raise ValueError("fast_stop_512 transfer claim limit differs from its frozen profile")
    expected_objective = foundation._fast_stop_objective_contract(_profile_config(profile))
    if json.loads(json.dumps(protocol.get("goal_objective_contract"))) != json.loads(json.dumps(expected_objective)):
        raise ValueError("fast_stop_512 stop/reward objective contract differs from its frozen profile")


def prepare(output: Path, *, method: str = "mastery", profile: str = PROFILE_LEGACY,
            input_schema: str = INPUT_SCHEMA,
            budget: int = 196_608,
            block_steps: int = 8_192, gamma: float | None = None,
            gae_lambda: float = 0.95) -> dict:
    """Freeze the complete protocol without creating a simulator or training."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse experiment directory: {output}")
    if method not in ("fixed", "mastery"):
        raise ValueError("method must be fixed or mastery")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if input_schema not in SUPPORTED_INPUT_SCHEMAS:
        raise ValueError(
            f"input_schema must be one of {SUPPORTED_INPUT_SCHEMAS}, got {input_schema!r}"
        )
    if profile == PROFILE_FAST_STOP_512 and method != "mastery":
        raise ValueError("fast_stop_512 requires method=mastery")
    stages = _stages_for_profile(profile)
    if gamma is None:
        gamma = foundation.FAST_STOP_GAMMA if profile == PROFILE_FAST_STOP_512 else 0.995
    elif profile == PROFILE_FAST_STOP_512 and gamma != foundation.FAST_STOP_GAMMA:
        raise ValueError("fast_stop_512 requires gamma=1.0")
    if profile == PROFILE_FAST_STOP_512 and gae_lambda != 0.95:
        raise ValueError("fast_stop_512 keeps the default gae_lambda=0.95")
    # SB3 rounds every learn call to a complete rollout: reject budgets that
    # could silently overshoot the protocol (128 steps * 4 environments).
    for name, value in (("budget", budget), ("block_steps", block_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 512 or value % 512:
            raise ValueError(f"{name} must be a positive multiple of 512")
    if method == "fixed" and budget % (512 * len(stages)):
        raise ValueError("fixed budget must divide into equal, complete stage rollouts")
    for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite in (0, 1]")
    # All arms share a fixed 60-second input AND termination contract. Historical
    # 20-second reports are not baselines for this experiment.
    cfg = Config(**_profile_config(profile, input_schema))
    paths = _sources()
    output.mkdir(parents=True)
    rows = []
    for name in paths:
        source, target = ROOT / name, output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": name, "snapshot_path": target.relative_to(output).as_posix(),
                     "sha256": foundation._sha256(source)})
    selection = {"stage_indices": list(range(len(stages))), "episodes_per_stage": 4,
                 "seed_banks": [list(range(89_000 + i * 100, 89_004 + i * 100)) for i in range(len(stages))],
                 "every_probes": 4, "also_on_promotion": True,
                 "criterion": "clean successes increase; damaged episodes, mean damage and collisions do not increase"}
    mastery = {"episodes_per_stage": 16, "min_clean_success_rate": 0.8,
               "required_consecutive_passes": 2, "probe_interval_timesteps": block_steps,
               "max_damaged_episodes": 0, "max_mean_damage": 0.0,
               "max_total_collisions": 0,
               "active_blocks_per_replay": 3,
               "seed_banks": {stage["name"]: list(range(88_100 + i * 100, 88_116 + i * 100))
                              for i, stage in enumerate(stages)}}
    protocol = {
        "schema": SCHEMA, "method": method, "profile": profile,
        "input_schema": input_schema, "config": asdict(cfg),
        "stages": stages, "budget": budget, "block_steps": block_steps,
        "initialization": {"seed": 96_001, "num_timesteps": 0,
                           "source": "new random RSL-RL actor/critic and optimizer; no trained checkpoint",
                           "cross_task_checkpoint_policy": "weights transfer with reset optimizer and local timestep count; parent contract/hash recorded"},
        "hyperparameters": {"n_steps": 128, "batch_size": 128, "n_epochs": 5,
                            "learning_rate": 3e-4, "gamma": gamma, "gae_lambda": gae_lambda,
                            "log_std_init": -2.0, "target_kl": 0.03,
                            **({"lstm_hidden_size": 512} if profile == PROFILE_FAST_STOP_512 else {})},
        "mastery": mastery,
        "selection": selection,
        "evaluation_environment_seed": 90_003,
        "training_seed_base": 97_000,
        "dataset_role": "development for curriculum and checkpoint selection, never held-out",
        "blind_seeds": {"range": [83_000, 83_999], "used": False},
        "source_snapshot": {"files": rows}, "runtime": foundation._package_versions(),
        "isaaclab_commit": foundation._isaac_commit(), "robot_geometry": foundation._robot_geometry(),
        "action_config": foundation._action_config(),
        "claim_limit": "Software validation and training completion do not establish a learned capability improvement.",
    }
    architecture = _policy_architecture(profile, input_schema)
    if architecture is not None:
        protocol["policy_architecture"] = architecture
    if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        protocol["observation_contract"] = _observation_contract(input_schema, protocol["config"])
    if profile == PROFILE_FAST_STOP_512:
        global_profile = foundation.PROFILE_RANDOM_ROUGH_FAST_STOP
        global_config = foundation._profile_config(global_profile)
        protocol["goal_objective_contract"] = foundation._fast_stop_objective_contract(protocol["config"])
        protocol["profile_physics_contract"] = foundation._profile_physics_contract(
            global_profile, global_config, foundation._robot_geometry())
        protocol["transfer_probe"] = _fast_stop_transfer_probe(input_schema)
        protocol["selection"].update(
            stage_indices=[],
            stage_names=[member["name"] for member in protocol["transfer_probe"]["members"]],
            seed_banks=[member["seeds"] for member in protocol["transfer_probe"]["members"]],
        )
        protocol["claim_limit"] = protocol["transfer_probe"]["claim_limit"]
        _validate_profile_protocol(protocol)
    elif input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        _validate_profile_protocol(protocol)
    from isaac_field.policy_contract import stamp_backend_identity

    stamp_backend_identity(protocol)
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {"status": "prepared", "num_timesteps": 0,
          "protocol_sha256": foundation._sha256(output / "protocol.json")})
    return {"status": "prepared", "output": str(output), "method": method, "profile": profile,
            "input_schema": input_schema,
            "source_files": len(rows), "training_started": False}


def preflight(output: Path) -> dict:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA or status.get("status") != "prepared":
        raise ValueError("only a prepared, never-launched mastery experiment may launch")
    if status["protocol_sha256"] != foundation._sha256(output / "protocol.json"):
        raise ValueError("prepared protocol hash changed")
    from isaac_field.policy_contract import require_backend_identity

    require_backend_identity(protocol)
    _validate_profile_protocol(protocol)
    verify_sources(output, protocol)
    if protocol["action_config"] != foundation._action_config():
        raise ValueError("current action contract differs from the protocol")
    if (protocol["runtime"] != foundation._package_versions()
            or protocol["isaaclab_commit"] != foundation._isaac_commit()
            or protocol["robot_geometry"] != json.loads(json.dumps(foundation._robot_geometry()))):
        raise ValueError("runtime dependencies, upstream revision or robot geometry changed after prepare")
    return protocol


def verify_sources(output: Path, protocol: dict) -> None:
    rows = protocol["source_snapshot"]["files"]
    if {row["path"] for row in rows} != set(_sources()):
        raise ValueError("source inventory differs from prepared experiment")
    for row in rows:
        if (foundation._sha256(ROOT / row["path"]) != row["sha256"]
                or foundation._sha256(output / row["snapshot_path"]) != row["sha256"]):
            raise ValueError(f"frozen source mismatch: {row['path']}")


def _contract(report: dict) -> dict:
    return {key: report.get(key) for key in CONTRACT_KEYS}


def _attach_probe(report: dict, *, purpose: str, stage: str) -> dict:
    report["probe"] = {"schema": "isaac-field-fixed-probe-v1", "purpose": purpose,
                       "probe_stage": stage, "seeds": report["seeds"], "config": _contract(report)}
    return report


def _open_env(output: Path, protocol: dict, stage: dict, *, training: bool, seed: int,
              task_config: dict | None = None):
    from isaac_field.runtime import create_env

    verify_sources(output, protocol)
    input_schema = protocol.get("input_schema", protocol["config"].get("input_schema", INPUT_SCHEMA))
    values = dict(task_config or protocol["config"])
    task_schema = values.get("input_schema", INPUT_SCHEMA)
    if task_schema != input_schema:
        raise ValueError(
            f"environment task input_schema {task_schema!r} differs from protocol {input_schema!r}"
        )
    cfg = replace(Config(**values), plants=stage["plants"])
    expected_task = asdict(cfg)
    physical_kwargs = {}
    if protocol.get("profile", PROFILE_LEGACY) == PROFILE_FAST_STOP_512:
        (front_guide_spec,
         body_stability_config) = foundation._rough_profile_components(
            foundation.PROFILE_RANDOM_ROUGH_FAST_STOP)
        physical_kwargs.update(front_guide_spec=front_guide_spec,
                               body_stability_config=body_stability_config)
    engine = create_env(cfg=cfg, mode=stage["mode"], num_envs=4 if training else 1,
                      seed=seed, auto_reset=training,
                      goal_distance=(None if stage["goal_distance"] is None
                                     else tuple(stage["goal_distance"])),
                      goal_angle=stage["goal_angle"], distance_cost=0.0,
                      rsl=True, **physical_kwargs)
    try:
        if input_schema == INPUT_SCHEMA:
            foundation._live_environment_contract(engine)
        else:
            foundation._live_environment_contract(engine, input_schema=input_schema)
        actual_distance = (None if engine.cfg.goal_distance is None
                           else list(engine.cfg.goal_distance))
        if (asdict(engine.field_cfg) != expected_task or engine.cfg.mode != stage["mode"]
                or actual_distance != stage["goal_distance"]
                or engine.cfg.goal_angle != stage["goal_angle"]
                or engine.cfg.distance_cost != 0.0 or engine.num_envs != (4 if training else 1)):
            raise ValueError("actual environment task or sampling differs from the frozen stage")
        if protocol.get("profile", PROFILE_LEGACY) == PROFILE_FAST_STOP_512:
            if input_schema == INPUT_SCHEMA:
                foundation._validate_rough_profile_environment(
                    engine, protocol["profile_physics_contract"])
            else:
                foundation._validate_rough_profile_environment(
                    engine, protocol["profile_physics_contract"], input_schema=input_schema)
        if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
            actual_contract = _observation_contract_projection(engine.observation_config)
            if actual_contract != protocol["observation_contract"]:
                raise ValueError(
                    "runtime observation contract differs from the frozen absolute-goal protocol"
                )
        _append(output / "runtime_assets.jsonl", {"training": training, "seed": seed,
                "stage": stage, "assets": foundation._runtime_assets(engine)})
    except BaseException:
        engine.close()
        raise
    return engine


def _evaluate_stage(output: Path, protocol: dict, checkpoint: Path, stage: dict,
                    seeds: list[int], *, environment_seed: int | None = None,
                    task_config: dict | None = None) -> dict:
    from isaac_field.env import SingleFieldAdapter
    from isaac_field.learning import evaluate_model

    engine = _open_env(output, protocol, stage, training=False,
                       seed=(protocol["evaluation_environment_seed"] if environment_seed is None
                             else environment_seed), task_config=task_config)
    try:
        evaluate_kwargs = {}
        if protocol.get("input_schema", INPUT_SCHEMA) == ABSOLUTE_GOAL_INPUT_SCHEMA:
            evaluate_kwargs.update(
                input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                expected_observation_config=engine.observation_config,
            )
        report = evaluate_model(SingleFieldAdapter(engine), checkpoint, seeds,
            goal_distance=(None if stage["goal_distance"] is None
                           else tuple(stage["goal_distance"])), goal_angle=stage["goal_angle"],
            mode=stage["mode"], record_initial_conditions=True, device=protocol["device"],
            **evaluate_kwargs)
        report.setdefault("input_schema", protocol.get("input_schema", INPUT_SCHEMA))
        report["evaluation_config"] = {"task": asdict(engine.field_cfg),
            "environment_seed": (protocol["evaluation_environment_seed"] if environment_seed is None
                                  else environment_seed), "stage": stage}
    finally:
        engine.close()
    return report


def _global_probe(output: Path, protocol: dict, checkpoint: Path) -> dict:
    if protocol.get("profile", PROFILE_LEGACY) == PROFILE_FAST_STOP_512:
        transfer = protocol["transfer_probe"]
        reports = []
        for member in transfer["members"]:
            stage = {key: member[key] for key in
                     ("name", "plants", "goal_distance", "goal_angle")}
            stage["mode"] = "goto"
            reports.append(_evaluate_stage(
                output, protocol, checkpoint, stage, member["seeds"],
                environment_seed=member["environment_seed"],
                task_config=transfer["environment_config"],
            ))
        result = {key: reports[0].get(key) for key in CONTRACT_KEYS}
        result["input_schema"] = reports[0].get("input_schema", protocol.get("input_schema", INPUT_SCHEMA))
        if (protocol.get("input_schema") == ABSOLUTE_GOAL_INPUT_SCHEMA
                and reports[0].get("input_schema") != ABSOLUTE_GOAL_INPUT_SCHEMA):
            raise ValueError("absolute-goal transfer report is missing its input_schema identity")
        for report in reports[1:]:
            for key in ("action_config", "drive_config", "input_schema", "observation_config", "damage_config", "damage_metric"):
                actual = report.get(key, protocol.get("input_schema", INPUT_SCHEMA) if key == "input_schema" else None)
                if actual != result[key]:
                    raise ValueError(f"global rough transfer suite has inconsistent {key}")
        result.update(schema="isaac-field-development-suite-v1", checkpoint=str(checkpoint),
                      goal_distance=None, goal_angle_rad=None, mode="goto_suite",
                      seeds=[seed for report in reports for seed in report["seeds"]],
                      rows=[row for report in reports for row in report["rows"]],
                      evaluation_config={"suite": transfer["purpose"],
                          "members": [_contract(report) for report in reports],
                          "claim_limit": transfer["claim_limit"]})
        # The controller schema reserves global_suite for every checkpoint-selection
        # probe. The concrete rough-transfer purpose and member contracts remain in
        # evaluation_config so this label does not misstate the evaluated scenes.
        return _attach_probe(result, purpose="best_checkpoint", stage="global_suite")
    reports = [_evaluate_stage(output, protocol, checkpoint, protocol["stages"][index], seeds)
               for index, seeds in zip(protocol["selection"]["stage_indices"],
                                       protocol["selection"]["seed_banks"])]
    # Record the actual contract of every suite member; never relabel a single
    # scene's report as if all suite episodes had that same scene.
    result = {key: reports[0].get(key) for key in CONTRACT_KEYS}
    result["input_schema"] = reports[0].get("input_schema", protocol.get("input_schema", INPUT_SCHEMA))
    if (protocol.get("input_schema") == ABSOLUTE_GOAL_INPUT_SCHEMA
            and reports[0].get("input_schema") != ABSOLUTE_GOAL_INPUT_SCHEMA):
        raise ValueError("absolute-goal suite report is missing its input_schema identity")
    for report in reports[1:]:
        for key in ("action_config", "drive_config", "input_schema", "observation_config", "damage_config", "damage_metric"):
            actual = report.get(key, protocol.get("input_schema", INPUT_SCHEMA) if key == "input_schema" else None)
            if actual != result[key]:
                raise ValueError(f"global suite has inconsistent {key}")
    result.update(schema="isaac-field-development-suite-v1", checkpoint=str(checkpoint),
                  goal_distance=None, goal_angle_rad=None, mode="goto_suite",
                  seeds=[seed for report in reports for seed in report["seeds"]],
                  rows=[row for report in reports for row in report["rows"]],
                  evaluation_config={"members": [_contract(report) for report in reports]})
    return _attach_probe(result, purpose="best_checkpoint", stage="global_suite")


def _mastery_specs(protocol: dict, baseline_report: dict) -> dict[str, ProbeSpec]:
    """Declare expected contracts; these are not simulated probe results."""
    specs = {}
    for stage in protocol["stages"]:
        contract = {key: baseline_report[key] for key in CONTRACT_KEYS}
        cfg = replace(Config(**protocol["config"]), plants=stage["plants"])
        contract.update(goal_distance=stage["goal_distance"], goal_angle_rad=stage["goal_angle"],
                        mode=stage["mode"], evaluation_config={"task": asdict(cfg),
                            "environment_seed": protocol["evaluation_environment_seed"], "stage": stage})
        specs[stage["name"]] = ProbeSpec(protocol["mastery"]["seed_banks"][stage["name"]], contract)
    return specs


def _controller(protocol: dict, mastery_specs: dict, baseline_report: dict) -> CurriculumController:
    return CurriculumController(protocol["stages"], CurriculumControlConfig(
        total_timesteps=protocol["budget"],
        probe_interval_timesteps=protocol["mastery"]["probe_interval_timesteps"],
        training_seeds=[protocol["training_seed_base"] + i for i in range(protocol["budget"] // 512 + 1)],
        mastery_probes=mastery_specs,
        best_probe=ProbeSpec(baseline_report["seeds"], _contract(baseline_report)),
        required_consecutive_passes=protocol["mastery"]["required_consecutive_passes"],
        min_clean_success_rate=protocol["mastery"]["min_clean_success_rate"],
        minimum_stage_timesteps=min(protocol["budget"], protocol["block_steps"]
            + protocol["mastery"]["required_consecutive_passes"]
            * protocol["mastery"]["probe_interval_timesteps"]),
        active_blocks_per_replay=protocol["mastery"]["active_blocks_per_replay"],
    ))


def _policy_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_best(protocol: dict, candidate: dict, best: dict) -> dict:
    """Prevent gains on easy scenes from concealing regressions on plant scenes."""
    comparison = compare_best_checkpoints(candidate, best)
    if not comparison["comparable"]:
        return comparison
    candidates = {row["seed"]: row for row in candidate["rows"]}
    incumbents = {row["seed"]: row for row in best["rows"]}
    members = []

    def metrics(rows):
        return {"clean_successes": sum(row["success"] and row["damage"] == 0 and row["collisions"] == 0 for row in rows),
                "damaged_episodes": sum(row["damage"] > 0 for row in rows),
                "total_damage": sum(row["damage"] for row in rows),
                "total_collisions": sum(row["collisions"] for row in rows)}

    if protocol.get("profile") == PROFILE_FAST_STOP_512:
        selection_members = list(zip(protocol["selection"]["stage_names"],
                                     protocol["selection"]["seed_banks"]))
    else:
        selection_members = [
            (protocol["stages"][index]["name"], seeds)
            for index, seeds in zip(protocol["selection"]["stage_indices"],
                                    protocol["selection"]["seed_banks"])
        ]
    for stage_name, seeds in selection_members:
        current = metrics([candidates[seed] for seed in seeds])
        previous = metrics([incumbents[seed] for seed in seeds])
        safe = (current["clean_successes"] >= previous["clean_successes"]
                and all(current[key] <= previous[key] for key in
                        ("damaged_episodes", "total_damage", "total_collisions")))
        members.append({"stage": stage_name, "candidate": current,
                        "best": previous, "non_regression": safe})
    comparison["member_checks"] = members
    if not all(member["non_regression"] for member in members):
        comparison.update(replace_best=False, assessment="suite_member_regression",
                          selected_checkpoint=best.get("checkpoint"))
    return comparison


def launch(output: Path, *, device: str = "cuda:0") -> dict:
    from isaac_field.runtime import launch_app

    with launch_app(device=device, headless=True):
        return _launch_workflow(output, device=device)


def _launch_workflow(output: Path, *, device: str = "cuda:0") -> dict:
    protocol = preflight(output)
    # Runtime-only device is recorded separately, never mutate the frozen file.
    protocol["device"] = device
    consumed = 0
    status_base = {"protocol_sha256": foundation._sha256(output / "protocol.json"), "device": device}

    def status(phase, **extra):
        _json(output / "run_status.json", {**status_base, "status": "running", "phase": phase,
              "num_timesteps": consumed, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **extra})

    status("initializing")
    try:
        from isaac_field.learning import train_stage
        from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

        stages = protocol["stages"]
        hp = protocol["hyperparameters"]
        baseline = output / "untrained_baseline.pt"
        env = _open_env(output, protocol, stages[0], training=True,
                        seed=protocol["initialization"]["seed"])
        try:
            input_schema = protocol.get("input_schema", INPUT_SCHEMA)
            observation_config = env.observation_config
            if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA and not _complete_observation_config(observation_config):
                raise ValueError("absolute-goal initialization requires a complete observation_config")
            if not _complete_observation_config(observation_config):
                observation_config = None
            initialized = train_stage(
                env, None, 0, baseline,
                seed=protocol["initialization"]["seed"], device=device,
                input_schema=input_schema, **hp,
            )
            contract = initialized.model.field_policy_contract
            _json(output / "initialization.json", {
                "num_timesteps": 0,
                "checkpoint_sha256": foundation._sha256(baseline),
                "initialization_seed": protocol["initialization"]["seed"],
                "field_policy_contract": contract,
                "optimizer_state_present": "optimizer_state_dict" in load_rsl_checkpoint_payload(baseline),
                "backend_identity": {key: contract[key] for key in
                                     ("simulator", "physics_backend", "learning_backend")},
            })
        finally:
            env.close()
        status("initial_probes")
        best_report = _global_probe(output, protocol, baseline)
        _json(output / "probes" / "global_000000.json", best_report)
        mastery_specs = _mastery_specs(protocol, best_report)
        _json(output / "probe_contracts.json", {"status": "declared_expected_contracts_not_results",
              "mastery": {name: asdict(spec) for name, spec in mastery_specs.items()},
              "global": _contract(best_report)})
        controller = _controller(protocol, mastery_specs, best_report)
        controller.validate_best_probe(best_report)
        checkpoint, best_checkpoint = baseline, baseline
        blocks, probe_count = [], 0
        while consumed < protocol["budget"]:
            verify_sources(output, protocol)
            if protocol["method"] == "mastery":
                plan = controller.next_training_plan(max_timesteps=protocol["block_steps"])
                stage = next(item for item in stages if item["name"] == plan["sampled_stage"])
                block_steps = plan["timesteps"]
            else:
                stage_size = protocol["budget"] // len(stages)
                index = min(consumed // stage_size, len(stages) - 1)
                stage = stages[index]
                block_steps = min(protocol["block_steps"], (index + 1) * stage_size - consumed)
                plan = {"sampled_stage": stage["name"], "timesteps": block_steps,
                        "sampling": {key: stage[key] for key in ("mode", "goal_distance", "goal_angle", "plants")}}
            block_index = len(blocks)
            status("training", sampled_stage=stage["name"], planned_timesteps=block_steps)
            env = _open_env(output, protocol, stage, training=True,
                            seed=protocol["training_seed_base"] + block_index)
            source_checkpoint_sha256 = foundation._sha256(checkpoint)
            try:
                source_payload = load_rsl_checkpoint_payload(checkpoint, map_location="cpu")
                source_steps = int(source_payload.get("num_timesteps", 0))
                weights_transfer = source_payload["field_policy_contract"].get("task_config") != asdict(env.field_cfg)
                result = train_stage(env, checkpoint, block_steps,
                    output / "checkpoints" / f"steps_{consumed + block_steps:07d}_{stage['name']}.pt",
                    seed=protocol["training_seed_base"] + block_index,
                    device=device, input_schema=input_schema,
                    weights_transfer=weights_transfer, **hp)
                actual_sampling = {"mode": env.cfg.mode,
                    "goal_distance": list(env.cfg.goal_distance), "goal_angle": env.cfg.goal_angle,
                    "plants": env.field_cfg.plants}
            finally:
                env.close()
            expected_segment_steps = (block_steps if weights_transfer else source_steps + block_steps)
            if result.trained_timesteps != block_steps or result.total_timesteps != expected_segment_steps:
                raise RuntimeError("actual RSL timestep count differs from the stage's continuation/transfer contract")
            consumed += result.trained_timesteps
            checkpoint = result.checkpoint
            if protocol["method"] == "mastery":
                decision = controller.record_training(result.trained_timesteps,
                    sampled_stage=stage["name"], actual_sampling=actual_sampling)
            else:
                decision = {"probe_due": True, "active_stage": stage, "active_stage_index": stages.index(stage)}
            blocks.append({"plan": plan, "actual_sampling": actual_sampling,
                           "checkpoint": str(checkpoint), "sha256": foundation._sha256(checkpoint),
                           "num_timesteps": consumed,
                           "checkpoint_segment_timesteps": result.total_timesteps,
                           "weights_transfer": weights_transfer,
                           "parent_checkpoint_sha256": source_checkpoint_sha256,
                           "optimizer_state": "reinitialized" if weights_transfer else "resumed"})
            promoted = False
            did_probe = decision["probe_due"]
            if decision["probe_due"]:
                status("mastery_probe")
                active = next(item for item in stages if item["name"] == decision["active_stage"]["name"])
                report = _evaluate_stage(output, protocol, checkpoint, active,
                                         protocol["mastery"]["seed_banks"][active["name"]])
                report = _attach_probe(report, purpose="mastery", stage=active["name"])
                _json(output / "probes" / f"mastery_{consumed:07d}_{active['name']}.json", report)
                probe_count += 1
                if protocol["method"] == "mastery":
                    decision = controller.record_probe(report)
                    promoted = decision["last_event"]["promotion"] == "advanced"
                _append(output / "decisions.jsonl", {"num_timesteps": consumed, "decision": decision})
            final_mastered = protocol["method"] == "mastery" and controller.decision()["mastered"]
            if promoted or final_mastered or (did_probe and probe_count % protocol["selection"]["every_probes"] == 0) or consumed == protocol["budget"]:
                status("global_selection_probe")
                report = _global_probe(output, protocol, checkpoint)
                controller.validate_best_probe(report)
                comparison = _select_best(protocol, report, best_report)
                if comparison["replace_best"]:
                    best_report, best_checkpoint = report, checkpoint
                _json(output / "probes" / f"global_{consumed:07d}.json", report)
                _append(output / "selection.jsonl", {"num_timesteps": consumed, **comparison})
            _json(output / "controller_state.json", controller.state_dict())
            _json(output / "checkpoint_manifest.json", {"baseline": str(baseline), "blocks": blocks,
                  "latest_checkpoint": str(checkpoint), "best_checkpoint": str(best_checkpoint)})
            # Stop a fully mastered final stage without spending the remaining
            # budget. A curriculum that runs out of budget stays incomplete.
            if protocol["method"] == "mastery" and controller.decision()["mastered"]:
                break
        summary = {"schema": SCHEMA, "method": protocol["method"],
            "profile": protocol.get("profile", PROFILE_LEGACY), "num_timesteps": consumed,
            "simulator": protocol["simulator"], "physics_backend": protocol["physics_backend"],
            "learning_backend": protocol["learning_backend"],
            "input_schema": protocol.get("input_schema", INPUT_SCHEMA),
            "budget": protocol["budget"], "latest_checkpoint": str(checkpoint),
            "best_checkpoint": str(best_checkpoint), "best_checkpoint_sha256": foundation._sha256(best_checkpoint),
            "controller": controller.decision() if protocol["method"] == "mastery" else None,
            "training_finished": True, "capability_improvement_established": False,
            "claim_limit": protocol["claim_limit"]}
        _json(output / "summary.json", summary)
        status("finished", status="completed", exit_code=0)
        return summary
    except BaseException as exc:
        status("failed", status="failed", exit_code=1, exception=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        # This is the Python outcome, not an observed operating-system exit.
        # run_isaac_mastery.sh independently writes the actual child exit code.
        exit_status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
        _json(output / "python_outcome.json", {"intended_exit_code": exit_status.get("exit_code", 1),
              "num_timesteps": consumed, "status": exit_status["status"],
              "timestamp_utc": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("fixed", "mastery"), default="mastery")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_LEGACY)
    parser.add_argument("--input-schema", choices=SUPPORTED_INPUT_SCHEMAS, default=INPUT_SCHEMA)
    parser.add_argument("--budget", type=int, default=196_608)
    parser.add_argument("--block-steps", type=int, default=8_192)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.output, method=args.method, profile=args.profile, budget=args.budget,
                         block_steps=args.block_steps, gamma=args.gamma, gae_lambda=args.gae_lambda,
                         input_schema=args.input_schema)
    elif args.preflight:
        protocol = preflight(args.output)
        result = {"status": "preflight_passed", "method": protocol["method"],
                  "profile": protocol.get("profile", PROFILE_LEGACY),
                  "input_schema": protocol.get("input_schema", INPUT_SCHEMA),
                  "training_started": False}
    else:
        result = launch(args.output, device=args.device)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
