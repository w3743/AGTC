#!/usr/bin/env python3
"""Fixed-action front-guide/step contact probe; no policy or baseline comparison."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = ROOT / "results/isaac_field_diagnosis/front_guide_step_probe_20261020"
OUTPUT_DIR = EXPERIMENT_DIR / "run_001"
SAMPLE_PATH = OUTPUT_DIR / "policy_step_samples.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
SEED = 80003
RESET_SEED = 87001
START = [2.0, 2.0, 0.0]
GOAL = [6.0, 2.0]
ACTION = [0.05, 0.05, 0.0, 0.0]
POLICY_STEPS = 200
POLICY_DT_S = 0.05
STEP_SIZE_M = [0.8, 2.0, 0.06]
STEP_CENTER_M = [3.0, 2.0, 0.03]


def as_numpy(value: Any) -> np.ndarray:
    """Convert Isaac/torch arrays without attempting implicit CUDA conversion."""
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape_inventory(model, newton_module, guide_body_names: set[str]) -> dict[str, Any]:
    shape_labels = [str(label) for label in model.shape_label]
    body_labels = [str(label) for label in model.body_label]
    shape_body = as_numpy(model.shape_body).reshape(-1).astype(np.int64)
    shape_flags = as_numpy(model.shape_flags).reshape(-1).astype(np.int64)
    shape_types = as_numpy(model.shape_type).reshape(-1).astype(np.int64)
    step_ids = [index for index, label in enumerate(shape_labels)
                if "DiagnosticStep" in label]

    guide_ids = []
    for shape_id, label in enumerate(shape_labels):
        body_id = int(shape_body[shape_id]) if shape_id < len(shape_body) else -1
        body_name = (body_labels[body_id].rsplit("/", 1)[-1]
                     if 0 <= body_id < len(body_labels) else "")
        if any(name in label for name in guide_body_names) or body_name in guide_body_names:
            guide_ids.append(shape_id)

    def material_values(field: str, indices: list[int]) -> dict[str, float | str]:
        raw = getattr(model, field, None)
        if raw is None:
            return {str(index): "unavailable" for index in indices}
        values = as_numpy(raw).reshape(-1)
        return {str(index): float(values[index]) if index < len(values) else "index_out_of_range"
                for index in indices}

    def shape_row(shape_id: int) -> dict[str, Any]:
        body_id = int(shape_body[shape_id]) if shape_id < len(shape_body) else -1
        body_label = body_labels[body_id] if 0 <= body_id < len(body_labels) else None
        raw_geo = int(shape_types[shape_id]) if shape_id < len(shape_types) else None
        try:
            geo_type = newton_module.GeoType(raw_geo).name if raw_geo is not None else "unavailable"
        except Exception:
            geo_type = raw_geo
        flags = int(shape_flags[shape_id]) if shape_id < len(shape_flags) else 0
        return {
            "shape_id": shape_id,
            "shape_label": shape_labels[shape_id],
            "shape_type": geo_type,
            "shape_flags": flags,
            "collide_shapes": bool(flags & int(newton_module.ShapeFlags.COLLIDE_SHAPES)),
            "shape_body_id": body_id,
            "shape_body_label": body_label,
            "shape_material_mu": material_values("shape_material_mu", [shape_id]).get(str(shape_id)),
        }

    guide_body_names_seen = sorted({
        record["shape_body_label"].rsplit("/", 1)[-1]
        for record in (shape_row(shape_id) for shape_id in guide_ids)
        if record["shape_body_label"] is not None
    })
    return {
        "shape_count": len(shape_labels),
        "body_count": len(body_labels),
        "diagnostic_step_shape_ids": step_ids,
        "front_guide_shape_ids": guide_ids,
        "front_guide_shape_body_names_seen": guide_body_names_seen,
        "diagnostic_step_shapes": [shape_row(index) for index in step_ids],
        "front_guide_shapes": [shape_row(index) for index in guide_ids],
        "step_mu_by_shape_id": material_values("shape_material_mu", step_ids),
        "guide_mu_by_shape_id": material_values("shape_material_mu", guide_ids),
        "relevant_shape_inventory": [shape_row(index) for index in sorted(set(step_ids + guide_ids))],
    }


def contact_snapshot(NewtonManager, model, shape_info: dict[str, Any]) -> dict[str, Any]:
    """Return public Newton pair identities only; never infer per-pair forces."""
    contacts = NewtonManager.get_contacts()
    if contacts is None:
        return {"available": False, "reason": "NewtonManager.get_contacts() returned None"}

    count_values = as_numpy(contacts.rigid_contact_count).reshape(-1)
    contact_count = int(count_values[0]) if count_values.size else 0
    shape0 = as_numpy(contacts.rigid_contact_shape0).reshape(-1)
    shape1 = as_numpy(contacts.rigid_contact_shape1).reshape(-1)
    labels = [str(label) for label in model.shape_label]
    body_labels = [str(label) for label in model.body_label]
    shape_body = as_numpy(model.shape_body).reshape(-1).astype(np.int64)
    step_ids = set(shape_info["diagnostic_step_shape_ids"])
    guide_ids = set(shape_info["front_guide_shape_ids"])
    available_records = min(contact_count, shape0.size, shape1.size)

    def shape_record(shape_id: int) -> dict[str, Any]:
        label = labels[shape_id] if 0 <= shape_id < len(labels) else None
        body_id = int(shape_body[shape_id]) if 0 <= shape_id < len(shape_body) else -1
        body_label = body_labels[body_id] if 0 <= body_id < len(body_labels) else None
        return {
            "shape_id": shape_id,
            "shape_label": label,
            "shape_body_id": body_id,
            "shape_body_label": body_label,
        }

    relevant_pairs = []
    guide_step_pairs = []
    for record_index in range(available_records):
        id0, id1 = int(shape0[record_index]), int(shape1[record_index])
        touches_step = id0 in step_ids or id1 in step_ids
        touches_guide = id0 in guide_ids or id1 in guide_ids
        if not (touches_step or touches_guide):
            continue
        pair = {
            "contact_record_index": record_index,
            "shape0": shape_record(id0),
            "shape1": shape_record(id1),
            "touches_diagnostic_step": touches_step,
            "touches_front_guide": touches_guide,
        }
        relevant_pairs.append(pair)
        if touches_step and touches_guide:
            guide_step_pairs.append(pair)

    return {
        "available": True,
        "recorded_contact_count": contact_count,
        "identity_records_available": available_records,
        "shape_array_lengths": [int(shape0.size), int(shape1.size)],
        "relevant_shape_pairs": relevant_pairs,
        "guide_step_pairs": guide_step_pairs,
        "semantics": (
            "Newton public contact identity snapshot from the final physics substep of this policy step; "
            "no per-pair force/normal or force direction is inferred."
        ),
    }


def main() -> int:
    if os.environ.get("RUN_ISAAC_PHYSICS") != "1":
        raise SystemExit("Prepared only; set RUN_ISAAC_PHYSICS=1 after the native test completes")
    if OUTPUT_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing probe output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    env = None
    summary: dict[str, Any] = {
        "status": "initializing",
        "probe": str(Path(__file__).resolve()),
        "sampling_contract": {
            "policy_steps": POLICY_STEPS,
            "physics_dt_s": 0.001,
            "decimation": 50,
            "policy_dt_s": POLICY_DT_S,
            "fixed_normalized_action_left_right_gimbal_yaw_pitch": ACTION,
            "baseline_comparison": False,
            "checkpoint_loaded": False,
            "claim_scope": "single fixed-action Newton contact diagnostic, not navigation competence or a continuous-track model",
        },
        "task_contract": {
            "mode": "goto",
            "plants": 0,
            "terrain_enabled": False,
            "terrain_amplitude_m": 0.0,
            "rock_density_per_m2": 0.0,
            "start_xy_yaw": START,
            "goal_xy": GOAL,
            "goal_distance_config": None,
            "max_time_s": 12.0,
            "auto_reset": False,
        },
        "obstacle_contract": {
            "primitive": "Isaac Lab USD Cuboid via AssetBaseCfg + sim_utils.CuboidCfg",
            "prim_path": "{ENV_REGEX_NS}/DiagnosticStep",
            "size_xyz_m": STEP_SIZE_M,
            "center_xyz_m": STEP_CENTER_M,
            "front_edge_x_m": STEP_CENTER_M[0] - STEP_SIZE_M[0] / 2.0,
            "collision_material_override": None,
            "collision_note": "Uses the actual configured Newton default shape material; value is recorded from the imported model.",
        },
        "source_sha256": {},
        "error": None,
    }
    try:
        import torch
        import newton
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab_newton.physics import NewtonManager

        from isaac_field.assets import FrontGuideSpec, ROBOT_SPEC
        from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
        from isaac_field.task import Config

        guide_spec = FrontGuideSpec()
        task_cfg = Config(
            size=12.0,
            plants=0,
            terrain_enabled=False,
            terrain_amplitude=0.0,
            rock_density=0.0,
            validate_scene=False,
            max_time=12.0,
        )
        cfg = FieldEnvCfg(task=task_cfg, mode="goto", auto_reset=False, seed=SEED)
        cfg.scene.num_envs = 1
        cfg.sim.dt = 0.001
        cfg.decimation = 50
        # FieldEnvCfg keeps immutable physics specs as plain mappings; the live
        # IsaacFieldEnv restores its public FrontGuideSpec object from this data.
        cfg.front_guide_spec = asdict(guide_spec)
        cfg.body_stability_config = None
        cfg.scene.diagnostic_step = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DiagnosticStep",
            spawn=sim_utils.CuboidCfg(
                size=tuple(STEP_SIZE_M),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(STEP_CENTER_M)),
        )

        env = IsaacFieldEnv(cfg)
        reset_observation, reset_info = env.reset(
            seed=RESET_SEED,
            options={
                "mode": "goto",
                "start": START,
                "goal": GOAL,
                "random_start": False,
            },
        )
        guide_wheels = env.front_guide_spec.wheel_specs(ROBOT_SPEC)
        guide_joint_names = [wheel.joint_name for wheel in guide_wheels]
        guide_body_names = {wheel.link_name for wheel in guide_wheels}
        joint_names = list(env.rover.joint_names)
        missing_guide_joints = [name for name in guide_joint_names if name not in joint_names]
        if missing_guide_joints:
            raise RuntimeError(f"front guide joints were not imported into the articulation: {missing_guide_joints}")
        guide_joint_ids = [joint_names.index(name) for name in guide_joint_names]
        drive_joint_ids = env._wheel_ids.detach().cpu().numpy().astype(np.int64).tolist()
        applied_joint_ids = sorted(set(drive_joint_ids + guide_joint_ids))
        model = env.sim.physics_manager.get_model()
        shapes = shape_inventory(model, newton, guide_body_names)

        default_shape_cfg = cfg.sim.physics.default_shape_cfg
        physics_contact = env.drive_config["physics_contact"]
        if (not math.isclose(float(cfg.sim.dt), 0.001, abs_tol=1e-12)
                or int(cfg.decimation) != 50
                or not math.isclose(float(env.step_dt), POLICY_DT_S, abs_tol=1e-12)
                or not math.isclose(float(cfg.drive_velocity_gain), 0.04, abs_tol=1e-12)
                or not math.isclose(
                    float(cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s), 0.1, abs_tol=1e-12
                )
                or not math.isclose(float(cfg.drive_effort_limit), 2.0, abs_tol=1e-12)
                or cfg.body_stability_config is not None):
            raise RuntimeError("probe initialization changed the approved timing/drive/reward defaults")
        expected_action = torch.tensor([ACTION], dtype=torch.float32, device=env.device)
        cfg_scene_assets = {
            "diagnostic_step": {
                "prim_path": cfg.scene.diagnostic_step.prim_path,
                "size_xyz_m": list(cfg.scene.diagnostic_step.spawn.size),
                "initial_center_xyz_m": list(cfg.scene.diagnostic_step.init_state.pos),
                "collidable_schema_configured": cfg.scene.diagnostic_step.spawn.collision_props is not None,
                "static_asset_class_type": str(cfg.scene.diagnostic_step.class_type),
            },
            "robot_usd_path": str(Path(cfg.scene.robot.spawn.usd_path).resolve()),
            "field_usd_path": str(Path(cfg.scene.field.spawn.usd_path).resolve()),
        }
        summary.update({
            "status": "sampling",
            "reset_observation_keys": sorted(reset_observation.keys()) if isinstance(reset_observation, dict) else None,
            "reset_info": reset_info,
            "runtime": {
                "device": str(env.device),
                "physics_dt_s": float(cfg.sim.dt),
                "decimation": int(cfg.decimation),
                "policy_dt_s": float(env.step_dt),
                "max_episode_steps": int(env.max_episode_length),
                "body_stability_config": None,
                "front_guide_spec": asdict(env.front_guide_spec),
                "guide_joint_names": guide_joint_names,
                "guide_body_names": sorted(guide_body_names),
                "drive_joint_count": len(drive_joint_ids),
                "drive_interface": env.drive_config.get("interface"),
                "drive_velocity_gain": float(cfg.drive_velocity_gain),
                "drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s": float(
                    cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s
                ),
                "drive_effort_limit_nm": float(cfg.drive_effort_limit),
                "newton_contact_backend": physics_contact.get("contact_generation"),
                "configured_default_shape_material": {
                    "mu": float(default_shape_cfg.mu),
                    "margin_m": float(default_shape_cfg.margin),
                    "gap_m": float(default_shape_cfg.gap),
                },
                "physics_contact_metadata": physics_contact,
            },
            "scene_assets": cfg_scene_assets,
            "robot_asset_sha256": sha256_file(Path(cfg_scene_assets["robot_usd_path"])),
            "field_asset_sha256": sha256_file(Path(cfg_scene_assets["field_usd_path"])),
            "newton_shape_inventory": shapes,
            "sampling_timing_note": (
                "Per-step pose and velocity are read after env.step at policy time; Newton contact identity "
                "is the public contact buffer snapshot after the final physics substep. Their different sampling "
                "points are retained and are not combined to infer force direction, sustained support, or cause."
            ),
        })
        source_paths = (
            Path(__file__).resolve(),
            ROOT / "isaac_field/env.py",
            ROOT / "isaac_field/assets.py",
            ROOT / "isaac_field/newton_drive.py",
            ROOT / "isaac_field/task.py",
        )
        summary["source_sha256"] = {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in source_paths if path.is_file()
        }

        guide_step_contact_frames = 0
        guide_any_contact_frames = 0
        step_any_contact_frames = 0
        finite_row_count = 0
        terminated_steps = []
        truncated_steps = []
        with SAMPLE_PATH.open("x", encoding="utf-8") as stream:
            for step_index in range(1, POLICY_STEPS + 1):
                _obs, reward, terminated, truncated, _info = env.step(expected_action)
                task = env.tasks[0]
                robot_state = as_numpy(task.robot).astype(np.float64).reshape(-1)
                body_pose = as_numpy(task.body_pose).astype(np.float64).reshape(-1)
                if robot_state.size < 5 or body_pose.size < 3:
                    raise RuntimeError(
                        f"task telemetry has unexpected shape: robot={robot_state.shape}, body_pose={body_pose.shape}"
                    )
                x, y, _yaw, forward_speed, yaw_rate = robot_state[:5]
                z, pitch, roll = body_pose[:3]
                joint_pos = as_numpy(env.rover.data.joint_pos)[0]
                joint_vel = as_numpy(env.rover.data.joint_vel)[0]
                applied_effort = as_numpy(env.rover.actuators.applied_effort)[0]
                reward_value = float(as_numpy(reward).reshape(-1)[0])
                if max(guide_joint_ids + applied_joint_ids, default=-1) >= min(
                    len(joint_pos), len(joint_vel), len(applied_effort)
                ):
                    raise RuntimeError(
                        "joint telemetry arrays do not cover all guide/drive articulation indices: "
                        f"joint_pos={len(joint_pos)}, joint_vel={len(joint_vel)}, "
                        f"applied_effort={len(applied_effort)}, requested={applied_joint_ids}"
                    )
                contact = contact_snapshot(NewtonManager, model, shapes)
                if contact.get("guide_step_pairs"):
                    guide_step_contact_frames += 1
                if any(pair["touches_front_guide"] for pair in contact.get("relevant_shape_pairs", [])):
                    guide_any_contact_frames += 1
                if any(pair["touches_diagnostic_step"] for pair in contact.get("relevant_shape_pairs", [])):
                    step_any_contact_frames += 1

                numeric_values = np.concatenate((
                    np.asarray([x, y, z, pitch, roll, forward_speed, yaw_rate], dtype=np.float64),
                    joint_pos[guide_joint_ids].astype(np.float64),
                    joint_vel[guide_joint_ids].astype(np.float64),
                    applied_effort[applied_joint_ids].astype(np.float64),
                    np.asarray([reward_value], dtype=np.float64),
                ))
                finite = bool(np.isfinite(numeric_values).all())
                if finite:
                    finite_row_count += 1
                terminated_value = bool(as_numpy(terminated).reshape(-1)[0])
                truncated_value = bool(as_numpy(truncated).reshape(-1)[0])
                if terminated_value:
                    terminated_steps.append(step_index)
                if truncated_value:
                    truncated_steps.append(step_index)
                row = {
                    "policy_step": step_index,
                    "simulation_time_s": float(step_index * env.step_dt),
                    "normalized_action_left_right_gimbal_yaw_pitch": ACTION,
                    "body_state": {
                        "x_m": float(x), "y_m": float(y), "z_m": float(z),
                        "pitch_rad": float(pitch), "roll_rad": float(roll),
                        "forward_speed_mps": float(forward_speed), "yaw_rate_rad_s": float(yaw_rate),
                    },
                    "front_guide_joints": {
                        name: {
                            "position_rad": float(joint_pos[joint_id]),
                            "velocity_rad_s": float(joint_vel[joint_id]),
                        }
                        for name, joint_id in zip(guide_joint_names, guide_joint_ids)
                    },
                    "actual_applied_effort_nm_by_joint": {
                        joint_names[joint_id]: float(applied_effort[joint_id])
                        for joint_id in applied_joint_ids
                    },
                    "reward": reward_value,
                    "terminated": terminated_value,
                    "truncated": truncated_value,
                    "finite": finite,
                    "contact_snapshot": contact,
                    "contact_sampling_point": "Newton final substep buffer; this policy-step post-state row",
                }
                stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()

        step_ids = shapes["diagnostic_step_shape_ids"]
        guide_ids = shapes["front_guide_shape_ids"]
        collidable_step_ids = [row["shape_id"] for row in shapes["diagnostic_step_shapes"]
                               if row["collide_shapes"]]
        summary["sampling_result"] = {
            "sample_count": POLICY_STEPS,
            "finite_row_count": finite_row_count,
            "terminated_policy_steps": terminated_steps,
            "truncated_policy_steps": truncated_steps,
            "guide_step_contact_policy_step_snapshots": guide_step_contact_frames,
            "guide_any_contact_policy_step_snapshots": guide_any_contact_frames,
            "step_any_contact_policy_step_snapshots": step_any_contact_frames,
            "contact_snapshot_semantics": "policy-step snapshots with a pair identity; not continuous contact duration",
            "checks": {
                "step_shape_imported_by_newton": bool(step_ids),
                "step_shape_has_collision_flag": bool(step_ids) and len(collidable_step_ids) == len(step_ids),
                "both_front_guide_shapes_imported_by_newton": guide_body_names <= set(
                    shapes["front_guide_shape_body_names_seen"]
                ),
                "all_200_rows_finite": finite_row_count == POLICY_STEPS,
                "guide_step_contact_observed": guide_step_contact_frames > 0,
            },
        }
        checks = summary["sampling_result"]["checks"]
        summary["status"] = "passed" if all(checks.values()) else "not_passed"
        return 0
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                summary["close_error"] = f"{type(close_exc).__name__}: {close_exc}"
                summary["status"] = "error"
        summary["finished_utc"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        if not SUMMARY_PATH.exists():
            with SUMMARY_PATH.open("x", encoding="utf-8") as stream:
                json.dump(json_safe(summary), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
