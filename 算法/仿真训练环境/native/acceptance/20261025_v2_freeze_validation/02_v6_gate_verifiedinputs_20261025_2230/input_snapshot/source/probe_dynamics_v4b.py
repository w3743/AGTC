"""Short multi-world eager probe for the external Newton MuJoCo Warp candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


MJW_STATE_FIELDS = (
    "time", "qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied",
    "xfrc_applied", "qacc", "xpos", "xquat", "geom_xpos", "geom_xmat",
)


def _arr(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _finite_fields(backend) -> dict[str, bool]:
    checks = {
        "state_0.body_q": _arr(backend.state_0.body_q),
        "state_0.body_qd": _arr(backend.state_0.body_qd),
        "state_1.body_q": _arr(backend.state_1.body_q),
        "state_1.body_qd": _arr(backend.state_1.body_qd),
        "joint_q": _arr(backend.joint_q),
        "joint_qd": _arr(backend.joint_qd),
        "control.joint_target_q": _arr(backend.control.joint_target_q),
        "control.joint_target_qd": _arr(backend.control.joint_target_qd),
        "mjw.qpos": _arr(backend.solver.mjw_data.qpos),
        "mjw.qvel": _arr(backend.solver.mjw_data.qvel),
        "mjw.qacc": _arr(backend.solver.mjw_data.qacc),
        "mjw.qacc_warmstart": _arr(backend.solver.mjw_data.qacc_warmstart),
        "mjw.ctrl": _arr(backend.solver.mjw_data.ctrl),
    }
    return {name: bool(np.isfinite(value).all()) for name, value in checks.items()}


def _contact_metrics(backend) -> dict:
    data = backend.solver.mjw_data
    count = int(_arr(data.nacon).reshape(-1)[0])
    worldids = _arr(data.contact.worldid).reshape(-1)[:count].astype(np.int64, copy=False)
    distances = _arr(data.contact.dist).reshape(-1)[:count].astype(np.float64, copy=False)
    counts = np.bincount(worldids, minlength=backend.nworld) if count else np.zeros(backend.nworld, dtype=np.int64)
    min_dist = np.full(backend.nworld, np.inf, dtype=np.float64)
    max_pen = np.zeros(backend.nworld, dtype=np.float64)
    for world in range(backend.nworld):
        selected = distances[worldids == world]
        if selected.size:
            min_dist[world] = float(np.min(selected))
            max_pen[world] = max(0.0, float(-np.min(selected)))
    overflow = _arr(data.overflow).reshape(-1).astype(np.int64, copy=False)
    return {
        "nacon_total": count,
        "contact_count_by_world": counts.astype(int).tolist(),
        "min_contact_distance_m_by_world": [None if not np.isfinite(x) else float(x) for x in min_dist],
        "max_penetration_m_by_world": max_pen.tolist(),
        "overflow_raw_by_world": overflow.tolist(),
        "overflow_names_by_world": [
            [name for value, name in ((1, "NEFC"), (2, "NJMAX_NNZ"), (4, "BROADPHASE"), (8, "NARROWPHASE"), (16, "CCD"), (32, "HFIELD"), (64, "CONTACT_MATCH"), (128, "NVMAX"), (256, "EPA_HORIZON"), (512, "ITERATIONS"), (1024, "LS_ITERATIONS"), (2048, "TACTILE")) if int(bits) & value]
            for bits in overflow
        ],
    }


def _mimic_diagnostics(backend) -> dict[str, np.ndarray | list[str]]:
    """Capture each mimic pair from stored Newton, IK-derived, and actual MJWarp qpos."""
    from physics_backend import GUIDE_RATIO, WHEELS_PER_SIDE

    state_q = _arr(backend.state_0.joint_q).astype(np.float64, copy=True)
    backend._newton.eval_ik(backend.model, backend.state_0, backend.joint_q, backend.joint_qd)
    ik_q = _arr(backend.joint_q).astype(np.float64, copy=True)
    q_starts = _arr(backend.model.joint_q_start).astype(np.int64, copy=False)
    qpos = _arr(backend.solver.mjw_data.qpos).astype(np.float64, copy=False)
    mj_q_starts = _arr(backend.solver.mj_q_start).astype(np.int64, copy=False)
    labels: list[str] = []
    relations_by_world: list[list[tuple[str, int, int, float, float]]] = []
    for world in range(backend.nworld):
        relations = []
        for side in ("left", "right"):
            joints = backend.wheel_joint_idx[side][world]
            leader = int(joints[0])
            for index, follower_value in enumerate(joints[1:], start=1):
                relations.append((f"{side}_wheel_{index}", leader, int(follower_value), 0.0, 1.0))
        for side, guide_joint in zip(("left", "right"), backend.guide_joint_idx[world]):
            leader = int(backend.wheel_joint_idx[side][world][0])
            relations.append((f"{side}_guide", leader, int(guide_joint), 0.0, float(GUIDE_RATIO)))
        if len(relations) != 2 * (WHEELS_PER_SIDE - 1) + 2:
            raise AssertionError(f"world {world}: unexpected mimic relation count {len(relations)}")
        if not labels:
            labels = [item[0] for item in relations]
        relations_by_world.append(relations)

    pair_values = {
        source: np.zeros((backend.nworld, len(labels), 2), dtype=np.float64)
        for source in ("state_joint_q", "eval_ik_q", "mjw_qpos")
    }
    for world, relations in enumerate(relations_by_world):
        local_joints = [int(j) for j in backend.world_joint_idx[world]]
        local_by_global = {joint: local for local, joint in enumerate(local_joints)}
        for row, (_, leader, follower, _, _) in enumerate(relations):
            for source, q in (("state_joint_q", state_q), ("eval_ik_q", ik_q)):
                pair_values[source][world, row] = (
                    q[q_starts[leader]], q[q_starts[follower]]
                )
            leader_local = local_by_global[leader]
            follower_local = local_by_global[follower]
            leader_qpos = int(mj_q_starts[leader_local])
            follower_qpos = int(mj_q_starts[follower_local])
            if leader_qpos < 0 or follower_qpos < 0:
                raise RuntimeError(f"world {world} mimic pair {labels[row]} has no MuJoCo qpos mapping")
            pair_values["mjw_qpos"][world, row] = (
                qpos[world, leader_qpos], qpos[world, follower_qpos]
            )

    result: dict[str, np.ndarray | list[str]] = {"mimic_pair_labels": labels}
    for source, values in pair_values.items():
        raw = np.zeros((backend.nworld, len(labels)), dtype=np.float64)
        for world, relations in enumerate(relations_by_world):
            for row, (_, _, _, offset, coefficient) in enumerate(relations):
                leader_value, follower_value = values[world, row]
                raw[world, row] = follower_value - (offset + coefficient * leader_value)
        folded = (raw + np.pi) % (2.0 * np.pi) - np.pi
        result[f"mimic_pair_values_{source}"] = values
        result[f"mimic_raw_residual_{source}"] = raw
        result[f"mimic_folded_residual_{source}"] = folded
    return result


def _world_control_indices(backend, world: int) -> tuple[np.ndarray, np.ndarray]:
    qd = sorted(
        dof
        for side in ("left", "right")
        for dof in backend.wheel_target_dof[side][world]
    )
    q = sorted(backend.gimbal_target_q_coord[world])
    return np.asarray(qd, dtype=np.int64), np.asarray(q, dtype=np.int64)


def _capture_unaffected(backend, worlds: tuple[int, ...]) -> dict:
    result = {}
    for state_name in ("state_0", "state_1"):
        state = getattr(backend, state_name)
        for field in ("body_q", "body_qd"):
            rows = _arr(getattr(state, field))
            for world in worlds:
                indices = np.asarray(backend.world_body_idx[world], dtype=np.int64)
                result[f"{state_name}.{field}.world{world}"] = np.ascontiguousarray(rows[indices]).tobytes()
        state_q = _arr(state.joint_q)
        state_qd = _arr(state.joint_qd)
        for world in worlds:
            slices = backend.world_joint_slices[world]
            qvals = np.concatenate([state_q[q0:q1] for q0, q1, _, _ in slices])
            qdvals = np.concatenate([state_qd[qd0:qd1] for _, _, qd0, qd1 in slices])
            result[f"{state_name}.joint_q.world{world}"] = np.ascontiguousarray(qvals).tobytes()
            result[f"{state_name}.joint_qd.world{world}"] = np.ascontiguousarray(qdvals).tobytes()
    target_qd = _arr(backend.control.joint_target_qd)
    target_q = _arr(backend.control.joint_target_q)
    for world in worlds:
        qd_indices, q_indices = _world_control_indices(backend, world)
        result[f"control.target_qd.world{world}"] = np.ascontiguousarray(target_qd[qd_indices]).tobytes()
        result[f"control.target_q.world{world}"] = np.ascontiguousarray(target_q[q_indices]).tobytes()
    data = backend.solver.mjw_data
    for field in MJW_STATE_FIELDS:
        value = getattr(data, field, None)
        if value is None:
            continue
        arr = _arr(value)
        if arr.ndim == 0 or arr.shape[0] != backend.nworld:
            continue
        for world in worlds:
            result[f"mjw.{field}.world{world}"] = np.ascontiguousarray(arr[world]).tobytes()
    return result


def _git_commit(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None
    except Exception:
        return None


def _source_manifest(paths: list[Path], repo: Path) -> list[dict]:
    rows = []
    for path in paths:
        raw = path.read_bytes()
        try:
            label = path.relative_to(repo).as_posix()
        except ValueError:
            label = str(path)
        rows.append({"path": label, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-proto", default=os.environ.get(
        "AGTC_NATIVE_PROTO",
        "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto",
    ))
    parser.add_argument("--nworld", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.nworld != 3:
        parser.error("this first dynamics probe is intentionally limited to exactly 3 worlds")
    if args.cpu_preflight:
        if args.device != "cpu" or not 1 <= args.steps <= 3:
            parser.error("CPU preflight requires --device cpu and 1 to 3 policy steps")
    elif not 100 <= args.steps <= 200:
        parser.error("GPU steps must be between 100 and 200 policy steps")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    proto = Path(args.repo_proto).resolve(strict=True)
    native = proto.parent
    repo = native.parents[2]
    sys.path.insert(0, str(proto))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import mujoco
    import mujoco_warp
    import newton
    import warp as wp
    from mjw_backend_candidate_v4 import NewtonMuJoCoWarpCandidate, SOLVER_CONFIG

    start_time = time.perf_counter()
    backend = NewtonMuJoCoWarpCandidate(
        nworld=3, dt=1e-3, decimation=50, seed=17, device=args.device,
    )
    backend._capture_solver_trace = True
    device = str(backend.model.device)
    if args.cpu_preflight and device != "cpu":
        raise RuntimeError(f"CPU preflight unexpectedly finalized on {device}")
    if not args.cpu_preflight and not device.startswith("cuda"):
        raise RuntimeError(f"requested GPU device but model finalized on {device}")
    device_inventory = {
        "newton_model": str(backend.model.device),
        "state_body_q": str(backend.state_0.body_q.device),
        "control_joint_target_qd": str(backend.control.joint_target_qd.device),
        "mjwarp_model_geom_gap": str(backend.solver.mjw_model.geom_gap.device),
        "mjwarp_data_qpos": str(backend.solver.mjw_data.qpos.device),
        "mjwarp_contact_dist": str(backend.solver.mjw_data.contact.dist.device),
    }
    wrong_devices = {name: value for name, value in device_inventory.items() if value != device}
    if args.cpu_preflight:
        # The probe records raw overflow bits in JSON; disable repeated warning text only.
        backend.solver.mjw_model.opt.warn_overflow = 0
    if wrong_devices:
        raise RuntimeError(f"candidate arrays are not co-located on {device}: {wrong_devices}")

    starts = np.asarray([
        [0.0, 0.0, 0.0, 0.025],
        [1.4, 0.25, 0.3, 0.025],
        [-1.5, -0.3, -0.25, 0.025],
    ], dtype=np.float64)
    backend.reset(env_ids=[0, 1, 2], poses=starts)
    errors = []
    checks = {}

    # Read actual actuator force after an intentionally over-commanded single physics step.
    target_qd = _arr(backend.control.joint_target_qd).copy()
    target_q = _arr(backend.control.joint_target_q).copy()
    for world in range(3):
        sign = (-1.0, 1.0, -1.0)[world]
        for side in ("left", "right"):
            for dof in backend.wheel_target_dof[side][world]:
                target_qd[dof] = sign * 1000.0
        for coord in backend.gimbal_target_q_coord[world]:
            target_q[coord] = sign * 10.0
    backend.control.joint_target_qd.assign(target_qd)
    backend.control.joint_target_q.assign(target_q)
    backend.state_0.clear_forces()
    backend.solver.step(
        backend.state_0, backend.state_1, backend.control, None, backend.dt,
        observables=backend._actuator_observables,
    )
    backend.state_0, backend.state_1 = backend.state_1, backend.state_0
    backend._snapshot_actuator_effort()
    limit_sample = backend.read()
    wheel_by_world = np.max(np.abs(limit_sample["effort"]), axis=1)
    gimbal_by_world = np.max(np.abs(limit_sample["gimbal_effort"]), axis=1)
    checks["wheel_qfrc_actuator_max_abs_nm_by_world"] = wheel_by_world.tolist()
    checks["gimbal_qfrc_actuator_max_abs_nm_by_world"] = gimbal_by_world.tolist()
    if not np.isfinite(wheel_by_world).all() or np.any(wheel_by_world > 2.0001):
        errors.append(f"wheel qfrc actuator limit check failed: {wheel_by_world.tolist()}")
    if not np.isfinite(gimbal_by_world).all() or np.any(gimbal_by_world > 1.0001):
        errors.append(f"gimbal qfrc actuator limit check failed: {gimbal_by_world.tolist()}")

    # Reset after the force-limit probe and verify per-world observation values diverge.
    backend.reset(env_ids=[0, 1, 2], poses=starts)
    rng = np.random.default_rng(20261025)
    history = {name: [] for name in (
        "pos", "wheel_qd", "gimbal_q", "gimbal_qd", "effort", "gimbal_effort",
        "contact_count_by_world", "max_penetration_m_by_world", "overflow_raw_by_world",
        "mimic_pair_values_state_joint_q", "mimic_raw_residual_state_joint_q",
        "mimic_folded_residual_state_joint_q", "mimic_pair_values_eval_ik_q",
        "mimic_raw_residual_eval_ik_q", "mimic_folded_residual_eval_ik_q",
        "mimic_pair_values_mjw_qpos", "mimic_raw_residual_mjw_qpos",
        "mimic_folded_residual_mjw_qpos", "solver_niter_by_substep",
        "solver_overflow_raw_by_substep", "solver_nefc_by_substep",
        "solver_max_equality_efc_pos_abs_by_substep", "base_z_by_world", "finite_all",
    )}
    fixed = np.asarray([
        [0.45, 0.45, 0.00, 0.00],
        [-0.35, -0.35, 0.25, -0.15],
        [0.20, -0.10, -0.10, 0.25],
    ], dtype=np.float64)
    mimic_pair_labels = None
    for step in range(args.steps):
        if step < 20:
            action = np.zeros((3, 4), dtype=np.float64)
            phase = "zero_settle"
        elif step < 70:
            action = fixed
            phase = "fixed"
        else:
            action = rng.uniform(-0.35, 0.35, size=(3, 4))
            phase = "deterministic_random"
        backend.step(action)
        obs = backend.read()
        contacts = _contact_metrics(backend)
        finite = _finite_fields(backend)
        mimic = _mimic_diagnostics(backend)
        if mimic_pair_labels is None:
            mimic_pair_labels = list(mimic["mimic_pair_labels"])
        history["pos"].append(obs["pos"].copy())
        history["wheel_qd"].append(obs["wheel_qd"].copy())
        history["gimbal_q"].append(obs["gimbal_q"].copy())
        history["gimbal_qd"].append(obs["gimbal_qd"].copy())
        history["effort"].append(obs["effort"].copy())
        history["gimbal_effort"].append(obs["gimbal_effort"].copy())
        history["contact_count_by_world"].append(contacts["contact_count_by_world"])
        history["max_penetration_m_by_world"].append(contacts["max_penetration_m_by_world"])
        history["overflow_raw_by_world"].append(contacts["overflow_raw_by_world"])
        for name in (
            "mimic_pair_values_state_joint_q", "mimic_raw_residual_state_joint_q",
            "mimic_folded_residual_state_joint_q", "mimic_pair_values_eval_ik_q",
            "mimic_raw_residual_eval_ik_q", "mimic_folded_residual_eval_ik_q",
            "mimic_pair_values_mjw_qpos", "mimic_raw_residual_mjw_qpos",
            "mimic_folded_residual_mjw_qpos",
        ):
            history[name].append(np.asarray(mimic[name]).copy())
        trace = backend.last_solver_substep_trace or {}
        for source, target in (
            ("solver_niter", "solver_niter_by_substep"),
            ("overflow_raw", "solver_overflow_raw_by_substep"),
            ("nefc", "solver_nefc_by_substep"),
            ("max_equality_efc_pos_abs", "solver_max_equality_efc_pos_abs_by_substep"),
        ):
            if source not in trace:
                raise RuntimeError(f"missing solver trace array {source}")
            history[target].append(np.asarray(trace[source]).copy())
        history["base_z_by_world"].append(obs["pos"][:, 2].copy())
        history["finite_all"].append(all(finite.values()))
        bad_fields = [name for name, passed in finite.items() if not passed]
        if bad_fields:
            errors.append(f"step {step + 1} ({phase}) non-finite fields: {bad_fields}")
            break

    trajectory = {name: np.asarray(values) for name, values in history.items()}
    steps_completed = int(trajectory["pos"].shape[0])
    if steps_completed != args.steps:
        errors.append(f"completed {steps_completed} of {args.steps} policy steps")
    if steps_completed:
        if not args.cpu_preflight:
            if not np.any(trajectory["contact_count_by_world"] > 0):
                errors.append("no internal MuJoCo Warp contacts were observed during the trajectory")
            raw_all = trajectory["solver_overflow_raw_by_substep"].astype(np.int64, copy=False)
            capacity_mask = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 2048
            if np.any(raw_all & capacity_mask):
                errors.append("MuJoCo Warp reported a contact/constraint/sensor capacity overflow bit")
            if np.any(trajectory["max_penetration_m_by_world"] > 0.02):
                errors.append("contact penetration exceeded 0.02 m")
            final_wheel = trajectory["wheel_qd"][-1]
            final_gimbal = trajectory["gimbal_q"][-1]
            if np.max(np.ptp(final_wheel, axis=0)) < 1e-3:
                errors.append("wheel readback did not distinguish the three worlds")
            if np.max(np.ptp(final_gimbal, axis=0)) < 1e-3:
                errors.append("gimbal readback did not distinguish the three worlds")

    # After trajectories, reset only world 1 and require all untouched world state and
    # MuJoCo Warp warmstart/control buffers to remain byte-identical.
    untouched = (0, 2)
    before_reset = _capture_unaffected(backend, untouched)
    world1_pose = np.asarray([[2.8, 0.9, 0.7, 0.06]], dtype=np.float64)
    backend.reset(env_ids=[1], poses=world1_pose)
    after_reset = _capture_unaffected(backend, untouched)
    changed = sorted(key for key in before_reset if before_reset[key] != after_reset.get(key))
    checks["partial_reset_untouched_world_fields_checked"] = sorted(before_reset)
    checks["partial_reset_untouched_world_changed_fields"] = changed
    checks["partial_reset_untouched_worlds_bitwise_unchanged"] = not changed
    if changed:
        errors.append(f"partial reset changed untouched world data: {changed}")
    # Save reset coordinates before read(), which intentionally overwrites the
    # backend's separate IK scratch arrays and need not preserve exact bit patterns.
    source_q = _arr(backend.joint_q).copy()
    source_qd = _arr(backend.joint_qd).copy()
    selected_sync = True
    for state_name in ("state_0", "state_1"):
        state = getattr(backend, state_name)
        state_q = _arr(state.joint_q)
        state_qd = _arr(state.joint_qd)
        for q0, q1, qd0, qd1 in backend.world_joint_slices[1]:
            selected_sync &= np.array_equal(state_q[q0:q1], source_q[q0:q1])
            selected_sync &= np.array_equal(state_qd[qd0:qd1], source_qd[qd0:qd1])
    checks["partial_reset_selected_reduced_joint_state_synced_in_both_states"] = bool(selected_sync)
    if not selected_sync:
        errors.append("selected reset-world reduced joint state was not copied into both ping-pong states")

    post_reset = backend.read()
    if not np.allclose(post_reset["pos"][1], [2.8, 0.9, 0.21], atol=2e-5):
        errors.append(f"selected world pose did not survive reset: {post_reset['pos'][1].tolist()}")
    if not np.array_equal(post_reset["effort_observation_valid"], [True, False, True]):
        errors.append(f"effort sample validity did not reset per world: {post_reset['effort_observation_valid'].tolist()}")

    first_substep_trace = backend._solver_substep()
    after_substep = backend.read()
    checks["reset_first_substep_position_world1"] = after_substep["pos"][1].tolist()
    checks["reset_first_substep_solver_trace"] = {
        key: np.asarray(value).tolist() for key, value in first_substep_trace.items()
    }
    if not np.allclose(after_substep["pos"][1], [2.8, 0.9, 0.21], atol=2e-3):
        errors.append(f"selected pose did not survive the first physics substep: {after_substep['pos'][1].tolist()}")
    backend.step(np.zeros((3, 4), dtype=np.float64))
    after_policy = backend.read()
    checks["reset_first_policy_position_world1"] = after_policy["pos"][1].tolist()
    if not np.allclose(after_policy["pos"][1, :2], [2.8, 0.9], atol=0.03):
        errors.append(f"selected reset pose drifted in the first policy step: {after_policy['pos'][1].tolist()}")
    if not np.isfinite(after_policy["pos"][1]).all():
        errors.append("selected reset pose became non-finite during the first policy step")

    identity = backend.identity()
    exported_topology = {
        "mimic_relations_per_world": identity["mimic_relations_per_world"],
        "exported_actuators": identity["solver"]["exported_actuators"],
        "exported_equalities": identity["solver"]["exported_equalities"],
    }
    if exported_topology != {
        "mimic_relations_per_world": 14,
        "exported_actuators": 16,
        "exported_equalities": 14,
    }:
        errors.append(f"Newton/MuJoCo mimic and actuator topology mismatch: {exported_topology}")
    runtime_diagnostics = {
        "mjwarp_naconmax_total": int(backend.solver.mjw_data.naconmax),
        "mjwarp_nconmax_requested_per_world": int(SOLVER_CONFIG["nconmax"]),
        "mjwarp_nconmax_actual_per_world": int(backend.solver.mjw_data.naconmax) // backend.nworld,
        "mjwarp_njmax_actual_per_world": int(backend.solver.mjw_data.njmax),
    }
    overflow_bit_sample_counts = {}
    solver_diagnostics = {}
    if steps_completed:
        raw_overflow = trajectory["solver_overflow_raw_by_substep"].astype(np.int64, copy=False)
        for bit, name in ((1, "NEFC"), (2, "NJMAX_NNZ"), (4, "BROADPHASE"), (8, "NARROWPHASE"), (16, "CCD"), (32, "HFIELD"), (64, "CONTACT_MATCH"), (128, "NVMAX"), (256, "EPA_HORIZON"), (512, "ITERATIONS"), (1024, "LS_ITERATIONS"), (2048, "TACTILE")):
            overflow_bit_sample_counts[name] = np.count_nonzero((raw_overflow & bit) != 0, axis=(0, 1)).astype(int).tolist()
        niter = trajectory["solver_niter_by_substep"].astype(np.float64, copy=False)
        eq_residual = trajectory["solver_max_equality_efc_pos_abs_by_substep"].astype(np.float64, copy=False)
        solver_diagnostics = {
            "policy_substeps_recorded": int(raw_overflow.shape[0] * raw_overflow.shape[1]),
            "solver_niter_min_by_world": np.min(niter, axis=(0, 1)).astype(int).tolist(),
            "solver_niter_mean_by_world": np.mean(niter, axis=(0, 1)).tolist(),
            "solver_niter_max_by_world": np.max(niter, axis=(0, 1)).astype(int).tolist(),
            "nefc_max_by_world": np.max(trajectory["solver_nefc_by_substep"], axis=(0, 1)).astype(int).tolist(),
            "max_equality_efc_pos_abs_by_world": np.max(eq_residual, axis=(0, 1)).tolist(),
            "outer_iterations_limit_samples_by_world": overflow_bit_sample_counts["ITERATIONS"],
            "line_search_limit_samples_by_world": overflow_bit_sample_counts["LS_ITERATIONS"],
            "capacity_overflow_samples_by_world": np.count_nonzero((raw_overflow & (1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 2048)) != 0, axis=(0, 1)).astype(int).tolist(),
            "line_search_limit_semantics": "MuJoCo Warp sets LS_ITERATIONS when its inner line search did not converge within configured ls_iterations; this is not a memory/contact capacity flag. solver_niter and equality efc.pos residual are recorded separately.",
        }
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("mjw_backend_candidate_v4.py").resolve(),
        proto / "newton_backend.py",
        proto / "physics_backend.py",
    ]
    result = {
        "schema": "agtc-newton-mujoco-warp-three-world-probe-v2",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - start_time,
        "requested_device": args.device,
        "actual_model_device": device,
        "mjwarp_model_device": device_inventory["mjwarp_model_geom_gap"],
        "device_inventory": device_inventory,
        "versions": {
            "newton": getattr(newton, "__version__", "unknown"),
            "warp": str(wp.config.version),
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(mujoco_warp.__version__),
        },
        "policy_steps_requested": args.steps,
        "policy_steps_completed": steps_completed,
        "physics_substeps_completed": steps_completed * backend.decimation,
        "phase_steps": {"zero_settle": min(20, steps_completed), "fixed": max(0, min(50, steps_completed - 20)), "deterministic_random": max(0, steps_completed - 70)},
        "identity": identity,
        "runtime_diagnostics": runtime_diagnostics,
        "exported_topology": exported_topology,
        "checks": checks,
        "mimic_pair_labels": mimic_pair_labels,
        "mimic_residual_note": "folded residual is a modulo-2pi branch diagnostic; for non-unit guide ratios the raw affine equality residual remains authoritative",
        "solver_diagnostics": solver_diagnostics,
        "trajectory_summary": {
            "max_contact_count_by_world": np.max(trajectory["contact_count_by_world"], axis=0).astype(int).tolist() if steps_completed else [],
            "max_penetration_m_by_world": np.max(trajectory["max_penetration_m_by_world"], axis=0).tolist() if steps_completed else [],
            "max_mimic_raw_residual_state_joint_q_by_world": np.max(np.abs(trajectory["mimic_raw_residual_state_joint_q"]), axis=(0, 2)).tolist() if steps_completed else [],
            "max_mimic_raw_residual_eval_ik_q_by_world": np.max(np.abs(trajectory["mimic_raw_residual_eval_ik_q"]), axis=(0, 2)).tolist() if steps_completed else [],
            "max_mimic_raw_residual_mjw_qpos_by_world": np.max(np.abs(trajectory["mimic_raw_residual_mjw_qpos"]), axis=(0, 2)).tolist() if steps_completed else [],
            "max_mimic_folded_residual_eval_ik_q_by_world": np.max(np.abs(trajectory["mimic_folded_residual_eval_ik_q"]), axis=(0, 2)).tolist() if steps_completed else [],
            "max_base_z_by_world_m": np.max(trajectory["base_z_by_world"], axis=0).tolist() if steps_completed else [],
            "min_base_z_by_world_m": np.min(trajectory["base_z_by_world"], axis=0).tolist() if steps_completed else [],
            "overflow_nonzero_policy_samples_by_world": np.count_nonzero(trajectory["overflow_raw_by_world"], axis=0).astype(int).tolist() if steps_completed else [],
            "capacity_overflow_physics_substeps_by_world": solver_diagnostics.get("capacity_overflow_samples_by_world", []),
            "max_overflow_raw_physics_substep_by_world": np.max(trajectory["solver_overflow_raw_by_substep"], axis=(0, 1)).astype(int).tolist() if steps_completed else [],
            "overflow_bit_sample_counts_by_world": overflow_bit_sample_counts,
            "all_finite_samples": bool(np.asarray(trajectory["finite_all"]).all()) if steps_completed else False,
        },
        "source_files": _source_manifest(source_files, repo),
        "git_commit_readonly": _git_commit(repo),
        "output_dir": str(out_dir),
        "gpu_used": not args.cpu_preflight,
    }
    npz_path = out_dir / "trajectory.npz"
    np.savez_compressed(npz_path, **trajectory)
    result["trajectory_npz"] = str(npz_path)
    result["trajectory_npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())