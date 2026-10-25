from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

PROTO = Path("/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")


def arr(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def world_control_indices(backend, world):
    qd = sorted(
        int(dof)
        for side in ("left", "right")
        for dof in backend.wheel_target_dof[side][world]
    )
    q = sorted(int(coord) for coord in backend.gimbal_target_q_coord[world])
    return np.asarray(qd, dtype=np.int64), np.asarray(q, dtype=np.int64)


def capture_worlds(backend, worlds):
    captured = {}
    for state_name in ("state_0", "state_1"):
        state = getattr(backend, state_name)
        for field in ("body_q", "body_qd"):
            data = arr(getattr(state, field))
            for world in worlds:
                captured[f"{state_name}.{field}.{world}"] = np.ascontiguousarray(
                    data[backend.world_body_idx[world]]
                ).tobytes()
        q = arr(state.joint_q)
        qd = arr(state.joint_qd)
        for world in worlds:
            qparts, qdparts = [], []
            for q0, q1, qd0, qd1 in backend.world_joint_slices[world]:
                qparts.append(q[q0:q1])
                qdparts.append(qd[qd0:qd1])
            captured[f"{state_name}.joint_q.{world}"] = np.ascontiguousarray(
                np.concatenate(qparts)
            ).tobytes()
            captured[f"{state_name}.joint_qd.{world}"] = np.ascontiguousarray(
                np.concatenate(qdparts)
            ).tobytes()
    target_q, target_qd = arr(backend.control.joint_target_q), arr(backend.control.joint_target_qd)
    for world in worlds:
        qd_idx, q_idx = world_control_indices(backend, world)
        captured[f"control.q.{world}"] = np.ascontiguousarray(target_q[q_idx]).tobytes()
        captured[f"control.qd.{world}"] = np.ascontiguousarray(target_qd[qd_idx]).tobytes()
    data = backend.solver.mjw_data
    for field in (
        "time", "qpos", "qvel", "ctrl", "qacc_warmstart",
        "qfrc_applied", "xfrc_applied",
    ):
        value = getattr(data, field, None)
        if value is None:
            continue
        rows = arr(value)
        if rows.ndim < 1 or rows.shape[0] != backend.nworld:
            continue
        for world in worlds:
            captured[f"mjw.{field}.{world}"] = np.ascontiguousarray(rows[world]).tobytes()
    return captured


def contacts(backend):
    data = backend.solver.mjw_data
    count = int(arr(data.nacon).reshape(-1)[0])
    worldids = arr(data.contact.worldid).reshape(-1)[:count].astype(np.int64, copy=False)
    distances = arr(data.contact.dist).reshape(-1)[:count].astype(np.float64, copy=False)
    per_world = np.bincount(worldids, minlength=backend.nworld) if count else np.zeros(backend.nworld, dtype=np.int64)
    min_distance = np.full(backend.nworld, np.inf, dtype=np.float64)
    for world in range(backend.nworld):
        selected = distances[worldids == world]
        if selected.size:
            min_distance[world] = float(selected.min())
    return {
        "nacon_total": count,
        "contact_count_by_world": per_world.astype(int).tolist(),
        "min_distance_by_world": [None if not np.isfinite(x) else float(x) for x in min_distance],
        "max_penetration_m_by_world": [
            0.0 if not np.isfinite(x) else max(0.0, float(-x)) for x in min_distance
        ],
    }


def finite_state(backend):
    values = {
        "state0.body_q": arr(backend.state_0.body_q),
        "state0.body_qd": arr(backend.state_0.body_qd),
        "state1.body_q": arr(backend.state_1.body_q),
        "state1.body_qd": arr(backend.state_1.body_qd),
        "state0.joint_q": arr(backend.state_0.joint_q),
        "state0.joint_qd": arr(backend.state_0.joint_qd),
        "control.q": arr(backend.control.joint_target_q),
        "control.qd": arr(backend.control.joint_target_qd),
        "mjw.qpos": arr(backend.solver.mjw_data.qpos),
        "mjw.qvel": arr(backend.solver.mjw_data.qvel),
        "mjw.qacc": arr(backend.solver.mjw_data.qacc),
        "mjw.overflow": arr(backend.solver.mjw_data.overflow),
    }
    return {name: bool(np.isfinite(value).all()) for name, value in values.items()}


def source_hash(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "pid.txt").write_text(str(os.getpid()) + "\n", encoding="ascii")
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    errors = []
    report = {
        "schema": "agtc-newton-mujoco-warp-repo-smoke-v1",
        "status": "running",
        "started_utc": started,
        "pid": os.getpid(),
        "requested_device": "cuda:0",
        "policy_steps_requested": 20,
        "policy_steps_completed": 0,
        "gpu_used": True,
        "errors": errors,
    }
    save_json(out / "progress.json", report)

    import mujoco
    import mujoco_warp
    import newton
    import warp as wp
    from newton_mujoco_backend import NewtonMuJoCoWarpBackend
    from physics_backend import GUIDE_RATIO

    backend = NewtonMuJoCoWarpBackend(
        nworld=3, dt=1e-3, decimation=50, seed=20261025, device="cuda:0"
    )
    device = str(backend.model.device)
    if device != "cuda:0":
        raise RuntimeError(f"model finalized on {device}, expected cuda:0")
    inventory = {
        "model": str(backend.model.device),
        "state_body_q": str(backend.state_0.body_q.device),
        "state_joint_q": str(backend.state_0.joint_q.device),
        "control_joint_target_qd": str(backend.control.joint_target_qd.device),
        "mjwarp_geom_gap": str(backend.solver.mjw_model.geom_gap.device),
        "mjwarp_qpos": str(backend.solver.mjw_data.qpos.device),
        "mjwarp_contact_dist": str(backend.solver.mjw_data.contact.dist.device),
        "mjwarp_overflow": str(backend.solver.mjw_data.overflow.device),
    }
    if set(inventory.values()) != {device}:
        raise RuntimeError(f"device buffers are not co-located: {inventory}")
    backend.solver.mjw_model.opt.warn_overflow = 0  # Diagnostics are counted in-device below.
    identity = backend.identity()
    if identity["solver"]["actual_capacity_per_world"] != {"nconmax": 64, "njmax": 256}:
        raise RuntimeError(f"unexpected solver capacity: {identity['solver']['actual_capacity_per_world']}")

    # One deliberately excessive target checks the real QFRC_ACTUATOR output and limits.
    qd_target = arr(backend.control.joint_target_qd).copy()
    q_target = arr(backend.control.joint_target_q).copy()
    for world, sign in enumerate((1.0, -1.0, 1.0)):
        for side in ("left", "right"):
            for dof in backend.wheel_target_dof[side][world]:
                qd_target[dof] = sign * 1000.0
        for coord in backend.gimbal_target_q_coord[world]:
            q_target[coord] = sign * 10.0
    backend.control.joint_target_qd.assign(qd_target)
    backend.control.joint_target_q.assign(q_target)
    backend._solver_substep()
    backend._snapshot_actuator_effort()
    torque = backend.read()
    wheel_nm = np.max(np.abs(torque["effort"]), axis=1)
    gimbal_nm = np.max(np.abs(torque["gimbal_effort"]), axis=1)
    overflow_one_step = arr(backend.solver.mjw_data.overflow).astype(np.int64).reshape(-1)
    capacity_mask = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 2048
    if np.any(overflow_one_step & capacity_mask):
        errors.append(f"capacity overflow in actuator probe: {overflow_one_step.tolist()}")
    if not np.isfinite(wheel_nm).all() or np.any(wheel_nm > 2.0001):
        errors.append(f"wheel actuator torque exceeds 2 Nm: {wheel_nm.tolist()}")
    if not np.isfinite(gimbal_nm).all() or np.any(gimbal_nm > 1.0001):
        errors.append(f"gimbal actuator torque exceeds 1 Nm: {gimbal_nm.tolist()}")

    poses = np.asarray([
        [0.0, 0.0, 0.0, 0.025],
        [1.4, 0.25, 0.3, 0.025],
        [-1.5, -0.3, -0.25, 0.025],
    ], dtype=np.float64)
    backend.reset(poses=poses)
    rng = np.random.default_rng(20261025)
    trajectory = {
        "pos": [],
        "wheel_qd": [],
        "gimbal_q": [],
        "gimbal_qd": [],
        "effort": [],
        "gimbal_effort": [],
        "contact_count_by_world": [],
        "max_penetration_m_by_world": [],
        "capacity_overflow_substeps_by_world": [],
        "outer_iteration_limit_substeps_by_world": [],
        "line_search_limit_substeps_by_world": [],
    }
    fixed = np.asarray([
        [0.45, 0.45, 0.2, -0.1],
        [-0.35, -0.35, -0.25, 0.1],
        [0.2, -0.2, -0.1, 0.25],
    ], dtype=np.float64)
    for index in range(19):
        if index < 3:
            action = np.zeros((3, 4), dtype=np.float64)
        elif index < 10:
            action = fixed
        else:
            action = rng.uniform(-0.2, 0.2, size=(3, 4))
        backend.step(action)
        obs = backend.read()
        contact = contacts(backend)
        finite = finite_state(backend)
        if not all(finite.values()):
            errors.append(f"nonfinite state at policy step {index + 1}: {finite}")
        if np.any(backend.last_solver_diagnostics["capacity_overflow_substeps_by_world"]):
            errors.append(
                f"capacity overflow at policy step {index + 1}: {backend.last_solver_diagnostics}"
            )
        trajectory["pos"].append(obs["pos"].copy())
        trajectory["wheel_qd"].append(obs["wheel_qd"].copy())
        trajectory["gimbal_q"].append(obs["gimbal_q"].copy())
        trajectory["gimbal_qd"].append(obs["gimbal_qd"].copy())
        trajectory["effort"].append(obs["effort"].copy())
        trajectory["gimbal_effort"].append(obs["gimbal_effort"].copy())
        trajectory["contact_count_by_world"].append(contact["contact_count_by_world"])
        trajectory["max_penetration_m_by_world"].append(contact["max_penetration_m_by_world"])
        for name in (
            "capacity_overflow_substeps_by_world",
            "outer_iteration_limit_substeps_by_world",
            "line_search_limit_substeps_by_world",
        ):
            trajectory[name].append(list(backend.last_solver_diagnostics[name]))
        report["policy_steps_completed"] = index + 1
        report["last_policy_step"] = index + 1
        save_json(out / "progress.json", report)

    # Verify reset world 1 does not alter body, reduced-coordinate, control, or
    # MuJoCo-Warp buffers belonging to worlds 0 and 2.
    untouched = (0, 2)
    before_reset = capture_worlds(backend, untouched)
    reset_pose = np.asarray([[2.8, 0.9, 0.7, 0.06]], dtype=np.float64)
    backend.reset(env_ids=[1], poses=reset_pose)
    after_reset = capture_worlds(backend, untouched)
    changed = sorted(key for key in before_reset if before_reset[key] != after_reset.get(key))
    if changed:
        errors.append(f"partial reset changed untouched world data: {changed}")
    reset_source_q = arr(backend.joint_q).copy()
    reset_source_qd = arr(backend.joint_qd).copy()
    selected_sync = True
    for name in ("state_0", "state_1"):
        state = getattr(backend, name)
        q, qd = arr(state.joint_q), arr(state.joint_qd)
        for q0, q1, qd0, qd1 in backend.world_joint_slices[1]:
            selected_sync &= np.array_equal(q[q0:q1], reset_source_q[q0:q1])
            selected_sync &= np.array_equal(qd[qd0:qd1], reset_source_qd[qd0:qd1])
    if not selected_sync:
        errors.append("selected reset reduced coordinates are not synchronized in both states")
    reset_read = backend.read()
    if not np.allclose(reset_read["pos"][1], [2.8, 0.9, 0.21], atol=2e-5):
        errors.append(f"reset pose was not installed: {reset_read['pos'][1].tolist()}")

    backend.step(np.zeros((3, 4), dtype=np.float64))
    obs = backend.read()
    contact = contacts(backend)
    finite = finite_state(backend)
    if not all(finite.values()):
        errors.append(f"nonfinite state after reset policy step: {finite}")
    if not np.allclose(obs["pos"][1, :2], [2.8, 0.9], atol=0.03):
        errors.append(f"selected pose drifted after its next policy step: {obs['pos'][1].tolist()}")
    if np.any(backend.last_solver_diagnostics["capacity_overflow_substeps_by_world"]):
        errors.append(f"capacity overflow on final policy step: {backend.last_solver_diagnostics}")
    for name in (
        "pos", "wheel_qd", "gimbal_q", "gimbal_qd", "effort", "gimbal_effort",
    ):
        trajectory[name].append(obs[name].copy())
    trajectory["contact_count_by_world"].append(contact["contact_count_by_world"])
    trajectory["max_penetration_m_by_world"].append(contact["max_penetration_m_by_world"])
    for name in (
        "capacity_overflow_substeps_by_world",
        "outer_iteration_limit_substeps_by_world",
        "line_search_limit_substeps_by_world",
    ):
        trajectory[name].append(list(backend.last_solver_diagnostics[name]))

    if not all(np.asarray(trajectory["contact_count_by_world"])[-1] > 0):
        errors.append(f"final step had no contacts in every world: {contact}")
    if max(contact["max_penetration_m_by_world"]) > 0.015:
        errors.append(f"contact penetration exceeded 15 mm: {contact}")
    if float(np.max(np.abs(obs["wheel_qd"][0] - obs["wheel_qd"][1]))) <= 1e-5:
        errors.append("world-specific wheel readbacks did not diverge")
    if float(np.max(np.abs(obs["gimbal_qd"]))) <= 1e-7:
        errors.append("gimbal velocity readback stayed zero")

    q_starts = arr(backend.model.joint_q_start).astype(np.int64, copy=False)
    q = arr(backend.state_0.joint_q)
    mimic_residuals = []
    for world in range(backend.nworld):
        residuals = []
        for side in ("left", "right"):
            joints = backend.wheel_joint_idx[side][world]
            leader = float(q[q_starts[joints[0]]])
            residuals.extend(float(q[q_starts[joint]]) - leader for joint in joints[1:])
            guide_joint = backend.guide_joint_idx[world][0 if side == "left" else 1]
            residuals.append(float(q[q_starts[guide_joint]]) - GUIDE_RATIO * leader)
        mimic_residuals.append(max(abs(value) for value in residuals))
    if max(mimic_residuals) > 0.05:
        errors.append(f"mimic state residual exceeds 0.05 rad: {mimic_residuals}")

    report.update({
        "status": "passed" if not errors else "failed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - timer,
        "actual_model_device": device,
        "device_inventory": inventory,
        "versions": {
            "newton": getattr(newton, "__version__", "unknown"),
            "warp": str(wp.config.version),
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(mujoco_warp.__version__),
        },
        "policy_steps_completed": backend.step_count.tolist()[0],
        "physics_substeps_completed": 20 * backend.decimation + 1,
        "actuator_limit_probe_substeps": 1,
        "identity": identity,
        "torque_limit_check": {
            "wheel_max_abs_nm_by_world": wheel_nm.tolist(),
            "gimbal_max_abs_nm_by_world": gimbal_nm.tolist(),
            "capacity_overflow_raw_one_substep": overflow_one_step.tolist(),
        },
        "partial_reset_untouched_worlds_bitwise_unchanged": not changed,
        "partial_reset_untouched_changed_fields": changed,
        "partial_reset_selected_state_joint_sync": bool(selected_sync),
        "reset_pose_after_next_policy_step": obs["pos"][1].tolist(),
        "final_contacts": contact,
        "finite_fields_after_final_step": finite,
        "mimic_max_abs_raw_residual_rad_by_world": mimic_residuals,
        "last_solver_diagnostics": backend.last_solver_diagnostics,
        "trajectory_summary": {
            "policy_steps": len(trajectory["pos"]),
            "capacity_overflow_substeps_by_world": np.sum(
                np.asarray(trajectory["capacity_overflow_substeps_by_world"]), axis=0
            ).astype(int).tolist(),
            "outer_iteration_limit_substeps_by_world": np.sum(
                np.asarray(trajectory["outer_iteration_limit_substeps_by_world"]), axis=0
            ).astype(int).tolist(),
            "line_search_limit_substeps_by_world": np.sum(
                np.asarray(trajectory["line_search_limit_substeps_by_world"]), axis=0
            ).astype(int).tolist(),
            "max_contact_count_by_world": np.max(
                np.asarray(trajectory["contact_count_by_world"]), axis=0
            ).astype(int).tolist(),
            "max_penetration_m_by_world": np.max(
                np.asarray(trajectory["max_penetration_m_by_world"]), axis=0
            ).tolist(),
            "all_finite": bool(all(
                np.isfinite(np.asarray(trajectory[name])).all()
                for name in ("pos", "wheel_qd", "gimbal_q", "gimbal_qd", "effort", "gimbal_effort")
            )),
        },
        "source_files": [
            source_hash(PROTO / "newton_backend.py"),
            source_hash(PROTO / "newton_mujoco_backend.py"),
        ],
        "output_dir": str(out),
        "gpu_used": True,
    })
    trajectory_path = out / "trajectory.npz"
    np.savez_compressed(trajectory_path, **{
        key: np.asarray(value) for key, value in trajectory.items()
    })
    report["trajectory_npz"] = str(trajectory_path)
    report["trajectory_npz_sha256"] = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    report["errors"] = errors
    save_json(out / "result.json", report)
    save_json(out / "progress.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())