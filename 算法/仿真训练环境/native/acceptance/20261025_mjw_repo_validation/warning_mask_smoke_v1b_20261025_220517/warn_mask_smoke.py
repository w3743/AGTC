"""GPU smoke for LS warning suppression while keeping per-substep counters."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def array(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def capture_unaffected(backend, worlds: tuple[int, ...]) -> dict[str, bytes]:
    result = {}
    for state_name in ("state_0", "state_1"):
        state = getattr(backend, state_name)
        for field in ("body_q", "body_qd"):
            values = array(getattr(state, field))
            for world in worlds:
                indices = np.asarray(backend.world_body_idx[world], dtype=np.int64)
                result[f"{state_name}.{field}.world{world}"] = np.ascontiguousarray(values[indices]).tobytes()
        q = array(state.joint_q)
        qd = array(state.joint_qd)
        for world in worlds:
            slices = backend.world_joint_slices[world]
            q_values = np.concatenate([q[q0:q1] for q0, q1, _, _ in slices])
            qd_values = np.concatenate([qd[qd0:qd1] for _, _, qd0, qd1 in slices])
            result[f"{state_name}.joint_q.world{world}"] = np.ascontiguousarray(q_values).tobytes()
            result[f"{state_name}.joint_qd.world{world}"] = np.ascontiguousarray(qd_values).tobytes()
    target_q = array(backend.control.joint_target_q)
    target_qd = array(backend.control.joint_target_qd)
    for world in worlds:
        q_indices = np.asarray(backend.gimbal_target_q_coord[world], dtype=np.int64)
        qd_indices = np.asarray(sorted(
            dof for side in ("left", "right") for dof in backend.wheel_target_dof[side][world]
        ), dtype=np.int64)
        result[f"control.q.world{world}"] = np.ascontiguousarray(target_q[q_indices]).tobytes()
        result[f"control.qd.world{world}"] = np.ascontiguousarray(target_qd[qd_indices]).tobytes()
    data = backend.solver.mjw_data
    for field in ("time", "qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        values = getattr(data, field, None)
        if values is None:
            continue
        values = array(values)
        if values.ndim == 0 or values.shape[0] != backend.nworld:
            continue
        for world in worlds:
            result[f"mjw.{field}.world{world}"] = np.ascontiguousarray(values[world]).tobytes()
    return result


def finite_fields(backend) -> dict[str, bool]:
    checks = {
        "state0.body_q": array(backend.state_0.body_q),
        "state0.body_qd": array(backend.state_0.body_qd),
        "state1.body_q": array(backend.state_1.body_q),
        "state1.body_qd": array(backend.state_1.body_qd),
        "state0.joint_q": array(backend.state_0.joint_q),
        "state0.joint_qd": array(backend.state_0.joint_qd),
        "mjw.qpos": array(backend.solver.mjw_data.qpos),
        "mjw.qvel": array(backend.solver.mjw_data.qvel),
        "mjw.qacc": array(backend.solver.mjw_data.qacc),
        "mjw.ctrl": array(backend.solver.mjw_data.ctrl),
        "effort": np.asarray(backend.last_effort),
        "gimbal_effort": np.asarray(backend.last_gimbal_effort),
    }
    return {name: bool(np.isfinite(value).all()) for name, value in checks.items()}


def main() -> int:
    out = Path(os.environ["AGTC_WARN_OUT"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    proto = Path(os.environ.get(
        "AGTC_NATIVE_PROTO",
        "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto",
    )).resolve(strict=True)
    sys.path.insert(0, str(proto))

    import mujoco
    import mujoco_warp
    import newton
    import warp as wp
    from newton_mujoco_backend import NewtonMuJoCoWarpBackend

    report = {
        "schema": "agtc-mjw-ls-warning-mask-smoke-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "requested_policy_steps": 20,
        "errors": [],
    }
    write_json(out / "progress.json", report)
    backend = None
    started = time.perf_counter()
    try:
        backend = NewtonMuJoCoWarpBackend(
            nworld=3, dt=1e-3, decimation=50, seed=20261025,
            device="cuda:0", use_cuda_graph=False,
        )
        if not str(backend.model.device).startswith("cuda"):
            raise RuntimeError(f"unexpected model device {backend.model.device}")
        if backend._cuda_graph_enabled:
            raise RuntimeError("CUDA graph was unexpectedly enabled")

        all_warnings = int(mujoco_warp.OverflowType.ALL)
        ls_bit = int(mujoco_warp.OverflowType.LS_ITERATIONS)
        actual_mask = int(backend.solver.mjw_model.opt.warn_overflow)
        expected_mask = all_warnings & ~ls_bit
        if actual_mask != expected_mask:
            raise AssertionError(f"warn_overflow mask {actual_mask} != LS-only suppression mask {expected_mask}")

        starts = np.asarray([
            [0.0, 0.0, 0.0, 0.025],
            [1.4, 0.25, 0.3, 0.025],
            [-1.5, -0.3, -0.25, 0.025],
        ], dtype=np.float64)
        backend.reset(poses=starts)

        # Over-command a single physical substep to verify actual QFRC actuator limits.
        target_qd = array(backend.control.joint_target_qd).copy()
        target_q = array(backend.control.joint_target_q).copy()
        for world, sign in enumerate((-1.0, 1.0, -1.0)):
            for side in ("left", "right"):
                for dof in backend.wheel_target_dof[side][world]:
                    target_qd[dof] = sign * 1000.0
            for coord in backend.gimbal_target_q_coord[world]:
                target_q[coord] = sign * 10.0
        backend.control.joint_target_qd.assign(target_qd)
        backend.control.joint_target_q.assign(target_q)
        backend._overflow_event_counts.zero_()
        backend._solver_substep()
        backend._snapshot_actuator_effort()
        limit_counts = np.asarray(backend._overflow_event_counts.numpy(), dtype=np.int64)
        wheel_max = np.max(np.abs(backend.last_effort), axis=1)
        gimbal_max = np.max(np.abs(backend.last_gimbal_effort), axis=1)
        if np.any(limit_counts[:, 0]):
            raise AssertionError(f"capacity overflow on torque-limit substep: {limit_counts[:, 0].tolist()}")
        if not np.isfinite(wheel_max).all() or np.any(wheel_max > 2.0001):
            raise AssertionError(f"wheel QFRC actuator limit failed: {wheel_max.tolist()}")
        if not np.isfinite(gimbal_max).all() or np.any(gimbal_max > 1.0001):
            raise AssertionError(f"gimbal QFRC actuator limit failed: {gimbal_max.tolist()}")
        if np.max(wheel_max) < 1.9 or np.max(gimbal_max) < 0.9:
            raise AssertionError(f"over-command did not exercise both effort clamps: {wheel_max.tolist()}, {gimbal_max.tolist()}")

        # Restore a clean episode, then run 20 policy steps with a partial reset.
        backend.reset(poses=starts)
        rng = np.random.default_rng(20261025)
        fixed = np.asarray([
            [0.45, 0.45, 0.2, -0.1],
            [-0.35, -0.35, -0.25, 0.1],
            [0.2, -0.2, -0.1, 0.25],
        ], dtype=np.float64)
        capacity_totals = np.zeros(3, dtype=np.int64)
        outer_totals = np.zeros(3, dtype=np.int64)
        line_search_totals = np.zeros(3, dtype=np.int64)
        reset_check = {"tested": False, "untouched_worlds_bitwise_unchanged": False}
        for index in range(20):
            if index == 10:
                before = capture_unaffected(backend, (0, 2))
                backend.reset(env_ids=np.asarray([1]), poses=np.asarray([[2.8, 0.9, 0.7, 0.06]]))
                after = capture_unaffected(backend, (0, 2))
                changed = sorted(key for key in before if before[key] != after.get(key))
                reset_check = {
                    "tested": True,
                    "untouched_worlds_bitwise_unchanged": not changed,
                    "changed_fields": changed,
                }
                if changed:
                    raise AssertionError(f"partial reset changed untouched-world state: {changed}")

            if index < 4:
                actions = np.zeros((3, 4), dtype=np.float64)
            elif index < 12:
                actions = fixed
            else:
                actions = rng.uniform(-0.2, 0.2, size=(3, 4))
            backend.step(actions)
            diagnostics = backend.last_solver_diagnostics
            capacity_totals += np.asarray(diagnostics["capacity_overflow_substeps_by_world"], dtype=np.int64)
            outer_totals += np.asarray(diagnostics["outer_iteration_limit_substeps_by_world"], dtype=np.int64)
            line_search_totals += np.asarray(diagnostics["line_search_limit_substeps_by_world"], dtype=np.int64)
            finite = finite_fields(backend)
            if not all(finite.values()):
                raise AssertionError(f"non-finite state after policy step {index + 1}: {finite}")
            if (index + 1) % 5 == 0:
                report.update({
                    "status": "running",
                    "policy_steps_completed": index + 1,
                    "capacity_overflow_substeps_by_world": capacity_totals.tolist(),
                    "outer_iteration_limit_substeps_by_world": outer_totals.tolist(),
                    "line_search_limit_substeps_by_world": line_search_totals.tolist(),
                })
                write_json(out / "progress.json", report)

        if int(np.sum(line_search_totals)) <= 0:
            raise AssertionError("LS counter was zero for all 20 steps; suppression did not verify visible nonzero reporting")
        report.update({
            "status": "passed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_wall_s": time.perf_counter() - started,
            "versions": {
                "newton": getattr(newton, "__version__", "unknown"),
                "warp": str(wp.config.version),
                "mujoco": str(mujoco.__version__),
                "mujoco_warp": str(mujoco_warp.__version__),
            },
            "actual_device": str(backend.model.device),
            "cuda_graph_enabled": backend._cuda_graph_enabled,
            "solver_warning_policy": backend.identity()["runtime_diagnostics"]["solver_warning_policy"],
            "expected_warn_overflow_mask": expected_mask,
            "actual_warn_overflow_mask": actual_mask,
            "torque_limit_single_substep": {
                "capacity_overflow_by_world": limit_counts[:, 0].tolist(),
                "wheel_qfrc_actuator_max_abs_nm_by_world": wheel_max.tolist(),
                "gimbal_qfrc_actuator_max_abs_nm_by_world": gimbal_max.tolist(),
            },
            "policy_steps_completed": 20,
            "physics_substeps_completed": 20 * backend.decimation + 1,
            "capacity_overflow_substeps_by_world": capacity_totals.tolist(),
            "outer_iteration_limit_substeps_by_world": outer_totals.tolist(),
            "line_search_limit_substeps_by_world": line_search_totals.tolist(),
            "partial_reset": reset_check,
            "final_finite_fields": finite_fields(backend),
            "errors": [],
        })
    except Exception as exc:
        report.update({
            "status": "failed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_wall_s": time.perf_counter() - started,
            "errors": [f"{type(exc).__name__}: {exc}"],
        })
        if backend is not None:
            try:
                report["actual_device"] = str(backend.model.device)
                report["last_solver_diagnostics"] = backend.last_solver_diagnostics
                report["final_finite_fields"] = finite_fields(backend)
            except Exception as inspect_error:
                report["diagnostic_error"] = f"{type(inspect_error).__name__}: {inspect_error}"
    finally:
        write_json(out / "result.json", report)
        write_json(out / "progress.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
