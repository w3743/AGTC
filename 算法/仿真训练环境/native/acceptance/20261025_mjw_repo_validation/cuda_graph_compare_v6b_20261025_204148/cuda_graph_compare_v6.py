from __future__ import annotations

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


def json_write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fields(backend):
    return {
        "state_0.joint_q": arr(backend.state_0.joint_q),
        "state_1.joint_q": arr(backend.state_1.joint_q),
        "state_0.joint_qd": arr(backend.state_0.joint_qd),
        "state_1.joint_qd": arr(backend.state_1.joint_qd),
        "state_0.body_q": arr(backend.state_0.body_q),
        "state_1.body_q": arr(backend.state_1.body_q),
        "state_0.body_qd": arr(backend.state_0.body_qd),
        "state_1.body_qd": arr(backend.state_1.body_qd),
        "mjw.qpos": arr(backend.solver.mjw_data.qpos),
        "mjw.qvel": arr(backend.solver.mjw_data.qvel),
        "mjw.qacc_warmstart": arr(backend.solver.mjw_data.qacc_warmstart),
        "mjw.qfrc_actuator": arr(backend._actuator_observables.qfrc_actuator),
        "control.q": arr(backend.control.joint_target_q),
        "control.qd": arr(backend.control.joint_target_qd),
        "effort": np.asarray(backend.last_effort).copy(),
        "gimbal_effort": np.asarray(backend.last_gimbal_effort).copy(),
    }


def compare(eager, graph, stage, maxima, thresholds):
    eager_fields, graph_fields = fields(eager), fields(graph)
    if eager_fields.keys() != graph_fields.keys():
        raise AssertionError(f"{stage}: backend field sets differ")
    checks = {}
    for name, left in eager_fields.items():
        right = graph_fields[name]
        atol, rtol = thresholds[name]
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        max_abs = float(np.max(difference)) if difference.size else 0.0
        maxima[name] = max(maxima.get(name, 0.0), max_abs)
        checks[name] = {
            "max_abs_diff": max_abs,
            "atol": atol,
            "rtol": rtol,
            "passed": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        }
    diagnostic_only = {"mjw.qacc_warmstart"}
    failed = [name for name, result in checks.items() if not result["passed"] and name not in diagnostic_only]
    diagnostic_failed = [name for name, result in checks.items() if not result["passed"] and name in diagnostic_only]
    return {
        "stage": stage,
        "passed": not failed,
        "failed_fields": failed,
        "diagnostic_failed_fields": diagnostic_failed,
        "fields": checks,
    }


def world_control_indices(backend, world):
    qd = sorted(
        int(dof)
        for side in ("left", "right")
        for dof in backend.wheel_target_dof[side][world]
    )
    q = sorted(int(coord) for coord in backend.gimbal_target_q_coord[world])
    return np.asarray(qd, dtype=np.int64), np.asarray(q, dtype=np.int64)


def capture_unaffected(backend, worlds):
    captured = {}
    for state_name in ("state_0", "state_1"):
        state = getattr(backend, state_name)
        for field in ("body_q", "body_qd"):
            value = arr(getattr(state, field))
            for world in worlds:
                captured[f"{state_name}.{field}.{world}"] = np.ascontiguousarray(
                    value[backend.world_body_idx[world]]
                ).tobytes()
        q, qd = arr(state.joint_q), arr(state.joint_qd)
        for world in worlds:
            qp, qdp = [], []
            for q0, q1, qd0, qd1 in backend.world_joint_slices[world]:
                qp.append(q[q0:q1])
                qdp.append(qd[qd0:qd1])
            captured[f"{state_name}.joint_q.{world}"] = np.ascontiguousarray(np.concatenate(qp)).tobytes()
            captured[f"{state_name}.joint_qd.{world}"] = np.ascontiguousarray(np.concatenate(qdp)).tobytes()
    target_q, target_qd = arr(backend.control.joint_target_q), arr(backend.control.joint_target_qd)
    for world in worlds:
        qd_idx, q_idx = world_control_indices(backend, world)
        captured[f"control.q.{world}"] = np.ascontiguousarray(target_q[q_idx]).tobytes()
        captured[f"control.qd.{world}"] = np.ascontiguousarray(target_qd[qd_idx]).tobytes()
    data = backend.solver.mjw_data
    for field_name in ("time", "qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        value = getattr(data, field_name, None)
        if value is None:
            continue
        value = arr(value)
        if value.ndim < 1 or value.shape[0] != backend.nworld:
            continue
        for world in worlds:
            captured[f"mjw.{field_name}.{world}"] = np.ascontiguousarray(value[world]).tobytes()
    return captured


def digest(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    out = Path(os.environ["AGTC_GRAPH_OUT"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "pid.txt").write_text(str(os.getpid()) + "\n", encoding="ascii")
    report = {
        "schema": "agtc-newton-mujoco-warp-cuda-graph-compare-v2",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu_used": True,
        "policy_steps_requested_per_backend": 60,
        "acceptance_criteria": "Compare joint/body q, joint/body qd, MuJoCo qpos/qvel, actual actuator torque and control targets at configured tolerances; qacc_warmstart is retained in every comparison as diagnostic-only solver scratch state.",
        "errors": [],
    }
    json_write(out / "progress.json", report)
    import mujoco
    import mujoco_warp
    import newton
    import warp as wp
    from contract_identity import canonical_physics_identity
    from newton_mujoco_backend import NewtonMuJoCoWarpBackend

    start = time.perf_counter()
    eager = NewtonMuJoCoWarpBackend(
        nworld=3, dt=1e-3, decimation=50, seed=20261025,
        device="cuda:0", use_cuda_graph=False,
    )
    eager.solver.mjw_model.opt.warn_overflow = 0
    # One eager warmup compiles all kernels before the measured comparison.
    eager.step(np.zeros((3, 4), dtype=np.float64))
    eager.reset()
    graph = NewtonMuJoCoWarpBackend(
        nworld=3, dt=1e-3, decimation=50, seed=20261025,
        device="cuda:0", use_cuda_graph=True,
    )
    if not graph._cuda_graph_enabled:
        raise RuntimeError("requested CUDA graph was not actually enabled")
    graph.solver.mjw_model.opt.warn_overflow = 0

    eager_identity, graph_identity = eager.identity(), graph.identity()
    eager_canonical = canonical_physics_identity(eager_identity)
    graph_canonical = canonical_physics_identity(graph_identity)
    if eager_canonical != graph_canonical:
        raise RuntimeError("eager and graph execution produced different canonical physics identity")

    poses = np.asarray([
        [0.0, 0.0, 0.0, 0.025],
        [1.4, 0.25, 0.3, 0.025],
        [-1.5, -0.3, -0.25, 0.025],
    ], dtype=np.float64)
    eager.reset(poses=poses)
    graph.reset(poses=poses)
    thresholds = {}
    for name in ("state_0.joint_q", "state_1.joint_q", "state_0.body_q", "state_1.body_q", "mjw.qpos", "control.q"):
        thresholds[name] = (1e-5, 1e-5)
    for name in ("state_0.joint_qd", "state_1.joint_qd", "state_0.body_qd", "state_1.body_qd",
                 "mjw.qvel", "mjw.qacc_warmstart", "mjw.qfrc_actuator",
                 "control.qd", "effort", "gimbal_effort"):
        thresholds[name] = (1e-4, 1e-5)

    comparisons = []
    max_diffs = {}
    initial = compare(eager, graph, "initial_reset", max_diffs, thresholds)
    comparisons.append(initial)
    if not initial["passed"]:
        report["errors"].append(f"initial reset differs: {initial['failed_fields']}")

    rng = np.random.default_rng(20261025)
    fixed = np.asarray([
        [0.45, 0.45, 0.2, -0.1],
        [-0.35, -0.35, -0.25, 0.1],
        [0.2, -0.2, -0.1, 0.25],
    ], dtype=np.float64)
    eager_times, graph_times = [], []
    flag_totals = {
        "eager": {name: np.zeros(3, dtype=np.int64) for name in (
            "capacity_overflow_substeps_by_world",
            "outer_iteration_limit_substeps_by_world",
            "line_search_limit_substeps_by_world",
        )},
        "graph": {name: np.zeros(3, dtype=np.int64) for name in (
            "capacity_overflow_substeps_by_world",
            "outer_iteration_limit_substeps_by_world",
            "line_search_limit_substeps_by_world",
        )},
    }
    partial_reset_checks = {}
    for index in range(60):
        if index == 30:
            for name, backend in (("eager", eager), ("graph", graph)):
                before = capture_unaffected(backend, (0, 2))
                backend.reset(env_ids=np.asarray([1]), poses=np.asarray([[2.8, 0.9, 0.7, 0.06]]))
                after = capture_unaffected(backend, (0, 2))
                changed = sorted(key for key in before if before[key] != after.get(key))
                partial_reset_checks[name] = {
                    "untouched_worlds_bitwise_unchanged": not changed,
                    "changed_fields": changed,
                    "graph_buffer_pointers_unchanged": (
                        backend._graph_buffer_pointer_snapshot() == backend._cuda_graph_buffer_ptrs
                        if name == "graph" else None
                    ),
                }
                if changed:
                    report["errors"].append(f"{name} partial reset changed untouched worlds: {changed}")
                if name == "graph" and not partial_reset_checks[name]["graph_buffer_pointers_unchanged"]:
                    report["errors"].append("graph partial reset replaced a captured buffer")
            reset_compare = compare(eager, graph, "after_partial_reset", max_diffs, thresholds)
            comparisons.append(reset_compare)
            if not reset_compare["passed"]:
                report["errors"].append(
                    f"partial reset states differ: {reset_compare['failed_fields']}"
                )

        if index < 8:
            action = np.zeros((3, 4), dtype=np.float64)
        elif index < 40:
            action = fixed
        else:
            action = rng.uniform(-0.2, 0.2, size=(3, 4))

        t0 = time.perf_counter()
        eager.step(action)
        eager_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        graph.step(action)
        graph_times.append(time.perf_counter() - t0)

        for label, backend in (("eager", eager), ("graph", graph)):
            diag = backend.last_solver_diagnostics
            for name in flag_totals[label]:
                flag_totals[label][name] += np.asarray(diag[name], dtype=np.int64)
            if np.any(diag["capacity_overflow_substeps_by_world"]):
                report["errors"].append(
                    f"{label} capacity overflow at policy step {index + 1}: {diag}"
                )
            current = fields(backend)
            if not all(np.isfinite(value).all() for value in current.values()):
                report["errors"].append(f"{label} nonfinite state at policy step {index + 1}")

        comparison = compare(eager, graph, f"policy_step_{index + 1}", max_diffs, thresholds)
        comparisons.append(comparison)
        if not comparison["passed"]:
            report["errors"].append(
                f"state comparison failed at policy step {index + 1}: {comparison['failed_fields']}"
            )
        report["policy_steps_completed_per_backend"] = index + 1
        if (index + 1) % 10 == 0:
            json_write(out / "progress.json", report)

    eager_last, graph_last = fields(eager), fields(graph)
    np.savez_compressed(out / "last_state.npz", **{
        "eager_" + key: value for key, value in eager_last.items()
    }, **{
        "graph_" + key: value for key, value in graph_last.items()
    })
    graph_summary = {
        "enabled": graph._cuda_graph_enabled,
        "requested": graph.use_cuda_graph_requested,
        "capture_elapsed_s": graph._cuda_graph_capture_s,
        "captured_buffer_count": len(graph._cuda_graph_buffer_ptrs),
        "captured_state_pointer_stable_after_reset": partial_reset_checks.get("graph", {}).get(
            "graph_buffer_pointers_unchanged"
        ),
    }
    report.update({
        "status": "passed" if not report["errors"] else "failed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_s": time.perf_counter() - start,
        "actual_device": str(eager.model.device),
        "versions": {
            "newton": getattr(newton, "__version__", "unknown"),
            "warp": str(wp.config.version),
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(mujoco_warp.__version__),
        },
        "eager_identity": eager_identity,
        "graph_identity": graph_identity,
        "canonical_identity_equal": eager_canonical == graph_canonical,
        "graph": graph_summary,
        "policy_steps_per_backend": 60,
        "physics_substeps_per_backend": 60 * eager.decimation,
        "partial_reset": partial_reset_checks,
        "comparisons": comparisons,
        "max_abs_diff_by_field": max_diffs,
        "tolerances": {name: {"atol": values[0], "rtol": values[1]} for name, values in thresholds.items()},
        "timing": {
            "comparison_includes": "backend.step calls only; per-step raw state comparisons occur outside timers",
            "eager_total_step_s": float(np.sum(eager_times)),
            "graph_total_step_s": float(np.sum(graph_times)),
            "eager_mean_policy_step_s": float(np.mean(eager_times)),
            "graph_mean_policy_step_s": float(np.mean(graph_times)),
            "eager_samples_per_s_3world": float(60 * 3 / np.sum(eager_times)),
            "graph_samples_per_s_3world": float(60 * 3 / np.sum(graph_times)),
            "speedup": float(np.sum(eager_times) / np.sum(graph_times)),
        },
        "overflow_events_by_backend": {
            name: {key: value.tolist() for key, value in values.items()}
            for name, values in flag_totals.items()
        },
        "source_files": [
            digest(PROTO / "newton_backend.py"),
            digest(PROTO / "newton_mujoco_backend.py"),
        ],
        "gpu_used": True,
        "output_dir": str(out),
    })
    report["last_state_npz"] = str(out / "last_state.npz")
    report["last_state_npz_sha256"] = hashlib.sha256(
        (out / "last_state.npz").read_bytes()
    ).hexdigest()
    json_write(out / "result.json", report)
    json_write(out / "progress.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())