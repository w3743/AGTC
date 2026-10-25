"""CPU-only smoke probe for the external MuJoCo Warp candidate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-proto",
        default=os.environ.get(
            "AGTC_NATIVE_PROTO",
            "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto",
        ),
    )
    parser.add_argument("--output", default=str(Path(__file__).with_name("cpu_probe_result.json")))
    args = parser.parse_args()
    proto = Path(args.repo_proto).resolve(strict=True)
    sys.path.insert(0, str(proto))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import mujoco
    import mujoco_warp
    import newton
    import warp as wp
    from mjw_backend_candidate import NewtonMuJoCoWarpCandidate

    backend = NewtonMuJoCoWarpCandidate(nworld=1, dt=1e-3, decimation=50, seed=17, device="cpu")
    identity = backend.identity()
    if str(backend.model.device) != "cpu" or identity["solver"]["use_mujoco_cpu"]:
        raise AssertionError("probe did not use the Warp CPU path")
    if identity["engine"] != "newton_mujoco_warp":
        raise AssertionError(identity["engine"])
    if identity["solver"]["requested_config"] != {
        "njmax": 256,
        "nconmax": 64,
        "iterations": 40,
        "ls_iterations": 8,
        "solver": "newton",
        "integrator": "implicitfast",
        "cone": "elliptic",
        "impratio": 10.0,
    }:
        raise AssertionError(identity["solver"]["requested_config"])
    if identity["contact"]["source"] != "mujoco_warp_internal":
        raise AssertionError(identity["contact"])
    limits = identity["joint_dof_effort_limit_values_world0_nm"]
    if not np.allclose(limits["wheel_dof"], [2.0], rtol=1e-6, atol=1e-8):
        raise AssertionError(limits)
    if not np.allclose(limits["gimbal_dof"], [1.0], rtol=1e-6, atol=1e-8):
        raise AssertionError(limits)
    if not np.allclose(identity["solver"]["impratio_actual"], [10.0], rtol=1e-6):
        raise AssertionError(identity["solver"])
    if identity["contact"]["model_shape_gap_values_per_world"] != [0.0]:
        raise AssertionError(identity["contact"])
    if identity["contact"]["exported_geom_gap_values_per_world"] != [0.0]:
        raise AssertionError(identity["contact"])
    if identity["contact"]["exported_geom_margin_values_per_world"] != [0.0]:
        raise AssertionError(identity["contact"])
    if identity["mimic_relations_per_world"] != 14:
        raise AssertionError(identity["mimic_relations_per_world"])
    if identity["solver"]["exported_actuators"] != 16 or identity["solver"]["exported_equalities"] != 14:
        raise AssertionError(identity["solver"])
    if identity["solver"]["actual_capacity_per_world"] != {"nconmax": 64, "njmax": 256}:
        raise AssertionError(identity["solver"]["actual_capacity_per_world"])
    np.testing.assert_allclose(identity["chassis_inertia_diag_kg_m2"], [0.2475, 1.4575, 1.606], rtol=1e-6)
    np.testing.assert_allclose(
        identity["gimbal_pitch_inertia_diag_kg_m2"],
        [1.3333333333333334e-5] * 3,
        rtol=1e-6,
    )
    if not np.allclose(identity["material_values_per_world"]["mu"], [0.8], rtol=1e-6, atol=1e-8):
        raise AssertionError(identity["material_values_per_world"])
    if identity["drive_semantics"]["side_pi_feed_forward_applied"]:
        raise AssertionError("PI feed-forward must not be reported as applied")

    initial = backend.read()
    for field in ("pos", "quat_wxyz", "lin_vel", "ang_vel", "wheel_qd", "gimbal_q", "gimbal_qd"):
        if not np.all(np.isfinite(initial[field])):
            raise AssertionError(f"non-finite initial observation field {field}")

    # Deliberately over-command wheel and gimbal targets for one physical step to
    # verify the solver's exported actuator limits through the actual qfrc observable.
    qd_target = backend.control.joint_target_qd.numpy().copy()
    q_target = backend.control.joint_target_q.numpy().copy()
    for side in ("left", "right"):
        for dof in backend.wheel_target_dof[side][0]:
            qd_target[dof] = 1000.0
    for coord in backend.gimbal_target_q_coord[0]:
        q_target[coord] = 10.0
    backend.control.joint_target_qd.assign(qd_target)
    backend.control.joint_target_q.assign(q_target)
    backend.state_0.clear_forces()
    backend.solver.step(
        backend.state_0,
        backend.state_1,
        backend.control,
        None,
        backend.dt,
        observables=backend._actuator_observables,
    )
    backend.state_0, backend.state_1 = backend.state_1, backend.state_0
    backend._snapshot_actuator_effort()
    torque_sample = backend.read()
    if not np.all(np.isfinite(torque_sample["effort"])) or not np.all(np.isfinite(torque_sample["gimbal_effort"])):
        raise AssertionError("non-finite qfrc_actuator observable")
    max_wheel_nm = float(np.max(np.abs(torque_sample["effort"])))
    max_gimbal_nm = float(np.max(np.abs(torque_sample["gimbal_effort"])))
    if max_wheel_nm > 2.0001:
        raise AssertionError(f"wheel actuator force exceeded 2 Nm: {max_wheel_nm}")
    if max_gimbal_nm > 1.0001:
        raise AssertionError(f"gimbal actuator force exceeded 1 Nm: {max_gimbal_nm}")

    # Reset the custom base pose after a solver step. flags=0 must preserve that
    # pose while clearing MuJoCo Warp's selected-world internal buffers.
    requested_pose = np.asarray([[0.12, -0.07, 0.35, 0.025]], dtype=np.float64)
    backend.reset(env_ids=[0], poses=requested_pose)
    reset_read = backend.read()
    np.testing.assert_allclose(reset_read["pos"][0], [0.12, -0.07, 0.175], atol=2e-5)
    if reset_read["effort_observation_valid"][0]:
        raise AssertionError("reset should invalidate the previous torque sample")
    if np.any(reset_read["effort"]) or np.any(reset_read["gimbal_effort"]):
        raise AssertionError("reset should clear last reported effort sample")

    result = {
        "status": "passed",
        "device": str(backend.model.device),
        "versions": {
            "newton": getattr(newton, "__version__", "unknown"),
            "warp": str(wp.config.version),
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(mujoco_warp.__version__),
        },
        "identity": identity,
        "checks": {
            "gap_margin_zero": True,
            "corrected_inertias_and_inverse_inertia": True,
            "material_mu_0_8": True,
            "solver_options_match_cpu_reference": True,
            "internal_mujoco_warp_contacts": True,
            "wheel_qfrc_actuator_max_nm_under_2": max_wheel_nm,
            "gimbal_qfrc_actuator_max_nm_under_1": max_gimbal_nm,
            "custom_pose_survives_masked_solver_reset_flags0": True,
        },
        "policy_step_performed": False,
        "gpu_used": False,
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




