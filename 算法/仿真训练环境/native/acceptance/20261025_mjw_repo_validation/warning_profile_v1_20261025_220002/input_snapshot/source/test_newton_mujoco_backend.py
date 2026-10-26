"""CPU regression checks for the Newton MuJoCo-Warp backend and masked reset."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import warp as wp

_PROTO = Path(__file__).resolve().parent
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from newton_mujoco_backend import (  # noqa: E402
    CHASSIS_INERTIA_DIAG,
    PITCH_INERTIA_DIAG,
    SOLVER_CONFIG,
    NewtonMuJoCoWarpBackend,
)
from physics_backend import GUIDE_RATIO, WHEELS_PER_SIDE  # noqa: E402


def _new_backend(nworld: int = 3, *, decimation: int = 50, deterministic=None):
    if importlib.util.find_spec("newton") is None or importlib.util.find_spec("warp") is None:
        raise unittest.SkipTest("Newton and Warp are required for the backend acceptance tests")
    return NewtonMuJoCoWarpBackend(
        nworld=nworld, dt=1e-3, decimation=decimation,
        seed=20261025, mu=0.8, device="cpu", deterministic=deterministic,
    )


class NewtonMuJoCoWarpBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = _new_backend()
        cls.backend.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.backend

    def test_identity_matches_exported_solver_and_model(self) -> None:
        import mujoco_warp

        backend = self.backend
        identity = backend.identity()
        self.assertEqual(str(backend.model.device), "cpu")
        self.assertEqual(identity["engine"], "newton_mujoco_warp")
        self.assertIn("v6_", identity["backend_revision"])
        self.assertEqual(identity["solver"]["class"], "SolverMuJoCo")
        self.assertFalse(identity["solver"]["use_mujoco_cpu"])
        self.assertTrue(identity["solver"]["use_mujoco_contacts"])
        self.assertTrue(identity["solver"]["separate_worlds"])
        self.assertEqual(identity["solver"]["update_data_interval"], 1)
        self.assertEqual(identity["solver"]["requested_config"], SOLVER_CONFIG)
        self.assertEqual(
            identity["solver"]["actual_capacity_per_world"],
            {"nconmax": 64, "njmax": 256},
        )
        self.assertEqual(identity["mimic_relations_per_world"], 14)
        self.assertEqual(identity["solver"]["exported_equalities"], 14)
        self.assertEqual(identity["solver"]["exported_actuators"], 16)
        self.assertEqual(identity["contact"]["source"], "mujoco_warp_internal")
        self.assertEqual(identity["contact"]["model_shape_gap_values_per_world"], [0.0])
        self.assertEqual(identity["contact"]["model_shape_margin_values_per_world"], [0.0])
        self.assertEqual(identity["contact"]["exported_geom_gap_values_per_world"], [0.0])
        self.assertEqual(identity["contact"]["exported_geom_margin_values_per_world"], [0.0])
        np.testing.assert_allclose(identity["chassis_inertia_diag_kg_m2"], CHASSIS_INERTIA_DIAG, rtol=1e-6)
        np.testing.assert_allclose(
            identity["gimbal_pitch_inertia_diag_kg_m2"], PITCH_INERTIA_DIAG, rtol=1e-6
        )
        np.testing.assert_allclose(
            identity["material_values_per_world"]["mu"], [0.8], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            identity["mimic_constraints"]["eq_solref_values_world0"], [[0.004, 1.0]],
            rtol=0.0, atol=1e-6,
        )
        np.testing.assert_allclose(
            identity["mimic_constraints"]["eq_solimp_values_world0"],
            [[0.999, 0.9999, 0.001, 0.5, 2.0]], rtol=0.0, atol=1e-6,
        )
        body_inertia = backend.model.body_inertia.numpy()
        body_inv_inertia = backend.model.body_inv_inertia.numpy()
        for world in range(backend.nworld):
            for body in (backend.base_body[world], backend._pitch_body_by_world[world]):
                np.testing.assert_allclose(
                    body_inertia[body] @ body_inv_inertia[body], np.eye(3), atol=2e-5, rtol=2e-5
                )
        self.assertEqual(str(backend.solver.mjw_model.geom_gap.device), "cpu")
        self.assertEqual(str(backend.solver.mjw_data.qpos.device), "cpu")
        self.assertEqual(str(backend.solver.mjw_data.overflow.device), "cpu")
        warning_policy = identity["runtime_diagnostics"]["solver_warning_policy"]
        warn_mask = int(backend.solver.mjw_model.opt.warn_overflow)
        self.assertEqual(warning_policy["actual_warn_overflow_mask"], warn_mask)
        self.assertFalse(warn_mask & int(mujoco_warp.OverflowType.LS_ITERATIONS))
        self.assertTrue(warn_mask & int(mujoco_warp.OverflowType.NEFC))
        self.assertFalse(warning_policy["ls_iterations_console_warning_enabled"])
        self.assertEqual(warning_policy["suppressed_console_warning_bits"], ["LS_ITERATIONS"])
        self.assertTrue(warning_policy["ls_iterations_overflow_flag_preserved"])
        self.assertEqual(
            warning_policy["line_search_limit_reporting"],
            "structured_counters_only",
        )
        self.assertEqual(
            warning_policy["line_search_limit_counter_field"],
            "line_search_limit_substeps_by_world",
        )

    def test_deterministic_mode_is_applied_and_recorded(self) -> None:
        backend = _new_backend(nworld=1, deterministic=wp.DeterministicMode.RUN_TO_RUN)
        try:
            identity = backend.identity()
            self.assertEqual(backend.solver._deterministic, wp.DeterministicMode.RUN_TO_RUN)
            self.assertEqual(identity["solver"]["deterministic_mode"], "RUN_TO_RUN")
            diagnostics = identity["runtime_diagnostics"]["determinism"]
            self.assertEqual(diagnostics["requested_mode"], "RUN_TO_RUN")
            self.assertEqual(diagnostics["actual_solver_mode"], "RUN_TO_RUN")
            self.assertTrue(diagnostics["module_options_match_solver_mode"])
            self.assertGreater(diagnostics["solver_max_records"], 0)
            module_options = diagnostics["mujoco_warp_module_options"]
            self.assertTrue(module_options)
            self.assertEqual(
                {item["deterministic_mode"] for item in module_options},
                {"RUN_TO_RUN"},
            )
            self.assertEqual(str(backend.model.device), "cpu")
        finally:
            del backend
    def test_policy_step_collects_substep_events_and_actual_torque_readback(self) -> None:
        backend = self.backend
        backend.reset()
        target_qd = backend.control.joint_target_qd.numpy().copy()
        target_q = backend.control.joint_target_q.numpy().copy()
        for world in range(backend.nworld):
            for side in ("left", "right"):
                for dof in backend.wheel_target_dof[side][world]:
                    target_qd[dof] = 1000.0 * (-1.0 if world == 1 else 1.0)
            for coord in backend.gimbal_target_q_coord[world]:
                target_q[coord] = 10.0 * (-1.0 if world == 1 else 1.0)
        backend.control.joint_target_qd.assign(target_qd)
        backend.control.joint_target_q.assign(target_q)
        backend._solver_substep()
        backend._snapshot_actuator_effort()
        saturated = backend.read()
        self.assertTrue(saturated["effort_observation_valid"].all())
        self.assertTrue(np.isfinite(saturated["effort"]).all())
        self.assertTrue(np.isfinite(saturated["gimbal_effort"]).all())
        self.assertLessEqual(float(np.max(np.abs(saturated["effort"]))), 2.0001)
        self.assertLessEqual(float(np.max(np.abs(saturated["gimbal_effort"]))), 1.0001)

        backend.reset()
        actions = np.asarray([
            [0.8, 0.8, 0.8, 0.0],
            [-0.8, -0.8, -0.8, 0.0],
            [0.8, -0.8, 0.5, -0.5],
        ], dtype=np.float64)
        backend.step(actions)
        result = backend.read()
        for name in ("pos", "lin_vel", "ang_vel", "wheel_qd", "gimbal_q", "gimbal_qd",
                     "effort", "gimbal_effort"):
            self.assertTrue(np.isfinite(result[name]).all(), name)
        self.assertTrue(result["effort_observation_valid"].all())
        self.assertEqual(
            backend.last_solver_diagnostics["capacity_overflow_substeps_by_world"],
            [0, 0, 0],
        )
        self.assertGreater(np.max(np.abs(result["wheel_qd"][0] - result["wheel_qd"][1])), 1e-5)
        self.assertGreater(np.max(np.abs(result["gimbal_qd"])), 1e-7)

    def test_partial_reset_preserves_other_worlds_and_continuous_mimic_coordinates(self) -> None:
        backend = self.backend
        starts = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [1.2, 0.2, 0.3, 0.0],
            [-1.3, -0.2, -0.3, 0.0],
        ], dtype=np.float64)
        backend.reset(poses=starts)
        backend.step(np.asarray([
            [0.4, 0.4, 0.0, 0.0],
            [-0.3, -0.3, 0.2, -0.1],
            [0.2, -0.2, -0.1, 0.2],
        ], dtype=np.float64))

        # Seed a valid continuous winding representation in world 0. eval_fk updates
        # body transforms; the explicit reduced state assignment retains the winding.
        q = backend.state_0.joint_q.numpy().copy()
        qd = backend.state_0.joint_qd.numpy().copy()
        q_starts = backend.model.joint_q_start.numpy()
        winding = 2.0 * np.pi + 0.25
        for side in ("left", "right"):
            for joint in backend.wheel_joint_idx[side][0]:
                q[int(q_starts[joint])] = winding
        for joint in backend.guide_joint_idx[0]:
            q[int(q_starts[joint])] = GUIDE_RATIO * winding
        backend.joint_q.assign(q)
        backend.joint_qd.assign(qd)
        articulation_ids = backend._wp.array(
            np.asarray(backend.world_articulation_idx[0], dtype=np.int32),
            dtype=backend._wp.int32,
            device=backend.model.device,
        )
        backend._newton.eval_fk(
            backend.model, backend.joint_q, backend.joint_qd,
            backend.state_0, indices=articulation_ids,
        )
        backend._newton.eval_fk(
            backend.model, backend.joint_q, backend.joint_qd,
            backend.state_1, indices=articulation_ids,
        )
        backend._sync_reset_joint_coordinates(np.asarray([0], dtype=np.int64), q, qd)

        # Match the solver's reduced coordinate row for world 0 so the next step tests
        # reset preservation of continuous MJWarp qpos, not an artificial state mismatch.
        qpos = backend.solver.mjw_data.qpos.numpy().copy()
        mj_q_starts = backend.solver.mj_q_start.numpy()
        local_joint_index = {joint: local for local, joint in enumerate(backend.world_joint_idx[0])}
        qd_starts = backend.model.joint_qd_start.numpy()
        for joint in backend.world_joint_idx[0]:
            local = local_joint_index[joint]
            mj_start = int(mj_q_starts[local])
            q0, q1 = int(q_starts[joint]), int(q_starts[joint + 1])
            qd0, qd1 = int(qd_starts[joint]), int(qd_starts[joint + 1])
            if mj_start >= 0:
                qpos[0, mj_start:mj_start + (q1 - q0)] = q[q0:q1]
        backend.solver.mjw_data.qpos.assign(qpos)

        before_qpos_world0 = backend.solver.mjw_data.qpos.numpy()[0].copy()
        before_state_q = {
            name: getattr(backend, name).joint_q.numpy().copy()
            for name in ("state_0", "state_1")
        }
        before_state_qd = {
            name: getattr(backend, name).joint_qd.numpy().copy()
            for name in ("state_0", "state_1")
        }
        before_body = {}
        for state_name in ("state_0", "state_1"):
            state = getattr(backend, state_name)
            before_body[(state_name, "body_q")] = state.body_q.numpy().copy()
            before_body[(state_name, "body_qd")] = state.body_qd.numpy().copy()
        before_targets = (
            backend.control.joint_target_q.numpy().copy(),
            backend.control.joint_target_qd.numpy().copy(),
        )

        reset_pose = np.asarray([[2.8, 0.9, 0.7, 0.06]], dtype=np.float64)
        backend.reset(env_ids=np.asarray([1]), poses=reset_pose)
        after_qpos_world0 = backend.solver.mjw_data.qpos.numpy()[0]
        np.testing.assert_array_equal(after_qpos_world0, before_qpos_world0)

        for state_name in ("state_0", "state_1"):
            state = getattr(backend, state_name)
            state_q = state.joint_q.numpy()
            state_qd = state.joint_qd.numpy()
            for q0, q1, qd0, qd1 in backend.world_joint_slices[0]:
                np.testing.assert_array_equal(state_q[q0:q1], before_state_q[state_name][q0:q1])
                np.testing.assert_array_equal(state_qd[qd0:qd1], before_state_qd[state_name][qd0:qd1])
            for body_world in (0, 2):
                indices = np.asarray(backend.world_body_idx[body_world], dtype=np.int64)
                np.testing.assert_array_equal(
                    state.body_q.numpy()[indices], before_body[(state_name, "body_q")][indices]
                )
                np.testing.assert_array_equal(
                    state.body_qd.numpy()[indices], before_body[(state_name, "body_qd")][indices]
                )

        reset_source_q = backend.joint_q.numpy()
        reset_source_qd = backend.joint_qd.numpy()
        for state_name in ("state_0", "state_1"):
            state = getattr(backend, state_name)
            state_q = state.joint_q.numpy()
            state_qd = state.joint_qd.numpy()
            for q0, q1, qd0, qd1 in backend.world_joint_slices[1]:
                np.testing.assert_array_equal(state_q[q0:q1], reset_source_q[q0:q1])
                np.testing.assert_array_equal(state_qd[qd0:qd1], reset_source_qd[qd0:qd1])

        target_q_mask = np.ones_like(before_targets[0], dtype=bool)
        target_qd_mask = np.ones_like(before_targets[1], dtype=bool)
        for joint in backend.world_joint_idx[1]:
            q0 = int(backend.model.joint_target_q_start.numpy()[joint])
            q1 = int(backend.model.joint_target_q_start.numpy()[joint + 1])
            qd0 = int(qd_starts[joint])
            qd1 = int(qd_starts[joint + 1])
            target_q_mask[q0:q1] = False
            target_qd_mask[qd0:qd1] = False
        np.testing.assert_array_equal(
            backend.control.joint_target_q.numpy()[target_q_mask],
            before_targets[0][target_q_mask],
        )
        np.testing.assert_array_equal(
            backend.control.joint_target_qd.numpy()[target_qd_mask],
            before_targets[1][target_qd_mask],
        )

        backend.step(np.zeros((backend.nworld, 4), dtype=np.float64))
        state = backend.read()
        self.assertTrue(np.isfinite(state["pos"]).all())
        self.assertTrue(np.allclose(state["pos"][1, :2], [2.8, 0.9], atol=0.03))
        post_q = backend.state_0.joint_q.numpy()
        leader = backend.wheel_joint_idx["left"][0][0]
        follower = backend.wheel_joint_idx["left"][0][1]
        self.assertGreater(post_q[int(q_starts[leader])], 2.0 * np.pi)
        self.assertGreater(
            float(backend.solver.mjw_data.qpos.numpy()[0, int(mj_q_starts[local_joint_index[leader]])]),
            2.0 * np.pi,
        )
        self.assertAlmostEqual(
            float(post_q[int(q_starts[follower])]),
            float(post_q[int(q_starts[leader])]),
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
