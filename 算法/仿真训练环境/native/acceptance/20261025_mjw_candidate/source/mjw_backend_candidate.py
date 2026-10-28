"""External prototype: replace Newton XPBD with the official Newton SolverMuJoCo/MJWarp path.

This prototype deliberately reuses the active NewtonXPBDGBackend's world maps, observation
assembly, and selected-articulation reset implementation. It patches the finalized model's
candidate-only inertias/contact distances before constructing SolverMuJoCo. Move those
literal corrections into _robot_builder before any final repository integration.
"""
from __future__ import annotations

import numpy as np

from newton_backend import NewtonXPBDGBackend
from physics_backend import (
    GIMBAL_JOINT_NAMES,
    GIMBAL_YAW_RATE,
    GIMBAL_PITCH_RATE,
    MAX_TRACK_SPEED,
    PITCH_MAX,
    PITCH_MIN,
    TRACK_ACCELERATION,
    WHEEL_RADIUS,
    WHEELS_PER_SIDE,
)


CHASSIS_INERTIA_DIAG = np.asarray((0.2475, 1.4575, 1.606), dtype=np.float64)
PITCH_INERTIA_DIAG = np.asarray((1.3333333333333334e-5,) * 3, dtype=np.float64)
SOLVER_CONFIG = {
    "njmax": 256,
    "nconmax": 64,
    "iterations": 40,
    "ls_iterations": 8,
    "solver": "newton",
    "integrator": "implicitfast",
    "cone": "elliptic",
    "impratio": 10.0,
}


def _name_tail(label: object) -> str:
    return str(label).replace("\\", "/").rsplit("/", 1)[-1]


def _as_json_value(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            return item.item()
        return item
    return arr.tolist()


def _unique_floats(values) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return sorted({float(x) for x in np.round(arr, 9)})


def _unique_rows(values, world: int, nworld: int, width: int) -> list[list[float]]:
    arr = _world_row(values, world, nworld).astype(np.float64, copy=False).reshape(-1, width)
    return sorted({tuple(float(x) for x in np.round(row, 9)) for row in arr})


def _world_row(values, world: int, nworld: int) -> np.ndarray:
    """Read shared templates or explicit per-world rows without reshaping guesses."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    if arr.shape[0] == 1:
        return arr[0].reshape(-1)
    if arr.shape[0] == nworld:
        return arr[world].reshape(-1)
    raise ValueError(f"cannot resolve world {world} from leading dimension of shape {arr.shape} for {nworld} worlds")


class NewtonMuJoCoWarpCandidate(NewtonXPBDGBackend):
    """Candidate backend for CPU validation first, then an explicitly authorized GPU probe."""

    BACKEND_REVISION = "newton_mujoco_warp_candidate_v3_capacity_shared_model_map"

    def __init__(self, *args, **kwargs):
        self._mjw_solver_ready = False
        super().__init__(*args, **kwargs)

        import newton

        self._patch_candidate_model()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            separate_worlds=True,
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=1,
            **SOLVER_CONFIG,
        )
        if self.solver.use_mujoco_cpu:
            raise RuntimeError("candidate unexpectedly selected native MuJoCo CPU")
        if not self.solver.mjw_model.opt.run_collision_detection:
            raise RuntimeError("candidate must use MuJoCo Warp's internal collision pipeline")

        self._actuator_observables = self.solver.observables(
            kinds={self.solver.ObservableKind.QFRC_ACTUATOR}
        )
        self.collision_pipeline = None
        self.contacts = None
        self.last_gimbal_effort = np.zeros((self.nworld, len(GIMBAL_JOINT_NAMES)), dtype=np.float64)
        self._effort_observation_valid = np.zeros(self.nworld, dtype=np.bool_)
        self._mjw_solver_ready = True
        self.reset()

    def _patch_candidate_model(self) -> None:
        """Correct the prototype's candidate model before the MJWarp model is exported."""
        labels = [_name_tail(value) for value in self.model.body_label]
        body_world = np.asarray(self.model.body_world.numpy(), dtype=np.int64)
        inertia = np.asarray(self.model.body_inertia.numpy()).copy()
        inv_inertia = np.asarray(self.model.body_inv_inertia.numpy()).copy()

        self._pitch_body_by_world: list[int] = []
        for world in range(self.nworld):
            base_id = int(self.base_body[world])
            pitch_ids = [
                body_id
                for body_id, label in enumerate(labels)
                if body_world[body_id] == world
                and (label == "gimbal_pitch" or label.endswith("_gimbal_pitch"))
            ]
            if len(pitch_ids) != 1:
                raise RuntimeError(f"world {world}: expected one gimbal pitch body, got {pitch_ids}")
            pitch_id = pitch_ids[0]
            self._pitch_body_by_world.append(pitch_id)
            for body_id, diagonal in (
                (base_id, CHASSIS_INERTIA_DIAG),
                (pitch_id, PITCH_INERTIA_DIAG),
            ):
                tensor = np.diag(diagonal).astype(inertia.dtype, copy=False)
                inertia[body_id] = tensor
                inv_inertia[body_id] = np.linalg.inv(tensor).astype(inv_inertia.dtype, copy=False)

        self.model.body_inertia.assign(inertia)
        self.model.body_inv_inertia.assign(inv_inertia)
        self.model.shape_gap.assign(np.zeros_like(self.model.shape_gap.numpy()))
        self.model.shape_margin.assign(np.zeros_like(self.model.shape_margin.numpy()))

    def reset(self, env_ids=None, poses: np.ndarray | None = None) -> None:
        ids = np.arange(self.nworld, dtype=np.int64) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        super().reset(env_ids=ids, poses=poses)
        self.last_effort[ids] = 0.0
        if hasattr(self, "last_gimbal_effort"):
            self.last_gimbal_effort[ids] = 0.0
        if not hasattr(self, "_effort_observation_valid"):
            self._effort_observation_valid = np.zeros(self.nworld, dtype=np.bool_)
        else:
            self._effort_observation_valid[ids] = False
        if not getattr(self, "_mjw_solver_ready", False) or not len(ids):
            return

        mask = np.zeros(self.nworld + 1, dtype=np.bool_)
        mask[ids] = True
        wp_mask = self._wp.array(mask, dtype=self._wp.bool, device=self.model.device)
        # flags=0 keeps the custom joint pose/velocity written by super().reset(); this
        # call clears only SolverMuJoCo's selected-world warmstart/control buffers.
        self.solver.reset(self.state_0, world_mask=wp_mask, flags=0)

    def _snapshot_actuator_effort(self) -> None:
        qfrc = np.asarray(self._actuator_observables.qfrc_actuator.numpy(), dtype=np.float64).reshape(-1)
        wheel = np.zeros((self.nworld, 2 * WHEELS_PER_SIDE), dtype=np.float64)
        gimbal = np.zeros((self.nworld, len(GIMBAL_JOINT_NAMES)), dtype=np.float64)
        for world in range(self.nworld):
            for side, base in (("left", 0), ("right", WHEELS_PER_SIDE)):
                for index, dof in enumerate(self.wheel_dof[side][world]):
                    wheel[world, base + index] = qfrc[dof]
            for index, dof in enumerate(self.gimbal_qd_dof[world]):
                gimbal[world, index] = qfrc[dof]
        self.last_effort = wheel
        self.last_gimbal_effort = gimbal
        self._effort_observation_valid[:] = True

    def step(self, actions: np.ndarray) -> None:
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        if actions.shape != (self.nworld, 4):
            raise ValueError(f"actions must be ({self.nworld}, 4), got {actions.shape}")

        desired = actions[:, :2] * MAX_TRACK_SPEED
        change = TRACK_ACCELERATION * self.policy_dt
        self.track_command += np.clip(desired - self.track_command, -change, change)
        self.wheel_target_qd = (self.track_command / WHEEL_RADIUS)[:, self.pi.wheel_side]
        self.gimbal_target += actions[:, 2:4] * np.asarray(
            [GIMBAL_YAW_RATE, GIMBAL_PITCH_RATE], dtype=np.float64
        ) * self.policy_dt
        self.gimbal_target[:, 0] = (self.gimbal_target[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
        self.gimbal_target[:, 1] = np.clip(self.gimbal_target[:, 1], PITCH_MIN, PITCH_MAX)

        qd_target = self.control.joint_target_qd.numpy().copy()
        q_target = self.control.joint_target_q.numpy().copy()
        for world in range(self.nworld):
            for side in ("left", "right"):
                base = 0 if side == "left" else WHEELS_PER_SIDE
                for index, dof in enumerate(self.wheel_target_dof[side][world]):
                    qd_target[dof] = self.wheel_target_qd[world, base + index]
            for index, coord in enumerate(self.gimbal_target_q_coord[world]):
                q_target[coord] = self.gimbal_target[world, index]
        self.control.joint_target_qd.assign(qd_target)
        self.control.joint_target_q.assign(q_target)

        # SolverMuJoCo runs its own MuJoCo Warp collision pipeline. Do not build a
        # second Newton CollisionPipeline or pass its contacts into this solver.
        for _ in range(self.decimation):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                None,
                self.dt,
                observables=self._actuator_observables,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.step_count += 1
        self._snapshot_actuator_effort()

    def read(self) -> dict[str, np.ndarray]:
        result = super().read()
        result["effort"] = self.last_effort.copy()
        result["gimbal_effort"] = self.last_gimbal_effort.copy()
        result["effort_observation_valid"] = self._effort_observation_valid.copy()
        return result

    def identity(self) -> dict:
        import mujoco
        import mujoco_warp

        shape_world = np.asarray(self.model.shape_world.numpy(), dtype=np.int64)
        materials = {}
        for field in ("mu", "ke", "kd"):
            values = np.asarray(getattr(self.model, f"shape_material_{field}").numpy())
            materials[field] = _unique_floats(values[shape_world == 0])
            for world in range(1, self.nworld):
                if _unique_floats(values[shape_world == world]) != materials[field]:
                    raise RuntimeError(f"world {world} has different shape material {field}")

        geom_gap = self.solver.mjw_model.geom_gap.numpy()
        geom_margin = self.solver.mjw_model.geom_margin.numpy()
        actual_gaps = _unique_floats(_world_row(geom_gap, 0, self.nworld))
        actual_margins = _unique_floats(_world_row(geom_margin, 0, self.nworld))
        for world in range(1, self.nworld):
            if _unique_floats(_world_row(geom_gap, world, self.nworld)) != actual_gaps:
                raise RuntimeError(f"world {world} exported geom gap differs from world 0")
            if _unique_floats(_world_row(geom_margin, world, self.nworld)) != actual_margins:
                raise RuntimeError(f"world {world} exported geom margin differs from world 0")

        body_inertia = np.asarray(self.model.body_inertia.numpy())
        inertia_summary = []
        for body_id in self.world_body_idx[0]:
            inertia_summary.append({
                "label": _name_tail(self.model.body_label[body_id]),
                "diag_kg_m2": np.diag(body_inertia[body_id]).astype(float).tolist(),
            })
        options = self.solver.mjw_model.opt
        impratio_invsqrt = np.asarray(options.impratio_invsqrt.numpy(), dtype=np.float64)
        impratio_actual = _unique_floats(1.0 / np.square(impratio_invsqrt))
        eq_solref = self.solver.mjw_model.eq_solref
        eq_solimp = self.solver.mjw_model.eq_solimp
        joint_effort_limit = np.asarray(self.model.joint_effort_limit.numpy(), dtype=np.float64)
        effort_limits = {
            "wheel_dof": _unique_floats([
                joint_effort_limit[dof]
                for side in ("left", "right")
                for dof in self.wheel_dof[side][0]
            ]),
            "gimbal_dof": _unique_floats([
                joint_effort_limit[dof] for dof in self.gimbal_qd_dof[0]
            ]),
        }
        return {
            "engine": "newton_mujoco_warp",
            "backend_revision": self.BACKEND_REVISION,
            "device": str(self.model.device),
            "newton": getattr(self._newton, "__version__", "unknown"),
            "warp": str(self._wp.config.version),
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(mujoco_warp.__version__),
            "nworld": self.nworld,
            "dt_s": self.dt,
            "decimation": self.decimation,
            "policy_dt_s": self.policy_dt,
            "dof": int(self.model.joint_dof_count),
            "joints": int(self.model.joint_count),
            "mimic_relations_per_world": int(self._mimic_relations),
            "bodies": int(self.model.body_count),
            "mimic_constraints": {
                "exported_equalities_per_world": int(self.solver.mjw_model.neq),
                "eq_solref_array_shape": list(eq_solref.shape),
                "eq_solimp_array_shape": list(eq_solimp.shape),
                "eq_solref_values_world0": _unique_rows(eq_solref.numpy(), 0, self.nworld, 2),
                "eq_solimp_values_world0": _unique_rows(eq_solimp.numpy(), 0, self.nworld, 5),
            },
            "solver": {
                "class": type(self.solver).__name__,
                "backend": "mujoco_warp",
                "use_mujoco_cpu": bool(self.solver.use_mujoco_cpu),
                "use_mujoco_contacts": bool(self.solver.mjw_model.opt.run_collision_detection),
                "separate_worlds": True,
                "update_data_interval": int(self.solver.update_data_interval),
                "iterations_actual": _as_json_value(options.iterations),
                "ls_iterations_actual": _as_json_value(options.ls_iterations),
                "solver_actual": _as_json_value(options.solver),
                "integrator_actual": _as_json_value(options.integrator),
                "cone_actual": _as_json_value(options.cone),
                "impratio_invsqrt_actual": _as_json_value(options.impratio_invsqrt),
                "impratio_actual": impratio_actual,
                "requested_config": dict(SOLVER_CONFIG),
                "requested_capacity_per_world": {
                    "nconmax": int(SOLVER_CONFIG["nconmax"]),
                    "njmax": int(SOLVER_CONFIG["njmax"]),
                },
                "actual_capacity_per_world": {
                    "nconmax": int(self.solver.mjw_data.naconmax) // self.nworld,
                    "njmax": int(self.solver.mjw_data.njmax),
                },
                "exported_actuators": int(self.solver.mjw_model.nactuator),
                "exported_equalities": int(self.solver.mjw_model.neq),
            },
            "contact": {
                "source": "mujoco_warp_internal",
                "model_shape_gap_values_per_world": _unique_floats(
                    self.model.shape_gap.numpy()[shape_world == 0]
                ),
                "model_shape_margin_values_per_world": _unique_floats(
                    self.model.shape_margin.numpy()[shape_world == 0]
                ),
                "exported_geom_gap_values_per_world": actual_gaps,
                "exported_geom_margin_values_per_world": actual_margins,
            "exported_geom_gap_array_shape": list(geom_gap.shape),
            "exported_geom_margin_array_shape": list(geom_margin.shape),
            },
            "material_values_per_world": materials,
            "inertia_diag_kg_m2_world0": inertia_summary,
            "chassis_inertia_diag_kg_m2": np.diag(body_inertia[self.base_body[0]]).astype(float).tolist(),
            "gimbal_pitch_inertia_diag_kg_m2": np.diag(
                body_inertia[self._pitch_body_by_world[0]]
            ).astype(float).tolist(),
            "joint_dof_effort_limit_values_world0_nm": effort_limits,
            "drive_semantics": {
                "wheel": "Newton joint_target_qd velocity control; no direct side PI torque feed-forward applied",
                "gimbal": "Newton joint_target_q position control",
                "effort_source": "SolverMuJoCo QFRC_ACTUATOR observable mapped through joint_qd_start",
                "side_pi_feed_forward_applied": False,
                "effort_observation_valid_after_policy_step": True,
            },
        }


