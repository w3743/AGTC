"""Newton's official MuJoCo-Warp solver backend for native vectorized training."""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from newton_backend import NewtonXPBDGBackend
from physics_backend import (
    DRIVE_KP,
    EFFORT_LIMIT,
    GIMBAL_JOINT_NAMES,
    GIMBAL_PITCH_RATE,
    GIMBAL_YAW_RATE,
    MAX_TRACK_SPEED,
    PITCH_MAX,
    PITCH_MIN,
    TRACK_ACCELERATION,
    WHEEL_RADIUS,
    WHEELS_PER_SIDE,
)


CHASSIS_INERTIA_DIAG = (
    26.4 * (0.3 ** 2 + 0.15 ** 2) / 12.0,
    26.4 * (0.8 ** 2 + 0.15 ** 2) / 12.0,
    26.4 * (0.8 ** 2 + 0.3 ** 2) / 12.0,
)
PITCH_INERTIA_DIAG = (0.05 * (0.04 ** 2 + 0.04 ** 2) / 12.0,) * 3
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

# MuJoCo-Warp 3.15 OverflowType bits: capacity flags exclude ITERATIONS and LS_ITERATIONS.
# These named Warp constants let the per-substep device counter classify the sticky bitmask.
_CAPACITY_OVERFLOW_MASK = wp.constant(0x9FF)
_OUTER_ITERATION_LIMIT_MASK = wp.constant(0x200)
_LINE_SEARCH_LIMIT_MASK = wp.constant(0x400)


@wp.kernel
def _clear_solver_event_counts(counts: wp.array2d(dtype=wp.int32)):
    world = wp.tid()
    counts[world, 0] = 0
    counts[world, 1] = 0
    counts[world, 2] = 0


@wp.kernel
def _accumulate_solver_event_counts(
    overflow: wp.array(dtype=wp.int32),
    counts: wp.array2d(dtype=wp.int32),
):
    world = wp.tid()
    bits = overflow[world]
    if (bits & _CAPACITY_OVERFLOW_MASK) != 0:
        counts[world, 0] = counts[world, 0] + 1
    if (bits & _OUTER_ITERATION_LIMIT_MASK) != 0:
        counts[world, 1] = counts[world, 1] + 1
    if (bits & _LINE_SEARCH_LIMIT_MASK) != 0:
        counts[world, 2] = counts[world, 2] + 1


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


def _world_row(values, world: int, nworld: int) -> np.ndarray:
    """Read a shared template or an explicit per-world row from a model array."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    if arr.shape[0] == 1:
        return arr[0].reshape(-1)
    if arr.shape[0] == nworld:
        return arr[world].reshape(-1)
    raise ValueError(
        f"cannot resolve world {world} from array shape {arr.shape} for {nworld} worlds"
    )


def _unique_rows(values, world: int, nworld: int, width: int) -> list[list[float]]:
    arr = _world_row(values, world, nworld).astype(np.float64, copy=False).reshape(-1, width)
    rows = {tuple(float(x) for x in np.round(row, 9)) for row in arr}
    return [list(row) for row in sorted(rows)]


def _unique_matrix_rows(values, width: int) -> list[list[float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, width)
    rows = {tuple(float(x) for x in np.round(row, 9)) for row in arr}
    return [list(row) for row in sorted(rows)]


class NewtonMuJoCoWarpBackend(NewtonXPBDGBackend):
    """Newton MuJoCo-Warp backend reusing the shared world maps and driver."""

    BACKEND_REVISION = "newton_mujoco_warp_v6_deterministic_identity"

    def __init__(
        self,
        nworld: int = 64,
        *,
        dt: float = 1e-3,
        decimation: int = 50,
        effort_limit: float = EFFORT_LIMIT,
        kp: float = DRIVE_KP,
        ki: float = 0.1,
        mu: float = 0.8,
        seed: int = 0,
        device: str | None = None,
        use_cuda_graph: bool = False,
        deterministic: wp.DeterministicMode | None = None,
    ):
        self.use_cuda_graph_requested = bool(use_cuda_graph)
        self._requested_deterministic_mode = deterministic
        self._cuda_graph = None
        self._cuda_graph_enabled = False
        self._cuda_graph_capture_s = 0.0
        self._cuda_graph_buffer_ptrs = {}
        super().__init__(
            nworld=nworld, dt=dt, decimation=decimation,
            effort_limit=effort_limit, kp=kp, ki=ki, mu=mu,
            seed=seed, device=device,
        )

    def _robot_builder_options(self) -> dict:
        return {
            "chassis_inertia": CHASSIS_INERTIA_DIAG,
            "pitch_inertia": PITCH_INERTIA_DIAG,
            "shape_gap": 0.0,
            "shape_margin": 0.0,
        }

    def _build_solver(self, newton):
        solver = newton.solvers.SolverMuJoCo(
            self.model,
            separate_worlds=True,
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=1,
            deterministic=self._requested_deterministic_mode,
            **SOLVER_CONFIG,
        )
        if solver.use_mujoco_cpu:
            raise RuntimeError("NewtonMuJoCoWarpBackend requires the MuJoCo-Warp solver")
        if not bool(solver.mjw_model.opt.run_collision_detection):
            raise RuntimeError("MuJoCo-Warp internal contact generation must be enabled")
        return solver

    def _build_contact_system(self, newton):
        # SolverMuJoCo owns its contacts; a second Newton collision pipeline would duplicate them.
        return None, None

    def _initialize_solver_state(self) -> None:
        import mujoco_warp

        # Silence only the repeated GPU printf for line-search exhaustion. The
        # solver still sets LS_ITERATIONS in mjw_data.overflow, which the
        # per-substep device counters collect below.
        self._ls_iterations_warning_bit = int(mujoco_warp.OverflowType.LS_ITERATIONS)
        self.solver.mjw_model.opt.warn_overflow &= ~self._ls_iterations_warning_bit
        self._solver_warn_overflow_mask = int(self.solver.mjw_model.opt.warn_overflow)
        if self._solver_warn_overflow_mask & self._ls_iterations_warning_bit:
            raise RuntimeError("MuJoCo-Warp LS_ITERATIONS console warning was not disabled")

        self._actuator_observables = self.solver.observables(
            kinds={self.solver.ObservableKind.QFRC_ACTUATOR}
        )
        self.last_gimbal_effort = np.zeros((self.nworld, len(GIMBAL_JOINT_NAMES)), dtype=np.float64)
        self._effort_observation_valid = np.zeros(self.nworld, dtype=np.bool_)
        self._overflow_event_counts = self._wp.zeros(
            shape=(self.nworld, 3), dtype=self._wp.int32, device=self.model.device
        )
        self.last_solver_diagnostics = {
            "policy_step_sampled": False,
            "capacity_overflow_substeps_by_world": [0] * self.nworld,
            "outer_iteration_limit_substeps_by_world": [0] * self.nworld,
            "line_search_limit_substeps_by_world": [0] * self.nworld,
        }
        self._mimic_eq_rows = np.empty(0, dtype=np.int64)
        self._mimic_eq_solref_requested = np.asarray([0.004, 1.0], dtype=np.float64)
        self._mimic_eq_solimp_requested = np.asarray(
            [0.999, 0.9999, 0.001, 0.5, 2.0], dtype=np.float64
        )
        labels = [_name_tail(label) for label in self.model.body_label]
        body_world = np.asarray(self.model.body_world.numpy(), dtype=np.int64)
        self._pitch_body_by_world = []
        for world in range(self.nworld):
            matches = [
                body
                for body, label in enumerate(labels)
                if body_world[body] == world and label == "gimbal_pitch"
            ]
            if len(matches) != 1:
                raise RuntimeError(f"world {world}: expected one gimbal_pitch body, got {matches}")
            self._pitch_body_by_world.append(int(matches[0]))
        self._set_mimic_constraint_parameters()
        self._assert_solver_buffers_on_requested_device()
        if self.use_cuda_graph_requested:
            self._initialize_cuda_graph()

    def _assert_solver_buffers_on_requested_device(self) -> None:
        expected = str(self.model.device)
        buffers = {
            "state_0.body_q": self.state_0.body_q,
            "state_0.body_qd": self.state_0.body_qd,
            "state_0.joint_q": self.state_0.joint_q,
            "state_0.joint_qd": self.state_0.joint_qd,
            "state_1.body_q": self.state_1.body_q,
            "control.joint_target_q": self.control.joint_target_q,
            "control.joint_target_qd": self.control.joint_target_qd,
            "mjwarp.geom_gap": self.solver.mjw_model.geom_gap,
            "mjwarp.qpos": self.solver.mjw_data.qpos,
            "mjwarp.qfrc_actuator": self._actuator_observables.qfrc_actuator,
            "mjwarp.overflow": self.solver.mjw_data.overflow,
            "event_counts": self._overflow_event_counts,
        }
        mismatches = {
            name: str(value.device)
            for name, value in buffers.items()
            if str(value.device) != expected
        }
        if mismatches:
            raise RuntimeError(f"physics buffers are not all on {expected}: {mismatches}")
        actual_ncon = int(self.solver.mjw_data.naconmax)
        actual_nj = int(self.solver.mjw_data.njmax)
        if actual_ncon % self.nworld or actual_ncon // self.nworld != SOLVER_CONFIG["nconmax"]:
            raise RuntimeError(f"unexpected MuJoCo-Warp nconmax total: {actual_ncon}")
        if actual_nj != SOLVER_CONFIG["njmax"]:
            raise RuntimeError(f"unexpected MuJoCo-Warp njmax: {actual_nj}")

    def _graph_buffer_pointer_snapshot(self) -> dict[str, int]:
        pointers = {}
        for state_name in ("state_0", "state_1"):
            state = getattr(self, state_name)
            for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
                pointers[f"{state_name}.{field}"] = int(getattr(state, field).ptr)
        for name, value in (
            ("control.q", self.control.joint_target_q),
            ("control.qd", self.control.joint_target_qd),
            ("mjw.qpos", self.solver.mjw_data.qpos),
            ("mjw.qvel", self.solver.mjw_data.qvel),
            ("mjw.ctrl", self.solver.mjw_data.ctrl),
            ("mjw.qacc", self.solver.mjw_data.qacc),
            ("mjw.qacc_warmstart", self.solver.mjw_data.qacc_warmstart),
            ("mjw.qfrc_applied", self.solver.mjw_data.qfrc_applied),
            ("mjw.xfrc_applied", self.solver.mjw_data.xfrc_applied),
            ("mjw.qfrc_actuator", self.solver.mjw_data.qfrc_actuator),
            ("mjw.nacon", self.solver.mjw_data.nacon),
            ("mjw.overflow", self.solver.mjw_data.overflow),
            ("mjw.contact.dist", self.solver.mjw_data.contact.dist),
            ("mjw.contact.worldid", self.solver.mjw_data.contact.worldid),
            ("mjw.efc.pos", self.solver.mjw_data.efc.pos),
            ("actuator_observable.qfrc", self._actuator_observables.qfrc_actuator),
            ("event_counts", self._overflow_event_counts),
        ):
            pointers[name] = int(value.ptr)
        return pointers

    def _initialize_cuda_graph(self) -> None:
        if not str(self.model.device).startswith("cuda"):
            raise ValueError("use_cuda_graph=True requires a CUDA model device")
        if self.decimation != 50 or self.decimation % 2:
            raise ValueError("CUDA graph capture requires an even decimation of exactly 50")
        if int(self.solver.update_data_interval) != 1:
            raise ValueError("CUDA graph capture requires update_data_interval=1")

        state_ids = (id(self.state_0), id(self.state_1))
        before = self._graph_buffer_pointer_snapshot()

        # Compile and warm the fixed solver/contact buffers eagerly once before capture.
        self._wp.launch(
            _clear_solver_event_counts,
            dim=self.nworld,
            inputs=[self._overflow_event_counts],
            device=self.model.device,
        )
        for _ in range(self.decimation):
            self._solver_substep()
        warmup_counts = np.asarray(self._overflow_event_counts.numpy(), dtype=np.int64)
        if np.any(warmup_counts[:, 0]):
            raise RuntimeError(f"capacity overflow during CUDA graph warmup: {warmup_counts[:, 0].tolist()}")
        if (id(self.state_0), id(self.state_1)) != state_ids:
            raise RuntimeError("even warmup decimation did not restore ping-pong state references")

        # Restore the initial state before recording, then capture the same 50 eager
        # substeps with the same stable state/control/MJWarp buffer addresses.
        self.reset()
        before_capture = self._graph_buffer_pointer_snapshot()
        if before_capture != before:
            raise RuntimeError("warmup/reset replaced a CUDA graph input or output buffer")

        capture_start = __import__("time").perf_counter()
        with self._wp.ScopedCapture(
            device=self.model.device,
            force_module_load=False,
        ) as capture:
            self._wp.launch(
                _clear_solver_event_counts,
                dim=self.nworld,
                inputs=[self._overflow_event_counts],
                device=self.model.device,
            )
            for _ in range(self.decimation):
                self._solver_substep()
        if capture.graph is None:
            raise RuntimeError("Warp returned no CUDA graph after capture")
        if (id(self.state_0), id(self.state_1)) != state_ids:
            raise RuntimeError("even graph capture did not restore ping-pong state references")
        self._cuda_graph = capture.graph
        self._wp.capture_launch(self._cuda_graph)
        self._wp.synchronize_device(self.model.device)
        capture_counts = np.asarray(self._overflow_event_counts.numpy(), dtype=np.int64)
        if np.any(capture_counts[:, 0]):
            raise RuntimeError(
                f"capacity overflow while warming the captured graph: {capture_counts[:, 0].tolist()}"
            )
        self._cuda_graph_capture_s = __import__("time").perf_counter() - capture_start
        self.reset()
        after_reset = self._graph_buffer_pointer_snapshot()
        if after_reset != before:
            raise RuntimeError("masked reset replaced a CUDA graph buffer")
        self.last_solver_diagnostics = {
            "policy_step_sampled": False,
            "capacity_overflow_substeps_by_world": [0] * self.nworld,
            "outer_iteration_limit_substeps_by_world": [0] * self.nworld,
            "line_search_limit_substeps_by_world": [0] * self.nworld,
        }
        self._cuda_graph_buffer_ptrs = before
        self._cuda_graph_enabled = True

    def _set_mimic_constraint_parameters(self) -> None:
        """Set exported direct-mimic equality values in CPU and MJWarp model arrays."""
        import mujoco

        cpu_model = self.solver.mj_model
        warp_model = self.solver.mjw_model
        if cpu_model is None or int(cpu_model.neq) != 14 or int(warp_model.neq) != 14:
            raise RuntimeError("expected exactly 14 exported direct-mimic equality rows")
        rows = np.flatnonzero(np.asarray(cpu_model.eq_type) == int(mujoco.mjtEq.mjEQ_JOINT))
        if rows.size != 14:
            raise RuntimeError(f"expected 14 JOINT equality rows, found {rows.size}")

        solref = self._mimic_eq_solref_requested
        solimp = self._mimic_eq_solimp_requested
        cpu_solref = np.asarray(cpu_model.eq_solref).copy()
        cpu_solimp = np.asarray(cpu_model.eq_solimp).copy()
        cpu_solref[rows, :] = solref
        cpu_solimp[rows, :] = solimp
        cpu_model.eq_solref[:] = cpu_solref
        cpu_model.eq_solimp[:] = cpu_solimp

        for name, target, width in (("eq_solref", solref, 2), ("eq_solimp", solimp, 5)):
            array = getattr(warp_model, name)
            values = np.asarray(array.numpy()).copy()
            if values.shape[-2] != int(cpu_model.neq) or values.shape[-1] != width:
                raise RuntimeError(f"unexpected MJWarp {name} shape {values.shape}")
            values[..., rows, :] = target.astype(values.dtype, copy=False)
            array.assign(values)
            actual = np.asarray(array.numpy()).reshape(-1, int(cpu_model.neq), width)
            if not np.allclose(actual[:, rows, :], target, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"MJWarp {name} readback differs from requested mimic values")

        if not np.allclose(cpu_model.eq_solref[rows], solref, rtol=0.0, atol=1e-7):
            raise RuntimeError("MuJoCo CPU equality solref readback differs from requested values")
        if not np.allclose(cpu_model.eq_solimp[rows], solimp, rtol=0.0, atol=1e-7):
            raise RuntimeError("MuJoCo CPU equality solimp readback differs from requested values")
        self._mimic_eq_rows = rows.astype(np.int64)

    def reset(self, env_ids=None, poses: np.ndarray | None = None) -> None:
        ids = (
            np.arange(self.nworld, dtype=np.int64)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int64)
        )
        super().reset(env_ids=ids, poses=poses)
        if not len(ids):
            return
        self.last_effort[ids] = 0.0
        self.last_gimbal_effort[ids] = 0.0
        self._effort_observation_valid[ids] = False

        # flags=0 preserves the selected custom Newton pose while clearing only the
        # selected worlds' MuJoCo-Warp control and warmstart buffers.
        mask = np.zeros(self.nworld + 1, dtype=np.bool_)
        mask[ids] = True
        world_mask = self._wp.array(mask, dtype=self._wp.bool, device=self.model.device)
        self.solver.reset(self.state_0, world_mask=world_mask, flags=0)

    def _solver_substep(self) -> None:
        # MuJoCo-Warp 3.15 ORs overflow flags and does not clear them per step.
        # Zero on-device before each physical substep so the following device counter
        # records independent events rather than repeated observations of a sticky bit.
        self.solver.mjw_data.overflow.zero_()
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
        self._wp.launch(
            _accumulate_solver_event_counts,
            dim=self.nworld,
            inputs=[self.solver.mjw_data.overflow, self._overflow_event_counts],
            device=self.model.device,
        )

    def _snapshot_actuator_effort(self) -> None:
        qfrc = np.asarray(
            self._actuator_observables.qfrc_actuator.numpy(), dtype=np.float64
        ).reshape(-1)
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
        self.gimbal_target[:, 0] = (self.gimbal_target[:, 0] + math.pi) % (2.0 * math.pi) - math.pi
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

        if self._cuda_graph_enabled:
            self._wp.capture_launch(self._cuda_graph)
        else:
            self._wp.launch(
                _clear_solver_event_counts,
                dim=self.nworld,
                inputs=[self._overflow_event_counts],
                device=self.model.device,
            )
            for _ in range(self.decimation):
                self._solver_substep()

        counts = np.asarray(self._overflow_event_counts.numpy(), dtype=np.int64)
        self.last_solver_diagnostics = {
            "policy_step_sampled": True,
            "capacity_overflow_substeps_by_world": counts[:, 0].tolist(),
            "outer_iteration_limit_substeps_by_world": counts[:, 1].tolist(),
            "line_search_limit_substeps_by_world": counts[:, 2].tolist(),
        }
        if np.any(counts[:, 0]):
            raise RuntimeError(
                "MuJoCo-Warp reported contact/constraint capacity overflow during "
                f"this policy step: {self.last_solver_diagnostics}"
            )

        self.step_count += 1
        self._snapshot_actuator_effort()

    def read(self) -> dict[str, np.ndarray]:
        result = super().read()
        result["effort"] = self.last_effort.copy()
        result["gimbal_effort"] = self.last_gimbal_effort.copy()
        result["effort_observation_valid"] = self._effort_observation_valid.copy()
        return result


    def _determinism_diagnostics(self) -> dict:
        import sys

        actual_mode = self.solver._deterministic
        module_items = []
        seen_modules = set()
        for module_name, module in sorted(sys.modules.items()):
            if module is None or not module_name.startswith("mujoco_warp._src."):
                continue
            module_items.append((module_name, module))
        solver_module = sys.modules.get(type(self.solver).__module__)
        solver_kernels = getattr(solver_module, "kernels", None)
        if solver_kernels is not None:
            module_items.append(("newton_solver_mujoco.kernels", solver_kernels))

        records = []
        for module_name, module in module_items:
            module_id = id(module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            options = self._wp.get_module_options(module)
            mode = options.get("deterministic")
            if mode is None:
                continue
            mode_name = getattr(mode, "name", str(mode))
            records.append({
                "module": module_name,
                "deterministic_mode": mode_name,
                "deterministic_max_records": int(options.get("deterministic_max_records", 0)),
            })

        requested = self._requested_deterministic_mode
        module_modes = {item["deterministic_mode"] for item in records}
        module_options_match_solver_mode = module_modes == {actual_mode.name}
        return {
            "requested_mode": None if requested is None else requested.name,
            "actual_solver_mode": actual_mode.name,
            "module_options_match_solver_mode": module_options_match_solver_mode,
            "solver_max_records": int(self.solver._deterministic_max_records),
            "mujoco_warp_module_options": records,
        }

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

        geom_gap_array = self.solver.mjw_model.geom_gap
        geom_margin_array = self.solver.mjw_model.geom_margin
        geom_gap = geom_gap_array.numpy()
        geom_margin = geom_margin_array.numpy()
        gaps = _unique_floats(_world_row(geom_gap, 0, self.nworld))
        margins = _unique_floats(_world_row(geom_margin, 0, self.nworld))
        for world in range(1, self.nworld):
            if _unique_floats(_world_row(geom_gap, world, self.nworld)) != gaps:
                raise RuntimeError(f"world {world} exported geom gap differs from world 0")
            if _unique_floats(_world_row(geom_margin, world, self.nworld)) != margins:
                raise RuntimeError(f"world {world} exported geom margin differs from world 0")

        options = self.solver.mjw_model.opt
        impratio_invsqrt = np.asarray(options.impratio_invsqrt.numpy(), dtype=np.float64)
        impratio_actual = _unique_floats(1.0 / np.square(impratio_invsqrt))
        eq_solref = self.solver.mjw_model.eq_solref
        eq_solimp = self.solver.mjw_model.eq_solimp
        body_inertia = np.asarray(self.model.body_inertia.numpy())
        inertia_summary = [
            {
                "label": _name_tail(self.model.body_label[body]),
                "diag_kg_m2": np.diag(body_inertia[body]).astype(float).tolist(),
            }
            for body in self.world_body_idx[0]
        ]
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
        naconmax_total = int(self.solver.mjw_data.naconmax)
        njmax_actual = int(self.solver.mjw_data.njmax)
        solver_config = {
            "class": type(self.solver).__name__,
            "backend": "mujoco_warp",
            "use_mujoco_cpu": bool(self.solver.use_mujoco_cpu),
            "use_mujoco_contacts": bool(self.solver.mjw_model.opt.run_collision_detection),
            "separate_worlds": bool(self.solver.mjw_data.qpos.shape[0] == self.nworld),
            "update_data_interval": int(self.solver.update_data_interval),
            "deterministic_mode": self.solver._deterministic.name,
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
                "nconmax": naconmax_total // self.nworld,
                "njmax": njmax_actual,
            },
            "exported_actuators": int(self.solver.mjw_model.nactuator),
            "exported_equalities": int(self.solver.mjw_model.neq),
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
            "bodies": int(self.model.body_count),
            "mimic_relations_per_world": int(self._mimic_relations),
            "mimic_constraints": {
                "exported_equalities_per_world": int(self.solver.mjw_model.neq),
                "eq_solref_array_shape": list(eq_solref.shape),
                "eq_solimp_array_shape": list(eq_solimp.shape),
                "eq_solref_values_world0": _unique_rows(eq_solref.numpy(), 0, self.nworld, 2),
                "eq_solimp_values_world0": _unique_rows(eq_solimp.numpy(), 0, self.nworld, 5),
                "mujoco_cpu_eq_solref_values": _unique_matrix_rows(self.solver.mj_model.eq_solref, 2),
                "mujoco_cpu_eq_solimp_values": _unique_matrix_rows(self.solver.mj_model.eq_solimp, 5),
                "parameter_source": (
                    "explicit post-export correction applied to MuJoCo CPU and MJWarp arrays; "
                    "not Newton custom attributes"
                ),
                "requested_cpu_proxy_solref": self._mimic_eq_solref_requested.tolist(),
                "requested_cpu_proxy_solimp": self._mimic_eq_solimp_requested.tolist(),
            },
            "solver": solver_config,
            "contact": {
                "source": "mujoco_warp_internal",
                "model_shape_gap_values_per_world": _unique_floats(
                    self.model.shape_gap.numpy()[shape_world == 0]
                ),
                "model_shape_margin_values_per_world": _unique_floats(
                    self.model.shape_margin.numpy()[shape_world == 0]
                ),
                "exported_geom_gap_values_per_world": gaps,
                "exported_geom_margin_values_per_world": margins,
                "exported_geom_gap_array_shape": list(geom_gap.shape),
                "exported_geom_margin_array_shape": list(geom_margin.shape),
            },
            "material_values_per_world": materials,
            "inertia_diag_kg_m2_world0": inertia_summary,
            "chassis_inertia_diag_kg_m2": np.diag(
                body_inertia[self.base_body[0]]
            ).astype(float).tolist(),
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
            "runtime_diagnostics": {
                **self.last_solver_diagnostics,
                "solver_warning_policy": {
                    "actual_warn_overflow_mask": self._solver_warn_overflow_mask,
                    "ls_iterations_console_warning_enabled": False,
                    "suppressed_console_warning_bits": ["LS_ITERATIONS"],
                    "ls_iterations_overflow_flag_preserved": True,
                    "line_search_limit_reporting": "structured_counters_only",
                    "line_search_limit_counter_field": "line_search_limit_substeps_by_world",
                },
                "cuda_graph_requested": self.use_cuda_graph_requested,
                "cuda_graph_enabled": self._cuda_graph_enabled,
                "cuda_graph_capture_s": self._cuda_graph_capture_s,
                "naconmax_total": naconmax_total,
                "naconmax_per_world": naconmax_total // self.nworld,
                "njmax_per_world": njmax_actual,
                "determinism": self._determinism_diagnostics(),
            },
            "naconmax_total": naconmax_total,
        }