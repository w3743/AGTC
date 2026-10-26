"""CPU-only unit tests for the native v2 task contract identity."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import unittest

TRAIN = pathlib.Path(__file__).resolve().parents[1] / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from contract_identity import (  # noqa: E402
    IDENTITY_SCHEMA_VERSION,
    ContractMismatchError,
    attach_contract_identity,
    canonical_contract_identity,
    canonical_physics_identity,
    contract_identity_sha256,
    validate_checkpoint_contract,
    validate_checkpoint_preflight,
)


def physics(nworld: int = 64, *, device: str = "cuda:0", dt_s: float = 1e-3,
            decimation: int = 50) -> dict:
    return {
        "engine": "newton_xpbd",
        "device": device,
        "newton": "1.7.0",
        "warp": "1.18.0",
        "mujoco": None,
        "nworld": nworld,
        "dt_s": dt_s,
        "decimation": decimation,
        "policy_dt_s": dt_s * decimation,
        "dof": 24 * nworld,
        "joints": 19 * nworld,
        "bodies": 19 * nworld,
        "mimic_relations_per_world": 14,
        "effort_limit_nm": 2.0,
        "drive_kp": 0.04,
        "drive_ki": 0.1,
        "mu": 0.8,
        "max_track_speed_mps": 15.0 / 3.6,
        "drive_semantics": "velocity targets with target_kd=0.04; side PI not applied",
        "gravity_mps2": [0.0, 0.0, -9.81],
        "solver": {"kind": "newton_xpbd", "iterations": 10, "shape_ke": 2500.0, "shape_kd": 100.0},
        "contact": {"shape_mu": 0.8},
    }


def mujoco_cpu_physics(nworld: int = 64) -> dict:
    return {
        "engine": "mujoco_cpu", "mujoco": "3.17.0", "nworld": nworld,
        "dt_s": 0.001, "decimation": 50, "policy_dt_s": 0.05,
        "dof": 24, "joints": 19, "equalities": 14, "actuators": 16,
        "effort_limit_nm": 2.0, "drive_kp": 0.04, "drive_ki": 0.1,
        "mu": 0.8, "max_track_speed_mps": 15.0 / 3.6, "mimic_relations": 14,
    }


def newton_mujoco_warp_physics(
    nworld: int = 64, *, backend_revision: str = "newton_mujoco_warp_candidate_v3_capacity_shared_model_map",
) -> dict:
    return {
        "engine": "newton_mujoco_warp",
        "backend_revision": backend_revision,
        "device": "cuda:0",
        "newton": "1.7.0",
        "warp": "1.18.0",
        "mujoco": "3.15.0",
        "mujoco_warp": "3.15.0",
        "nworld": nworld,
        "dt_s": 0.001,
        "decimation": 50,
        "policy_dt_s": 0.05,
        "dof": 24 * nworld,
        "joints": 19 * nworld,
        "bodies": 19 * nworld,
        "mimic_relations_per_world": 14,
        "solver": {
            "class": "SolverMuJoCo",
            "backend": "mujoco_warp",
            "use_mujoco_cpu": False,
            "use_mujoco_contacts": True,
            "separate_worlds": True,
            "update_data_interval": 1,
            "iterations_actual": 40,
            "ls_iterations_actual": 8,
            "solver_actual": 2,
            "integrator_actual": 3,
            "cone_actual": 1,
            "impratio_invsqrt_actual": [0.3162277638912201] * nworld,
            "impratio_actual": [10.000000134],
            "requested_config": {
                "njmax": 256, "nconmax": 64, "iterations": 40,
                "ls_iterations": 8, "solver": "newton", "integrator": "implicitfast",
                "cone": "elliptic", "impratio": 10.0,
            },
            "requested_capacity_per_world": {"nconmax": 64, "njmax": 256},
            "actual_capacity_per_world": {"nconmax": 64, "njmax": 256},
            "exported_actuators": 16,
            "exported_equalities": 14,
        },
        "contact": {
            "source": "mujoco_warp_internal",
            "model_shape_gap_values_per_world": [0.0] * nworld,
            "model_shape_margin_values_per_world": [0.0] * nworld,
            "exported_geom_gap_values_per_world": [0.0] * nworld,
            "exported_geom_margin_values_per_world": [0.0] * nworld,
            "exported_geom_gap_array_shape": [nworld, 20],
            "exported_geom_margin_array_shape": [nworld, 20],
        },
        "mimic_constraints": {
            "exported_equalities_per_world": 14,
            "eq_solref_array_shape": [nworld, 14],
            "eq_solimp_array_shape": [nworld, 14],
            "eq_solref_values_world0": [[0.02, 1.0]],
            "eq_solimp_values_world0": [[0.899999976, 0.949999988, 0.001, 0.5, 2.0]],
        },
        "material_values_per_world": {
            "mu": [0.800000012] * nworld,
            "ke": [2500.0] * nworld,
            "kd": [100.0] * nworld,
        },
        "inertia_diag_kg_m2_world0": [
            {"label": "base", "diag_kg_m2": [0.2475, 1.4575, 1.606]},
            {"label": "gimbal_pitch", "diag_kg_m2": [0.0000133333] * 3},
        ],
        "chassis_inertia_diag_kg_m2": [0.2475, 1.4575, 1.606],
        "gimbal_pitch_inertia_diag_kg_m2": [0.0000133333] * 3,
        "joint_dof_effort_limit_values_world0_nm": {"wheel_dof": [2.0], "gimbal_dof": [1.0]},
        "drive_semantics": {
            "wheel": "Newton joint_target_qd velocity control",
            "gimbal": "Newton joint_target_q position control",
            "side_pi_feed_forward_applied": False,
        },
        "runtime_diagnostics": {
            "mjwarp_naconmax_total": nworld * 64,
            "mjwarp_nconmax_requested_per_world": 64,
            "mjwarp_nconmax_actual_per_world": 64,
            "mjwarp_njmax_actual_per_world": 256,
        },
    }


def contract(*, nworld: int = 64, profile: bool = False,
             task_backend: str = "fast_stop_v2") -> dict:
    task = {
        "backend": task_backend,
        "episode_length_s": 60.0,
        "reset_sampling": {"field_size_m": 100.0, "spawn_clearance_m": 0.01},
        "reward_termination": {"goal_tolerance": 0.4, "goal_hold": 0.3,
                               "goal_stop_speed": 0.2, "goal_stop_angular_speed": 0.3,
                               "timeout_behavior": "terminal_failure"},
    }
    channels = {
        "lidar": {"shape": [1, 400], "mode": "real", "backend": "warp_kernel"},
        "depth": {"shape": [1, 160, 88], "mode": "placeholder", "backend": "constant"},
    }
    return {
        "run_id": f"run-{nworld}",
        "phase": "train",
        "profile_enabled": profile,
        "policy_dt_s": 0.05,
        "physics_dt_s": 1e-3,
        "decimation": 50,
        "physics_experimental": False,
        "task_backend": task_backend,
        "task": task,
        "physics": physics(nworld),
        "channels": channels,
        "action_components": ["left_track_speed", "right_track_speed"],
    }


def checkpoint(task_contract: dict) -> dict:
    attached = attach_contract_identity(copy.deepcopy(task_contract))
    identity = {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "task_identity_sha256": attached["task_identity_sha256"],
        "task_backend": attached["task_backend"],
        "env": {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "task_identity_sha256": attached["task_identity_sha256"],
            "task_backend": attached["task_backend"],
            "physics": attached["physics"],
        },
    }
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "task_backend": attached["task_backend"],
        "task_identity": attached["task_identity"],
        "task_identity_sha256": attached["task_identity_sha256"],
        "observation_contract_sha256": attached["observation_contract_sha256"],
        "identity": identity,
    }


class TaskContractTests(unittest.TestCase):
    def test_existing_engine_canonical_hashes_remain_unchanged(self):
        self.assertEqual(contract_identity_sha256(contract()),
                         "cfa2fcdc1239cdca944fb6db0125f0b0e8ea995ea7f9e5353d5b53a38ead361e")
        cpu_contract = contract()
        cpu_contract["physics"] = mujoco_cpu_physics()
        self.assertEqual(contract_identity_sha256(cpu_contract),
                         "3d55b1878d48b513b9d31888c803687a1681560c6b589a5fd4ecc86b4faee622")

    def test_newton_mujoco_warp_world_topology_and_runtime_fields_normalize(self):
        train_contract = contract(nworld=64)
        train_contract["physics"] = newton_mujoco_warp_physics(64)
        eval_contract = contract(nworld=1)
        eval_contract["physics"] = newton_mujoco_warp_physics(1)
        train_identity = canonical_contract_identity(train_contract)
        self.assertEqual(train_identity, canonical_contract_identity(eval_contract))
        self.assertEqual(train_identity["physics"]["world_topology"], {
            "dof_per_world": 24, "joints_per_world": 19, "bodies_per_world": 19,
            "mimic_relations_per_world": 14, "equalities_per_world": 14,
            "actuators_per_world": 16,
        })
        self.assertNotIn("runtime_diagnostics", train_identity["physics"])
        self.assertNotIn("exported_geom_gap_array_shape", train_identity["physics"]["contact"])
        self.assertNotIn("eq_solref_array_shape", train_identity["physics"]["mimic_constraints"])

    def test_newton_mujoco_warp_revision_changes_task_identity(self):
        base = contract(nworld=64)
        base["physics"] = newton_mujoco_warp_physics(64)
        changed = copy.deepcopy(base)
        changed["physics"]["backend_revision"] = "newton_mujoco_warp_candidate_next"
        saved = checkpoint(base)
        self.assertNotEqual(contract_identity_sha256(base), contract_identity_sha256(changed))
        with self.assertRaisesRegex(ContractMismatchError, "does not match live SHA"):
            validate_checkpoint_contract(saved, contract_identity_sha256(changed))

    def test_newton_mujoco_warp_checkpoint_preflight_and_64_to_1_match(self):
        train_contract = contract(nworld=64)
        train_contract["physics"] = newton_mujoco_warp_physics(64)
        saved = checkpoint(train_contract)
        self.assertEqual(validate_checkpoint_preflight(
            saved, task_backend="fast_stop_v2", physics="newton_mujoco_warp",
            device="cuda:0", sim_dt=1e-3, decimation=50,
        ), "preflight_match")
        eval_contract = contract(nworld=1)
        eval_contract["physics"] = newton_mujoco_warp_physics(1)
        self.assertEqual(validate_checkpoint_contract(
            saved, contract_identity_sha256(eval_contract)), "strict_match")
        with self.assertRaisesRegex(ContractMismatchError, "engine"):
            validate_checkpoint_preflight(
                saved, task_backend="fast_stop_v2", physics="newton_xpbd",
                device="cuda:0", sim_dt=1e-3, decimation=50,
            )

    def test_newton_mujoco_warp_counts_capacities_and_per_world_values_are_strict(self):
        bad_count = newton_mujoco_warp_physics(64)
        bad_count["dof"] += 1
        with self.assertRaisesRegex(ValueError, "not divisible|topology"):
            canonical_physics_identity(bad_count)

        bad_export = newton_mujoco_warp_physics(64)
        bad_export["solver"]["exported_equalities"] = 13
        with self.assertRaisesRegex(ValueError, "equality counts disagree"):
            canonical_physics_identity(bad_export)

        bad_capacity = newton_mujoco_warp_physics(64)
        bad_capacity["solver"]["actual_capacity_per_world"]["njmax"] = 128
        with self.assertRaisesRegex(ValueError, "actual_capacity_per_world"):
            canonical_physics_identity(bad_capacity)

        bad_world_value = newton_mujoco_warp_physics(64)
        bad_world_value["material_values_per_world"]["mu"][-1] = 0.9
        with self.assertRaisesRegex(ValueError, "differs across worlds"):
            canonical_physics_identity(bad_world_value)

        bad_boolean_capacity = newton_mujoco_warp_physics(64)
        bad_boolean_capacity["solver"]["actual_capacity_per_world"]["nconmax"] = True
        with self.assertRaisesRegex(ValueError, "actual_capacity_per_world.nconmax"):
            canonical_physics_identity(bad_boolean_capacity)

        bad_fractional_topology = newton_mujoco_warp_physics(64)
        bad_fractional_topology["dof"] = 24.0 * 64
        with self.assertRaisesRegex(ValueError, "dof.*integer"):
            canonical_physics_identity(bad_fractional_topology)

        bad_policy_period = newton_mujoco_warp_physics(64)
        bad_policy_period["policy_dt_s"] = 0.1
        with self.assertRaisesRegex(ValueError, "policy_dt_s"):
            canonical_physics_identity(bad_policy_period)

        bad_solver_value = newton_mujoco_warp_physics(64)
        bad_solver_value["solver"]["impratio_actual"] = [float("inf")]
        with self.assertRaisesRegex(ValueError, "impratio_actual"):
            canonical_physics_identity(bad_solver_value)

    def test_profile_and_world_count_are_runtime_metadata(self):
        train_contract = contract(nworld=64, profile=True)
        eval_contract = contract(nworld=1, profile=False)
        self.assertEqual(contract_identity_sha256(train_contract),
                         contract_identity_sha256(eval_contract))

    def test_task_backend_and_reward_changes_change_identity(self):
        base = contract()
        self.assertNotEqual(contract_identity_sha256(base),
                            contract_identity_sha256(contract(task_backend="simplified")))
        changed = copy.deepcopy(base)
        changed["task"]["reward_termination"]["goal_hold"] = 0.5
        self.assertNotEqual(contract_identity_sha256(base), contract_identity_sha256(changed))

    def test_real_physics_changes_change_identity(self):
        base = contract()
        changed_device = copy.deepcopy(base)
        changed_device["physics"]["device"] = "cpu"
        changed_solver = copy.deepcopy(base)
        changed_solver["physics"]["solver"]["iterations"] = 12
        changed_dt = copy.deepcopy(base)
        changed_dt["physics"]["dt_s"] = 0.0005
        for changed in (changed_device, changed_solver, changed_dt):
            self.assertNotEqual(contract_identity_sha256(base), contract_identity_sha256(changed))

    def test_invalid_world_topology_is_rejected(self):
        invalid = contract(nworld=1)
        invalid["physics"]["bodies"] = 20
        with self.assertRaisesRegex(ValueError, "topology"):
            canonical_contract_identity(invalid)

    def test_checkpoint_preflight_and_full_live_contract_match(self):
        saved = checkpoint(contract(nworld=64, profile=True))
        status = validate_checkpoint_preflight(
            saved, task_backend="fast_stop_v2", physics="newton_xpbd",
            device="cuda:0", sim_dt=1e-3, decimation=50,
        )
        self.assertEqual(status, "preflight_match")
        live = contract(nworld=1, profile=False)
        self.assertEqual(validate_checkpoint_contract(
            saved, contract_identity_sha256(live)), "strict_match")

    def test_checkpoint_preflight_rejects_task_physics_and_tampering(self):
        saved = checkpoint(contract())
        with self.assertRaises(ContractMismatchError):
            validate_checkpoint_preflight(
                saved, task_backend="simplified", physics="newton_xpbd",
                device="cuda:0", sim_dt=1e-3, decimation=50,
            )
        with self.assertRaises(ContractMismatchError):
            validate_checkpoint_preflight(
                saved, task_backend="fast_stop_v2", physics="newton_xpbd",
                device="cpu", sim_dt=1e-3, decimation=50,
            )
        changed = copy.deepcopy(saved)
        changed["task_identity"]["task"]["episode_length_s"] = 30.0
        with self.assertRaisesRegex(ContractMismatchError, "content"):
            validate_checkpoint_preflight(
                changed, task_backend="fast_stop_v2", physics="newton_xpbd",
                device="cuda:0", sim_dt=1e-3, decimation=50,
            )

    def test_legacy_checkpoint_requires_explicit_opt_in_and_task_choice(self):
        legacy = {"identity": {"env": {"physics": physics(nworld=1)}}}
        kwargs = dict(task_backend="simplified", physics="newton_xpbd", device="cuda:0",
                      sim_dt=1e-3, decimation=50)
        with self.assertRaisesRegex(ContractMismatchError, "legacy"):
            validate_checkpoint_preflight(legacy, **kwargs)
        with self.assertRaisesRegex(ContractMismatchError, "explicit"):
            validate_checkpoint_preflight(legacy, allow_legacy=True, **kwargs)
        self.assertEqual(validate_checkpoint_preflight(
            legacy, allow_legacy=True, task_backend_explicit=True, **kwargs), "legacy_opt_in")


if __name__ == "__main__":
    unittest.main()
