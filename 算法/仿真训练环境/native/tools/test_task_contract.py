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