"""Versioned, canonical task/physics identity shared by native training and evaluation."""
from __future__ import annotations

import hashlib
import json
import copy
import math
from typing import Any

IDENTITY_SCHEMA_VERSION = "native_task_contract_v2"
_RUNTIME_CONTRACT_FIELDS = {
    "run_id", "phase", "note", "sha_scope", "observation_contract_sha256",
    "task_identity_sha256", "task_identity", "identity_schema_version",
    "provider_contract_sha256", "profile_enabled", "physics_note", "verification",
    "reason", "capability_metrics_allowed", "capability_blockers",
}
_PHYSICS_RUNTIME_FIELDS = {
    "nworld", "dof", "joints", "bodies", "equalities", "actuators",
    "mimic_relations", "mimic_relations_per_world", "note",
}
_EXPECTED_WORLD_TOPOLOGY = {
    "newton_xpbd": {
        "dof_per_world": 24,
        "joints_per_world": 19,
        "bodies_per_world": 19,
        "mimic_relations_per_world": 14,
    },
    "mujoco_cpu": {
        "dof_per_world": 24,
        "joints_per_world": 19,
        "equalities_per_world": 14,
        "actuators_per_world": 16,
        "mimic_relations_per_world": 14,
    },
    "newton_mujoco_warp": {
        "dof_per_world": 24,
        "joints_per_world": 19,
        "bodies_per_world": 19,
        "mimic_relations_per_world": 14,
        "equalities_per_world": 14,
        "actuators_per_world": 16,
    },
}


class ContractMismatchError(ValueError):
    """Raised when a checkpoint does not match the requested native task contract."""


def _divide_total(name: str, value: Any, worlds: int) -> int:
    total = int(value)
    if worlds < 1 or total < 0 or total % worlds:
        raise ValueError(f"physics field {name}={value!r} is not divisible by nworld={worlds}")
    return total // worlds


def _strict_count(name: str, value: Any, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"physics field {name}={value!r} must be an integer >= {minimum}")
    return value


def _normalize_per_world_values(name: str, values: Any, worlds: int) -> list[Any]:
    """Collapse a shared scalar template or identical per-world copies to one value."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"physics field {name} must be a non-empty JSON array")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(float(value)) for value in values):
        raise ValueError(f"physics field {name} must contain finite numeric values")
    if len(values) == 1:
        return copy.deepcopy(values)
    if len(values) != worlds:
        raise ValueError(f"physics field {name} has {len(values)} values; expected 1 or nworld={worlds}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"physics field {name} differs across worlds")
    return [copy.deepcopy(first)]


def _newton_mujoco_warp_topology(physics: dict[str, Any], worlds: int) -> dict[str, int]:
    for name in ("backend_revision", "newton", "warp", "mujoco", "mujoco_warp"):
        value = physics.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"newton_mujoco_warp physics identity is missing {name}")

    solver = physics.get("solver")
    if not isinstance(solver, dict):
        raise ValueError("newton_mujoco_warp physics identity is missing solver details")
    if not isinstance(solver.get("class"), str) or not solver["class"].strip():
        raise ValueError("newton_mujoco_warp solver class is missing")
    if solver.get("backend") != "mujoco_warp":
        raise ValueError("newton_mujoco_warp solver backend must be mujoco_warp")
    if solver.get("use_mujoco_cpu") is not False or solver.get("use_mujoco_contacts") is not True:
        raise ValueError("newton_mujoco_warp must use MuJoCo-Warp contacts without the CPU solver")
    if solver.get("separate_worlds") is not True:
        raise ValueError("newton_mujoco_warp must report separate per-world models")

    expected_capacity = {"nconmax": 64, "njmax": 256}
    for name in ("requested_capacity_per_world", "actual_capacity_per_world"):
        capacity = solver.get(name)
        if not isinstance(capacity, dict) or set(capacity) != set(expected_capacity):
            raise ValueError(f"solver.{name} must equal {expected_capacity!r}")
        for key, expected_value in expected_capacity.items():
            actual_value = _strict_count(f"solver.{name}.{key}", capacity[key], minimum=1)
            if actual_value != expected_value:
                raise ValueError(f"solver.{name} must equal {expected_capacity!r}")
    requested_config = solver.get("requested_config")
    if not isinstance(requested_config, dict):
        raise ValueError(f"solver.requested_config must request per-world capacity {expected_capacity!r}")
    for key, expected_value in expected_capacity.items():
        actual_value = _strict_count(f"solver.requested_config.{key}", requested_config.get(key), minimum=1)
        if actual_value != expected_value:
            raise ValueError(f"solver.requested_config must request per-world capacity {expected_capacity!r}")
    for name, minimum in (
        ("update_data_interval", 1), ("iterations_actual", 1), ("ls_iterations_actual", 1),
        ("solver_actual", 0), ("integrator_actual", 0), ("cone_actual", 0),
    ):
        _strict_count(f"solver.{name}", solver.get(name), minimum=minimum)
    for name in ("impratio_invsqrt_actual", "impratio_actual"):
        values = solver.get(name)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"solver.{name} must be a non-empty numeric array")

    exported_equalities = _strict_count("solver.exported_equalities", solver.get("exported_equalities"))
    exported_actuators = _strict_count("solver.exported_actuators", solver.get("exported_actuators"))
    mimic = physics.get("mimic_constraints")
    if not isinstance(mimic, dict):
        raise ValueError("newton_mujoco_warp physics identity is missing exported mimic constraints")
    exported_mimic_equalities = _strict_count(
        "mimic_constraints.exported_equalities_per_world",
        mimic.get("exported_equalities_per_world"),
    )
    if exported_equalities != exported_mimic_equalities:
        raise ValueError("solver and exported mimic equality counts disagree")
    for name in ("eq_solref_values_world0", "eq_solimp_values_world0"):
        values = mimic.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"newton_mujoco_warp mimic constraints are missing {name}")
        def finite_tree(value: Any) -> bool:
            if isinstance(value, list):
                return bool(value) and all(finite_tree(item) for item in value)
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)))
        if not finite_tree(values):
            raise ValueError(f"newton_mujoco_warp mimic constraints {name} must be finite numeric arrays")

    contact = physics.get("contact")
    if (not isinstance(contact, dict) or not isinstance(contact.get("source"), str)
            or not contact["source"].strip()):
        raise ValueError("newton_mujoco_warp physics identity is missing contact source/details")
    for name in (
        "model_shape_gap_values_per_world", "model_shape_margin_values_per_world",
        "exported_geom_gap_values_per_world", "exported_geom_margin_values_per_world",
    ):
        if name not in contact:
            raise ValueError(f"newton_mujoco_warp contact identity is missing {name}")

    materials = physics.get("material_values_per_world")
    if not isinstance(materials, dict) or not {"mu", "ke", "kd"}.issubset(materials):
        raise ValueError("newton_mujoco_warp physics identity is missing per-world material values")

    device = physics.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("newton_mujoco_warp physics identity is missing device")
    dt_s, policy_dt_s = physics.get("dt_s"), physics.get("policy_dt_s")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(float(value)) or value <= 0 for value in (dt_s, policy_dt_s)):
        raise ValueError("newton_mujoco_warp dt_s and policy_dt_s must be finite positive numbers")
    decimation = _strict_count("decimation", physics.get("decimation"), minimum=1)
    if not math.isclose(float(policy_dt_s), float(dt_s) * decimation, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("newton_mujoco_warp policy_dt_s must equal dt_s * decimation")

    semantic_fields = (
        "inertia_diag_kg_m2_world0", "chassis_inertia_diag_kg_m2",
        "gimbal_pitch_inertia_diag_kg_m2", "joint_dof_effort_limit_values_world0_nm",
        "drive_semantics",
    )
    for name in semantic_fields:
        if name not in physics or physics[name] in (None, [], {}):
            raise ValueError(f"newton_mujoco_warp physics identity is missing {name}")

    topology = {
        "dof_per_world": _divide_total("dof", _strict_count("dof", physics.get("dof")), worlds),
        "joints_per_world": _divide_total("joints", _strict_count("joints", physics.get("joints")), worlds),
        "bodies_per_world": _divide_total("bodies", _strict_count("bodies", physics.get("bodies")), worlds),
        "mimic_relations_per_world": _strict_count(
            "mimic_relations_per_world", physics.get("mimic_relations_per_world")),
        "equalities_per_world": exported_equalities,
        "actuators_per_world": exported_actuators,
    }
    return topology


def canonical_physics_identity(physics: dict[str, Any]) -> dict[str, Any]:
    """Normalize aggregate multi-world counts into a strict per-world topology."""
    if not isinstance(physics, dict):
        raise ValueError("physics identity must be a dictionary")
    engine = physics.get("engine")
    if engine not in _EXPECTED_WORLD_TOPOLOGY:
        raise ValueError(f"unsupported physics engine in identity: {engine!r}")
    worlds = (_strict_count("nworld", physics.get("nworld", 1), minimum=1)
              if engine == "newton_mujoco_warp" else int(physics.get("nworld", 1)))
    if worlds < 1:
        raise ValueError("physics identity nworld must be positive")

    if engine == "newton_xpbd":
        topology = {
            "dof_per_world": _divide_total("dof", physics.get("dof"), worlds),
            "joints_per_world": _divide_total("joints", physics.get("joints"), worlds),
            "bodies_per_world": _divide_total("bodies", physics.get("bodies"), worlds),
            "mimic_relations_per_world": int(physics.get("mimic_relations_per_world", 0)),
        }
    elif engine == "newton_mujoco_warp":
        topology = _newton_mujoco_warp_topology(physics, worlds)
    else:
        # MuJoCo reports model topology once even when many independent MjData worlds exist.
        topology = {
            "dof_per_world": int(physics.get("dof", 0)),
            "joints_per_world": int(physics.get("joints", 0)),
            "equalities_per_world": int(physics.get("equalities", 0)),
            "actuators_per_world": int(physics.get("actuators", 0)),
            "mimic_relations_per_world": int(physics.get("mimic_relations", 0)),
        }

    expected = _EXPECTED_WORLD_TOPOLOGY[engine]
    if topology != expected:
        raise ValueError(f"unexpected per-world {engine} topology: {topology!r}; expected {expected!r}")

    runtime_fields = _PHYSICS_RUNTIME_FIELDS
    if engine == "newton_mujoco_warp":
        runtime_fields = runtime_fields | {"runtime_diagnostics", "naconmax_total", "mjwarp_naconmax_total"}
    canonical = copy.deepcopy({k: v for k, v in physics.items() if k not in runtime_fields})
    canonical["device"] = str(physics.get("device", "cpu"))
    canonical["world_topology"] = topology
    if engine == "newton_mujoco_warp":
        solver = canonical["solver"]
        for name in ("impratio_invsqrt_actual", "impratio_actual"):
            solver[name] = _normalize_per_world_values(f"solver.{name}", solver.get(name), worlds)
        materials = canonical["material_values_per_world"]
        for name, values in materials.items():
            materials[name] = _normalize_per_world_values(f"material_values_per_world.{name}", values, worlds)
        contact = canonical["contact"]
        for name in (
            "model_shape_gap_values_per_world", "model_shape_margin_values_per_world",
            "exported_geom_gap_values_per_world", "exported_geom_margin_values_per_world",
        ):
            contact[name] = _normalize_per_world_values(f"contact.{name}", contact[name], worlds)
        # Array shapes vary with world count and describe storage, not the exported model.
        contact.pop("exported_geom_gap_array_shape", None)
        contact.pop("exported_geom_margin_array_shape", None)
        mimic = canonical["mimic_constraints"]
        mimic.pop("eq_solref_array_shape", None)
        mimic.pop("eq_solimp_array_shape", None)
    return canonical


def canonical_contract_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """Return stable semantic fields; profiling and per-run metadata never affect identity."""
    if not isinstance(contract, dict):
        raise ValueError("observation contract must be a dictionary")
    task_backend = contract.get("task_backend")
    task = contract.get("task")
    if task_backend not in ("simplified", "fast_stop_v2"):
        raise ValueError(f"contract has missing or unsupported task_backend: {task_backend!r}")
    if not isinstance(task, dict) or task.get("backend") != task_backend:
        raise ValueError("contract task configuration does not match task_backend")
    source = {k: v for k, v in contract.items() if k not in _RUNTIME_CONTRACT_FIELDS}
    source["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    source["task_backend"] = task_backend
    source["task"] = task
    source["physics"] = canonical_physics_identity(contract.get("physics"))
    return source


def _identity_sha256(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_identity_sha256(contract: dict[str, Any]) -> str:
    return _identity_sha256(canonical_contract_identity(contract))


def attach_contract_identity(contract: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_contract_identity(contract)
    digest = _identity_sha256(canonical)
    contract["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    contract["task_identity"] = canonical
    contract["task_identity_sha256"] = digest
    # Preserve the existing field name for metrics and checkpoint readers; it now uses v2 scope.
    contract["observation_contract_sha256"] = digest
    contract["sha_scope"] = ("native_task_contract_v2: observation channels, task/reward/termination, "
                              "and normalized per-world physics; excludes profiling, world count, "
                              "verification evidence, and run metadata")
    return contract


def validate_checkpoint_preflight(
    checkpoint: dict[str, Any], *, task_backend: str, physics: str, device: str,
    sim_dt: float, decimation: int, allow_legacy: bool = False,
    task_backend_explicit: bool = False,
) -> str:
    """Reject known task/physics mismatches before allocating a physics environment."""
    if not isinstance(checkpoint, dict):
        raise ContractMismatchError("checkpoint payload is not a dictionary")
    identity = checkpoint.get("identity") if isinstance(checkpoint.get("identity"), dict) else {}
    env = identity.get("env") if isinstance(identity.get("env"), dict) else {}
    schema = checkpoint.get("identity_schema_version") or identity.get("identity_schema_version")
    saved_sha = checkpoint.get("task_identity_sha256") or identity.get("task_identity_sha256")
    legacy = schema != IDENTITY_SCHEMA_VERSION or not saved_sha
    if legacy:
        if not allow_legacy:
            raise ContractMismatchError(
                "legacy checkpoint has no versioned native task contract; pass "
                "--allow-legacy-checkpoint-contract only for an explicitly selected exploratory evaluation"
            )
        if not task_backend_explicit:
            raise ContractMismatchError("legacy checkpoint opt-in requires explicit --task-backend")
    else:
        identity_schema = identity.get("identity_schema_version")
        identity_sha = identity.get("task_identity_sha256")
        if identity_schema != schema or identity_sha != saved_sha:
            raise ContractMismatchError("checkpoint contract fields disagree between payload and identity")

    backends = [checkpoint.get("task_backend"), identity.get("task_backend"), env.get("task_backend")]
    known_backends = [value for value in backends if value is not None]
    if known_backends and (len(set(known_backends)) != 1 or known_backends[0] != task_backend):
        raise ContractMismatchError(
            f"checkpoint task_backend={known_backends[0]!r} does not match requested {task_backend!r}"
        )
    if not known_backends and not legacy:
        raise ContractMismatchError("versioned checkpoint is missing task_backend identity")

    if not legacy:
        task_identity = checkpoint.get("task_identity")
        if not isinstance(task_identity, dict):
            raise ContractMismatchError("versioned checkpoint is missing task_identity")
        if _identity_sha256(task_identity) != saved_sha:
            raise ContractMismatchError("checkpoint task_identity content does not match task_identity_sha256")
        if task_identity.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
            raise ContractMismatchError("checkpoint task_identity has an unsupported identity schema")
        if task_identity.get("task_backend") != task_backend:
            raise ContractMismatchError("checkpoint task_identity task_backend does not match the requested backend")
        env_sha = env.get("task_identity_sha256")
        if env_sha != saved_sha:
            raise ContractMismatchError("checkpoint identity.env task_identity_sha256 disagrees with payload")
        alias_sha = checkpoint.get("observation_contract_sha256")
        if alias_sha not in (None, saved_sha):
            raise ContractMismatchError("checkpoint observation/task identity SHA fields disagree")

    physics_identity = env.get("physics")
    if not isinstance(physics_identity, dict):
        if legacy and allow_legacy:
            return "legacy_opt_in_unverified_physics"
        raise ContractMismatchError("checkpoint is missing environment physics identity")
    actual_engine = physics_identity.get("engine")
    if actual_engine != physics:
        raise ContractMismatchError(f"checkpoint physics engine={actual_engine!r} does not match {physics!r}")
    actual_device = str(physics_identity.get("device", "cpu"))
    if actual_device != device:
        raise ContractMismatchError(f"checkpoint physics device={actual_device!r} does not match {device!r}")
    actual_dt = float(physics_identity.get("dt_s", env.get("sim_dt_s", -1.0)))
    actual_decimation = int(physics_identity.get("decimation", env.get("decimation", -1)))
    if abs(actual_dt - float(sim_dt)) > 1e-12 or actual_decimation != int(decimation):
        raise ContractMismatchError(
            f"checkpoint physics {actual_dt:g}s/{actual_decimation} does not match "
            f"requested {float(sim_dt):g}s/{int(decimation)}"
        )
    topology = canonical_physics_identity(physics_identity)["world_topology"]
    expected_topology = _EXPECTED_WORLD_TOPOLOGY[physics]
    if topology != expected_topology:
        raise ContractMismatchError(f"checkpoint per-world topology does not match {physics}")
    return "legacy_opt_in" if legacy else "preflight_match"


def validate_checkpoint_contract(
    checkpoint: dict[str, Any], expected_sha256: str, *, allow_legacy: bool = False,
) -> str:
    """Compare a safely loaded checkpoint to the rebuilt live contract."""
    identity = checkpoint.get("identity") if isinstance(checkpoint.get("identity"), dict) else {}
    schema = checkpoint.get("identity_schema_version") or identity.get("identity_schema_version")
    saved_sha = checkpoint.get("task_identity_sha256") or identity.get("task_identity_sha256")
    if schema != IDENTITY_SCHEMA_VERSION or not saved_sha:
        if allow_legacy:
            return "legacy_opt_in"
        raise ContractMismatchError("checkpoint has no versioned native task contract")
    task_identity = checkpoint.get("task_identity")
    if not isinstance(task_identity, dict) or _identity_sha256(task_identity) != saved_sha:
        raise ContractMismatchError("checkpoint task_identity content does not match task_identity_sha256")
    if identity.get("task_identity_sha256") != saved_sha:
        raise ContractMismatchError("checkpoint identity task_identity_sha256 disagrees with payload")
    if checkpoint.get("observation_contract_sha256") not in (None, saved_sha):
        raise ContractMismatchError("checkpoint observation/task identity SHA fields disagree")
    if saved_sha != expected_sha256:
        raise ContractMismatchError(
            f"checkpoint task identity SHA {saved_sha} does not match live SHA {expected_sha256}"
        )
    return "strict_match"
