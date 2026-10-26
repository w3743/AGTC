"""Versioned, canonical task/physics identity shared by native training and evaluation."""
from __future__ import annotations

import hashlib
import json
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
}


class ContractMismatchError(ValueError):
    """Raised when a checkpoint does not match the requested native task contract."""


def _divide_total(name: str, value: Any, worlds: int) -> int:
    total = int(value)
    if worlds < 1 or total < 0 or total % worlds:
        raise ValueError(f"physics field {name}={value!r} is not divisible by nworld={worlds}")
    return total // worlds


def canonical_physics_identity(physics: dict[str, Any]) -> dict[str, Any]:
    """Normalize aggregate multi-world counts into a strict per-world topology."""
    if not isinstance(physics, dict):
        raise ValueError("physics identity must be a dictionary")
    engine = physics.get("engine")
    if engine not in _EXPECTED_WORLD_TOPOLOGY:
        raise ValueError(f"unsupported physics engine in identity: {engine!r}")
    worlds = int(physics.get("nworld", 1))
    if worlds < 1:
        raise ValueError("physics identity nworld must be positive")

    if engine == "newton_xpbd":
        topology = {
            "dof_per_world": _divide_total("dof", physics.get("dof"), worlds),
            "joints_per_world": _divide_total("joints", physics.get("joints"), worlds),
            "bodies_per_world": _divide_total("bodies", physics.get("bodies"), worlds),
            "mimic_relations_per_world": int(physics.get("mimic_relations_per_world", 0)),
        }
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

    canonical = {k: v for k, v in physics.items() if k not in _PHYSICS_RUNTIME_FIELDS}
    canonical["device"] = str(physics.get("device", "cpu"))
    canonical["world_topology"] = topology
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