"""Prequential runner for the controlled 20-neuron learning ablation.

Each selected group/seed/sequence receives a fresh model and its own output
directory. Predictions are flushed to JSONL before the matching digit is
passed to ``observe``. The runner never reads live key input or starts the web
server.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

try:  # Support both direct script execution and package-style imports.
    from .ablation_window import WindowControlRNN
    from .model import CONTEXT_LENGTH, INPUT_COUNT, PlasticRNN
    from .online_model import DEFAULT_RANK, OnlinePlasticRNN
except ImportError:  # pragma: no cover - selected by direct script execution
    from ablation_window import WindowControlRNN
    from model import CONTEXT_LENGTH, INPUT_COUNT, PlasticRNN
    from online_model import DEFAULT_RANK, OnlinePlasticRNN


DEFAULT_SEEDS = (0, 20261027, 20261028)
DEFAULT_SEQUENCES = ("cycle", "triple")
DEFAULT_STEPS = 600
DEFAULT_CHECKPOINT_EVERY = 100
GRADIENT_CLIP_MAX_NORM = 1.0
SOURCE_SNAPSHOT_FILES = (
    "model.py",
    "online_model.py",
    "ablation_reference.py",
    "ablation_window.py",
    "run_ablation.py",
    "test_ablation_reference.py",
    "requirements.txt",
)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    model_kind: str
    effective: Mapping[str, Any]
    description: str


def _group(
    name: str,
    model_kind: str,
    *,
    learning_rate: float = 0.015,
    updates_per_press: int = 3,
    loss_scope: str = "all",
    training_context_alignment: str = "legacy_including_target",
    weight_decay: float = 1e-4,
    sparsity_strength: float = 0.0,
    state_refresh_policy: str = "replay_from_zero_window",
    credit_horizon: int | None = CONTEXT_LENGTH,
    gradient_estimator: str = "exact",
    description: str,
) -> GroupSpec:
    return GroupSpec(
        name=name,
        model_kind=model_kind,
        effective={
            "learning_rate": learning_rate,
            "updates_per_press": updates_per_press,
            "loss_scope": loss_scope,
            "training_context_alignment": training_context_alignment,
            "weight_decay": weight_decay,
            "sparsity_strength": sparsity_strength,
            "state_refresh_policy": state_refresh_policy,
            "credit_horizon": credit_horizon,
            "gradient_estimator": gradient_estimator,
            "gradient_clip_max_norm": GRADIENT_CLIP_MAX_NORM,
            "clip_rule": "production_float64_global_norm_no_epsilon",
            "context_length": CONTEXT_LENGTH,
        },
        description=description,
    )


GROUPS: dict[str, GroupSpec] = {
    "window_original": _group(
        "window_original",
        "window",
        description="W0: recent-window baseline, lr .015, three Adam updates, all-pair loss.",
    ),
    "window_sparse_only": _group(
        "window_sparse_only",
        "window",
        sparsity_strength=0.02,
        description="WS: W0 with input-weight proximal sparsity .02.",
    ),
    "window_updates1": _group(
        "window_updates1",
        "window",
        updates_per_press=1,
        description="W1: W0 with one Adam update per observed target.",
    ),
    "window_latest_only": _group(
        "window_latest_only",
        "window",
        updates_per_press=1,
        loss_scope="latest",
        description="W2: W1 with loss only on the newest target.",
    ),
    "window_lr003": _group(
        "window_lr003",
        "window",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        description="W3: W2 with learning rate .003.",
    ),
    "window_decay0": _group(
        "window_decay0",
        "window",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        weight_decay=0.0,
        description="W4: W3 with Adam weight decay disabled; matched window control.",
    ),
    "window_aligned": _group(
        "window_aligned",
        "window",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        training_context_alignment="sealed_prefix",
        weight_decay=0.0,
        description=(
            "WA: W4 with sealed-prefix training alignment; the target is scored "
            "from the exact pre-observation window."
        ),
    ),
    "exact_online_trunc32": _group(
        "exact_online_trunc32",
        "truncated_online",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        training_context_alignment="sealed_prefix",
        weight_decay=0.0,
        state_refresh_policy="persistent_forward_state",
        credit_horizon=CONTEXT_LENGTH,
        description="T32: persistent forward state with exact credit through 32 transitions.",
    ),
    "exact_online_matched": _group(
        "exact_online_matched",
        "exact_online",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        training_context_alignment="sealed_prefix",
        weight_decay=0.0,
        state_refresh_policy="persistent_forward_state",
        credit_horizon=None,
        description="E: persistent forward state with full-history exact RTRL credit.",
    ),
    "uoro_dense": _group(
        "uoro_dense",
        "uoro",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        training_context_alignment="sealed_prefix",
        weight_decay=0.0,
        state_refresh_policy="persistent_forward_state",
        credit_horizon=None,
        gradient_estimator="uoro_rank4",
        description="U0: E with rank-4 UORO gradient estimation.",
    ),
    "uoro_current": _group(
        "uoro_current",
        "uoro",
        learning_rate=0.003,
        updates_per_press=1,
        loss_scope="latest",
        training_context_alignment="sealed_prefix",
        weight_decay=0.0,
        sparsity_strength=0.02,
        state_refresh_policy="persistent_forward_state",
        credit_horizon=None,
        gradient_estimator="uoro_rank4",
        description="US: U0 with input-weight proximal sparsity .02.",
    ),
}


# The compared controls form a bridge chain. Each edge must change exactly the
# named effective field; this catches accidental confounds before any run.
EXPECTED_PAIR_DIFFS: dict[tuple[str, str], str] = {
    ("window_original", "window_sparse_only"): "sparsity_strength",
    ("window_original", "window_updates1"): "updates_per_press",
    ("window_updates1", "window_latest_only"): "loss_scope",
    ("window_latest_only", "window_lr003"): "learning_rate",
    ("window_lr003", "window_decay0"): "weight_decay",
    ("window_decay0", "window_aligned"): "training_context_alignment",
    ("window_aligned", "exact_online_trunc32"): "state_refresh_policy",
    ("exact_online_trunc32", "exact_online_matched"): "credit_horizon",
    ("exact_online_matched", "uoro_dense"): "gradient_estimator",
    ("uoro_dense", "uoro_current"): "sparsity_strength",
}


def validate_group_protocol() -> None:
    """Assert the declared bridge chain changes one effective field per edge."""

    for pair, expected_field in EXPECTED_PAIR_DIFFS.items():
        left, right = (GROUPS[name].effective for name in pair)
        keys = set(left) | set(right)
        differences = {
            key for key in keys if left.get(key) != right.get(key)
        }
        if differences != {expected_field}:
            raise AssertionError(
                f"invalid single-variable pair {pair}: expected only "
                f"{expected_field}, found {sorted(differences)}"
            )


def generate_sequence(name: str, steps: int) -> list[int]:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if name == "cycle":
        pattern = tuple(range(INPUT_COUNT))
    elif name == "triple":
        pattern = tuple(digit for digit in range(INPUT_COUNT) for _ in range(3))
    else:
        raise ValueError(f"unknown deterministic sequence: {name}")
    return [pattern[index % len(pattern)] for index in range(steps)]


def _reference_class(name: str):
    """Load the isolated exact-gradient reference when an online group is used."""

    module = importlib.import_module("ablation_reference")
    class_name = {
        "exact_online": "ExactOnlineRNN",
        "truncated_online": "TruncatedOnlineRNN",
    }[name]
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"ablation_reference is missing required {class_name}"
        ) from exc


def _construct_reference(model_class: Any, spec: GroupSpec, seed: int):
    """Pass supported common settings to the reference implementation."""

    candidate_kwargs: dict[str, Any] = {
        "seed": seed,
        "rank": DEFAULT_RANK,
        "learning_rate": spec.effective["learning_rate"],
        "sparsity_strength": spec.effective["sparsity_strength"],
        "weight_decay": spec.effective["weight_decay"],
    }
    if spec.model_kind == "truncated_online":
        candidate_kwargs["credit_horizon"] = spec.effective["credit_horizon"]

    signature = inspect.signature(model_class)
    parameters = signature.parameters
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = (
        candidate_kwargs
        if accepts_arbitrary
        else {
            name: value
            for name, value in candidate_kwargs.items()
            if name in parameters
        }
    )
    missing = [
        parameter.name
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"cannot construct {model_class.__name__}; unsupported required arguments: "
            f"{missing}"
        )
    model = model_class(**kwargs)
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise TypeError(f"{model_class.__name__} must expose its Adam optimizer")
    groups = getattr(optimizer, "param_groups", [])
    if len(groups) != 1:
        raise TypeError(f"{model_class.__name__} must use one optimizer parameter group")
    group = groups[0]
    if float(group.get("lr", math.nan)) != float(spec.effective["learning_rate"]):
        raise ValueError(f"{model_class.__name__} did not apply the configured learning rate")
    if float(group.get("weight_decay", 0.0)) != float(spec.effective["weight_decay"]):
        raise ValueError(f"{model_class.__name__} did not apply the configured weight decay")
    if int(spec.effective["updates_per_press"]) != 1:
        raise AssertionError("online models are configured for exactly one update per press")
    if float(getattr(model, "sparsity_strength", 0.0)) != float(
        spec.effective["sparsity_strength"]
    ):
        raise ValueError(f"{model_class.__name__} did not apply sparsity_strength")
    return model


def create_model(spec: GroupSpec, seed: int):
    if spec.model_kind == "window":
        effective = spec.effective
        return WindowControlRNN(
            seed=seed,
            learning_rate=effective["learning_rate"],
            updates_per_press=effective["updates_per_press"],
            weight_decay=effective["weight_decay"],
            sparsity_strength=effective["sparsity_strength"],
            loss_scope=effective["loss_scope"],
            training_context_alignment=effective["training_context_alignment"],
            context_length=effective["context_length"],
        )
    if spec.model_kind == "uoro":
        effective = spec.effective
        return OnlinePlasticRNN(
            seed=seed,
            rank=DEFAULT_RANK,
            learning_rate=effective["learning_rate"],
            sparsity_strength=effective["sparsity_strength"],
            exact_readout_step=True,
        )
    if spec.model_kind in {"exact_online", "truncated_online"}:
        return _construct_reference(_reference_class(spec.model_kind), spec, seed)
    raise ValueError(f"unsupported model kind: {spec.model_kind}")


def _parameter_vector(model: Any) -> torch.Tensor:
    method = getattr(model, "parameter_vector", None)
    if callable(method):
        source = method()
    elif callable(getattr(model, "parameters", None)):
        # The frozen legacy PlasticRNN exposes parameters only through
        # nn.Module; preserve its natural named-parameter order for smoke QA.
        source = torch.cat(
            [parameter.detach().reshape(-1) for parameter in model.parameters()]
        )
    else:
        raise TypeError(f"{type(model).__name__} must expose parameters")
    vector = torch.as_tensor(source).detach().cpu().contiguous().reshape(-1)
    if not bool(torch.isfinite(vector).all().item()):
        raise FloatingPointError(f"{type(model).__name__} has non-finite initial parameters")
    return vector.clone()


def _state_vector(model: Any) -> torch.Tensor:
    method = getattr(model, "state_vector", None)
    if not callable(method):
        raise TypeError(f"{type(model).__name__} must expose state_vector()")
    vector = torch.as_tensor(method()).detach().cpu().contiguous().reshape(-1)
    if not bool(torch.isfinite(vector).all().item()):
        raise FloatingPointError(f"{type(model).__name__} has non-finite recurrent state")
    return vector.clone()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_payload(
    model: Any,
    *,
    group: str,
    seed: int,
    sequence: str,
    step: int,
    initial_parameter_sha256: str,
) -> dict[str, Any]:
    if not callable(getattr(model, "state_dict", None)):
        raise TypeError(f"{type(model).__name__} must be a torch module")
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None or not callable(getattr(optimizer, "state_dict", None)):
        raise TypeError(f"{type(model).__name__} must expose optimizer.state_dict()")
    full_state = getattr(model, "checkpoint_state", None)
    training_state = full_state() if callable(full_state) else None
    return {
        "format_version": 1,
        "model_kind": type(model).__name__,
        "group": group,
        "seed": seed,
        "sequence": sequence,
        "step": step,
        "initial_parameter_sha256": initial_parameter_sha256,
        "model_state": _cpu_clone(model.state_dict()),
        "optimizer_state": _cpu_clone(optimizer.state_dict()),
        "training_state": _cpu_clone(training_state),
    }


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _model_diagnostics(model: Any) -> dict[str, Any]:
    method = getattr(model, "diagnostics", None)
    if not callable(method):
        raise TypeError(f"{type(model).__name__} must expose diagnostics()")
    data = method()
    if not isinstance(data, Mapping):
        raise TypeError(f"{type(model).__name__}.diagnostics() must return a mapping")
    return dict(data)


def _predict_probabilities(model: Any) -> list[float]:
    result = model.predict_next()
    if not isinstance(result, (tuple, list)) or not result:
        raise TypeError(f"{type(model).__name__}.predict_next() returned no probabilities")
    probabilities = [float(value) for value in result[0]]
    if len(probabilities) != INPUT_COUNT:
        raise ValueError(f"predict_next must return {INPUT_COUNT} probabilities")
    if not all(math.isfinite(value) and value >= 0 for value in probabilities):
        raise FloatingPointError("predict_next returned invalid probabilities")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError(f"predict_next probabilities sum to {sum(probabilities)}")
    return probabilities


def _training_updates(diagnostics: Mapping[str, Any]) -> int:
    value = diagnostics.get("optimizer_updates")
    if type(value) is not int or value < 0:
        raise ValueError("diagnostics must include a non-negative optimizer_updates integer")
    return value


def _pairwise_tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(
        left, right
    )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _case_dir(root: Path, group: str, sequence: str, seed: int) -> Path:
    return root / group / sequence / f"seed_{seed}"


def _status(path: Path, state: str, **fields: Any) -> None:
    _write_json(path, {"state": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields})


def _snapshot_sources(run_root: Path) -> dict[str, str]:
    """Copy the exact harness sources and return hashes of copied bytes."""

    source_root = Path(__file__).resolve().parent
    snapshot_root = run_root / "source_snapshot"
    snapshot_root.mkdir(parents=False, exist_ok=False)
    hashes: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for filename in SOURCE_SNAPSHOT_FILES:
        source_path = source_root / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"required source for frozen snapshot is missing: {source_path}")
        raw = source_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        target_path = snapshot_root / filename
        with target_path.open("xb") as stream:
            stream.write(raw)
        copied = target_path.read_bytes()
        copied_digest = hashlib.sha256(copied).hexdigest()
        if copied_digest != digest:
            raise IOError(f"source snapshot verification failed for {filename}")
        hashes[filename] = digest
        entries.append(
            {
                "file": filename,
                "size_bytes": len(raw),
                "sha256": digest,
            }
        )
    _write_json(
        snapshot_root / "manifest.json",
        {
            "manifest_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(source_root),
            "copy_method": "byte_for_byte",
            "files": entries,
        },
    )
    return hashes


def _window_legacy_smoke(steps: int = 40) -> dict[str, Any]:
    """Compare W0 against legacy PlasticRNN using float32 tolerances."""

    seed = 20261027
    old = PlasticRNN(
        seed=seed,
        context_length=CONTEXT_LENGTH,
        updates_per_press=3,
        learning_rate=0.015,
    )
    candidate = create_model(GROUPS["window_original"], seed)
    if not _pairwise_tensor_equal(_parameter_vector(old), _parameter_vector(candidate)):
        raise AssertionError("W0 does not bitwise match legacy initial parameters")
    sequence = generate_sequence("cycle", steps)
    history: list[int] = []
    prediction_checks = 0
    parameter_checks = 0
    max_prediction_abs_error = 0.0
    max_parameter_abs_error = 0.0
    for digit in sequence:
        old_prediction = old.predict_next(history)
        new_prediction = candidate.predict_next()
        old_tensors = (
            torch.tensor(old_prediction[0]),
            torch.tensor(old_prediction[1]),
            torch.tensor(old_prediction[2]),
        )
        new_tensors = (
            torch.tensor(new_prediction[0]),
            torch.tensor(new_prediction[1]),
            torch.tensor(new_prediction[2]),
        )
        step_prediction_error = max(
            float((a - b).abs().max().item()) for a, b in zip(old_tensors, new_tensors)
        )
        max_prediction_abs_error = max(max_prediction_abs_error, step_prediction_error)
        if not all(
            torch.allclose(a, b, rtol=1e-5, atol=1e-5)
            for a, b in zip(old_tensors, new_tensors)
        ):
            raise AssertionError(f"W0 prequential prediction diverged at key {len(history) + 1}")
        old_argmax = int(max(range(INPUT_COUNT), key=old_prediction[0].__getitem__))
        new_argmax = int(max(range(INPUT_COUNT), key=new_prediction[0].__getitem__))
        if old_argmax != new_argmax:
            raise AssertionError(
                f"W0 argmax diverged at key {len(history) + 1}: "
                f"legacy={old_argmax}, candidate={new_argmax}"
            )
        prediction_checks += 1

        history.append(digit)
        old_loss = old.train_on_history(history)
        new_loss = candidate.observe(digit)
        if old_loss is None or new_loss is None:
            same_loss = old_loss is None and new_loss is None
        else:
            same_loss = math.isclose(old_loss, new_loss, rel_tol=1e-5, abs_tol=1e-5)
        if not same_loss:
            raise AssertionError(f"W0 loss diverged at key {len(history)}: {old_loss} != {new_loss}")
        old_parameters = _parameter_vector(old)
        new_parameters = _parameter_vector(candidate)
        parameter_error = float((old_parameters - new_parameters).abs().max().item())
        max_parameter_abs_error = max(max_parameter_abs_error, parameter_error)
        if not torch.allclose(old_parameters, new_parameters, rtol=1e-5, atol=1e-5):
            raise AssertionError(f"W0 parameter update diverged at key {len(history)}")
        parameter_checks += 1

    return {
        "seed": seed,
        "sequence": "cycle",
        "steps": steps,
        "prediction_checks": prediction_checks,
        "parameter_checks": parameter_checks,
        "initial_parameters_bitwise_equal": True,
        "legacy_training_comparison": "allclose(rtol=1e-5, atol=1e-5)",
        "argmax_equal_every_step": True,
        "max_prediction_abs_error": max_prediction_abs_error,
        "max_parameter_abs_error": max_parameter_abs_error,
        "clip_rule": "production_float64_global_norm_no_epsilon",
        "final_parameter_sha256": _tensor_sha256(_parameter_vector(candidate)),
        "optimizer_updates": candidate.optimizer_updates,
        "status": "passed",
    }


def _sequence_sha256(digits: Sequence[int]) -> str:
    raw = bytes(int(value) for value in digits)
    return hashlib.sha256(raw).hexdigest()


def _prepare_cases(
    run_root: Path,
    group_names: Sequence[str],
    seeds: Sequence[int],
    sequence_names: Sequence[str],
    steps: int,
    checkpoint_every: int,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    cases: dict[tuple[str, str, int], dict[str, Any]] = {}
    pair_vectors: dict[tuple[str, int], torch.Tensor] = {}
    pair_hashes: dict[tuple[str, int], str] = {}
    group_initial_hashes: dict[str, dict[str, str]] = {}

    for seed in seeds:
        for sequence_name in sequence_names:
            digits = generate_sequence(sequence_name, steps)
            key = (sequence_name, seed)
            for group_name in group_names:
                spec = GROUPS[group_name]
                directory = _case_dir(run_root, group_name, sequence_name, seed)
                directory.mkdir(parents=True, exist_ok=False)
                model = create_model(spec, seed)
                parameter_vector = _parameter_vector(model)
                parameter_sha = _tensor_sha256(parameter_vector)
                state_vector = _state_vector(model)
                if not bool(torch.equal(state_vector, torch.zeros_like(state_vector))):
                    raise AssertionError(f"{group_name} starts with a non-zero recurrent state")
                prior = pair_vectors.get(key)
                if prior is None:
                    pair_vectors[key] = parameter_vector.clone()
                    pair_hashes[key] = parameter_sha
                elif not _pairwise_tensor_equal(prior, parameter_vector):
                    raise AssertionError(
                        f"initial parameters differ for seed={seed}, sequence={sequence_name}, "
                        f"group={group_name}; first SHA {pair_hashes[key]}, current {parameter_sha}"
                    )

                group_initial_hashes.setdefault(group_name, {})[
                    f"{sequence_name}/seed_{seed}"
                ] = parameter_sha
                config = spec.effective
                _write_json(
                    directory / "manifest.json",
                    {
                        "manifest_version": 1,
                        "frozen_before_training": True,
                        "group": group_name,
                        "description": spec.description,
                        "model_kind": spec.model_kind,
                        "seed": seed,
                        "sequence": sequence_name,
                        "steps": steps,
                        "sequence_sha256": _sequence_sha256(digits),
                        "effective_config": config,
                        "initial_parameter_sha256": parameter_sha,
                        "initial_state_sha256": _tensor_sha256(state_vector),
                        "training_order": "predict_and_flush_then_observe",
                        "checkpoint_every": checkpoint_every,
                        "torch_version": torch.__version__,
                    },
                )
                _atomic_torch_save(
                    directory / "initial_checkpoint.pt",
                    _checkpoint_payload(
                        model,
                        group=group_name,
                        seed=seed,
                        sequence=sequence_name,
                        step=0,
                        initial_parameter_sha256=parameter_sha,
                    ),
                )
                _status(directory / "status.json", "prepared")
                cases[(group_name, sequence_name, seed)] = {
                    "directory": directory,
                    "initial_parameter_sha256": parameter_sha,
                }
                del model

    if len(pair_vectors) != len(seeds) * len(sequence_names):
        raise AssertionError("initial-parameter pairing did not cover every seed/sequence")

    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_training": True,
        "run_id": run_root.name,
        "groups": [
            {
                "name": name,
                "model_kind": GROUPS[name].model_kind,
                "description": GROUPS[name].description,
                "effective_config": GROUPS[name].effective,
            }
            for name in group_names
        ],
        "seeds": list(seeds),
        "sequences": list(sequence_names),
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "cpu_threads": 1,
        "inter_op_threads": 1,
        "sequence_sha256": {
            f"{name}/{steps}": _sequence_sha256(generate_sequence(name, steps))
            for name in sequence_names
        },
        "initial_parameter_sha256": group_initial_hashes,
        "initial_parameters_bitwise_equal_by_seed_sequence": True,
        "single_variable_pairs": {
            f"{left}->{right}": field
            for (left, right), field in EXPECTED_PAIR_DIFFS.items()
            if left in group_names and right in group_names
        },
        "torch_version": torch.__version__,
        "python_version": sys.version,
    }
    return cases, manifest


def _state_norms(diagnostics: Mapping[str, Any]) -> dict[str, float | None]:
    value = diagnostics.get("norms", {})
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float | None] = {}
    for key in ("activity", "fast_edges", "parameters"):
        raw = value.get(key)
        result[key] = float(raw) if isinstance(raw, (int, float)) else None
    return result


def _run_case(
    group_name: str,
    sequence_name: str,
    seed: int,
    steps: int,
    checkpoint_every: int,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    spec = GROUPS[group_name]
    directory: Path = case["directory"]
    initial_sha = str(case["initial_parameter_sha256"])
    model = create_model(spec, seed)
    if _tensor_sha256(_parameter_vector(model)) != initial_sha:
        raise AssertionError(f"{group_name} initial parameters changed after preflight")

    digits = generate_sequence(sequence_name, steps)
    prequential_path = directory / "prequential.jsonl"
    observe_path = directory / "observe_metrics.jsonl"
    predictions: list[dict[str, Any]] = []
    update_count = 0
    clipped_update_count = 0
    training_milliseconds: list[float] = []
    activity_norms: list[float] = []
    fast_edge_norms: list[float] = []
    last_diagnostics: dict[str, Any] = _model_diagnostics(model)
    _status(directory / "status.json", "running", step=0, total_steps=steps)

    try:
        for index, digit in enumerate(digits):
            predicted = _predict_probabilities(model)
            target_probability = max(predicted[digit], 1e-30)
            prediction = {
                "step": index + 1,
                "digit": digit,
                "probabilities": predicted,
                "predicted_digit": int(max(range(INPUT_COUNT), key=predicted.__getitem__)),
                "correct": int(max(range(INPUT_COUNT), key=predicted.__getitem__)) == digit,
                "nll": -math.log(target_probability),
                "updates_before_observe": _training_updates(last_diagnostics),
            }
            # This line is durable before observe sees the target.
            _append_jsonl(prequential_path, prediction)
            predictions.append(prediction)

            updates_before = _training_updates(last_diagnostics)
            started = time.perf_counter()
            train_loss = model.observe(digit)
            training_ms = (time.perf_counter() - started) * 1000.0
            training_milliseconds.append(training_ms)
            last_diagnostics = _model_diagnostics(model)
            updates_after = _training_updates(last_diagnostics)
            updates_this_step = updates_after - updates_before
            if updates_this_step < 0:
                raise AssertionError("optimizer update count moved backwards")
            update_count += updates_this_step

            was_clipped = last_diagnostics.get("last_gradient_was_clipped")
            if spec.model_kind == "window":
                # WindowControlRNN reports its cumulative count so multiple
                # clipped updates in one observe are accounted exactly.
                clipped_update_count = int(last_diagnostics.get("clipped_updates", 0))
            elif was_clipped is True:
                # Each persistent online model performs one update per press.
                clipped_update_count += updates_this_step

            norms = _state_norms(last_diagnostics)
            if norms.get("activity") is not None:
                activity_norms.append(float(norms["activity"]))
            if norms.get("fast_edges") is not None:
                fast_edge_norms.append(float(norms["fast_edges"]))
            _append_jsonl(
                observe_path,
                {
                    "step": index + 1,
                    "digit": digit,
                    "training_ms": training_ms,
                    "training_loss": train_loss,
                    "updates_this_step": updates_this_step,
                    "optimizer_updates": updates_after,
                    "last_gradient_was_clipped": was_clipped,
                    "clipped_updates": last_diagnostics.get("clipped_updates"),
                    "clip_fraction": (
                        float(clipped_update_count / update_count) if update_count else 0.0
                    ),
                    "state_norms": norms,
                    "input_nonzero_total": last_diagnostics.get("input_nonzero_total"),
                    "input_nonzero_by_digit": last_diagnostics.get(
                        "input_nonzero_by_digit"
                    ),
                },
            )

            if (index + 1) % checkpoint_every == 0 or index + 1 == steps:
                checkpoint_path = directory / f"checkpoint_step_{index + 1:06d}.pt"
                _atomic_torch_save(
                    checkpoint_path,
                    _checkpoint_payload(
                        model,
                        group=group_name,
                        seed=seed,
                        sequence=sequence_name,
                        step=index + 1,
                        initial_parameter_sha256=initial_sha,
                    ),
                )
            if (index + 1) % 10 == 0 or index + 1 == steps:
                _status(
                    directory / "status.json",
                    "running",
                    step=index + 1,
                    total_steps=steps,
                )
    except Exception as exc:
        _status(
            directory / "status.json",
            "failed",
            step=len(predictions),
            total_steps=steps,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    final_diagnostics = _model_diagnostics(model)
    if not final_diagnostics.get("finite", True):
        raise FloatingPointError(f"{group_name} finished with non-finite model state")
    total = len(predictions)

    def segment(start: int, end: int) -> dict[str, Any]:
        values = predictions[start:end]
        if not values:
            return {"start_step": start + 1, "end_step": end, "count": 0, "accuracy": None, "mean_nll": None}
        return {
            "start_step": start + 1,
            "end_step": end,
            "count": len(values),
            "correct": sum(bool(value["correct"]) for value in values),
            "accuracy": sum(bool(value["correct"]) for value in values) / len(values),
            "mean_nll": statistics.fmean(float(value["nll"]) for value in values),
        }

    head_end = min(60, total)
    last90_start = max(0, total - 90)
    last100_start = max(0, total - 100)
    metrics = {
        "group": group_name,
        "sequence": sequence_name,
        "seed": seed,
        "steps": total,
        "correct": sum(bool(value["correct"]) for value in predictions),
        "accuracy": sum(bool(value["correct"]) for value in predictions) / total,
        "mean_nll": statistics.fmean(float(value["nll"]) for value in predictions),
        "head57": segment(0, min(57, total)),
        "head60": segment(0, head_end),
        "last90": segment(last90_start, total),
        "last100": segment(last100_start, total),
        "updates": update_count,
        "clipped_updates": clipped_update_count,
        "clip_fraction": float(clipped_update_count / update_count) if update_count else 0.0,
        "training_ms": {
            "total": sum(training_milliseconds),
            "mean_per_key": statistics.fmean(training_milliseconds),
            "median_per_key": statistics.median(training_milliseconds),
        },
        "state_norms": {
            "activity_mean": statistics.fmean(activity_norms) if activity_norms else None,
            "activity_final": activity_norms[-1] if activity_norms else None,
            "fast_edges_mean": statistics.fmean(fast_edge_norms) if fast_edge_norms else None,
            "fast_edges_final": fast_edge_norms[-1] if fast_edge_norms else None,
            "parameters_final": _state_norms(final_diagnostics).get("parameters"),
        },
        "input_nonzero_total_final": final_diagnostics.get("input_nonzero_total"),
        "input_nonzero_by_digit_final": final_diagnostics.get("input_nonzero_by_digit"),
        "optimizer_updates_final": _training_updates(final_diagnostics),
        "paired_10key_blocks": [
            segment(start, min(start + 10, total)) for start in range(0, total, 10)
        ],
        "final_parameter_sha256": _tensor_sha256(_parameter_vector(model)),
        "final_state_sha256": _tensor_sha256(_state_vector(model)),
    }
    _write_json(directory / "metrics.json", metrics)
    _status(directory / "status.json", "completed", step=steps, total_steps=steps)
    return metrics


def _paired_block_metrics(
    all_metrics: Mapping[tuple[str, str, int], Mapping[str, Any]],
    group_names: Sequence[str],
    sequence_names: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    block_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    selected_pairs = [
        pair for pair in EXPECTED_PAIR_DIFFS if pair[0] in group_names and pair[1] in group_names
    ]
    block_count = max(
        (
            len(metrics.get("paired_10key_blocks", []))
            for metrics in all_metrics.values()
        ),
        default=0,
    )

    for sequence_name in sequence_names:
        for block_index in range(block_count):
            seed_values: dict[int, dict[str, dict[str, Any]]] = {}
            for seed in seeds:
                by_group: dict[str, dict[str, Any]] = {}
                for group_name in group_names:
                    metrics = all_metrics.get((group_name, sequence_name, seed))
                    blocks = metrics.get("paired_10key_blocks", []) if metrics else []
                    if block_index < len(blocks):
                        by_group[group_name] = blocks[block_index]
                seed_values[seed] = by_group
                block_rows.append(
                    {
                        "sequence": sequence_name,
                        "seed": seed,
                        "block_index": block_index,
                        "start_step": block_index * 10 + 1,
                        "end_step": (block_index + 1) * 10,
                        "groups": by_group,
                    }
                )

            for left, right in selected_pairs:
                deltas: list[tuple[float, float]] = []
                for seed in seeds:
                    values = seed_values.get(seed, {})
                    a = values.get(left)
                    b = values.get(right)
                    if not a or not b:
                        continue
                    deltas.append(
                        (
                            float(b["accuracy"]) - float(a["accuracy"]),
                            float(b["mean_nll"]) - float(a["mean_nll"]),
                        )
                    )
                if not deltas:
                    continue
                accuracy_delta = [item[0] for item in deltas]
                nll_delta = [item[1] for item in deltas]
                aggregates.append(
                    {
                        "sequence": sequence_name,
                        "block_index": block_index,
                        "start_step": block_index * 10 + 1,
                        "end_step": (block_index + 1) * 10,
                        "left_group": left,
                        "right_group": right,
                        "n_paired_seeds": len(deltas),
                        "right_minus_left_accuracy_mean": statistics.fmean(accuracy_delta),
                        "right_minus_left_accuracy_sd": (
                            statistics.stdev(accuracy_delta) if len(accuracy_delta) > 1 else 0.0
                        ),
                        "right_minus_left_nll_mean": statistics.fmean(nll_delta),
                        "right_minus_left_nll_sd": (
                            statistics.stdev(nll_delta) if len(nll_delta) > 1 else 0.0
                        ),
                        "accuracy_win_fraction": sum(value > 0 for value in accuracy_delta)
                        / len(accuracy_delta),
                        "nll_win_fraction": sum(value < 0 for value in nll_delta) / len(nll_delta),
                    }
                )
    return {
        "format_version": 1,
        "pair_direction": "right_minus_left",
        "single_variable_pairs": [
            {"left": left, "right": right, "field": EXPECTED_PAIR_DIFFS[(left, right)]}
            for left, right in selected_pairs
        ],
        "per_seed_blocks": block_rows,
        "paired_seed_aggregates_by_10key_block": aggregates,
    }


def run_experiment(
    *,
    group_names: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sequence_names: Sequence[str] = DEFAULT_SEQUENCES,
    steps: int = DEFAULT_STEPS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    data_root: Path,
) -> Path:
    validate_group_protocol()
    if not group_names or any(name not in GROUPS for name in group_names):
        raise ValueError("group_names must contain known non-empty group names")
    if len(set(group_names)) != len(group_names):
        raise ValueError("group_names cannot contain duplicates")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be unique integers")
    if not sequence_names or len(set(sequence_names)) != len(sequence_names):
        raise ValueError("sequence_names must be unique and non-empty")
    if any(name not in DEFAULT_SEQUENCES for name in sequence_names):
        raise ValueError(f"sequence_names must be selected from {DEFAULT_SEQUENCES}")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")

    configure_cpu_threads()
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    run_root = data_root / _run_id()
    run_root.mkdir(parents=False, exist_ok=False)
    _status(run_root / "run_status.json", "preparing")

    try:
        source_sha256 = _snapshot_sources(run_root)
        cases, manifest = _prepare_cases(
            run_root,
            group_names,
            seeds,
            sequence_names,
            steps,
            checkpoint_every,
        )
        manifest["source_sha256"] = source_sha256
        manifest["source_snapshot_manifest"] = "source_snapshot/manifest.json"
        _write_json(run_root / "run_manifest.json", manifest)
        _status(run_root / "run_status.json", "prepared", case_count=len(cases))
    except Exception as exc:
        _status(
            run_root / "run_status.json",
            "preparation_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    _status(run_root / "run_status.json", "running", case_count=len(cases))
    metrics: dict[tuple[str, str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        for sequence_name in sequence_names:
            for group_name in group_names:
                key = (group_name, sequence_name, seed)
                try:
                    metrics[key] = _run_case(
                        group_name,
                        sequence_name,
                        seed,
                        steps,
                        checkpoint_every,
                        cases[key],
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "group": group_name,
                            "sequence": sequence_name,
                            "seed": seed,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    # Keep the actual failed attempt and continue independent cases.

    paired = _paired_block_metrics(metrics, group_names, sequence_names, seeds)
    _write_json(run_root / "paired_10key_blocks.json", paired)
    _write_json(
        run_root / "summary.json",
        {
            "format_version": 1,
            "run_id": run_root.name,
            "completed_case_count": len(metrics),
            "failed_case_count": len(failures),
            "failures": failures,
            "case_metrics": [
                metrics[key]
                for seed in seeds
                for sequence_name in sequence_names
                for group_name in group_names
                if (key := (group_name, sequence_name, seed)) in metrics
            ],
        },
    )
    final_state = "completed" if not failures else "completed_with_failures"
    _status(
        run_root / "run_status.json",
        final_state,
        case_count=len(cases),
        completed_case_count=len(metrics),
        failed_case_count=len(failures),
    )
    return run_root


def configure_cpu_threads() -> None:
    """Use one CPU thread for each run process, including smoke checks."""

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        help="group names or 'all'; groups run serially inside one output bundle",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=DEFAULT_SEQUENCES,
        default=list(DEFAULT_SEQUENCES),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="parent directory for a new immutable run bundle; required unless --dry-run/--legacy-smoke",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the frozen group protocol")
    parser.add_argument(
        "--legacy-smoke",
        action="store_true",
        help="compare W0 to legacy PlasticRNN for 40 cycle keys; no files are written",
    )
    parser.add_argument("--list-groups", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_cpu_threads()
    if args.list_groups:
        for name, spec in GROUPS.items():
            print(f"{name}: {spec.description}")
        return 0
    try:
        validate_group_protocol()
        if args.legacy_smoke:
            result = _window_legacy_smoke(steps=40)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.dry_run:
            names = list(GROUPS) if args.groups == ["all"] else args.groups
            unknown = [name for name in names if name not in GROUPS]
            if unknown:
                raise ValueError(f"unknown groups: {unknown}")
            print(
                json.dumps(
                    {
                        "groups": [
                            {
                                "name": name,
                                "model_kind": GROUPS[name].model_kind,
                                "effective_config": GROUPS[name].effective,
                            }
                            for name in names
                        ],
                        "single_variable_pairs": {
                            f"{left}->{right}": field
                            for (left, right), field in EXPECTED_PAIR_DIFFS.items()
                        },
                        "seeds": args.seeds,
                        "sequences": args.sequences,
                        "steps": args.steps,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.data_root is None:
            raise ValueError("--data-root is required for a training run")
        names = list(GROUPS) if args.groups == ["all"] else args.groups
        unknown = [name for name in names if name not in GROUPS]
        if unknown:
            raise ValueError(f"unknown groups: {unknown}")
        output = run_experiment(
            group_names=names,
            seeds=args.seeds,
            sequence_names=args.sequences,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            data_root=args.data_root,
        )
        print(output)
        return 0
    except Exception as exc:
        print(f"run_ablation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
