"""Trial-local adapter for the established RSL full-episode evaluator.

This keeps the production evaluator and its sampling/reporting flow intact,
while selecting and verifying the same task backend used by a06 training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


TRIAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = TRIAL_DIR.parents[2]
TASK_BACKEND = "torch_batched"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from scripts import evaluate_fast_stop_mastery as mastery
from scripts import evaluate_rsl_curriculum as evaluator


_OBSERVED_BACKENDS: list[str] = []


def _create_rsl_evaluation_adapter(contract: dict[str, Any], seed: int, *, task_backend: str):
    """Reuse the standard contract reconstruction with an explicit backend."""
    from isaac_field.env import SingleFieldAdapter
    from isaac_field.runtime import create_env

    if task_backend != TASK_BACKEND:
        raise ValueError(f"a06 only supports task_backend={TASK_BACKEND!r}")
    task, front_guides, body_stability = mastery._validate_rsl_task_contract(contract)
    engine = create_env(
        cfg=task,
        mode="goto",
        num_envs=1,
        seed=seed,
        auto_reset=False,
        goal_distance=None,
        goal_angle=math.pi,
        distance_cost=0.0,
        front_guide_spec=front_guides,
        body_stability_config=body_stability,
        task_backend=task_backend,
        rsl=True,
    )
    observed = getattr(engine, "task_backend", None)
    if observed != task_backend:
        engine.close()
        raise RuntimeError(f"environment task_backend is {observed!r}, expected {task_backend!r}")
    _OBSERVED_BACKENDS.append(observed)
    return engine, SingleFieldAdapter(engine)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _record_backend(output: Path, task_backend: str) -> None:
    if _OBSERVED_BACKENDS != [task_backend]:
        raise RuntimeError(f"expected one evaluation environment using {task_backend!r}; observed {_OBSERVED_BACKENDS!r}")
    for name in ("evaluation.json", "run_manifest.json"):
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(f"standard evaluator did not write {path}")
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        report["task_backend"] = task_backend
        report["task_backend_verified"] = True
        _write_json_atomic(path, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-backend", choices=(TASK_BACKEND,), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--scenario-count", type=int, required=True)
    parser.add_argument("--goal-angle-deg", type=float, required=True)
    parser.add_argument("--goal-distance", type=float, nargs=2, default=None)
    args = parser.parse_args(argv)

    mastery._create_rsl_evaluation_adapter = lambda contract, seed: _create_rsl_evaluation_adapter(
        contract, seed, task_backend=args.task_backend,
    )
    try:
        result = evaluator.run_evaluation(
            args.checkpoint,
            args.output,
            seed_start=args.seed_start,
            scenario_count=args.scenario_count,
            goal_distance=None if args.goal_distance is None else tuple(args.goal_distance),
            goal_angle_deg=args.goal_angle_deg,
        )
        _record_backend(args.output.resolve(), args.task_backend)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"torch_batched evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
