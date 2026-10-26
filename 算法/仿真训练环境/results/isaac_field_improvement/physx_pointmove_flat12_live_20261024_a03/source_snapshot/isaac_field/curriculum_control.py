"""Pure-Python control decisions for ability-gated curricula.

This module does not create environments or train policies.  The caller must
use the returned sampling configuration to build the actual task environment,
then report the environment's effective sampling values to ``record_training``.
Probe reports are deliberately strict: missing safety or pairing evidence is
an error, never an implicit zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from typing import Any, Mapping, Sequence


STATE_SCHEMA = "isaac-field-curriculum-control-v1"
PLAN_SCHEMA = "isaac-field-training-plan-v1"
PROBE_SCHEMA = "isaac-field-fixed-probe-v1"
FORBIDDEN_SEED_RANGE = (83_000, 83_999)
_REPORT_CONTRACT_KEYS = (
    "action_config",
    "drive_config",
    "observation_config",
    "damage_config",
    "damage_metric",
    "distance_cost",
    "goal_distance",
    "goal_angle_rad",
    "mode",
    "deterministic",
    "evaluation_config",
)
_INITIAL_CONDITION_KEYS = ("plant_xy_sha256", "start_pose_xy_yaw", "goal_xy")


@dataclass(frozen=True)
class ProbeSpec:
    """One fixed probe bank and its complete observed evaluation contract."""

    seeds: Sequence[int]
    config: Mapping[str, Any]


@dataclass(frozen=True)
class CurriculumControlConfig:
    """Budget, promotion, replay, and fixed-probe policy.

    ``mastery_probes`` maps every promotable stage name to its own fixed probe
    bank. ``best_probe`` is a separate, shared global suite with stage name
    ``global_suite``. The training seed list is recorded and checked against
    every probe bank so a probe cannot accidentally reuse training episodes.
    """

    total_timesteps: int
    probe_interval_timesteps: int
    training_seeds: Sequence[int]
    mastery_probes: Mapping[str, ProbeSpec]
    best_probe: ProbeSpec | None = None
    enable_curriculum_promotion: bool = True
    enable_best_checkpoint_selection: bool = True
    required_consecutive_passes: int = 2
    min_clean_success_rate: float = 0.75
    max_damaged_episodes: int = 0
    max_mean_damage: float = 0.0
    max_total_collisions: int = 0
    minimum_stage_timesteps: int = 8_192
    active_blocks_per_replay: int = 3


def _plain_json(value: Any, *, name: str) -> Any:
    """Return a detached JSON value and reject non-finite/non-serializable data."""
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _seeds(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integer seeds")
    result = tuple(_integer(value, name=f"{name} seed") for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one seed")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate seeds")
    _reject_forbidden_seeds(result, name=name)
    return result


def _reject_forbidden_seeds(seeds: Sequence[int], *, name: str) -> None:
    lo, hi = FORBIDDEN_SEED_RANGE
    forbidden = [seed for seed in seeds if lo <= seed <= hi]
    if forbidden:
        raise ValueError(f"{name} must not use held-out 830xx seeds: {forbidden}")


def _stage_dict(stage: Any) -> dict[str, Any]:
    if is_dataclass(stage):
        raw = asdict(stage)
    elif isinstance(stage, Mapping):
        raw = dict(stage)
    else:
        names = ("name", "mode", "goal_distance", "goal_angle")
        if not all(hasattr(stage, name) for name in names):
            raise TypeError("each stage must expose name, mode, goal_distance, and goal_angle")
        raw = {name: getattr(stage, name) for name in names}
    name = raw.get("name")
    mode = raw.get("mode")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("curriculum stage name must be a non-empty string")
    if mode not in ("mixed", "goto", "search"):
        raise ValueError(f"invalid curriculum stage mode: {mode!r}")
    distance = raw.get("goal_distance")
    if distance is None:
        normalized_distance = None
    else:
        if not isinstance(distance, Sequence) or isinstance(distance, (str, bytes)) or len(distance) != 2:
            raise ValueError(f"stage {name!r} goal_distance must be a pair or None")
        lo, hi = (float(item) for item in distance)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0 or hi < lo:
            raise ValueError(f"stage {name!r} goal_distance must be finite, non-negative, and ordered")
        normalized_distance = [lo, hi]
    try:
        angle = float(raw.get("goal_angle"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stage {name!r} goal_angle must be finite") from exc
    if not math.isfinite(angle) or not 0.0 <= angle <= math.pi:
        raise ValueError(f"stage {name!r} goal_angle must be in [0, pi]")
    standard = {"name", "mode", "goal_distance", "goal_angle"}
    options: dict[str, Any] = {}
    for key in (
        "sampling_options", "sampling", "environment_overrides", "task_config", "cfg_overrides"
    ):
        value = raw.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"stage {name!r} {key} must be a mapping")
            options.update(dict(value))
            standard.add(key)
    options.update({key: value for key, value in raw.items() if key not in standard})
    return {
        "name": name,
        "mode": mode,
        "goal_distance": normalized_distance,
        "goal_angle": angle,
        "sampling_options": _plain_json(options, name=f"stage {name!r} sampling_options"),
    }


def _probe_spec_dict(value: ProbeSpec | Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if isinstance(value, ProbeSpec):
        raw = {"seeds": value.seeds, "config": value.config}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError(f"{name} must be a ProbeSpec or mapping")
    seeds = _seeds(raw.get("seeds", ()), name=f"{name}.seeds")
    config = raw.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ValueError(f"{name}.config must be a non-empty mapping of actual evaluation settings")
    return {"seeds": list(seeds), "config": _plain_json(dict(config), name=f"{name}.config")}


def _config_dict(config: CurriculumControlConfig | Mapping[str, Any]) -> dict[str, Any]:
    if is_dataclass(config):
        raw = asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raise TypeError("config must be CurriculumControlConfig or a mapping")
    total = _integer(raw.get("total_timesteps"), name="total_timesteps", minimum=1)
    interval = _integer(raw.get("probe_interval_timesteps"), name="probe_interval_timesteps", minimum=1)
    train_seeds = _seeds(raw.get("training_seeds", ()), name="training_seeds")
    probe_raw = raw.get("mastery_probes", {})
    if not isinstance(probe_raw, Mapping):
        raise ValueError("mastery_probes must map stage names to ProbeSpec values")
    mastery_probes = {
        str(stage): _probe_spec_dict(spec, name=f"mastery_probes[{stage!r}]")
        for stage, spec in probe_raw.items()
    }
    best_raw = raw.get("best_probe")
    best_probe = None if best_raw is None else _probe_spec_dict(best_raw, name="best_probe")
    enable_promotion = raw.get("enable_curriculum_promotion", True)
    enable_best = raw.get("enable_best_checkpoint_selection", True)
    if not isinstance(enable_promotion, bool) or not isinstance(enable_best, bool):
        raise ValueError("curriculum and best-checkpoint enable flags must be booleans")
    if enable_best and best_probe is None:
        raise ValueError("best_probe is required when best-checkpoint selection is enabled")
    if enable_promotion and not mastery_probes:
        raise ValueError("mastery_probes are required when curriculum promotion is enabled")
    passes = _integer(raw.get("required_consecutive_passes", 2), name="required_consecutive_passes", minimum=1)
    max_damage_episodes = _integer(raw.get("max_damaged_episodes", 0), name="max_damaged_episodes")
    max_collisions = _integer(raw.get("max_total_collisions", 0), name="max_total_collisions")
    minimum_stage = _integer(raw.get("minimum_stage_timesteps", 8_192), name="minimum_stage_timesteps", minimum=1)
    replay_blocks = _integer(raw.get("active_blocks_per_replay", 3), name="active_blocks_per_replay", minimum=1)
    try:
        clean_rate = float(raw.get("min_clean_success_rate", 0.75))
        mean_damage = float(raw.get("max_mean_damage", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("promotion safety thresholds must be finite numbers") from exc
    if not math.isfinite(clean_rate) or not 0.0 <= clean_rate <= 1.0:
        raise ValueError("min_clean_success_rate must be in [0, 1]")
    if not math.isfinite(mean_damage) or mean_damage < 0.0:
        raise ValueError("max_mean_damage must be finite and non-negative")
    if enable_promotion and total < minimum_stage:
        # A smaller global budget can train a stage, but cannot claim it can
        # support a complete next-stage block after a successful probe.
        minimum_stage = total
    train_seed_set = set(train_seeds)
    for label, spec in [*[(f"mastery:{key}", value) for key, value in mastery_probes.items()],
                        *([("best_checkpoint", best_probe)] if best_probe is not None else [])]:
        overlap = sorted(train_seed_set.intersection(spec["seeds"]))
        if overlap:
            raise ValueError(f"{label} probe seeds overlap training seeds: {overlap}")
    return {
        "total_timesteps": total,
        "probe_interval_timesteps": interval,
        "training_seeds": list(train_seeds),
        "mastery_probes": mastery_probes,
        "best_probe": best_probe,
        "enable_curriculum_promotion": enable_promotion,
        "enable_best_checkpoint_selection": enable_best,
        "required_consecutive_passes": passes,
        "min_clean_success_rate": clean_rate,
        "max_damaged_episodes": max_damage_episodes,
        "max_mean_damage": mean_damage,
        "max_total_collisions": max_collisions,
        "minimum_stage_timesteps": minimum_stage,
        "active_blocks_per_replay": replay_blocks,
    }


def _sampling_for_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": stage["mode"],
        "goal_distance": stage["goal_distance"],
        "goal_angle": stage["goal_angle"],
        **dict(stage.get("sampling_options", {})),
    }


def _sampling_equal(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    expected_sampling = _sampling_for_stage(expected)
    if set(actual) != set(expected_sampling):
        return False
    if actual.get("mode") != expected_sampling["mode"]:
        return False
    actual_distance = actual.get("goal_distance")
    expected_distance = expected_sampling["goal_distance"]
    if actual_distance is None or expected_distance is None:
        distance_equal = actual_distance is None and expected_distance is None
    else:
        try:
            values = tuple(float(value) for value in actual_distance)
            distance_equal = len(values) == 2 and all(
                math.isfinite(value) and math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)
                for value, target in zip(values, expected_distance)
            )
        except (TypeError, ValueError):
            distance_equal = False
    try:
        angle = float(actual.get("goal_angle"))
    except (TypeError, ValueError):
        return False
    if not distance_equal or not math.isfinite(angle) or not math.isclose(
        angle, expected_sampling["goal_angle"], rel_tol=0.0, abs_tol=1e-12
    ):
        return False
    expected_extras = {key: value for key, value in expected_sampling.items()
                       if key not in ("mode", "goal_distance", "goal_angle")}
    actual_extras = {key: value for key, value in actual.items()
                     if key not in ("mode", "goal_distance", "goal_angle")}
    return _plain_json(actual_extras, name="actual sampling extras") == expected_extras


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    return _integer(value, name=name, minimum=0)


def _finite_vector(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def _row_map(report: Mapping[str, Any], expected_seeds: Sequence[int]) -> dict[int, Mapping[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_seeds):
        raise ValueError("probe rows must contain exactly one row per configured seed")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every probe row must be a mapping")
        seed = _integer(row.get("seed"), name="probe row seed")
        if seed in result:
            raise ValueError(f"probe rows contain duplicate seed {seed}")
        if seed not in expected_seeds:
            raise ValueError(f"probe row has unconfigured seed {seed}")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"probe row {seed} is missing a boolean success value")
        _strict_nonnegative_int(row.get("damage"), name=f"probe row {seed} damage")
        _strict_nonnegative_int(row.get("collisions"), name=f"probe row {seed} collisions")
        initial = row.get("reset_initial_condition")
        if not isinstance(initial, Mapping):
            raise ValueError(f"probe row {seed} is missing reset_initial_condition")
        if not isinstance(initial.get("plant_xy_sha256"), str) or not initial["plant_xy_sha256"]:
            raise ValueError(f"probe row {seed} is missing plant_xy_sha256")
        _finite_vector(initial.get("start_pose_xy_yaw"), name=f"probe row {seed} start_pose_xy_yaw", length=3)
        _finite_vector(initial.get("goal_xy"), name=f"probe row {seed} goal_xy", length=2)
        if row.get("trajectory_start_matches_reset") is not True:
            raise ValueError(f"probe row {seed} does not prove trajectory start matches reset")
        result[seed] = row
    if set(result) != set(expected_seeds):
        raise ValueError("probe rows do not match the fixed probe seed bank")
    return result


def _probe_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    meta = report.get("probe")
    if not isinstance(meta, Mapping):
        raise ValueError("probe report is missing fixed-probe metadata")
    if meta.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"probe report schema must be {PROBE_SCHEMA!r}")
    purpose = meta.get("purpose")
    if purpose not in ("mastery", "best_checkpoint"):
        raise ValueError("probe purpose must be mastery or best_checkpoint")
    stage = meta.get("probe_stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("probe_stage must be a non-empty string")
    seeds = _seeds(meta.get("seeds", ()), name="probe.seeds")
    config = meta.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ValueError("probe.config must contain the fixed observed evaluation contract")
    return {
        "schema": PROBE_SCHEMA,
        "purpose": purpose,
        "probe_stage": stage,
        "seeds": list(seeds),
        "config": _plain_json(dict(config), name="probe.config"),
    }


def _observed_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _REPORT_CONTRACT_KEYS if key not in report]
    if missing:
        raise ValueError(f"probe report is missing observed evaluation contract fields: {missing}")
    contract = {key: report[key] for key in _REPORT_CONTRACT_KEYS}
    if not isinstance(contract["mode"], str) or not contract["mode"]:
        raise ValueError("probe report mode must be present")
    if not isinstance(contract["deterministic"], bool):
        raise ValueError("probe report deterministic must be a boolean")
    if isinstance(contract["distance_cost"], bool) or not isinstance(contract["distance_cost"], (int, float)):
        raise ValueError("probe report distance_cost must be numeric")
    if not math.isfinite(float(contract["distance_cost"])):
        raise ValueError("probe report distance_cost must be finite")
    return _plain_json(contract, name="observed evaluation contract")


def _validate_report_for_spec(
    report: Mapping[str, Any],
    *,
    purpose: str,
    probe_stage: str,
    spec: Mapping[str, Any],
) -> tuple[dict[int, Mapping[str, Any]], dict[str, float | int]]:
    if not isinstance(report, Mapping):
        raise TypeError("probe report must be a mapping")
    meta = _probe_metadata(report)
    if meta["purpose"] != purpose:
        raise ValueError(f"expected {purpose!r} probe report, found {meta['purpose']!r}")
    if meta["probe_stage"] != probe_stage:
        raise ValueError(f"expected probe_stage {probe_stage!r}, found {meta['probe_stage']!r}")
    expected_seeds = tuple(spec["seeds"])
    if tuple(meta["seeds"]) != expected_seeds:
        raise ValueError("probe report seed bank differs from the configured fixed bank")
    expected_config = _plain_json(spec["config"], name="configured probe contract")
    if meta["config"] != expected_config:
        raise ValueError("probe metadata config differs from the configured fixed contract")
    actual_contract = _observed_contract(report)
    if actual_contract != expected_config:
        raise ValueError("observed evaluation contract differs from configured probe contract")
    rows = _row_map(report, expected_seeds)
    damages = []
    clean_successes = 0
    damaged_episodes = 0
    total_collisions = 0
    for seed, row in rows.items():
        damage = int(row["damage"])
        collisions = int(row["collisions"])
        damages.append(damage)
        damaged_episodes += int(damage > 0)
        total_collisions += collisions
        clean_successes += int(bool(row["success"]) and damage == 0 and collisions == 0)
    metrics: dict[str, float | int] = {
        "episodes": len(rows),
        "clean_successes": clean_successes,
        "clean_success_rate": clean_successes / len(rows),
        "damaged_episodes": damaged_episodes,
        "mean_damage": sum(damages) / len(damages),
        "total_collisions": total_collisions,
    }
    summary = report.get("summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise ValueError("probe summary must be a mapping when present")
        consistency = {
            "episodes": metrics["episodes"],
            "clean_successes": metrics["clean_successes"],
            "damaged_episodes": metrics["damaged_episodes"],
            "total_collisions": metrics["total_collisions"],
        }
        for key, expected in consistency.items():
            if key in summary and summary[key] != expected:
                raise ValueError(f"probe summary {key} conflicts with per-seed rows")
        if "mean_damage" in summary:
            try:
                observed_mean = float(summary["mean_damage"])
            except (TypeError, ValueError) as exc:
                raise ValueError("probe summary mean_damage must be numeric") from exc
            if not math.isfinite(observed_mean) or not math.isclose(
                observed_mean, metrics["mean_damage"], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("probe summary mean_damage conflicts with per-seed rows")
    return rows, metrics


def _initial_conditions_match(
    left: Mapping[int, Mapping[str, Any]], right: Mapping[int, Mapping[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    mismatches = []
    for seed in sorted(left):
        a = left[seed]["reset_initial_condition"]
        b = right[seed]["reset_initial_condition"]
        differences = []
        if a["plant_xy_sha256"] != b["plant_xy_sha256"]:
            differences.append("plant_xy_sha256")
        for key, length in (("start_pose_xy_yaw", 3), ("goal_xy", 2)):
            va = _finite_vector(a[key], name=f"seed {seed} {key}", length=length)
            vb = _finite_vector(b[key], name=f"seed {seed} {key}", length=length)
            if any(not math.isclose(x, y, rel_tol=0.0, abs_tol=1e-8) for x, y in zip(va, vb)):
                differences.append(key)
        if differences:
            mismatches.append({"seed": seed, "differences": differences})
    return not mismatches, mismatches


class CurriculumController:
    """Stateful, serializable decision maker; environment/training stay external."""

    def __init__(
        self,
        stages: Sequence[Any],
        config: CurriculumControlConfig | Mapping[str, Any],
    ) -> None:
        self._stages = tuple(_stage_dict(stage) for stage in stages)
        if not self._stages:
            raise ValueError("at least one curriculum stage is required")
        names = [stage["name"] for stage in self._stages]
        if len(set(names)) != len(names):
            raise ValueError("curriculum stage names must be unique")
        self._config = _config_dict(config)
        if self._config["enable_curriculum_promotion"]:
            if set(self._config["mastery_probes"]) != set(names):
                raise ValueError("mastery_probes must define exactly one fixed bank for every stage")
        self._consumed_timesteps = 0
        self._active_stage_timesteps = 0
        self._active_index = 0
        self._consecutive_passes = 0
        self._next_probe_at = self._config["probe_interval_timesteps"]
        self._probe_due = False
        self._mixture_cursor = 0
        self._block_count = 0
        self._pending_plan: dict[str, Any] | None = None
        self._last_probe_metrics: dict[str, Any] | None = None
        self._last_probe_passed: bool | None = None
        self._promotion_blocked_reason: str | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def active_stage(self) -> Mapping[str, Any]:
        return dict(self._stages[self._active_index])

    @property
    def consumed_timesteps(self) -> int:
        return self._consumed_timesteps

    @property
    def remaining_timesteps(self) -> int:
        return self._config["total_timesteps"] - self._consumed_timesteps

    def _cycle(self) -> tuple[int, ...]:
        active = self._active_index
        if active == 0:
            return (0,)
        blocks = self._config["active_blocks_per_replay"]
        cycle: list[int] = []
        # Begin with the most recently mastered prior stage.  This guarantees
        # a successful two-probe promotion path still trains a retention block
        # before the active stage can promote again.  A complete cycle retains
        # the configured 3:1 active-to-replay target, with the replay share
        # divided equally among all mastered prior stages.
        for earlier in reversed(range(active)):
            cycle.append(earlier)
            cycle.extend([active] * blocks)
        return tuple(cycle)

    def _distribution(self) -> list[dict[str, Any]]:
        cycle = self._cycle()
        counts: dict[int, int] = {}
        for index in cycle:
            counts[index] = counts.get(index, 0) + 1
        return [
            {"stage": dict(self._stages[index]), "weight": count / len(cycle)}
            for index, count in sorted(counts.items())
        ]

    def _long_run_cycle(self) -> dict[str, Any]:
        cycle = self._cycle()
        return {
            "block_order": [self._stages[index]["name"] for index in cycle],
            "target_distribution": self._distribution(),
            "semantics": "target weights over complete cycles; actual finite-run exposure comes from completed training blocks",
        }

    def decision(self) -> dict[str, Any]:
        remaining = self.remaining_timesteps
        mastered = self._active_index == len(self._stages) - 1 and (
            self._last_probe_passed is True
            and self._consecutive_passes >= self._config["required_consecutive_passes"]
        )
        return {
            "schema": STATE_SCHEMA,
            "consumed_timesteps": self._consumed_timesteps,
            "total_timesteps": self._config["total_timesteps"],
            "remaining_timesteps": remaining,
            "budget_exhausted": remaining == 0,
            "can_train": remaining > 0 and not self._probe_due and self._pending_plan is None,
            "probe_due": self._probe_due,
            "active_stage_timesteps": self._active_stage_timesteps,
            "next_probe_at_active_timesteps": self._next_probe_at,
            "active_timesteps_until_probe": max(0, self._next_probe_at - self._active_stage_timesteps),
            "active_stage": dict(self._stages[self._active_index]),
            "active_stage_index": self._active_index,
            "mastered": mastered,
            "consecutive_passes": self._consecutive_passes,
            "required_consecutive_passes": self._config["required_consecutive_passes"],
            "last_probe_passed": self._last_probe_passed,
            "last_probe_metrics": None if self._last_probe_metrics is None else dict(self._last_probe_metrics),
            "promotion_blocked_reason": self._promotion_blocked_reason,
            "enable_curriculum_promotion": self._config["enable_curriculum_promotion"],
            "enable_best_checkpoint_selection": self._config["enable_best_checkpoint_selection"],
            "long_run_cycle": self._long_run_cycle(),
            "pending_plan": None if self._pending_plan is None else dict(self._pending_plan),
        }

    def next_training_plan(self, max_timesteps: int | None = None) -> dict[str, Any]:
        """Return the next real environment task block, capped at probe/budget."""
        if max_timesteps is not None:
            max_timesteps = _integer(max_timesteps, name="max_timesteps", minimum=1)
        if self._pending_plan is not None:
            return dict(self._pending_plan)
        if self._probe_due:
            return {**self.decision(), "can_train": False, "reason": "probe_due"}
        if self.remaining_timesteps == 0:
            return {**self.decision(), "can_train": False, "reason": "budget_exhausted"}
        cycle = self._cycle()
        stage_index = cycle[self._mixture_cursor % len(cycle)]
        stage = dict(self._stages[stage_index])
        block = self.remaining_timesteps
        if stage_index == self._active_index and self._config["enable_curriculum_promotion"]:
            until_active_probe = max(1, self._next_probe_at - self._active_stage_timesteps)
            block = min(block, until_active_probe)
        if max_timesteps is not None:
            block = min(block, max_timesteps)
        sampling = _sampling_for_stage(stage)
        self._pending_plan = {
            "schema": PLAN_SCHEMA,
            "block_index": self._block_count,
            "timesteps": block,
            "budget_before": self._consumed_timesteps,
            "budget_after": self._consumed_timesteps + block,
            "active_stage": dict(self._stages[self._active_index]),
            "sampled_stage": stage["name"],
            "sampling": sampling,
            "long_run_cycle": self._long_run_cycle(),
            "next_probe_at": self._next_probe_at,
        }
        self._events.append({"event": "training_plan", **self._pending_plan})
        return dict(self._pending_plan)

    def record_training(
        self,
        timesteps: int,
        *,
        sampled_stage: str,
        actual_sampling: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit one completed block after checking the task actually sampled."""
        if self._pending_plan is None:
            raise RuntimeError("call next_training_plan before recording training")
        completed = _integer(timesteps, name="timesteps", minimum=1)
        plan = self._pending_plan
        if completed != plan["timesteps"]:
            raise ValueError(f"training block must consume exactly the planned {plan['timesteps']} timesteps")
        if sampled_stage != plan["sampled_stage"]:
            raise ValueError(
                f"sampled stage {sampled_stage!r} differs from planned stage {plan['sampled_stage']!r}"
            )
        expected_stage = next(stage for stage in self._stages if stage["name"] == sampled_stage)
        if not isinstance(actual_sampling, Mapping) or not _sampling_equal(expected_stage, actual_sampling):
            raise ValueError("actual environment sampling values differ from the planned CurriculumStage")
        if completed > self.remaining_timesteps:
            raise ValueError("training block would exceed the fixed global budget")
        self._consumed_timesteps += completed
        self._mixture_cursor += 1
        self._block_count += 1
        if sampled_stage == self._stages[self._active_index]["name"]:
            self._active_stage_timesteps += completed
        self._pending_plan = None
        if (
            self._config["enable_curriculum_promotion"]
            and self._active_stage_timesteps >= self._next_probe_at
        ):
            self._probe_due = True
        self._events.append({
            "event": "training_completed",
            "timesteps": completed,
            "sampled_stage": sampled_stage,
            "actual_sampling": _plain_json(dict(actual_sampling), name="actual_sampling"),
            "consumed_timesteps": self._consumed_timesteps,
            "active_stage_timesteps": self._active_stage_timesteps,
            "probe_due": self._probe_due,
        })
        return self.decision()

    def record_probe(self, report: Mapping[str, Any]) -> dict[str, Any]:
        """Consume one due stage probe and update only promotion state."""
        if not self._config["enable_curriculum_promotion"]:
            raise RuntimeError("curriculum promotion is disabled")
        if not self._probe_due:
            raise RuntimeError("a mastery probe is accepted only when the controller marks one due")
        stage = self._stages[self._active_index]
        spec = self._config["mastery_probes"][stage["name"]]
        rows, metrics = _validate_report_for_spec(
            report,
            purpose="mastery",
            probe_stage=stage["name"],
            spec=spec,
        )
        del rows
        passed = (
            metrics["clean_success_rate"] >= self._config["min_clean_success_rate"]
            and metrics["damaged_episodes"] <= self._config["max_damaged_episodes"]
            and metrics["mean_damage"] <= self._config["max_mean_damage"]
            and metrics["total_collisions"] <= self._config["max_total_collisions"]
        )
        self._last_probe_metrics = dict(metrics)
        self._last_probe_passed = bool(passed)
        self._promotion_blocked_reason = None
        self._consecutive_passes = self._consecutive_passes + 1 if passed else 0
        self._probe_due = False
        while self._next_probe_at <= self._active_stage_timesteps:
            self._next_probe_at += self._config["probe_interval_timesteps"]
        event: dict[str, Any] = {
            "event": "mastery_probe",
            "stage": stage["name"],
            "passed": bool(passed),
            "metrics": dict(metrics),
            "consecutive_passes": self._consecutive_passes,
            "consumed_timesteps": self._consumed_timesteps,
            "active_stage_timesteps": self._active_stage_timesteps,
            "promotion": "not_promoted",
        }
        threshold_met = self._consecutive_passes >= self._config["required_consecutive_passes"]
        has_next_stage = self._active_index + 1 < len(self._stages)
        if self._config["enable_curriculum_promotion"] and threshold_met and has_next_stage:
            if self.remaining_timesteps < self._config["minimum_stage_timesteps"]:
                self._promotion_blocked_reason = "insufficient_global_budget_for_next_stage"
                event["promotion"] = "blocked_by_budget"
            else:
                previous = stage["name"]
                self._active_index += 1
                self._active_stage_timesteps = 0
                self._next_probe_at = self._config["probe_interval_timesteps"]
                self._mixture_cursor = 0
                self._consecutive_passes = 0
                self._last_probe_passed = None
                self._last_probe_metrics = None
                self._promotion_blocked_reason = None
                event["promotion"] = "advanced"
                event["from_stage"] = previous
                event["to_stage"] = self._stages[self._active_index]["name"]
        self._events.append(event)
        return {**self.decision(), "last_event": dict(event)}

    def validate_best_probe(self, report: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a report against the separate global checkpoint-selection suite."""
        if not self._config["enable_best_checkpoint_selection"]:
            raise RuntimeError("best-checkpoint selection is disabled")
        spec = self._config["best_probe"]
        if spec is None:
            raise RuntimeError("no global best-checkpoint probe is configured")
        _, metrics = _validate_report_for_spec(
            report,
            purpose="best_checkpoint",
            probe_stage="global_suite",
            spec=spec,
        )
        self._events.append({"event": "best_probe_validated", "metrics": dict(metrics)})
        return {"probe_stage": "global_suite", "metrics": dict(metrics)}

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable controller state, including audit decisions."""
        return {
            "schema": STATE_SCHEMA,
            "stages": [dict(stage) for stage in self._stages],
            "config": _plain_json(self._config, name="controller config"),
            "consumed_timesteps": self._consumed_timesteps,
            "active_stage_timesteps": self._active_stage_timesteps,
            "active_index": self._active_index,
            "consecutive_passes": self._consecutive_passes,
            "next_probe_at": self._next_probe_at,
            "probe_due": self._probe_due,
            "mixture_cursor": self._mixture_cursor,
            "block_count": self._block_count,
            "pending_plan": None if self._pending_plan is None else dict(self._pending_plan),
            "last_probe_metrics": None if self._last_probe_metrics is None else dict(self._last_probe_metrics),
            "last_probe_passed": self._last_probe_passed,
            "promotion_blocked_reason": self._promotion_blocked_reason,
            "events": _plain_json(self._events, name="controller events"),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CurriculumController":
        if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"state schema must be {STATE_SCHEMA!r}")
        controller = cls(state["stages"], state["config"])
        consumed = _integer(state.get("consumed_timesteps"), name="consumed_timesteps")
        if consumed > controller._config["total_timesteps"]:
            raise ValueError("serialized state exceeds the fixed global budget")
        active_timesteps = _integer(
            state.get("active_stage_timesteps", 0), name="active_stage_timesteps"
        )
        active = _integer(state.get("active_index"), name="active_index")
        if active >= len(controller._stages):
            raise ValueError("serialized active_index is outside the curriculum")
        consecutive = _integer(state.get("consecutive_passes"), name="consecutive_passes")
        next_probe = _integer(state.get("next_probe_at"), name="next_probe_at", minimum=1)
        cursor = _integer(state.get("mixture_cursor"), name="mixture_cursor")
        block_count = _integer(state.get("block_count", cursor), name="block_count")
        probe_due = state.get("probe_due")
        if not isinstance(probe_due, bool):
            raise ValueError("serialized probe_due must be a boolean")
        controller._consumed_timesteps = consumed
        controller._active_stage_timesteps = active_timesteps
        controller._active_index = active
        controller._consecutive_passes = consecutive
        controller._next_probe_at = next_probe
        controller._probe_due = probe_due
        controller._mixture_cursor = cursor
        controller._block_count = block_count
        pending = state.get("pending_plan")
        controller._pending_plan = None if pending is None else _plain_json(pending, name="pending_plan")
        metrics = state.get("last_probe_metrics")
        controller._last_probe_metrics = None if metrics is None else _plain_json(metrics, name="last_probe_metrics")
        passed = state.get("last_probe_passed")
        if passed is not None and not isinstance(passed, bool):
            raise ValueError("serialized last_probe_passed must be a boolean or None")
        controller._last_probe_passed = passed
        reason = state.get("promotion_blocked_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("serialized promotion_blocked_reason must be a string or None")
        controller._promotion_blocked_reason = reason
        events = state.get("events", [])
        if not isinstance(events, list):
            raise ValueError("serialized events must be a list")
        controller._events = _plain_json(events, name="events")
        if controller._pending_plan is not None and controller._probe_due:
            raise ValueError("serialized state cannot have both a pending training block and a due probe")
        return controller


def compare_best_checkpoints(
    candidate_report: Mapping[str, Any],
    best_report: Mapping[str, Any],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Select only a safe improvement on one fixed global probe suite.

    Course-specific mastery probes are intentionally rejected here.  The
    caller should persist the candidate report as best only when
    ``replace_best`` is true.
    """
    if not enabled:
        return {"comparable": False, "assessment": "selection_disabled", "replace_best": False}
    left_meta = _probe_metadata(candidate_report)
    right_meta = _probe_metadata(best_report)
    if left_meta["purpose"] != "best_checkpoint" or right_meta["purpose"] != "best_checkpoint":
        return {"comparable": False, "assessment": "not_global_checkpoint_probes", "replace_best": False}
    if left_meta["probe_stage"] != right_meta["probe_stage"]:
        return {
            "comparable": False,
            "assessment": "incomparable_probe_stage",
            "replace_best": False,
            "candidate_probe_stage": left_meta["probe_stage"],
            "best_probe_stage": right_meta["probe_stage"],
        }
    if left_meta["probe_stage"] != "global_suite":
        return {
            "comparable": False,
            "assessment": "incomparable_probe_stage",
            "replace_best": False,
            "candidate_probe_stage": left_meta["probe_stage"],
            "best_probe_stage": right_meta["probe_stage"],
        }
    if left_meta["seeds"] != right_meta["seeds"]:
        return {"comparable": False, "assessment": "incomparable_probe_seeds", "replace_best": False}
    if left_meta["config"] != right_meta["config"]:
        return {"comparable": False, "assessment": "incomparable_probe_config", "replace_best": False}
    candidate_rows, candidate_metrics = _validate_report_for_spec(
        candidate_report,
        purpose="best_checkpoint",
        probe_stage="global_suite",
        spec={"seeds": left_meta["seeds"], "config": left_meta["config"]},
    )
    best_rows, best_metrics = _validate_report_for_spec(
        best_report,
        purpose="best_checkpoint",
        probe_stage="global_suite",
        spec={"seeds": right_meta["seeds"], "config": right_meta["config"]},
    )
    paired, mismatches = _initial_conditions_match(candidate_rows, best_rows)
    if not paired:
        return {
            "comparable": False,
            "assessment": "invalid_initial_conditions",
            "replace_best": False,
            "mismatches": mismatches,
            "candidate_metrics": candidate_metrics,
            "best_metrics": best_metrics,
        }
    if candidate_report["damage_config"] != best_report["damage_config"]:
        return {"comparable": False, "assessment": "incomparable_damage_geometry", "replace_best": False}
    if candidate_report["damage_metric"] != best_report["damage_metric"]:
        return {"comparable": False, "assessment": "incomparable_damage_metric", "replace_best": False}
    if candidate_report["drive_config"] != best_report["drive_config"] or candidate_report["observation_config"] != best_report["observation_config"]:
        return {"comparable": False, "assessment": "incomparable_environment_contract", "replace_best": False}

    candidate_safe = dict(candidate_report)
    best_safe = dict(best_report)
    candidate_safe["summary"] = {**dict(candidate_report.get("summary", {})), **candidate_metrics}
    best_safe["summary"] = {**dict(best_report.get("summary", {})), **best_metrics}
    # Reuse the project's already-declared clean-success/safety ordering.
    from scripts.improve_isaac_policy import _safe_goto_assessment

    assessment = _safe_goto_assessment(best_safe, candidate_safe)
    return {
        "comparable": True,
        "assessment": assessment,
        "replace_best": assessment == "improved_under_predeclared_safety_criterion",
        "candidate_metrics": candidate_metrics,
        "best_metrics": best_metrics,
        "paired_seed_count": len(candidate_rows),
        "candidate_checkpoint": candidate_report.get("checkpoint"),
        "best_checkpoint": best_report.get("checkpoint"),
        "selected_checkpoint": (
            candidate_report.get("checkpoint")
            if assessment == "improved_under_predeclared_safety_criterion"
            else best_report.get("checkpoint")
        ),
    }

