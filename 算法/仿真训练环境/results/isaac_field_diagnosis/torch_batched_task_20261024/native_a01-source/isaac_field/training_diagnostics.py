"""Optional, append-only training diagnostics for SB3-Contrib RecurrentPPO.

The pure accumulator in this module accepts the local values exposed by an
SB3 on-policy callback. It deliberately does not depend on Isaac Lab or SB3 so
the data handling can be checked on CPU without creating a simulator.

SB3's rollout-end callback runs after collection and before the optimizer
update for that rollout. In the pinned on-policy loop, ``dump_logs`` runs after
that callback and before ``train``; ``Logger.dump`` clears ``name_to_value``.
When numeric values remain at rollout end, the record reports the snapshot as
seen and identifies its expected source. With the default ``log_interval=1``
this is normally the preceding train; a custom interval can change the source
window. An empty/non-numeric snapshot remains missing rather than being
reconstructed. A separate training-end event records the final logger snapshot
after ``learn`` returns.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .action_contract import ACTION_COMPONENTS, ACTION_SCHEMA


SCHEMA = "isaac-field-training-diagnostics-v1"
ACTION_ABSOLUTE_THRESHOLD = 0.95
ACTION_BOUND_LOW = -1.0
ACTION_BOUND_HIGH = 1.0
REWARD_COMPONENT_ABS_TOLERANCE = 1e-5


def _as_python(value: Any) -> Any:
    """Convert common NumPy/Torch scalar or array containers without importing them."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError, RuntimeError):
            return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return value
    return value


def _numbers(value: Any) -> tuple[list[float], int]:
    """Return finite numeric leaves and the number of non-finite numeric leaves."""
    value = _as_python(value)
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes, Mapping)):
        return [], 0
    if isinstance(value, Sequence):
        result: list[float] = []
        non_finite = 0
        for child in value:
            values, invalid = _numbers(child)
            result.extend(values)
            non_finite += invalid
        return result, non_finite
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return [], 0
    if not math.isfinite(numeric):
        return [], 1
    return [numeric], 0


class _Stats:
    def __init__(self) -> None:
        self.count = 0
        self.non_finite_count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, value: Any) -> None:
        values, invalid = _numbers(value)
        self.non_finite_count += invalid
        for item in values:
            self.count += 1
            self.total += item
            self.square_total += item * item
            self.minimum = item if self.minimum is None else min(self.minimum, item)
            self.maximum = item if self.maximum is None else max(self.maximum, item)

    def add_number(self, value: float) -> None:
        self.add(value)

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {
                "count": 0,
                "non_finite_count": self.non_finite_count,
                "sum": None,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.square_total / self.count - mean * mean)
        return {
            "count": self.count,
            "non_finite_count": self.non_finite_count,
            "sum": self.total,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _items(value: Any) -> list[Any]:
    value = _as_python(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _shape(value: Any) -> tuple[Any, ...]:
    """Infer a nested-list shape so equal flattened lengths cannot fake pairing."""
    value = _as_python(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return (0,)
        child_shapes = [_shape(child) for child in value]
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            return (len(value), "ragged")
        return (len(value),) + child_shapes[0]
    return ()


def _scalar_mapping(value: Any) -> dict[str, float] | None:
    """Keep only finite scalar entries so JSON logs never contain NaN/Inf."""
    value = _as_python(value)
    if not isinstance(value, Mapping):
        return None
    output: dict[str, float] = {}
    for key, raw in value.items():
        values, invalid = _numbers(raw)
        if len(values) == 1 and invalid == 0:
            output[str(key)] = values[0]
    return output or None


def _strict_scalar_mapping(value: Any) -> tuple[dict[str, float] | None, str | None]:
    """Validate that a non-empty mapping contains only finite scalar values."""
    value = _as_python(value)
    if not isinstance(value, Mapping):
        return None, "missing_or_non_mapping"
    if not value:
        return None, "empty_mapping"
    output: dict[str, float] = {}
    for key, raw in value.items():
        values, invalid = _numbers(raw)
        if len(values) != 1 or invalid:
            return None, "non_scalar_or_non_finite_component"
        output[str(key)] = values[0]
    return output, None


class _ActionComponentStats:
    """Per-coordinate action statistics, enabled only for the v1 4D contract."""

    def __init__(self) -> None:
        self.expected_steps = 0
        self.observed_steps = 0
        self.missing_steps = 0
        self.invalid_shape_steps = 0
        self.invalid_shapes: set[tuple[Any, ...]] = set()
        self.invalid_shape_reasons: dict[str, int] = {}
        self.env_counts: set[int] = set()
        self.expected_env_counts: set[int] = set()
        self.stats = {name: _Stats() for name in ACTION_COMPONENTS}
        self.absolute_threshold_counts = {name: 0 for name in ACTION_COMPONENTS}
        self.exact_bound_counts = {name: 0 for name in ACTION_COMPONENTS}
        self.over_bound_counts = {name: 0 for name in ACTION_COMPONENTS}

    def add(self, raw: Any, *, expected_env_count: int | None) -> None:
        self.expected_steps += 1
        if expected_env_count is not None:
            self.expected_env_counts.add(expected_env_count)
        if raw is None:
            self.missing_steps += 1
            return

        matrix = _as_python(raw)
        shape = _shape(raw)
        valid = len(shape) == 2 and isinstance(shape[0], int) and shape[0] > 0 and shape[1] == len(ACTION_COMPONENTS)
        invalid_reason = None if valid else "shape_not_[num_env,4]"
        if valid and expected_env_count is not None and shape[0] != expected_env_count:
            valid = False
            invalid_reason = "num_env_does_not_match_callback_reward_stream"
        if not valid:
            self.invalid_shape_steps += 1
            self.invalid_shapes.add(shape)
            reason = invalid_reason or "shape_not_[num_env,4]"
            self.invalid_shape_reasons[reason] = self.invalid_shape_reasons.get(reason, 0) + 1
            return

        self.observed_steps += 1
        self.env_counts.add(shape[0])
        for component_index, component_name in enumerate(ACTION_COMPONENTS):
            column = [row[component_index] for row in matrix]
            finite_values, non_finite_count = _numbers(column)
            component_stats = self.stats[component_name]
            component_stats.non_finite_count += non_finite_count
            for value in finite_values:
                component_stats.add_number(value)
                if abs(value) >= ACTION_ABSOLUTE_THRESHOLD:
                    self.absolute_threshold_counts[component_name] += 1
                if value == ACTION_BOUND_LOW or value == ACTION_BOUND_HIGH:
                    self.exact_bound_counts[component_name] += 1
                if value < ACTION_BOUND_LOW or value > ACTION_BOUND_HIGH:
                    self.over_bound_counts[component_name] += 1

    @staticmethod
    def _rate(count: int, denominator: int) -> float | None:
        return count / denominator if denominator else None

    def as_dict(self) -> dict[str, Any]:
        if self.observed_steps == 0:
            status = "unavailable_invalid_shape" if self.invalid_shape_steps else "unavailable_missing"
        elif self.missing_steps or self.invalid_shape_steps:
            status = "partial"
        else:
            status = "observed"
        components: list[dict[str, Any]] | None = None
        if self.observed_steps:
            components = []
            for name in ACTION_COMPONENTS:
                stats = self.stats[name]
                denominator = stats.count
                components.append({
                    "name": name,
                    "statistics": stats.as_dict(),
                    "absolute_value_at_least_threshold": {
                        "threshold": ACTION_ABSOLUTE_THRESHOLD,
                        "count": self.absolute_threshold_counts[name],
                        "denominator_finite_values": denominator,
                        "fraction": self._rate(self.absolute_threshold_counts[name], denominator),
                    },
                    "exact_bound": {
                        "low": ACTION_BOUND_LOW,
                        "high": ACTION_BOUND_HIGH,
                        "count": self.exact_bound_counts[name],
                        "denominator_finite_values": denominator,
                        "fraction": self._rate(self.exact_bound_counts[name], denominator),
                    },
                    "over_bound": {
                        "low": ACTION_BOUND_LOW,
                        "high": ACTION_BOUND_HIGH,
                        "count": self.over_bound_counts[name],
                        "denominator_finite_values": denominator,
                        "fraction": self._rate(self.over_bound_counts[name], denominator),
                    },
                })
        return {
            "status": status,
            "action_schema": ACTION_SCHEMA,
            "component_order": list(ACTION_COMPONENTS),
            "expected_shape": "[num_env, 4]",
            "normalized_bound": [ACTION_BOUND_LOW, ACTION_BOUND_HIGH],
            "absolute_value_threshold": ACTION_ABSOLUTE_THRESHOLD,
            "expected_steps": self.expected_steps,
            "observed_steps": self.observed_steps,
            "missing_steps": self.missing_steps,
            "invalid_shape_steps": self.invalid_shape_steps,
            "invalid_shapes": [list(shape) for shape in sorted(self.invalid_shapes, key=repr)],
            "invalid_shape_reasons": dict(sorted(self.invalid_shape_reasons.items())) or None,
            "observed_num_env_values": sorted(self.env_counts),
            "expected_num_env_values": sorted(self.expected_env_counts),
            "components": components,
        }


class _EpisodeRewardComponents:
    """Keep actual per-env component values across rollouts, without zero fill."""

    def __init__(self) -> None:
        self.expected_steps = 0
        self.provided_steps = 0
        self.missing_steps = 0
        self.invalid_mapping_steps = 0
        self.stats: dict[str, _Stats] = {}
        self.present_steps: dict[str, int] = {}
        self.invalid_component_steps: dict[str, int] = {}

    def observe(self, info: Any) -> None:
        self.expected_steps += 1
        if not isinstance(info, Mapping) or not isinstance(info.get("reward_components"), Mapping):
            self.missing_steps += 1
            return
        raw_components = _as_python(info["reward_components"])
        if not isinstance(raw_components, Mapping) or not raw_components:
            self.invalid_mapping_steps += 1
            return
        self.provided_steps += 1
        has_invalid_value = False
        for raw_name, raw_value in raw_components.items():
            name = str(raw_name)
            self.present_steps[name] = self.present_steps.get(name, 0) + 1
            values, invalid = _numbers(raw_value)
            if len(values) == 1 and not invalid:
                self.stats.setdefault(name, _Stats()).add_number(values[0])
            else:
                self.invalid_component_steps[name] = self.invalid_component_steps.get(name, 0) + 1
                has_invalid_value = True
        if has_invalid_value:
            self.invalid_mapping_steps += 1

    def as_dict(self) -> dict[str, Any]:
        if self.expected_steps == 0:
            status = "unavailable"
        elif self.provided_steps == 0:
            status = "missing"
        elif (
            self.missing_steps
            or self.invalid_mapping_steps
            or any(self.stats.get(name, _Stats()).count < self.expected_steps for name in self.present_steps)
        ):
            status = "partial"
        else:
            status = "complete"

        components: dict[str, dict[str, Any]] = {}
        for name in sorted(self.present_steps):
            stats = self.stats.get(name, _Stats())
            valid_steps = stats.count
            complete = self.expected_steps > 0 and valid_steps == self.expected_steps
            components[name] = {
                "status": "complete" if complete else "partial",
                "total": stats.total if complete else None,
                "observed_sum": stats.total if valid_steps else None,
                "present_steps": self.present_steps[name],
                "valid_steps": valid_steps,
                "missing_steps": max(0, self.expected_steps - self.present_steps[name]),
                "invalid_steps": self.invalid_component_steps.get(name, 0),
                "observed_statistics": stats.as_dict(),
            }
        return {
            "status": status,
            "expected_env_steps": self.expected_steps,
            "provided_env_steps": self.provided_steps,
            "missing_env_steps": self.missing_steps,
            "invalid_env_steps": self.invalid_mapping_steps,
            "components": components or None,
            "note": "Incomplete component totals are null; observed_sum includes only values actually supplied and is never zero-filled.",
        }


class _RewardComponentConsistency:
    """Compare supplied per-env component sums with the callback reward stream."""

    def __init__(self) -> None:
        self.expected_env_steps = 0
        self.compared_env_steps = 0
        self.within_tolerance_env_steps = 0
        self.mismatch_rows: list[dict[str, Any]] = []
        self.unavailable_reasons: dict[str, int] = {}

    def _unavailable(self, reason: str) -> None:
        self.unavailable_reasons[reason] = self.unavailable_reasons.get(reason, 0) + 1

    def observe(self, callback_step: int, rewards: list[Any], infos: list[Any]) -> None:
        expected_count = max(len(rewards), len(infos))
        self.expected_env_steps += expected_count
        for env_index in range(expected_count):
            if env_index >= len(infos) or not isinstance(infos[env_index], Mapping):
                self._unavailable("info_missing_or_non_mapping")
                continue
            info = infos[env_index]
            if "reward_components" not in info:
                self._unavailable("reward_components_missing")
                continue
            components, reason = _strict_scalar_mapping(info.get("reward_components"))
            if reason is not None or components is None:
                self._unavailable(reason or "invalid_reward_components")
                continue
            if env_index >= len(rewards):
                self._unavailable("callback_reward_missing")
                continue
            reward_values, invalid_reward_count = _numbers(rewards[env_index])
            if len(reward_values) != 1 or invalid_reward_count:
                self._unavailable("callback_reward_non_scalar_or_non_finite")
                continue

            component_sum = sum(components.values())
            reward = reward_values[0]
            absolute_error = abs(component_sum - reward)
            self.compared_env_steps += 1
            if absolute_error <= REWARD_COMPONENT_ABS_TOLERANCE:
                self.within_tolerance_env_steps += 1
            else:
                self.mismatch_rows.append({
                    "callback_step": callback_step,
                    "env_index": env_index,
                    "callback_reward": reward,
                    "reward_components_sum": component_sum,
                    "absolute_error": absolute_error,
                    "component_names": sorted(components),
                })

    def as_dict(self) -> dict[str, Any]:
        unavailable_count = sum(self.unavailable_reasons.values())
        mismatch_count = len(self.mismatch_rows)
        if self.expected_env_steps == 0 or self.compared_env_steps == 0:
            status = "unavailable"
        elif mismatch_count and unavailable_count:
            status = "partial_with_mismatches"
        elif mismatch_count:
            status = "mismatched"
        elif unavailable_count:
            status = "partial"
        else:
            status = "matched"
        return {
            "status": status,
            "absolute_tolerance": REWARD_COMPONENT_ABS_TOLERANCE,
            "expected_env_steps": self.expected_env_steps,
            "compared_env_steps": self.compared_env_steps,
            "within_tolerance_env_steps": self.within_tolerance_env_steps,
            "mismatch_env_steps": mismatch_count,
            "unavailable_env_steps": unavailable_count,
            "unavailable_reasons": dict(sorted(self.unavailable_reasons.items())) or None,
            "mismatches": self.mismatch_rows or None,
            "note": "Read-only comparison; callback rewards are neither replaced nor adjusted.",
        }


def _logger_snapshot(logger: Any) -> dict[str, float] | None:
    name_to_value = getattr(logger, "name_to_value", None)
    return _scalar_mapping(name_to_value)


def _compact_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only task outcome fields useful at terminal transitions."""
    result: dict[str, Any] = {}
    for name in ("mode", "reason", "is_success", "damage", "collisions", "distance", "time"):
        if name not in info:
            continue
        value = _as_python(info[name])
        if isinstance(value, (str, bool, int)) or value is None:
            result[name] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[name] = value
    return result


class TrainingDiagnosticsAccumulator:
    """Aggregate actual callback observations into one record per PPO rollout.

    ``observe_step`` expects the relevant mapping from SB3's callback locals:
    ``actions``, ``clipped_actions``, ``rewards``, ``dones``, and ``infos``.
    The raw sampled actions and the clipped values passed to VecEnv are kept as
    separate measurements. Episode returns are summed from the rewards seen by
    this callback, including any environment-side shaping or terminal penalty.
    """

    def __init__(self, stage: str | None = None) -> None:
        self.stage = None if stage is None else str(stage)
        self.rollout_index = 0
        self.last_rollout_end_num_timesteps: int | None = None
        self._episode_returns: list[float] | None = None
        self._episode_reward_components: list[_EpisodeRewardComponents] | None = None
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        self._callback_steps = 0
        self._sampled_actions = _Stats()
        self._vecenv_actions = _Stats()
        self._sampled_action_components = _ActionComponentStats()
        self._vecenv_action_components = _ActionComponentStats()
        self._action_pair_count = 0
        self._action_changed_count = 0
        self._action_pairing_statuses: set[str] = set()
        self._rewards = _Stats()
        self._completed_returns = _Stats()
        self._completed_episodes: list[dict[str, Any]] = []
        self._reward_component_totals: dict[str, _Stats] = {}
        self._reward_component_observed_steps = 0
        self._reward_component_expected_steps = 0
        self._reward_component_consistency = _RewardComponentConsistency()
        self._info_sources: set[str] = set()

    def observe_step(
        self,
        callback_locals: Mapping[str, Any],
        *,
        info_source: str = "sb3_callback_locals",
    ) -> None:
        """Consume one vector-environment step from an SB3 callback."""
        if not isinstance(callback_locals, Mapping):
            raise TypeError("callback_locals must be a mapping")
        self._callback_steps += 1
        self._info_sources.add(str(info_source))

        sampled_raw = callback_locals.get("actions")
        env_actions_raw = callback_locals.get("clipped_actions")
        sampled, sampled_non_finite = _numbers(sampled_raw)
        env_actions, env_non_finite = _numbers(env_actions_raw)
        self._sampled_actions.add(sampled_raw)
        self._vecenv_actions.add(env_actions_raw)
        has_sampled = sampled_raw is not None
        has_env_actions = env_actions_raw is not None
        if has_sampled or has_env_actions:
            if not has_sampled or not has_env_actions:
                self._action_pairing_statuses.add("one_action_array_missing")
            elif sampled_non_finite or env_non_finite:
                # Finite-only flattening cannot preserve element alignment once
                # a NaN/Inf is present, so do not manufacture a pairwise rate.
                self._action_pairing_statuses.add("non_finite_action_values")
            elif len(sampled) == len(env_actions) and _shape(sampled_raw) == _shape(env_actions_raw):
                self._action_pairing_statuses.add("paired")
                for before, after in zip(sampled, env_actions):
                    self._action_pair_count += 1
                    if before != after:
                        self._action_changed_count += 1
            else:
                # Preserve the separate distributions; no paired clipping
                # claim is valid when the callback-local shapes disagree.
                self._action_pairing_statuses.add("action_array_size_mismatch")

        rewards = _numbers(callback_locals.get("rewards"))[0]
        rewards_raw = callback_locals.get("rewards")
        reward_items = _items(rewards_raw)
        self._rewards.add(callback_locals.get("rewards"))
        dones_raw = _items(callback_locals.get("dones"))
        dones = [bool(_as_python(value)) for value in dones_raw]
        infos = _items(callback_locals.get("infos"))
        expected_env_count = len(reward_items) or None
        self._sampled_action_components.add(sampled_raw, expected_env_count=expected_env_count)
        self._vecenv_action_components.add(env_actions_raw, expected_env_count=expected_env_count)
        self._reward_component_consistency.observe(self._callback_steps, reward_items, infos)

        if rewards:
            if self._episode_returns is None:
                self._episode_returns = [0.0] * len(rewards)
            elif len(self._episode_returns) != len(rewards):
                raise ValueError("number of reward streams changed within one training stage")
            for index, reward in enumerate(rewards):
                self._episode_returns[index] += reward

        if reward_items:
            if self._episode_reward_components is None:
                self._episode_reward_components = [
                    _EpisodeRewardComponents() for _ in reward_items
                ]
            elif len(self._episode_reward_components) != len(reward_items):
                raise ValueError("number of reward streams changed within one training stage")
            for index, state in enumerate(self._episode_reward_components):
                state.observe(infos[index] if index < len(infos) else None)

        self._reward_component_expected_steps += max(len(infos), len(rewards))
        for info in infos:
            if not isinstance(info, Mapping):
                continue
            components = info.get("reward_components")
            if not isinstance(components, Mapping):
                continue
            self._reward_component_observed_steps += 1
            for key, value in components.items():
                stats = self._reward_component_totals.setdefault(str(key), _Stats())
                stats.add(value)

        if self._episode_returns is None:
            return
        for index, done in enumerate(dones):
            if not done or index >= len(self._episode_returns):
                continue
            episode_return = self._episode_returns[index]
            self._completed_returns.add_number(episode_return)
            info = infos[index] if index < len(infos) else {}
            if isinstance(info, Mapping):
                terminal_info = info.get("terminal_info", info.get("final_info", info))
                summary = _compact_info(terminal_info) if isinstance(terminal_info, Mapping) else {}
            else:
                summary = {}
            row = {"env_index": index, "callback_reward_return": episode_return}
            if self._episode_reward_components is not None and index < len(self._episode_reward_components):
                row["reward_components"] = self._episode_reward_components[index].as_dict()
                self._episode_reward_components[index] = _EpisodeRewardComponents()
            else:
                row["reward_components"] = {
                    "status": "unavailable",
                    "expected_env_steps": 0,
                    "provided_env_steps": 0,
                    "missing_env_steps": 0,
                    "invalid_env_steps": 0,
                    "components": None,
                    "note": "No reward-component observations were available for this environment episode.",
                }
            if summary:
                row["terminal_info"] = summary
            self._completed_episodes.append(row)
            self._episode_returns[index] = 0.0

    def finish_rollout(
        self,
        *,
        num_timesteps: int,
        logger_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return and reset a rollout record at SB3's rollout-end callback."""
        self.rollout_index += 1
        current = int(num_timesteps)
        source_timestep = self.last_rollout_end_num_timesteps
        source_rollout = self.rollout_index - 1 if source_timestep is not None else None
        metrics = _scalar_mapping(logger_metrics)
        logger_status = (
            "no_numeric_metrics_available" if metrics is None
            else "previous_rollout_optimizer" if source_rollout is not None
            else "source_rollout_unknown_before_callback"
        )
        changed_fraction = (
            self._action_changed_count / self._action_pair_count
            if self._action_pair_count else None
        )
        component_totals = {
            key: stats.as_dict() for key, stats in sorted(self._reward_component_totals.items())
        }
        if not self._reward_component_observed_steps:
            component_status = "not_provided_by_environment"
        elif self._reward_component_observed_steps < self._reward_component_expected_steps:
            component_status = "partial"
        else:
            component_status = "observed"
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "event_type": "rollout",
            "stage": self.stage,
            "rollout_index": self.rollout_index,
            "num_timesteps": current,
            "callback_vector_steps": self._callback_steps,
            "sampled_policy_actions": self._sampled_actions.as_dict(),
            "sampled_policy_action_components": self._sampled_action_components.as_dict(),
            "actions_passed_to_vec_env": self._vecenv_actions.as_dict(),
            "actions_passed_to_vec_env_components": self._vecenv_action_components.as_dict(),
            "action_clip_observation": {
                "source": "SB3 collect_rollouts locals: actions versus clipped_actions",
                "pairing_status": (
                    next(iter(self._action_pairing_statuses))
                    if len(self._action_pairing_statuses) == 1
                    else "mixed" if self._action_pairing_statuses else "unavailable"
                ),
                "paired_finite_elements": self._action_pair_count,
                "changed_elements": self._action_changed_count,
                "fraction_changed_by_sb3_action_clip": changed_fraction,
                "note": "clipped_actions is the array passed to VecEnv.step; this is not measured actuator or track feedback",
            },
            "callback_rewards": self._rewards.as_dict(),
            "completed_episode_returns": self._completed_returns.as_dict(),
            "completed_episodes": self._completed_episodes,
            "terminal_info_source": (
                next(iter(self._info_sources)) if len(self._info_sources) == 1
                else "mixed" if self._info_sources else None
            ),
            "reward_components": {
                "status": component_status,
                "observed_env_steps": self._reward_component_observed_steps,
                "expected_env_steps": self._reward_component_expected_steps,
                "totals": component_totals or None,
                "source_info_key": "reward_components",
                "per_step_reconciliation": self._reward_component_consistency.as_dict(),
            },
            "logger_metrics": metrics,
            "logger_metric_alignment": {
                "sampled_at": "on_rollout_end_before_current_rollout_optimizer_update",
                "expected_lag_rollouts": 1,
                "source_rollout_index": source_rollout,
                "source_num_timesteps": source_timestep,
                "status": logger_status,
                "snapshot_status": "numeric_values_observed" if metrics is not None else "empty_or_non_numeric_snapshot",
                "dump_order_note": (
                    "SB3 calls on_rollout_end before dump_logs and the current train; logger.dump clears name_to_value. "
                    "With the default log_interval=1, numeric values are normally from the preceding train. "
                    "Custom logging intervals may change the source window. This is the direct callback-time snapshot; "
                    "missing values are not reconstructed."
                ),
            },
        }
        self.last_rollout_end_num_timesteps = current
        self._reset_rollout()
        return record

    def finish_training(
        self,
        *,
        num_timesteps: int,
        logger_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a final post-learn logger snapshot as its own JSONL event."""
        current = int(num_timesteps)
        source_timestep = self.last_rollout_end_num_timesteps
        metrics = _scalar_mapping(logger_metrics)
        return {
            "schema": SCHEMA,
            "event_type": "final_optimizer_metrics",
            "stage": self.stage,
            "rollout_index": self.rollout_index,
            "num_timesteps": current,
            "logger_metrics": metrics,
            "logger_metric_alignment": {
                "sampled_at": "on_training_end_after_learn_loop",
                "expected_lag_rollouts": 0 if source_timestep is not None else None,
                "source_rollout_index": self.rollout_index if source_timestep is not None else None,
                "source_num_timesteps": source_timestep,
                "status": "no_numeric_metrics_available" if metrics is None else "final_completed_update",
                "snapshot_status": "numeric_values_observed" if metrics is not None else "empty_or_non_numeric_snapshot",
                "dump_order_note": (
                    "SB3's final train follows the preceding dump_logs call; the training-end callback records the "
                    "direct logger snapshot available after learn returns and does not reconstruct missing values."
                ),
            },
        }


def make_sb3_diagnostics_callback(
    path: str | Path,
    stage: str | None = None,
    info_provider: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
):
    """Create an optional SB3 callback that appends JSONL rollout diagnostics.

    SB3 is imported only when this factory is called. Callers may combine the
    returned callback with checkpoint and progress callbacks using CallbackList.
    The target is append-only so interruption never replaces earlier records.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    target = Path(path)

    class TrainingDiagnosticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.accumulator = TrainingDiagnosticsAccumulator(stage=stage)
            self._stream = None

        def _on_training_start(self) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._stream = target.open("a", encoding="utf-8", newline="\n")

        def _write(self, record: Mapping[str, Any]) -> None:
            if self._stream is None:
                raise RuntimeError("diagnostics callback stream is not open")
            self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._stream.flush()

        def _on_step(self) -> bool:
            info_source = "sb3_callback_locals"
            callback_locals = self.locals
            if info_provider is not None:
                callback_locals = dict(self.locals)
                callback_locals["infos"] = info_provider()
                info_source = "environment_info_provider"
            self.accumulator.observe_step(callback_locals, info_source=info_source)
            return True

        def _on_rollout_end(self) -> None:
            self._write(self.accumulator.finish_rollout(
                num_timesteps=self.num_timesteps,
                logger_metrics=_logger_snapshot(self.model.logger),
            ))

        def _on_training_end(self) -> None:
            try:
                self._write(self.accumulator.finish_training(
                    num_timesteps=self.num_timesteps,
                    logger_metrics=_logger_snapshot(self.model.logger),
                ))
            finally:
                if self._stream is not None:
                    self._stream.close()
                    self._stream = None

    return TrainingDiagnosticsCallback()


__all__ = [
    "TrainingDiagnosticsAccumulator",
    "make_sb3_diagnostics_callback",
]
