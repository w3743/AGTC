"""A Gymnasium map designer trained with the official RSL-RL PPO runner.

This is a one-sided, naive minimax teacher: its terminal reward is the negative
student score supplied by ``evaluator``. It does not implement PAIRED.
"""
from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from isaac_field.layouts import row_positions, scatter_positions
from isaac_field.task import Config, Scene, _make_terrain


TEACHER_RUNTIME_IDENTITY = {
    "simulator": "gymnasium",
    "physics_backend": "none",
    "learning_backend": "rsl_rl",
}
TARGET_STUDENT_BACKEND_IDENTITY = {
    "simulator": "isaac_sim",
    "physics_backend": "physx",
    "learning_backend": "rsl_rl",
}


@dataclass(frozen=True)
class DesignerTrainingResult:
    model: Any
    checkpoint: Path
    summary_path: Path
    episodes_path: Path
    summary: dict[str, Any]
    actual_timesteps: int = 0


class MapDesignerEnv(gym.Env):
    """Edit a fixed-shape field layout, then score it once when editing ends.

    Each continuous action is ``[plant_index, x, y]`` with values in ``[-1,1]``.
    The index selects a fixed plant row; x/y select a field grid cell. A layout
    is committed only when plant bounds, rock clearance, pair spacing, a
    feasible sampled start, and ``Scene.validate_connectivity`` all pass.

    At termination, ``scene.spawn_pose`` and ``scene.start_pose`` contain a
    random pose obtained from ``Scene.sample_start``. An evaluator can use that
    pose when launching a student run. No robot steps occur while the teacher
    is editing.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Config,
        evaluator: Callable[[Scene], float],
        seed: int = 7,
        *,
        layout_template: Scene | None = None,
        terrain=None,
        edit_budget: int = 12,
        invalid_penalty: float = 1.0,
        max_layout_attempts: int = 64,
        max_episode_records: int = 128,
        evaluator_kind: str = "score_callback",
        student_physx_evaluated: bool = False,
    ):
        super().__init__()
        if not callable(evaluator):
            raise TypeError("evaluator must be callable and return a student score")
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(edit_budget, bool) or not 8 <= int(edit_budget) <= 16:
            raise ValueError("edit_budget must be between 8 and 16")
        if not math.isfinite(invalid_penalty) or invalid_penalty <= 0:
            raise ValueError("invalid_penalty must be finite and positive")
        if max_layout_attempts < 1 or max_episode_records < 1:
            raise ValueError("layout attempts and episode record limits must be positive")
        if not isinstance(evaluator_kind, str) or not evaluator_kind.strip():
            raise ValueError("evaluator_kind must be a non-empty string")
        if type(student_physx_evaluated) is not bool:
            raise ValueError("student_physx_evaluated must be boolean")
        if cfg.plants < 1:
            raise ValueError("MapDesignerEnv requires at least one plant")

        self.cfg = cfg
        self.evaluator = evaluator
        self.evaluator_kind = evaluator_kind
        self.student_physx_evaluated = student_physx_evaluated
        self.seed_value = int(seed)
        self.edit_budget = int(edit_budget)
        self.invalid_penalty = float(invalid_penalty)
        self.max_layout_attempts = int(max_layout_attempts)
        self.max_episode_records = int(max_episode_records)
        self._rng = np.random.default_rng(self.seed_value)
        self._episode_index = 0
        self._episode_records: deque[dict[str, Any]] = deque(maxlen=self.max_episode_records)
        self._geometry_seed = int(self._rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        # These geometry constants are needed while constructing the first
        # template layout as well as for later resets and edits.
        self._plant_count = int(cfg.plants)
        if cfg.layout == "rows":
            # Preserve realistic crop spacing while allowing teacher edits to
            # approach to within 0.28 m of another plant.
            self._plant_spacing = 0.28
            self._plant_margin = max(2.0, float(cfg.stem_radius) + 0.03)
            self._plant_rock_clearance = 0.18
            self._edit_grid_resolution = 0.1
        else:
            self._plant_spacing = max(
                1.4,
                math.hypot(cfg.robot_length, cfg.robot_width) + 0.85,
            )
            self._plant_margin = max(1.2, self._plant_spacing * 0.7)
            self._plant_rock_clearance = 0.15
            self._edit_grid_resolution = float(cfg.cell)
        if 2.0 * self._plant_margin >= cfg.size:
            raise ValueError("field is too small for the configured plant margin")

        n_grid = int(math.ceil(cfg.size / self._edit_grid_resolution))
        self._grid_axis = (np.arange(n_grid, dtype=np.float64) + 0.5) * self._edit_grid_resolution
        allowed = np.flatnonzero(
            (self._grid_axis >= self._plant_margin)
            & (self._grid_axis <= cfg.size - self._plant_margin)
        )
        if len(allowed) < 2:
            raise ValueError("field has fewer than two legal plant grid cells per axis")
        self._allowed_grid_indices = allowed

        if layout_template is not None:
            if len(layout_template.xy) != cfg.plants:
                raise ValueError("layout_template plant count does not match cfg.plants")
            if layout_template.cfg.leaves_per_plant != cfg.leaves_per_plant:
                raise ValueError("layout_template leaves_per_plant does not match cfg")
            expected_dry = self._dry_count(cfg)
            if int(np.asarray(layout_template.targets, dtype=bool).sum()) != expected_dry:
                raise ValueError("layout_template dry target count does not match cfg.dry_fraction")
            self._template_scene = layout_template
            self._targets = np.asarray(layout_template.targets, dtype=bool).copy()
            self.terrain = terrain if terrain is not None else layout_template.terrain
        else:
            self._targets = self._make_targets(cfg.plants, cfg.dry_fraction, self._rng)
            self.terrain = terrain
            if self.terrain is None and cfg.terrain_enabled:
                self.terrain = _make_terrain(cfg, self._geometry_seed, clear_disks=[])
            self._template_scene = self._build_geometry_template()

        if terrain is not None:
            self.terrain = terrain
        self._scene_template = self._prepare_template(self._template_scene)
        self._plant_count = len(self._scene_template.xy)

        # One row per plant: x, y, and the fixed target bit. Two scalars encode
        # edit progress and whether the previous edit passed its constraints.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 * self._plant_count + 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.scene: Scene | None = None
        self.spawn_pose: np.ndarray | None = None
        self._edit_count = 0
        self._valid_edits = 0
        self._invalid_edits = 0
        self._episode_return = 0.0
        self._last_valid = True
        self._episode_checkpoints: list[list[list[float]]] = []
        self._scored = False

    @staticmethod
    def _dry_count(cfg: Config) -> int:
        n_dry = int(round(cfg.plants * cfg.dry_fraction))
        if cfg.plants and cfg.dry_fraction > 0:
            n_dry = max(1, n_dry)
        return min(cfg.plants, n_dry)

    @classmethod
    def _make_targets(cls, plants: int, dry_fraction: float, rng: np.random.Generator) -> np.ndarray:
        n_dry = int(round(plants * dry_fraction))
        if plants and dry_fraction > 0:
            n_dry = max(1, n_dry)
        targets = np.zeros(plants, dtype=bool)
        if n_dry:
            targets[rng.choice(plants, size=n_dry, replace=False)] = True
        return targets

    def _build_geometry_template(self) -> Scene:
        positions = self._random_positions()
        geometry_rng = np.random.default_rng(self._geometry_seed ^ 0x51A7E)
        scene = Scene(
            self.cfg,
            positions,
            self._targets,
            geometry_rng,
            terrain=self.terrain,
            terrain_seed=getattr(self.terrain, "seed", None),
        )
        valid, reason = self._layout_valid(scene.xy, scene=scene)
        if not valid:
            raise ValueError(f"could not build a valid layout template: {reason}")
        return scene

    def _prepare_template(self, template: Scene) -> Scene:
        if len(template.xy) != self.cfg.plants:
            raise ValueError("layout template shape does not match cfg.plants")
        if self.terrain is None:
            self.terrain = template.terrain
        scene = copy.copy(template)
        scene.cfg = self.cfg
        scene.xy = np.asarray(template.xy, dtype=np.float32).copy()
        scene.targets = self._targets.copy()
        scene.terrain = self.terrain
        if self.terrain is not None:
            scene.terrain_seed = int(self.terrain.seed) if self.terrain.seed is not None else None
        return scene

    def _random_positions(self) -> np.ndarray:
        """Sample plants using the configured field layout and rock geometry."""
        cfg = self.cfg
        if cfg.layout == "rows":
            return row_positions(cfg, self._rng, terrain=self.terrain)
        if cfg.layout == "scatter":
            return scatter_positions(cfg, self._rng, terrain=self.terrain)
        raise ValueError("cfg.layout must be 'rows' or 'scatter'")

    def _scene_at_positions(self, positions: np.ndarray) -> Scene:
        """Reuse the fixed shape/target/terrain template and shift only roots/leaves."""
        positions = np.asarray(positions, dtype=np.float32).reshape(self._plant_count, 2)
        scene = copy.copy(self._scene_template)
        scene.cfg = self.cfg
        scene.xy = positions.copy()
        scene.targets = self._targets.copy()
        scene.terrain = self.terrain
        if self.terrain is None:
            new_root_z = np.zeros(self._plant_count, dtype=np.float32)
            scene.terrain_seed = None
        else:
            new_root_z = np.asarray(self.terrain.height(positions), dtype=np.float32).reshape(-1)
            scene.terrain_seed = int(self.terrain.seed) if self.terrain.seed is not None else None
        old_root_z = np.asarray(self._scene_template.root_z, dtype=np.float32)
        root_shift = new_root_z - old_root_z
        plant_ids = self._scene_template.leaf_plant
        scene.root_z = new_root_z.copy()
        scene.centers = self._scene_template.centers.copy()
        scene.centers[:, :2] += positions[plant_ids] - self._scene_template.xy[plant_ids]
        scene.centers[:, 2] += root_shift[plant_ids]
        return scene

    def _layout_valid(self, positions: np.ndarray, *, scene: Scene | None = None) -> tuple[bool, str]:
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        if positions.shape != (self.cfg.plants, 2) or not np.isfinite(positions).all():
            return False, "invalid_position_shape_or_value"
        margin = self._plant_margin
        if np.any(positions < margin) or np.any(positions > self.cfg.size - margin):
            return False, "plant_out_of_bounds"
        if len(positions) > 1:
            delta = positions[:, None, :] - positions[None, :, :]
            distance2 = np.sum(delta * delta, axis=2)
            distance2[np.diag_indices_from(distance2)] = np.inf
            if np.any(distance2 < self._plant_spacing * self._plant_spacing - 1e-8):
                return False, "plant_spacing"
        if self.terrain is not None and len(self.terrain.rock_xy):
            delta = positions[:, None, :] - self.terrain.rock_xy[None, :, :]
            min_distance = self.terrain.rock_radius[None, :] + self._plant_rock_clearance
            if np.any(np.sum(delta * delta, axis=2) <= min_distance * min_distance):
                return False, "plant_overlaps_rock"

        candidate_scene = scene if scene is not None else self._scene_at_positions(positions)
        spawn = None
        for _ in range(8):
            try:
                sampled = candidate_scene.sample_start(
                    self._rng,
                    max_attempts=max(2_000, int(self.cfg.size * self.cfg.size)),
                )
            except ValueError:
                break
            if candidate_scene.navigable(sampled[:2]):
                spawn = sampled
                break
        if spawn is None:
            return False, "insufficient_spawn_area"
        connectivity = candidate_scene.validate_connectivity(spawn[:2], spawn[:2])
        if not connectivity.get("all_free_connected", connectivity.get("connected", False)):
            return False, "vehicle_clearance_disconnects_free_space"
        if not connectivity.get("start_goal_connected", connectivity.get("connected", False)):
            return False, "insufficient_spawn_area"
        return True, ""

    def _decode_action(self, action: np.ndarray) -> tuple[int, np.ndarray | None, str]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all():
            return 0, None, "non_finite_or_wrong_shape_action"
        action = np.clip(action, -1.0, 1.0)
        plant_index = int(np.rint((action[0] + 1.0) * 0.5 * (self._plant_count - 1)))
        grid_slots = self._allowed_grid_indices
        x_index = int(np.rint((action[1] + 1.0) * 0.5 * (len(grid_slots) - 1)))
        y_index = int(np.rint((action[2] + 1.0) * 0.5 * (len(grid_slots) - 1)))
        xy = np.asarray(
            [self._grid_axis[grid_slots[x_index]], self._grid_axis[grid_slots[y_index]]],
            dtype=np.float32,
        )
        return plant_index, xy, ""

    def _observation(self) -> np.ndarray:
        assert self.scene is not None
        xy = np.asarray(self.scene.xy, dtype=np.float32)
        normalized_xy = (xy / np.float32(self.cfg.size) * 2.0 - 1.0).clip(-1.0, 1.0)
        targets = np.where(self._targets, 1.0, -1.0).astype(np.float32)
        rows = np.column_stack((normalized_xy, targets)).reshape(-1)
        progress = np.float32(self._edit_count / self.edit_budget)
        valid = np.float32(1.0 if self._last_valid else -1.0)
        return np.concatenate((rows, np.asarray([progress * 2.0 - 1.0, valid], dtype=np.float32)))

    def _positions_list(self) -> list[list[float]]:
        assert self.scene is not None
        return np.asarray(self.scene.xy, dtype=np.float32).round(5).tolist()

    def _append_record(self, record: dict[str, Any]) -> None:
        self._episode_records.append(record)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = self.np_random
        positions = None
        scene = None
        last_reason = "no_valid_random_layout"
        for _ in range(self.max_layout_attempts):
            try:
                positions = self._random_positions()
                scene = self._scene_at_positions(positions)
            except ValueError as exc:
                last_reason = str(exc)
                continue
            valid, last_reason = self._layout_valid(positions, scene=scene)
            if valid:
                break
            scene = None
        if scene is None:
            raise RuntimeError(f"could not sample a connected random layout: {last_reason}")

        self._episode_index += 1
        self.scene = scene
        self.spawn_pose = None
        self._edit_count = 0
        self._valid_edits = 0
        self._invalid_edits = 0
        self._episode_return = 0.0
        self._last_valid = True
        self._episode_checkpoints = [self._positions_list()]
        self._scored = False
        info = {
            "episode_index": self._episode_index,
            "initial_positions": self._positions_list(),
            "target_count": int(self._targets.sum()),
            "plant_count": self._plant_count,
            "terrain_seed": getattr(self.terrain, "seed", None),
            "layout_valid": True,
            "fully_connected": True,
        }
        return self._observation(), info

    def step(self, action):
        if self.scene is None:
            raise RuntimeError("reset must be called before step")
        if self._edit_count >= self.edit_budget:
            raise RuntimeError("episode is complete; call reset before another step")

        self._edit_count += 1
        plant_index, new_xy, decode_error = self._decode_action(action)
        reward = 0.0
        info: dict[str, Any] = {"edit": self._edit_count, "edit_budget": self.edit_budget}
        valid = False
        reason = decode_error
        if new_xy is not None:
            proposed = np.asarray(self.scene.xy, dtype=np.float32).copy()
            proposed[plant_index] = new_xy
            candidate_scene = self._scene_at_positions(proposed)
            valid, reason = self._layout_valid(proposed, scene=candidate_scene)
            if valid:
                self.scene = candidate_scene
                self._valid_edits += 1
                self._episode_checkpoints.append(self._positions_list())
                info.update({"selected_plant": int(plant_index), "position": new_xy.tolist()})

        if not valid:
            self._invalid_edits += 1
            reward -= self.invalid_penalty
            info["invalid_reason"] = reason
            info["rolled_back"] = True
        else:
            info["rolled_back"] = False
        self._last_valid = valid

        terminated = self._edit_count >= self.edit_budget
        truncated = False
        if terminated:
            try:
                # Spawn only after editing is complete; the student remains still
                # throughout the teacher episode.
                self.spawn_pose = self.scene.sample_start(
                    self._rng,
                    max_attempts=max(10_000, int(self.cfg.size * self.cfg.size * 10)),
                )
                self.scene.spawn_pose = self.spawn_pose.copy()
                self.scene.start_pose = self.spawn_pose.copy()
            except ValueError:
                # A connected free grid should provide a continuous spawn area;
                # if numerical sampling still fails, score it as an invalid map.
                self.spawn_pose = None
                reward -= self.invalid_penalty
                self._invalid_edits += 1
                info["spawn_error"] = "Scene.sample_start could not find a free vehicle pose"

            student_score: float | None = None
            objective_reward = 0.0
            if self.spawn_pose is not None:
                student_score = float(self.evaluator(self.scene))
                if not math.isfinite(student_score):
                    raise ValueError("evaluator must return a finite student score")
                objective_reward = -student_score
                reward += objective_reward
            self._scored = True
            info.update(
                {
                    "student_score": student_score,
                    "teacher_objective_reward": objective_reward,
                    "teacher_objective": "naive_minimax",
                    "paired": False,
                    "spawn_pose": None if self.spawn_pose is None else self.spawn_pose.tolist(),
                    "positions": self._positions_list(),
                    "targets": self._targets.tolist(),
                    "terrain_seed": getattr(self.terrain, "seed", None),
                    "checkpoints": self._episode_checkpoints,
                    "valid_edits": self._valid_edits,
                    "invalid_edits": self._invalid_edits,
                }
            )

        self._episode_return += float(reward)
        info["teacher_reward"] = float(reward)
        info["episode_teacher_return"] = float(self._episode_return)
        info["edit_count"] = self._edit_count
        observation = self._observation()

        if terminated:
            record = {
                "episode_index": self._episode_index,
                "student_score": student_score,
                "teacher_objective_reward": objective_reward,
                "teacher_reward": float(self._episode_return),
                "valid_edits": self._valid_edits,
                "invalid_edits": self._invalid_edits,
                "positions": self._positions_list(),
                "targets": self._targets.tolist(),
                "spawn_pose": None if self.spawn_pose is None else self.spawn_pose.tolist(),
                "terrain_seed": getattr(self.terrain, "seed", None),
                "checkpoints": self._episode_checkpoints,
            }
            self._append_record(record)
            info["episode_summary"] = {
                key: record[key]
                for key in (
                    "episode_index",
                    "student_score",
                    "teacher_objective_reward",
                    "teacher_reward",
                    "valid_edits",
                    "invalid_edits",
                    "terrain_seed",
                )
            }
        return observation, float(reward), terminated, truncated, info

    def summary(self) -> dict[str, Any]:
        records = list(self._episode_records)
        scores = [record["student_score"] for record in records if record["student_score"] is not None]
        return {
            "algorithm": "Isaac Lab official RSL-RL OnPolicyRunner PPO with MLPModel",
            "teacher_runtime_identity": dict(TEACHER_RUNTIME_IDENTITY),
            "target_student_backend_identity": dict(TARGET_STUDENT_BACKEND_IDENTITY),
            "evaluator_kind": self.evaluator_kind,
            "student_physx_evaluated": self.student_physx_evaluated,
            "teacher_objective": "naive_minimax: teacher_reward = -student_score",
            "paired": False,
            "episodes_recorded": len(records),
            "episode_record_limit": self.max_episode_records,
            "mean_student_score": None if not scores else float(np.mean(scores)),
            "mean_teacher_reward": None
            if not records
            else float(np.mean([record["teacher_reward"] for record in records])),
            "mean_valid_edits": None
            if not records
            else float(np.mean([record["valid_edits"] for record in records])),
            "mean_invalid_edits": None
            if not records
            else float(np.mean([record["invalid_edits"] for record in records])),
            "records_retained": len(records),
        }

    def save_artifacts(self, output: str | Path, *, extra_summary: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
        """Write serializable recent episode layouts and the aggregate summary."""
        prefix = Path(output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        summary_path = Path(f"{prefix}.summary.json")
        episodes_path = Path(f"{prefix}.episodes.json")
        summary = self.summary()
        if extra_summary:
            summary.update(dict(extra_summary))
        summary["episodes_path"] = str(episodes_path)
        summary["limitation"] = (
            "The teacher uses a one-sided naive minimax reward. Training claims depend on the supplied evaluator."
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        episodes_path.write_text(
            json.dumps(list(self._episode_records), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary_path, episodes_path


def train_designer(
    env: MapDesignerEnv,
    output: str | Path,
    timesteps: int = 512,
    *,
    seed: int = 7,
    device: str = "cpu",
    ppo_kwargs: Mapping[str, Any] | None = None,
) -> DesignerTrainingResult:
    """Train the existing naive-minimax map designer with official RSL-RL PPO."""
    if not isinstance(env, MapDesignerEnv):
        raise TypeError("train_designer requires the existing MapDesignerEnv")
    if isinstance(timesteps, bool) or int(timesteps) != timesteps or int(timesteps) < 1:
        raise ValueError("timesteps must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")

    settings = {
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 3,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.0,
        "max_grad_norm": 0.5,
        "log_std_init": 0.0,
        "hidden_dims": (64, 64),
        "activation": "tanh",
        "verbose": 0,
        "target_kl": None,
    }
    if ppo_kwargs:
        unknown = set(ppo_kwargs) - set(settings)
        if unknown:
            raise ValueError(f"unsupported RSL-RL map-designer settings: {sorted(unknown)}")
        settings.update(dict(ppo_kwargs))
    if int(settings["verbose"]) != 0:
        raise ValueError("verbose is an SB3-only option; RSL-RL logging is handled by OnPolicyRunner")
    n_steps = int(settings["n_steps"])
    batch_size = int(settings["batch_size"])
    n_epochs = int(settings["n_epochs"])
    total_timesteps = int(timesteps)
    if n_steps < 1 or batch_size < 1 or n_epochs < 1:
        raise ValueError("n_steps, batch_size, and n_epochs must be positive")
    if total_timesteps % n_steps:
        raise ValueError("timesteps must be divisible by RSL-RL n_steps for exact teacher updates")
    if n_steps % batch_size:
        raise ValueError("RSL-RL rollout length must be divisible by batch_size")
    hidden_dims = tuple(int(value) for value in settings["hidden_dims"])
    if not hidden_dims or any(value < 1 for value in hidden_dims):
        raise ValueError("hidden_dims must contain positive layer widths")
    if not math.isfinite(float(settings["learning_rate"])) or float(settings["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(settings["gamma"])) or not 0 <= float(settings["gamma"]) <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not math.isfinite(float(settings["gae_lambda"])) or not 0 <= float(settings["gae_lambda"]) <= 1:
        raise ValueError("gae_lambda must be finite and in [0, 1]")

    import torch
    from rsl_rl.runners import OnPolicyRunner

    output_base = Path(output)
    checkpoint = output_base if output_base.suffix.lower() == ".pt" else Path(f"{output_base}.pt")
    summary_path = Path(f"{output_base}.summary.json")
    episodes_path = Path(f"{output_base}.episodes.json")
    for path in (checkpoint, summary_path, episodes_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite map-designer artifact: {path}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "seed": int(seed),
        "num_steps_per_env": n_steps,
        "max_iterations": total_timesteps // n_steps,
        "save_interval": total_timesteps // n_steps + 1,
        "experiment_name": "isaac_field_map_designer_rsl",
        "run_name": checkpoint.stem,
        "obs_groups": {"actor": ["obs"], "critic": ["obs"]},
        "check_for_nan": True,
        "actor": {
            "class_name": "rsl_rl.models.mlp_model:MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": str(settings["activation"]),
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": math.exp(float(settings["log_std_init"])),
                "std_type": "log",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.mlp_model:MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": str(settings["activation"]),
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo:PPO",
            "num_learning_epochs": n_epochs,
            "num_mini_batches": n_steps // batch_size,
            "clip_param": float(settings["clip_param"]),
            "gamma": float(settings["gamma"]),
            "lam": float(settings["gae_lambda"]),
            "value_loss_coef": float(settings["value_loss_coef"]),
            "entropy_coef": float(settings["entropy_coef"]),
            "learning_rate": float(settings["learning_rate"]),
            "max_grad_norm": float(settings["max_grad_norm"]),
            "schedule": "fixed",
            "desired_kl": settings["target_kl"],
            "use_clipped_value_loss": False,
        },
    }

    from rsl_rl.env import VecEnv
    from tensordict import TensorDict

    class _MapDesignerRslVecEnv(VecEnv):
        def __init__(self, source: MapDesignerEnv):
            self.env = source
            self.num_envs = 1
            self.num_actions = 3
            self.device = torch.device(device)
            self.max_episode_length = int(source.edit_budget)
            self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
            self.cfg = {"name": "MapDesignerEnv", "edit_budget": int(source.edit_budget)}
            self._observation, _ = source.reset(seed=int(seed))

        def get_observations(self):
            observation = torch.as_tensor(self._observation, dtype=torch.float32, device=self.device)
            return TensorDict({"obs": observation.unsqueeze(0)}, batch_size=[1])

        def step(self, actions):
            if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (1, self.num_actions):
                raise ValueError("RSL map-designer actions must have shape (1, 3)")
            action = actions.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
            self._observation, reward, terminated, truncated, info = self.env.step(
                np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
            )
            done = bool(terminated or truncated)
            self.episode_length_buf.add_(1)
            logs = {}
            if done:
                episode = info.get("episode_summary", {})
                for source_key, log_key in (
                    ("student_score", "/map_designer/student_score"),
                    ("teacher_reward", "/map_designer/teacher_reward"),
                    ("valid_edits", "/map_designer/valid_edits"),
                    ("invalid_edits", "/map_designer/invalid_edits"),
                ):
                    value = episode.get(source_key)
                    if value is not None:
                        logs[log_key] = torch.tensor(float(value), device=self.device)
                self._observation, _ = self.env.reset()
                self.episode_length_buf.zero_()
            extras = {
                "time_outs": torch.tensor([bool(truncated and not terminated)], device=self.device),
                "log": logs,
            }
            observation = torch.as_tensor(self._observation, dtype=torch.float32, device=self.device)
            return (
                TensorDict({"obs": observation.unsqueeze(0)}, batch_size=[1]),
                torch.tensor([float(reward)], dtype=torch.float32, device=self.device),
                torch.tensor([done], dtype=torch.long, device=self.device),
                extras,
            )

        def close(self):
            close = getattr(self.env, "close", None)
            if callable(close):
                close()

    vec_env = _MapDesignerRslVecEnv(env)
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    runner = OnPolicyRunner(vec_env, cfg, log_dir=None, device=str(device))
    started = time.perf_counter()
    runner.learn(num_learning_iterations=total_timesteps // n_steps, init_at_random_ep_len=False)
    learn_wall = time.perf_counter() - started
    actual_timesteps = n_steps * (total_timesteps // n_steps)
    model = runner.alg.get_policy()
    model.num_timesteps = actual_timesteps
    runner.save(str(checkpoint), infos={
        "teacher_runtime_identity": dict(TEACHER_RUNTIME_IDENTITY),
        "target_student_backend_identity": dict(TARGET_STUDENT_BACKEND_IDENTITY),
        "evaluator_kind": env.evaluator_kind,
        "student_physx_evaluated": env.student_physx_evaluated,
        "policy_contract": {
            "observation": "flattened_map_layout_v1",
            "action": "plant_index_x_y_clipped_minus1_plus1_v1",
            "objective": "naive_minimax_teacher_reward_equals_negative_student_score",
            "paired": False,
            "actor_hidden_dims": list(hidden_dims),
        },
    })
    summary_path, episodes_path = env.save_artifacts(
        output_base,
        extra_summary={
            "requested_timesteps": total_timesteps,
            "actual_timesteps": actual_timesteps,
            "learn_wall_seconds": float(learn_wall),
            "checkpoint": str(checkpoint),
            "scorer": getattr(env.evaluator, "__qualname__", type(env.evaluator).__name__),
            "algorithm": "Isaac Lab official RSL-RL OnPolicyRunner PPO with MLPModel",
            "teacher_runtime_identity": dict(TEACHER_RUNTIME_IDENTITY),
            "target_student_backend_identity": dict(TARGET_STUDENT_BACKEND_IDENTITY),
            "evaluator_kind": env.evaluator_kind,
            "student_physx_evaluated": env.student_physx_evaluated,
            "optimizer": "RSL-RL PPO Adam",
        },
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return DesignerTrainingResult(
        model=model,
        checkpoint=checkpoint,
        summary_path=summary_path,
        episodes_path=episodes_path,
        summary=summary,
        actual_timesteps=actual_timesteps,
    )
