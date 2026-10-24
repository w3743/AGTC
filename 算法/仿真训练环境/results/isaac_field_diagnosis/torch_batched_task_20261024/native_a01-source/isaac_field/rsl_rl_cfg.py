"""RSL-RL configs for the fast-stop-v2 absolute-goal task."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import sys

from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

from .env import FieldEnvCfg
from .feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from .policy_contract import ABSOLUTE_GOAL_ENCODER_SCHEMA, OBSERVATION_SHAPES
from .task import Config


def _fast_stop_env_values() -> tuple[Config, dict, dict]:
    """Reuse the fast-stop physics profile with the v2 reward semantics."""
    # These helpers are the source of truth for the existing fast-stop profile.
    # Import lazily so importing the task registration does not pull the training
    # protocol into the normal Isaac Field environment path.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from scripts.train_isaac_foundation_fresh import (
        FAST_STOP_CONFIG,
        PROFILE_RANDOM_ROUGH_FAST_STOP,
        _config_dict,
        _rough_profile_components,
    )

    task_values = _config_dict()
    task_values.update(FAST_STOP_CONFIG)
    task_values.update(
        max_time=60.0,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        goto_reward_profile="fast_stop_v2",
        rock_density=0.0,
        fast_stop_failure_remaining_time_cost=True,
    )
    front_guides, body_stability = _rough_profile_components(PROFILE_RANDOM_ROUGH_FAST_STOP)
    return Config(**task_values), asdict(front_guides), body_stability.as_dict()


_FAST_STOP_TASK, _FAST_STOP_FRONT_GUIDES, _FAST_STOP_BODY_STABILITY = _fast_stop_env_values()


@configclass
class FieldFastStopRslEnvCfg(FieldEnvCfg):
    """Fast-stop-v2 task: full-map absolute goal with a terminal 60 s horizon."""

    task: Config = _FAST_STOP_TASK
    mode: str = "goto"
    episode_length_s: float = 60.0
    is_finite_horizon: bool = True
    front_guide_spec: dict | None = _FAST_STOP_FRONT_GUIDES
    body_stability_config: dict | None = _FAST_STOP_BODY_STABILITY

    def __post_init__(self):
        super().__post_init__()
        # Defaults only customize an explicitly requested visualizer. Headless
        # runs still create no viewport, and every training environment stays
        # visible to its own sensors. Follow the actual first training robot.
        if self.sim.default_visualizer_cfg is None:
            self.sim.default_visualizer_cfg = KitVisualizerCfg(
                origin_type="asset", origin_env_index=0,
                origin_track_path="robot", eye=(3.0, -4.0, 2.0),
                lookat=(0.0, 0.0, 0.15), streaming_view=False,
                max_visible_envs=None,
            )


_OBSERVATION_KEYS = ["lidar", "depth", "detections", "state", "feedback"]
_OBSERVATION_SHAPES = {key: list(OBSERVATION_SHAPES[key]) for key in _OBSERVATION_KEYS}


@configclass
class FieldFastStopRslModelCfg(RslRlRNNModelCfg):
    """RSL recurrent model config carrying the field input/checkpoint schemas."""

    input_schema: str = ABSOLUTE_GOAL_INPUT_SCHEMA
    encoder_schema: str = ABSOLUTE_GOAL_ENCODER_SCHEMA
    goal_gain: float = 1.0
    cache_inference: bool = True
    observation_shapes: dict[str, list[int]] = _OBSERVATION_SHAPES


@configclass
class FieldFastStopRslRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Official RSL-RL runner config; custom model/PPO implementations are local contracts."""

    seed = 69_001
    num_steps_per_env = 128
    init_at_random_ep_len = False
    max_iterations = 384
    save_interval = 16
    experiment_name = "isaac_field_fast_stop_rsl"
    obs_groups = {"actor": list(_OBSERVATION_KEYS), "critic": list(_OBSERVATION_KEYS)}
    clip_actions = 1.0
    actor = FieldFastStopRslModelCfg(
        class_name="isaac_field.rsl_rl_policy:RslFieldRNNModel",
        hidden_dims=[64],
        activation="tanh",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="rsl_rl.modules.distribution.GaussianDistribution",
            init_std=0.05,
            std_type="log",
        ),
        rnn_type="lstm",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        encoder_schema=ABSOLUTE_GOAL_ENCODER_SCHEMA,
        goal_gain=1.0,
        cache_inference=True,
        observation_shapes=_OBSERVATION_SHAPES,
    )
    critic = FieldFastStopRslModelCfg(
        class_name="isaac_field.rsl_rl_policy:RslFieldRNNModel",
        hidden_dims=[64],
        activation="tanh",
        obs_normalization=False,
        distribution_cfg=None,
        rnn_type="lstm",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        encoder_schema=ABSOLUTE_GOAL_ENCODER_SCHEMA,
        goal_gain=1.0,
        cache_inference=True,
        observation_shapes=_OBSERVATION_SHAPES,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="isaac_field.rsl_rl_policy:RslFieldPPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=1.0,
        lam=0.99,
        desired_kl=0.01,
        max_grad_norm=0.5,
        share_cnn_encoders=True,
    )
