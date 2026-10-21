"""RSL-RL configs for the fast-stop absolute-goal task."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import sys

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

from .env import FieldEnvCfg
from .feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from .policy import ABSOLUTE_GOAL_ENCODER_SCHEMA, OBSERVATION_SHAPES
from .task import Config


def _fast_stop_env_values() -> tuple[Config, dict, dict]:
    """Reuse the project's frozen fast-stop baseline and physics components."""
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
        rock_density=0.0,
    )
    front_guides, body_stability = _rough_profile_components(PROFILE_RANDOM_ROUGH_FAST_STOP)
    return Config(**task_values), asdict(front_guides), body_stability.as_dict()


_FAST_STOP_TASK, _FAST_STOP_FRONT_GUIDES, _FAST_STOP_BODY_STABILITY = _fast_stop_env_values()


@configclass
class FieldFastStopRslEnvCfg(FieldEnvCfg):
    """Fast-stop task config: full-map baseline, absolute goal, terminal 60 s horizon."""

    task: Config = _FAST_STOP_TASK
    mode: str = "goto"
    episode_length_s: float = 60.0
    is_finite_horizon: bool = True
    front_guide_spec: dict | None = _FAST_STOP_FRONT_GUIDES
    body_stability_config: dict | None = _FAST_STOP_BODY_STABILITY


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
            init_std=math.exp(-2),
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
        schedule="fixed",
        gamma=1.0,
        lam=0.95,
        desired_kl=0.03,
        max_grad_norm=0.5,
        share_cnn_encoders=True,
    )
