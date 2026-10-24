"""CPU checks for the shared, SB3-independent sensor encoder boundary."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from isaac_field.policy_contract import OBSERVATION_SHAPES
from isaac_field.sensor_encoder import SensorEncoderCore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            key: gym.spaces.Box(-1.0, 1.0, shape=shape, dtype=float)
            for key, shape in OBSERVATION_SHAPES.items()
        }
    )


def _observation(batch: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(4021)
    values = {
        key: torch.rand(batch, *shape, dtype=torch.float32)
        for key, shape in OBSERVATION_SHAPES.items()
    }
    values["detections"][:, :5, 2:4] = 0.0
    return values


def test_torch_core_loads_legacy_encoder_tensors_and_matches_outputs():
    pytest.importorskip(
        "stable_baselines3",
        reason="the compatibility side of this encoder test requires the optional legacy SB3 extra",
    )
    from isaac_field.policy import SensorEncoder as LegacySensorEncoder

    space = _observation_space()
    legacy = LegacySensorEncoder(space, cache_inference=False).cpu().eval()
    core = SensorEncoderCore(space, cache_inference=False).cpu().eval()

    legacy_keys = tuple(legacy.state_dict())
    assert tuple(core.state_dict()) == legacy_keys
    expected_tensor_shapes = {
        "laser.0.weight": (16, 1, 5),
        "laser.0.bias": (16,),
        "laser.2.weight": (32, 16, 5),
        "laser.2.bias": (32,),
        "laser.6.weight": (32, 1600),
        "laser.6.bias": (32,),
        "depth.0.weight": (8, 1, 3, 3),
        "depth.0.bias": (8,),
        "depth.2.weight": (16, 8, 3, 3),
        "depth.2.bias": (16,),
        "depth.6.weight": (64, 768),
        "depth.6.bias": (64,),
        "boxes.0.weight": (32, 8),
        "boxes.0.bias": (32,),
        "boxes.2.weight": (32, 32),
        "boxes.2.bias": (32,),
        "proprioception.0.weight": (32, 33),
        "proprioception.0.bias": (32,),
        "feedback.0.weight": (32, 37),
        "feedback.0.bias": (32,),
    }
    assert {key: tuple(value.shape) for key, value in legacy.state_dict().items()} == expected_tensor_shapes
    core.load_state_dict(legacy.state_dict(), strict=True)
    assert all(tuple(core.state_dict()[key].shape) == tuple(legacy.state_dict()[key].shape) for key in legacy_keys)

    observations = _observation()
    with torch.no_grad():
        old_path_features = legacy(observations)
        core_features = core(observations)
    torch.testing.assert_close(core_features, old_path_features, rtol=0, atol=0)
    assert core_features.shape == (2, 192)


def test_rsl_field_policy_imports_constructs_and_predicts_with_sb3_blocked():
    code = r'''
import builtins
import importlib
import math
import numpy as np
import torch
import gymnasium as gym
from tensordict import TensorDict

original_import = builtins.__import__
def block_sb3(name, *args, **kwargs):
    if name == "stable_baselines3" or name.startswith("stable_baselines3."):
        raise AssertionError("Stable-Baselines3 was imported by the RSL core")
    if name == "sb3_contrib" or name.startswith("sb3_contrib."):
        raise AssertionError("SB3-Contrib was imported by the RSL core")
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_sb3

from isaac_field.policy_contract import OBSERVATION_SHAPES, validate_policy_spaces
from isaac_field.rsl_rl_policy import RslFieldRNNModel

for module_name in (
    "isaac_field.viewer",
    "isaac_field.visualization.inspection",
    "isaac_field.visualization.network_panel",
    "scripts.lassc_consumer",
    "scripts.lassc_replay",
    "scripts.view_fast_stop_goal",
    "scripts.evaluate_fast_stop_mastery",
    "isaac_field.__main__",
):
    importlib.import_module(module_name)

torch.set_num_threads(1)
observation_space = gym.spaces.Dict({
    key: gym.spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)
    for key, shape in OBSERVATION_SHAPES.items()
})
action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
validate_policy_spaces(observation_space, action_space)
batch = 2
observations = TensorDict(
    {key: torch.zeros(batch, *shape, dtype=torch.float32) for key, shape in OBSERVATION_SHAPES.items()},
    batch_size=[batch],
    device="cpu",
)
model = RslFieldRNNModel(
    obs=observations,
    obs_groups={"actor": list(OBSERVATION_SHAPES), "critic": list(OBSERVATION_SHAPES)},
    obs_set="actor",
    output_dim=4,
    distribution_cfg={
        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
        "init_std": math.exp(-2),
        "std_type": "log",
        "std_range": [1e-6, 1e6],
    },
)
with torch.no_grad():
    action = model(observations, stochastic_output=False)
assert tuple(action.shape) == (batch, 4), action.shape
assert torch.isfinite(action).all()
'''
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["RUN_ISAAC_PHYSICS"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
