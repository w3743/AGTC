from __future__ import annotations

import json
from threading import Lock
from types import MappingProxyType, SimpleNamespace
import zipfile

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from isaac_field.action_contract import action_config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA
from isaac_field.policy import (
    ENCODER_SCHEMA,
    PolicyRunner,
    _field_encoder_config,
    make_model,
)
from isaac_field.task import state_fields_for_schema
from isaac_field.viewer import FieldViewerControls
from isaac_field.visualization.activation_maps import STATE_LABELS
from isaac_field.visualization.inspection import PolicyInspector
from isaac_field.visualization.network_2d import Network2DServer, _fields_for


def _observation_config(input_schema: str) -> dict:
    config = {
        "schema": input_schema,
        "state_dim": 33,
        "state_fields": list(state_fields_for_schema(input_schema)),
        "observation_shapes": {
            "lidar": [1, 400],
            "depth": [1, 160, 88],
            "detections": [16, 8],
            "state": [33],
            "feedback": [37],
        },
        "feedback": {"input_schema": input_schema},
    }
    if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        config["absolute_goal_coordinates"] = {
            "source": "task.goal",
            "coordinate_frame": "field_world_xy",
            "origin_xy_m": [0.0, 0.0],
            "unit": "m",
            "field_size_m": 100.0,
            "normalization": "world_xy / field_size_m * 2 - 1",
            "state_clip": [-1.0, 1.0],
        }
    return config


class _ContractEnv(gym.Env):
    def __init__(self, input_schema: str):
        self.observation_config = _observation_config(input_schema)
        self.observation_space = gym.spaces.Dict(
            {
                "lidar": gym.spaces.Box(0.0, 1.0, shape=(1, 400), dtype=np.float32),
                "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 160, 88), dtype=np.float32),
                "detections": gym.spaces.Box(-1.0, 1.0, shape=(16, 8), dtype=np.float32),
                "state": gym.spaces.Box(-1.0, 1.0, shape=(33,), dtype=np.float32),
                "feedback": gym.spaces.Box(-np.inf, np.inf, shape=(37,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _observation(self):
        return {
            "lidar": np.zeros((1, 400), dtype=np.float32),
            "depth": np.zeros((1, 160, 88), dtype=np.float32),
            "detections": np.zeros((16, 8), dtype=np.float32),
            "state": np.zeros((33,), dtype=np.float32),
            "feedback": np.zeros((37,), dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, False, False, {}


def _controls(adapter) -> FieldViewerControls:
    controls = object.__new__(FieldViewerControls)
    controls.adapter = adapter
    controls._network_panel = None
    controls._runner = None
    controls._checkpoint = None
    return controls


def test_viewer_load_passes_absolute_environment_contract_and_keeps_legacy_call_shape(monkeypatch):
    import isaac_field.policy as policy_module

    calls = []

    class _MockRunner:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.model = SimpleNamespace(num_timesteps=1)

        def reset(self):
            pass

    monkeypatch.setattr(policy_module, "PolicyRunner", _MockRunner)
    absolute_config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    absolute_adapter = SimpleNamespace(
        observation_config=None,
        engine=SimpleNamespace(observation_config=absolute_config),
    )
    _controls(absolute_adapter)._load_checkpoint("absolute.zip", select_policy=False)

    assert calls[-1] == (
        (absolute_adapter, "absolute.zip"),
        {
            "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
            "expected_observation_config": absolute_config,
        },
    )

    legacy_config = _observation_config(INPUT_SCHEMA)
    legacy_adapter = SimpleNamespace(observation_config=legacy_config)
    _controls(legacy_adapter)._load_checkpoint("legacy.zip", select_policy=False)
    assert calls[-1] == ((legacy_adapter, "legacy.zip"), {})


def test_viewer_rejects_v6_checkpoint_before_sb3_load_for_absolute_environment(tmp_path, monkeypatch):
    from sb3_contrib import RecurrentPPO

    checkpoint = tmp_path / "legacy.zip"
    metadata = {
        "field_action_config": action_config(),
        "policy_kwargs": {
            "features_extractor_kwargs": {"encoder_schema": ENCODER_SCHEMA},
        },
        "field_encoder_config": _field_encoder_config(INPUT_SCHEMA),
        "field_observation_config": _observation_config(INPUT_SCHEMA),
    }
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps(metadata))

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("SB3 load must not run before the absolute-goal schema check")

    monkeypatch.setattr(RecurrentPPO, "load", classmethod(unexpected_load))
    adapter = _ContractEnv(ABSOLUTE_GOAL_INPUT_SCHEMA)
    controls = _controls(adapter)

    with pytest.raises(ValueError, match="encoder schema is missing or incompatible"):
        controls._load_checkpoint(str(checkpoint), select_policy=False)
    assert controls._runner is None


def _offline_network_server() -> Network2DServer:
    """Build the response owner without opening a socket or starting a service."""
    server = object.__new__(Network2DServer)
    server._lock = Lock()
    server._cache_lock = Lock()
    server._revision = 0
    server._frame = None
    server._parameters = MappingProxyType({})
    server._parameter_max_abs = 0.0
    server._adaptive_scale = 1.0
    server._has_frame_token = False
    server._last_frame_token = None
    server._last_sequence_number = None
    server._last_sim_time_number = None
    server._image_cache = {}
    server._network_layout = SimpleNamespace(get=lambda _nodes: {})
    return server


def test_absolute_snapshot_contract_selects_absolute_goal_labels_in_detail_fields():
    env = DummyVecEnv([lambda: _ContractEnv(ABSOLUTE_GOAL_INPUT_SCHEMA)])
    try:
        config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
        runner = PolicyRunner(
            env,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
            expected_observation_config=config,
            device="cpu",
        )
        inspector = PolicyInspector(runner)
        inspector.preview(env.reset(), 0.0)

        assert inspector.snapshot["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
        server = _offline_network_server()
        server.publish(
            inspector.snapshot,
            [
                {
                    "key": "input.state",
                    "title": "状态输入",
                    "description": "33维状态",
                    "branch": "proprioception",
                    "column": 0.0,
                    "parents": (),
                    "summary": False,
                    "source": "actual forward",
                    "values": np.zeros(33, dtype=np.float32),
                }
            ],
        )
        response = server._make_frame_response(
            {"detail": ["1"], "selected": ["input.state"]}
        )

        fields = response["detail"]["fields"]
        assert fields[1:3] == [
            "目标绝对世界 X 坐标（场地坐标映射到 [-1, 1]；仅到点模式）",
            "目标绝对世界 Y 坐标（场地坐标映射到 [-1, 1]；仅到点模式）",
        ]
        assert len(fields) == 33
        assert _fields_for("input.state") == STATE_LABELS
    finally:
        env.close()
