"""Backend-neutral consumer tests for the production RSL runner boundary."""
from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.task import Config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from scripts.lassc_consumer import LASSCConsumer, SourcePose
from scripts.lassc_replay import (
    SYNTHETIC_CLOCK,
    SYNTHETIC_SESSION,
    build_depth_fixture,
    make_qa_adapter,
)


class FakeRslRunner:
    """Duck-typed RslPolicyRunner: NumPy LSTM state and no SB3 dependency."""

    def __init__(self):
        self.state = None
        self.episode_start = np.ones(1, dtype=bool)
        self.actions = 0
        self.resets = 0
        self.return_invalid_action = False

    def reset(self, done=None):
        assert done is None
        self.resets += 1
        self.state = None
        self.episode_start = np.ones(1, dtype=bool)

    def action(self, observation):
        assert observation == {"sensor": "prepared"}
        self.actions += 1
        value = np.full((1, 1, 3), self.actions, dtype=np.float32)
        self.state = (value, value.copy())
        self.episode_start = np.zeros(1, dtype=bool)
        action = np.asarray([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)
        if self.return_invalid_action:
            action[0, 0] = 1.1
        return action


class MetadataRslRunner(FakeRslRunner):
    """Runner carrying the same checkpoint-bound metadata as production RSL."""

    def __init__(self, task_cfg):
        super().__init__()
        from policy_contract_helpers import observation_config as _observation_config

        self.input_schema = task_cfg.input_schema
        self.observation_config = _observation_config(task_cfg.input_schema)
        if task_cfg.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
            self.observation_config["absolute_goal_coordinates"]["field_size_m"] = task_cfg.size
        self.model = SimpleNamespace(contract={
            "backend": "rsl_rl",
            "input_schema": task_cfg.input_schema,
            "observation_config": self.observation_config,
            "task_config": asdict(task_cfg),
        })


def _frame(*, seq: int, epoch: int, source_ns: int):
    metadata, payload, _raster, depth_module, _wire = build_depth_fixture(
        "constant", seq=seq, pipeline_epoch=epoch, source_mono_ns=source_ns,
    )
    pose = SourcePose(
        SYNTHETIC_SESSION, epoch, SYNTHETIC_CLOCK, source_ns,
        (2.0, 2.0, 0.0, 0.1, 0.2), True,
    )
    adapter = make_qa_adapter(depth_module)
    return metadata, payload, pose, adapter


def test_consumer_accepts_runner_protocol_and_rolls_back_failed_prediction(monkeypatch):
    source_ns = 100_750_000_000
    metadata, payload, pose, adapter = _frame(seq=1, epoch=0, source_ns=source_ns)
    runner = FakeRslRunner()
    consumer = LASSCConsumer(
        adapter, cfg=Config(), max_depth_age_s=0.5, max_other_age_s=0.5,
        runner=runner,
    )
    monkeypatch.setattr(
        consumer, "_observation", lambda *_args: {"sensor": "prepared"},
    )

    result = consumer.consume(
        metadata, payload, context=None, source_pose=pose,
        consume_mono_ns=source_ns + 250_000_000,
        consume_clock_id=SYNTHETIC_CLOCK,
    )
    assert result.ready and result.action is not None
    np.testing.assert_allclose(result.action, [0.1, -0.2, 0.3, -0.4])
    assert consumer.accepted == 1
    assert runner.actions == 1
    assert runner.state is not None
    assert not runner.episode_start.any()

    # Switching stream resets the same injected recurrent runner before action.
    next_ns = source_ns + 500_000_000
    metadata2, payload2, pose2, _ = _frame(seq=1, epoch=1, source_ns=next_ns)
    reset_result = consumer.consume(
        metadata2, payload2, context=None, source_pose=pose2,
        consume_mono_ns=next_ns + 250_000_000,
        consume_clock_id=SYNTHETIC_CLOCK,
    )
    assert reset_result.ready and reset_result.reset_recurrent
    assert runner.resets == 1
    assert runner.actions == 2

    state_before = tuple(value.copy() for value in runner.state)
    starts_before = runner.episode_start.copy()
    accepted_before = consumer.accepted
    runner.return_invalid_action = True
    next_ns += 500_000_000
    metadata3, payload3, pose3, _ = _frame(seq=2, epoch=1, source_ns=next_ns)
    with pytest.raises(ValueError, match="policy action must be four finite values"):
        consumer.consume(
            metadata3, payload3, context=None, source_pose=pose3,
            consume_mono_ns=next_ns + 250_000_000,
            consume_clock_id=SYNTHETIC_CLOCK,
        )
    assert consumer.accepted == accepted_before
    np.testing.assert_array_equal(runner.episode_start, starts_before)
    for actual, expected in zip(runner.state, state_before):
        np.testing.assert_array_equal(actual, expected)


def test_consumer_rejects_checkpoint_schema_mismatch():
    source_ns = 100_750_000_000
    _metadata, _payload, _pose, adapter = _frame(seq=1, epoch=0, source_ns=source_ns)
    runner = MetadataRslRunner(Config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA))
    with pytest.raises(ValueError, match="input_schema"):
        LASSCConsumer(
            adapter, cfg=Config(), max_depth_age_s=0.5, max_other_age_s=0.5,
            runner=runner,
        )


def test_consumer_rejects_recorded_normalization_mismatch():
    source_ns = 100_750_000_000
    _metadata, _payload, _pose, adapter = _frame(seq=1, epoch=0, source_ns=source_ns)
    checkpoint_cfg = Config(
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        depth_range=4.0,
    )
    runner = MetadataRslRunner(checkpoint_cfg)
    with pytest.raises(ValueError, match="normalization"):
        LASSCConsumer(
            adapter, cfg=Config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA),
            max_depth_age_s=0.5, max_other_age_s=0.5, runner=runner,
        )


def test_consumer_rejects_runner_sensor_shape_mismatch():
    source_ns = 100_750_000_000
    _metadata, _payload, _pose, adapter = _frame(seq=1, epoch=0, source_ns=source_ns)
    runner = MetadataRslRunner(Config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA))
    with pytest.raises(ValueError, match="sensor dimensions"):
        LASSCConsumer(
            adapter, cfg=Config(
                input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                lidar_rays=300,
            ),
            max_depth_age_s=0.5, max_other_age_s=0.5, runner=runner,
        )


def test_production_scripts_have_no_module_level_sb3_policy_import():
    for relative in ("scripts/lassc_consumer.py", "scripts/lassc_replay.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        module_imports = [
            node.module for node in tree.body
            if isinstance(node, ast.ImportFrom)
        ]
        assert "isaac_field.policy" not in module_imports
