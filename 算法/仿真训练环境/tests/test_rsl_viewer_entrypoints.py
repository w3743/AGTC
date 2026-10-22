"""CPU checks for native RSL checkpoints through the existing viewer entrypoints."""
from __future__ import annotations

import copy
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaac_field.action_contract import action_config
from isaac_field.assets import FrontGuideSpec
from isaac_field.body_stability import BodyStabilityConfig
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from isaac_field.policy import OBSERVATION_SHAPES
from isaac_field.rsl_rl_policy import RslFieldRNNModel, build_field_policy_contract
from isaac_field.task import Config
from isaac_field.viewer import FieldViewerControls
import isaac_field.rsl_rl_inference as rsl_inference
import isaac_field.policy as sb3_policy
import isaac_field.viewer as viewer_module
import scripts.view_fast_stop_goal as goal_viewer
from test_absolute_goal_policy import _observation_config


def _native_checkpoint(path):
    torch.set_num_threads(1)
    torch.manual_seed(1822)
    observation_config = _observation_config(ABSOLUTE_GOAL_INPUT_SCHEMA)
    observations = TensorDict(
        {key: torch.zeros((1, *shape), dtype=torch.float32)
         for key, shape in OBSERVATION_SHAPES.items()},
        batch_size=[1],
    )
    groups = {key: list(observations.keys()) for key in ("actor", "critic")}
    actor = RslFieldRNNModel(
        observations,
        groups,
        "actor",
        4,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": float(np.exp(-2.0)),
            "std_type": "log",
            "std_range": [1e-6, 1e6],
        },
    )
    critic = RslFieldRNNModel(
        observations,
        groups,
        "critic",
        1,
        cnns=actor.cnns,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    env = SimpleNamespace(
        num_envs=1,
        observation_config=observation_config,
        action_config=action_config(),
        drive_config={"interface": "fixture-track-speed", "velocity_gain": 0.04},
        damage_config={"metric": "stem-contact-v2", "plants": 0},
        field_cfg=Config(
            plants=0, terrain_enabled=False, rock_density=0.0,
            input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        ),
        front_guide_spec=FrontGuideSpec(x_m=0.43, radius_m=0.047),
        body_stability_config=BodyStabilityConfig(tilt_weight=3.0),
    )
    contract = build_field_policy_contract(
        actor, critic, obs_groups=groups, env=env, num_steps_per_env=128
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "field_policy_contract": contract,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "num_timesteps": 128,
        "iter": 1,
    }, path)
    return path, observation_config, contract


@pytest.fixture
def checkpoint(tmp_path):
    return _native_checkpoint(tmp_path / "native" / "model.pt")


def _viewer_controls(adapter):
    controls = object.__new__(FieldViewerControls)
    controls.adapter = adapter
    controls._network_panel = None
    controls._runner = None
    controls._checkpoint = None
    return controls


def test_viewer_routes_native_pt_to_rsl_and_zip_to_explicit_legacy(monkeypatch, checkpoint):
    path, observation_config, _contract = checkpoint
    calls = []

    class FakeRslRunner:
        def __init__(self, env, selected, **kwargs):
            calls.append(("rsl", env, selected, kwargs))
            self.model = SimpleNamespace(backend="rsl_rl", num_timesteps=128)

        def reset(self):
            calls.append(("reset-rsl",))

    class FakeSb3Runner:
        def __init__(self, env, selected, **kwargs):
            calls.append(("sb3", env, selected, kwargs))
            self.model = SimpleNamespace(num_timesteps=2048)

        def reset(self):
            calls.append(("reset-sb3",))

    monkeypatch.setattr(rsl_inference, "RslPolicyRunner", FakeRslRunner)
    monkeypatch.setattr(sb3_policy, "PolicyRunner", FakeSb3Runner)
    adapter = SimpleNamespace(observation_config=observation_config)

    controls = _viewer_controls(adapter)
    controls._load_checkpoint(str(path), select_policy=False)
    assert calls[0] == (
        "rsl", adapter, path,
        {
            "device": "cpu",
            "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
            "expected_observation_config": observation_config,
        },
    )
    assert "RSL-RL原生 .pt" in controls._status

    old = _viewer_controls(adapter)
    old._load_checkpoint("legacy-policy.zip", select_policy=False)
    assert calls[-2] == ("sb3", adapter, "legacy-policy.zip", {
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "expected_observation_config": observation_config,
    })
    assert "Legacy SB3 .zip" in old._status


def test_rsl_model_identity_uses_real_contract_without_sb3_policy():
    model = SimpleNamespace(
        backend="rsl_rl",
        num_timesteps=128,
        contract={
            "format": "agtc-field-policy-v1",
            "version": 1,
            "actor_config": {"rnn_hidden_dim": 512, "rnn_num_layers": 1},
            "critic_config": {"rnn_hidden_dim": 512, "rnn_num_layers": 1},
        },
    )
    content = goal_viewer.FastStopGoalControls._model_identity_content(
        "model.pt", SimpleNamespace(model=model), {"selection": "rsl-checkpoint"}
    )
    assert "RSL-RL原生 field_policy_contract" in content
    assert "actor 512×1；critic 512×1" in content
    assert "agtc-field-policy-v1" in content
    assert "迁移 smoke（≤512步），未完成控制能力验收" in content
    assert "Legacy SB3" not in content


@pytest.mark.parametrize(
    ("steps", "expected_status", "unexpected_status"),
    (
        (0, "未训练（0步）", "迁移 smoke"),
        (1, "迁移 smoke（≤512步），未完成控制能力验收", "已记录检查点；能力以独立评估为准"),
        (512, "迁移 smoke（≤512步），未完成控制能力验收", "已记录检查点；能力以独立评估为准"),
        (513, "已记录检查点；能力以独立评估为准", "迁移 smoke"),
        (None, "训练步数未记录；能力以独立评估为准", "迁移 smoke"),
    ),
)
def test_model_identity_labels_smoke_maturity_and_unknown_steps(
    steps, expected_status, unexpected_status
):
    model = SimpleNamespace(
        backend="rsl_rl",
        num_timesteps=steps,
        contract={
            "actor_config": {"rnn_hidden_dim": 512, "rnn_num_layers": 1},
            "critic_config": {"rnn_hidden_dim": 512, "rnn_num_layers": 1},
        },
    )
    content = goal_viewer.FastStopGoalControls._model_identity_content(
        "model.pt", SimpleNamespace(model=model), {"selection": "rsl-checkpoint"}
    )
    assert f"**模型状态：** {expected_status}" in content
    assert unexpected_status not in content


@pytest.mark.parametrize(
    ("steps", "expected_status"),
    (
        (0, "未训练（0步）；未完成控制能力验收"),
        (1, "已记录检查点；能力以独立评估为准"),
        (512, "已记录检查点；能力以独立评估为准"),
        (513, "已记录检查点；能力以独立评估为准"),
        (None, "训练步数未记录；能力以独立评估为准"),
    ),
)
def test_legacy_sb3_step_count_never_gets_rsl_migration_smoke_label(steps, expected_status):
    model = SimpleNamespace(num_timesteps=steps)
    content = goal_viewer.FastStopGoalControls._model_identity_content(
        "legacy.zip", SimpleNamespace(model=model), {"selection": "legacy-checkpoint"}
    )

    assert "Legacy SB3 ZIP" in content
    assert f"**模型状态：** {expected_status}" in content
    assert "迁移 smoke（≤512步）" not in content


def test_check_loads_and_validates_native_rsl_on_cpu_without_ports_or_output(
    checkpoint, tmp_path, monkeypatch, capsys
):
    path, _observation_config, contract = checkpoint
    output = tmp_path / "not-created"

    def no_ports(_port):
        pytest.fail("CPU --check must not probe/bind viewer ports")

    monkeypatch.setattr(goal_viewer, "_require_viewer_ports", no_ports)
    result = goal_viewer.main(["--check", "--checkpoint", str(path), "--output", str(output)])
    assert result == 0
    response = json.loads(capsys.readouterr().out)
    assert response["backend"] == "rsl_rl"
    assert response["launch_allowed"] is False
    assert response["checkpoint"]["sha256"] == goal_viewer._sha256(path)
    assert response["checkpoint"]["num_timesteps"] == 128
    assert response["checkpoint"]["contract_sha256"] == goal_viewer._stable_hash(contract)
    assert response["checkpoint"]["field_policy_contract"]["task_config"]["plants"] == 0
    assert not output.exists()


def test_old_512_rsl_checkpoint_reconstructs_missing_failure_cost_on_cpu(checkpoint, tmp_path):
    path, _observation_config, contract = checkpoint
    assert contract["lstm"]["hidden_dim"] == 512

    from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload
    from scripts.lassc_replay import _policy_env, _rsl_config_from_contract

    payload = load_rsl_checkpoint_payload(path, map_location="cpu")
    old_contract = copy.deepcopy(payload["field_policy_contract"])
    old_contract["task_config"].pop("fast_stop_failure_remaining_time_cost")
    old_path = tmp_path / "legacy_512_without_failure_cost.pt"
    payload["field_policy_contract"] = old_contract
    torch.save(payload, old_path)

    loaded = load_rsl_checkpoint_payload(old_path, map_location="cpu")
    loaded_contract = loaded["field_policy_contract"]
    assert "fast_stop_failure_remaining_time_cost" not in loaded_contract["task_config"]

    record = goal_viewer._rsl_checkpoint_record(old_path)
    viewer_cfg, _guide, _stability = goal_viewer._rsl_environment_configs(
        record["field_policy_contract"]
    )
    replay_cfg = _rsl_config_from_contract(loaded_contract)
    assert viewer_cfg.fast_stop_failure_remaining_time_cost is False
    assert replay_cfg.fast_stop_failure_remaining_time_cost is False
    assert loaded_contract["task_config"] == old_contract["task_config"]

    replay_env = _policy_env(old_path)
    try:
        assert replay_env.field_policy_contract["task_config"] == old_contract["task_config"]
        assert "fast_stop_failure_remaining_time_cost" not in replay_env.task_config
    finally:
        replay_env.close()

    explicit_true = copy.deepcopy(loaded_contract)
    explicit_true["task_config"]["fast_stop_failure_remaining_time_cost"] = True
    assert goal_viewer._rsl_environment_configs(explicit_true)[0].fast_stop_failure_remaining_time_cost is True
    assert _rsl_config_from_contract(explicit_true).fast_stop_failure_remaining_time_cost is True

    for bad_config in (
        {key: value for key, value in old_contract["task_config"].items() if key != "max_time"},
        dict(old_contract["task_config"], unexpected_setting=1),
    ):
        invalid = copy.deepcopy(loaded_contract)
        invalid["task_config"] = bad_config
        with pytest.raises(ValueError, match="task_config|complete"):
            goal_viewer._rsl_environment_configs({
                **invalid,
                "front_guide_spec": loaded_contract["front_guide_spec"],
                "body_stability_config": loaded_contract["body_stability_config"],
            })
        with pytest.raises(ValueError, match="task_config|complete"):
            _rsl_config_from_contract(invalid)


def test_rsl_launch_dispatches_without_legacy_trial_gate(checkpoint, tmp_path, monkeypatch, capsys):
    path, _observation_config, _contract = checkpoint
    output = tmp_path / "paused-preview"
    observed = {}
    monkeypatch.setattr(goal_viewer, "_require_viewer_ports", lambda port: observed.setdefault("port", port))
    monkeypatch.setattr(
        goal_viewer,
        "_launch_rsl_viewer",
        lambda out, port, record: observed.update(output=out, launch_port=port, checkpoint=record),
    )
    assert goal_viewer.main([
        "--launch", "--checkpoint", str(path), "--output", str(output), "--port", "8791"
    ]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["backend"] == "rsl_rl"
    assert observed["port"] == observed["launch_port"] == 8791
    assert observed["checkpoint"]["sha256"] == goal_viewer._sha256(path)
    assert output.parent.is_dir() and not output.exists()


def test_check_requires_explicit_pt_and_zip_requires_legacy_switch(tmp_path, capsys):
    assert goal_viewer.main(["--check"]) == 2
    assert "必须显式传入真实检查点" in capsys.readouterr().err

    legacy_zip = tmp_path / "old.zip"
    legacy_zip.write_bytes(b"not an SB3 model")
    assert goal_viewer.main(["--check", "--checkpoint", str(legacy_zip)]) == 2
    assert "显式添加 --legacy-sb3" in capsys.readouterr().err


def test_contract_reconstruction_requires_all_physics_metadata(checkpoint):
    _path, _observation_config, contract = checkpoint
    cfg, guide, stability = goal_viewer._rsl_environment_configs(contract)
    assert cfg.plants == 0 and cfg.terrain_enabled is False
    assert isinstance(guide, FrontGuideSpec) and guide.x_m == pytest.approx(0.43)
    assert isinstance(stability, BodyStabilityConfig) and stability.tilt_weight == 3.0
    for missing in ("task_config", "front_guide_spec", "body_stability_config"):
        incomplete = dict(contract)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match="缺少可复现环境字段"):
            goal_viewer._rsl_environment_configs(incomplete)


def test_launch_builds_existing_nested_adapter_and_records_paused_smoke(
    checkpoint, tmp_path, monkeypatch
):
    path, _observation_config, contract = checkpoint
    checkpoint_record = goal_viewer._rsl_checkpoint_record(path)
    output = tmp_path / "launch-output"
    observed = {}

    class FakeTask:
        time = 0.0
        _hold = 0.0

    class FakeAdapter:
        def __init__(self, engine):
            self.engine = engine
            self.task = FakeTask()
            self.steps = 0

        def reset(self, *, seed, options):
            observed["reset"] = (seed, options)

        def state_payload(self):
            return {"robot": {"x": 0.0, "y": 0.0}, "goal": [1.0, 1.0]}

        def close(self):
            observed["closed"] = True

    def make_env(**kwargs):
        observed["make_env"] = kwargs
        return SimpleNamespace(
            sim=SimpleNamespace(visualizers=[
                SimpleNamespace(cfg=SimpleNamespace(visualizer_type="viser"))
            ]),
            drive_config=contract["drive_config"],
            observation_config=contract["observation_config"],
            action_config=contract["action_config"],
            damage_config=contract["damage_config"],
        )

    env_module = types.ModuleType("isaac_field.env")
    env_module.make_env = make_env
    env_module.SingleFieldAdapter = FakeAdapter
    monkeypatch.setitem(sys.modules, "isaac_field.env", env_module)

    class FakeControls:
        def __init__(self, *_args, **kwargs):
            metadata = kwargs["checkpoint_metadata"]
            self._runner = SimpleNamespace(model=SimpleNamespace(
                backend="rsl_rl",
                checkpoint_sha256=metadata["sha256"],
                contract=contract,
            ))
            self._running = False
            self._control_mode = "policy"
            self.policy_loaded = True

        def process_pending(self):
            pass

    monkeypatch.setattr(goal_viewer, "FastStopGoalControls", FakeControls)
    original_run = viewer_module.run_viewer

    def run_viewer(adapter, visualizer, selected, *, reset_seed):
        observed["run_viewer"] = (visualizer, selected, reset_seed)
        attached = viewer_module.attach_field_controls(visualizer, adapter, selected, reset_seed=reset_seed)
        assert attached.policy_loaded and not attached._running

    monkeypatch.setattr(viewer_module, "run_viewer", run_viewer)
    try:
        goal_viewer._launch_rsl_viewer(output, 8783, checkpoint_record)
    finally:
        monkeypatch.setattr(viewer_module, "run_viewer", original_run)

    configured = observed["make_env"]
    assert configured["cfg"].plants == 0
    assert configured["num_envs"] == 1 and configured["auto_reset"] is False
    assert isinstance(configured["front_guide_spec"], FrontGuideSpec)
    assert isinstance(configured["body_stability_config"], BodyStabilityConfig)
    assert observed["reset"] == (90003, {"mode": "goto"})
    assert observed["closed"] is True
    assert output.exists()
    record = json.loads((output / "initial_state.json").read_text(encoding="utf-8"))
    assert record["backend"] == "rsl_rl"
    assert record["checkpoint_sha256"] == goal_viewer._sha256(path)
    assert record["num_timesteps"] == 128
    assert record["field_policy_contract"] == contract
    assert record["loaded"] is True and record["paused"] is True
    assert record["smoke_max_steps"] == 512
    assert record["smoke_only"] is True and record["control_acceptance"] is False


def test_initial_record_labels_only_real_short_checkpoints_as_smoke(checkpoint, tmp_path):
    path, _observation_config, contract = checkpoint
    base_record = goal_viewer._rsl_checkpoint_record(path)

    class FakeAdapter:
        task = SimpleNamespace(time=0.0, _hold=0.0)

        def state_payload(self):
            return {"time": 0.0}

    class FakeControls:
        policy_loaded = True
        _running = False

        def __init__(self, sha):
            self._runner = SimpleNamespace(model=SimpleNamespace(
                backend="rsl_rl", checkpoint_sha256=sha, contract=contract,
            ))

    expected = {
        128: ("smoke_checkpoint", 512, True),
        512: ("smoke_checkpoint", 512, True),
        513: ("recorded_checkpoint", None, False),
        0: ("unrecorded", None, None),
        None: ("unrecorded", None, None),
    }
    for index, (steps, (label, max_steps, smoke_only)) in enumerate(expected.items()):
        record = dict(base_record, num_timesteps=steps)
        output = tmp_path / f"record-{index}"
        goal_viewer._record_rsl_preview(output, record, FakeAdapter(), FakeControls(record["sha256"]))
        saved = json.loads((output / "initial_state.json").read_text(encoding="utf-8"))
        assert saved["checkpoint_stage"] == label
        assert saved["smoke_max_steps"] == max_steps
        assert saved["smoke_only"] is smoke_only
        assert saved["smoke_step_source"] == ("unrecorded" if label == "unrecorded" else "checkpoint.num_timesteps")
