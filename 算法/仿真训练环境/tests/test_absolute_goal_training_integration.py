"""CPU-only checks for absolute-goal schema propagation through mastery/training."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile
from dataclasses import asdict

import pytest

import scripts.train_isaac_mastery as mastery
from isaac_field import learning
from isaac_field.action_contract import action_config
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA
from isaac_field.task import Config


FAKE_SOURCES = (
    "isaac_field/frozen_component.py",
    "scripts/train_isaac_mastery.py",
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for relative in FAKE_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen: {relative}\n", encoding="utf-8")
    monkeypatch.setattr(mastery, "ROOT", root)
    monkeypatch.setattr(mastery, "_sources", lambda: FAKE_SOURCES)
    monkeypatch.setattr(mastery.foundation, "_package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(mastery.foundation, "_isaac_commit", lambda: "test-commit")
    monkeypatch.setattr(mastery.foundation, "_robot_geometry", lambda: {"spec": "test-proxy"})
    monkeypatch.setattr(mastery.foundation, "_action_config", lambda: action_config())
    return root


def _prepared(root: Path, name: str, **kwargs) -> Path:
    output = root / "results" / name
    mastery.prepare(output, budget=512, block_steps=512, **kwargs)
    return output


def _absolute_observation(config=None):
    return mastery._observation_contract(ABSOLUTE_GOAL_INPUT_SCHEMA, config or asdict(Config()))


def _absolute_config_dict():
    return asdict(Config(plants=0, terrain_enabled=False, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA))


def _install_encoder_checkpoint(path: Path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"field_action_config": action_config()}))


def test_prepare_and_cli_freeze_absolute_schema_without_starting_training(workspace, monkeypatch, capsys):
    monkeypatch.setattr(
        mastery, "_open_env", lambda *args, **kwargs: pytest.fail("prepare must not start an environment")
    )
    output = workspace / "results" / "cli_absolute"
    monkeypatch.setattr(sys, "argv", [
        "train_isaac_mastery.py", "--prepare", "--output", str(output),
        "--input-schema", ABSOLUTE_GOAL_INPUT_SCHEMA,
        "--budget", "512", "--block-steps", "512",
    ])

    mastery.main()
    result = json.loads(capsys.readouterr().out)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))

    assert result["training_started"] is False
    assert result["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert protocol["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert protocol["config"]["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert protocol["policy_architecture"]["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert protocol["policy_architecture"]["encoder_schema"].endswith("absolute-goal-v4")
    assert protocol["observation_contract"]["schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert protocol["observation_contract"]["state_fields"][1:3] == [
        "goal_world_x_normalized", "goal_world_y_normalized"
    ]
    assert mastery.preflight(output)["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert not (output / "untrained_baseline.pt").exists()


def test_absolute_fast_stop_transfer_protocol_keeps_same_schema(workspace, monkeypatch):
    output = _prepared(
        workspace,
        "absolute_transfer",
        profile=mastery.PROFILE_FAST_STOP_512,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    protocol = mastery.preflight(output)
    transfer = protocol["transfer_probe"]

    assert protocol["config"]["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert protocol["policy_architecture"]["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert transfer["environment_config"]["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert transfer["environment_config"]["input_schema"] == protocol["input_schema"]

    calls = []

    def evaluate(_output, protocol_arg, _checkpoint, stage, seeds, *, environment_seed, task_config):
        calls.append(task_config)
        return {
            "action_config": action_config(),
            "drive_config": {"schema": "drive"},
            "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
            "observation_config": protocol_arg["observation_contract"],
            "damage_config": {"metric": "stem-contact-v2"},
            "damage_metric": "stem-contact-v2",
            "distance_cost": 0.0,
            "goal_distance": stage["goal_distance"],
            "goal_angle_rad": stage["goal_angle"],
            "mode": "goto",
            "deterministic": True,
            "evaluation_config": {"environment_seed": environment_seed, "task": task_config},
            "seeds": list(seeds),
            "rows": [{"seed": seed} for seed in seeds],
        }

    # _global_probe uses the actual frozen transfer member environment config.
    monkeypatch.setattr(mastery, "_evaluate_stage", evaluate)
    result = mastery._global_probe(output, protocol, Path("baseline.zip"))

    assert len(calls) == 2
    assert all(cfg["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA for cfg in calls)
    assert result["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA


@pytest.mark.parametrize("field,value", [
    ("input_schema", "future-schema"),
    ("config", {"input_schema": INPUT_SCHEMA}),
    ("policy_architecture", {"input_schema": INPUT_SCHEMA}),
    ("observation_contract", {"schema": INPUT_SCHEMA}),
    ("transfer_probe", {"environment_config": {"input_schema": INPUT_SCHEMA}}),
])
def test_absolute_protocol_rejects_unknown_or_mismatched_schema_identity(workspace, field, value):
    output = _prepared(
        workspace,
        f"mismatch_{field}",
        profile=mastery.PROFILE_FAST_STOP_512,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if field == "input_schema":
        protocol[field] = value
    elif field == "config":
        protocol[field]["input_schema"] = value["input_schema"]
    elif field == "policy_architecture":
        protocol[field]["input_schema"] = value["input_schema"]
    elif field == "observation_contract":
        protocol[field]["schema"] = value["schema"]
    else:
        protocol[field]["environment_config"]["input_schema"] = value["environment_config"]["input_schema"]

    with pytest.raises(ValueError, match="input_schema|absolute-goal|transfer probe"):
        mastery._validate_profile_protocol(protocol)


def test_open_env_passes_transfer_schema_into_both_training_and_evaluation(workspace, monkeypatch):
    output = _prepared(
        workspace,
        "open_transfer",
        profile=mastery.PROFILE_FAST_STOP_512,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    protocol = mastery.preflight(output)
    member = protocol["transfer_probe"]["members"][0]
    stage = {key: member[key] for key in ("name", "plants", "goal_distance", "goal_angle")}
    stage["mode"] = "goto"
    observation_config = protocol["observation_contract"]
    opened = []

    class FakeEngine:
        def __init__(self, cfg, kwargs):
            self.field_cfg = cfg
            self.cfg = SimpleNamespace(
                goal_distance=None if kwargs["goal_distance"] is None else tuple(kwargs["goal_distance"]),
                mode=kwargs["mode"], goal_angle=kwargs["goal_angle"], distance_cost=0.0,
            )
            self.num_envs = kwargs["num_envs"]
            self.observation_config = observation_config
            self.closed = False

        def close(self):
            self.closed = True

    def make_env(**kwargs):
        opened.append(kwargs)
        return FakeEngine(kwargs["cfg"], kwargs)

    fake_env = ModuleType("isaac_field.env")
    fake_env.make_env = make_env
    monkeypatch.setitem(sys.modules, "isaac_field.env", fake_env)
    monkeypatch.setattr(mastery.foundation, "_live_environment_contract", lambda _env, **_kwargs: {})
    monkeypatch.setattr(
        mastery.foundation, "_validate_rough_profile_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(mastery.foundation, "_rough_profile_components", lambda *_args: (None, None))
    monkeypatch.setattr(mastery.foundation, "_runtime_assets", lambda _env: [])

    for training in (True, False):
        engine = mastery._open_env(
            output, protocol, stage, training=training, seed=member["environment_seed"],
            task_config=protocol["transfer_probe"]["environment_config"],
        )
        assert engine.field_cfg.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
        assert engine.num_envs == (4 if training else 1)
        engine.close()

    assert [item["cfg"].input_schema for item in opened] == [
        ABSOLUTE_GOAL_INPUT_SCHEMA, ABSOLUTE_GOAL_INPUT_SCHEMA
    ]


def _absolute_env():
    observation_config = _absolute_observation(_absolute_config_dict())
    return SimpleNamespace(
        cfg=SimpleNamespace(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA, distance_cost=0.0,
                            dt=0.05, max_time=1.0),
        field_cfg=Config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA),
        observation_config=observation_config,
        action_config=action_config(),
        observation_space=object(),
        action_space=object(),
        num_envs=4,
        backend_identity={
            "simulator": "isaac_sim",
            "physics_backend": "physx",
            "learning_backend": "rsl_rl",
        },
        drive_config={
            "interface": "independent_left_right_track_speed_coupled_pi_v7",
            "same_side_wheel_coupling": {"enabled": True},
            "physics_contact": {
                "simulator": "isaac_sim",
                "physics_backend": "physx",
                "learning_backend": "rsl_rl",
                "backend": "PhysxCfg",
                "solver": "TGS",
                "physics_manager": "isaaclab_physx.SomePhysicsManager",
                "physics_dt_s": 0.001,
                "decimation": 50,
                "policy_dt_s": 0.05,
                "contact_offset_m": 0.005,
                "rest_offset_m": 0.0,
                "calibrated_hardware": False,
                "usd_authored_materials": {
                    f"/World/envs/env_{index}/{surface}/material": {
                        "static_friction": 0.8,
                        "dynamic_friction": 0.7,
                        "restitution": 0.0,
                    }
                    for index in range(4)
                    for surface in ("Field", "Robot")
                },
            },
        },
        damage_config={"metric": "stem-contact-v2"},
    )


def test_live_environment_contract_accepts_absolute_schema_without_mocking_helper():
    from scripts import train_isaac_foundation_fresh as foundation

    env = _absolute_env()
    env.drive_config = {
        "interface": "independent_left_right_track_speed_coupled_pi_v7",
        "same_side_wheel_coupling": {"enabled": True},
        "physics_contact": {
            "simulator": "isaac_sim",
            "physics_backend": "physx",
            "learning_backend": "rsl_rl",
            "backend": "PhysxCfg",
            "solver": "TGS",
            "physics_manager": "isaaclab_physx.SomePhysicsManager",
            "physics_dt_s": 0.001,
            "decimation": 50,
            "policy_dt_s": 0.05,
            "contact_offset_m": 0.005,
            "rest_offset_m": 0.0,
            "calibrated_hardware": False,
            "usd_authored_materials": {
                f"/World/envs/env_{index}/{surface}/material": {
                    "static_friction": 0.8,
                    "dynamic_friction": 0.7,
                    "restitution": 0.0,
                }
                for index in range(4)
                for surface in ("Field", "Robot")
            },
        },
    }

    contract = foundation._live_environment_contract(
        env, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA
    )

    assert contract["observation_config"] == env.observation_config
    with pytest.raises(ValueError, match="expected schema 'sensor-minimal-v6'"):
        foundation._live_environment_contract(env)
    with pytest.raises(ValueError, match="unsupported expected live observation schema"):
        foundation._live_environment_contract(env, input_schema="future-schema")


def test_rough_profile_environment_forwards_absolute_schema(monkeypatch):
    from scripts import train_isaac_foundation_fresh as foundation

    observed = {}

    class StopAfterContractCheck(Exception):
        pass

    def inspect_schema(_env, *, input_schema):
        observed["input_schema"] = input_schema
        raise StopAfterContractCheck

    monkeypatch.setattr(foundation, "_live_environment_contract", inspect_schema)
    with pytest.raises(StopAfterContractCheck):
        foundation._validate_rough_profile_environment(
            object(), {}, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA
        )

    assert observed["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA


def test_public_training_helper_delegates_to_official_rsl_and_defaults_to_512(tmp_path, monkeypatch):
    from isaac_field import training

    env = _absolute_env()
    calls = {}

    def fake_train_stage(environment, checkpoint, timesteps, output, callback=None, **kwargs):
        calls.update(
            env=environment,
            checkpoint=checkpoint,
            timesteps=timesteps,
            output=Path(output),
            callback=callback,
            kwargs=kwargs,
        )
        return SimpleNamespace(
            model=object(),
            checkpoint=Path(output),
            trained_timesteps=timesteps,
        )

    monkeypatch.setattr(learning, "train_stage", fake_train_stage)
    result = training.train(env, tmp_path / "fresh", timesteps=256, device="cpu")

    assert calls["env"] is env
    assert calls["checkpoint"] is None
    assert calls["timesteps"] == 256
    assert calls["output"] == tmp_path / "fresh.pt"
    assert calls["kwargs"]["n_steps"] == 128
    assert calls["kwargs"]["batch_size"] == 128
    assert calls["kwargs"]["n_epochs"] == 5
    assert calls["kwargs"]["lstm_hidden_size"] == 512
    assert result.actual_timesteps == 256
    assert result.checkpoint == tmp_path / "fresh.pt"


@pytest.mark.parametrize("entrypoint", ["train_path", "evaluate_path"])
def test_legacy_zip_is_rejected_from_rsl_resume_and_default_evaluation(tmp_path, entrypoint):
    path = tmp_path / "old-v6.zip"
    path.write_bytes(b"legacy SB3 checkpoint")
    env = _absolute_env()

    if entrypoint == "train_path":
        with pytest.raises(ValueError, match="SB3 checkpoints are legacy-only"):
            learning.train_stage(
                env, path, 128, tmp_path / "resumed", device="cpu",
                n_steps=128, batch_size=128, n_epochs=1,
            )
    else:
        adapter = SimpleNamespace(
            engine=env,
            cfg=SimpleNamespace(dt=0.05, max_time=1.0),
            task=None,
        )
        with pytest.raises(ValueError, match="SB3 policy loading is legacy-only"):
            learning.evaluate_model(
                adapter, path, (5,), device="cpu", input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                expected_observation_config=env.observation_config,
            )


@pytest.mark.parametrize("entrypoint", ["train_object", "evaluate_object"])
def test_resume_and_evaluate_reject_metadata_free_absolute_objects_before_use(
    tmp_path, monkeypatch, entrypoint,
):
    env = _absolute_env()
    touched = []

    class OldModel:
        field_action_config = action_config()

        def predict(self, *_args, **_kwargs):
            touched.append("predict")
            raise AssertionError("predict must not run")

        def set_env(self, _env):
            touched.append("set_env")
            raise AssertionError("set_env must not run")

    if entrypoint == "train_object":
        with pytest.raises(TypeError, match="RSL stage resume requires an explicit .pt"):
            learning.train_stage(
                env, OldModel(), 128, tmp_path / "resume-object", device="cpu",
                n_steps=128, batch_size=128, n_epochs=1,
            )
    else:
        adapter = SimpleNamespace(engine=env, cfg=SimpleNamespace(dt=0.05, max_time=1.0), task=None)
        with pytest.raises(ValueError, match="absolute-goal evaluation requires a native RSL-RL .pt"):
            learning.evaluate_model(
                adapter, OldModel(), (5,), device="cpu", input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                expected_observation_config=env.observation_config,
            )
    assert touched == []
