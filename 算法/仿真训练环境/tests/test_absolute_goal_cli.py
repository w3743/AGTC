"""Check real CLI schema propagation before any native simulation is opened."""
from __future__ import annotations

import sys
import types

import pytest

from isaac_field import __main__ as cli
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA


class _BeforePhysics(Exception):
    pass


@pytest.mark.parametrize("command", ["viewer", "train-sb3", "evaluate"])
@pytest.mark.parametrize("schema", [INPUT_SCHEMA, ABSOLUTE_GOAL_INPUT_SCHEMA])
def test_cli_passes_selected_schema_to_native_environment(monkeypatch, tmp_path, command, schema):
    def make_env(cfg, *_args, **_kwargs):
        assert cfg.input_schema == schema
        raise _BeforePhysics

    monkeypatch.setitem(sys.modules, "isaac_field.env", types.SimpleNamespace(
        make_env=make_env, SingleFieldAdapter=object))
    arguments = ["isaac_field", command, "--plants", "0", "--flat",
                 "--output", str(tmp_path / command)]
    if command == "evaluate":
        arguments.extend(["--checkpoint", str(tmp_path / "legacy.zip"), "--legacy-sb3"])
    if schema != INPUT_SCHEMA:
        arguments.extend(["--input-schema", schema])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(_BeforePhysics):
        cli.main()


def test_cli_rejects_unknown_schema_before_importing_native_environment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isaac_field", "viewer", "--input-schema", "unknown"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2


def test_new_train_delegates_to_official_rsl_cli_without_legacy_environment(monkeypatch):
    observed = []
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints", types.SimpleNamespace(
        run_train_cli=lambda arguments: observed.append(arguments) or 7,
    ))
    monkeypatch.setattr(sys, "argv", ["isaac_field", "train", "--num_envs", "4", "--max_iterations", "384"])
    assert cli.main() == 7
    assert observed == [[
        "--rl_library", "rsl_rl", "--task", "Isaac-Field-FastStop-Rsl-Direct",
        "--num_envs", "4", "--max_iterations", "384",
    ]]


def test_resume_delegates_explicit_rsl_checkpoint_to_official_cli(monkeypatch):
    observed = []
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints", types.SimpleNamespace(
        run_train_cli=lambda arguments: observed.append(arguments) or 0,
    ))
    monkeypatch.setattr(sys, "argv", ["isaac_field", "resume", "--checkpoint", "model_12.pt",
                                       "--max_iterations", "384"])

    assert cli.main() == 0
    assert observed == [[
        "--rl_library", "rsl_rl", "--task", "Isaac-Field-FastStop-Rsl-Direct",
        "--checkpoint", "model_12.pt", "--max_iterations", "384",
    ]]


def test_resume_rejects_sb3_zip_before_official_runner(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isaac_field", "resume", "--checkpoint", "old.zip"])
    with pytest.raises(SystemExit):
        cli.main()


def test_pt_evaluation_thin_forwards_explicit_smoke_without_generic_env(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setitem(sys.modules, "scripts.evaluate_fast_stop_mastery", types.SimpleNamespace(
        main=lambda arguments: observed.append(arguments) or 6,
    ))
    monkeypatch.setitem(sys.modules, "isaac_field.env", types.SimpleNamespace(
        make_env=lambda *_args, **_kwargs: pytest.fail("generic Config env must not be constructed"),
    ))
    report_dir = tmp_path / "rsl-smoke"
    monkeypatch.setattr(sys, "argv", [
        "isaac_field", "evaluate", "--checkpoint", str(tmp_path / "model.pt"),
        "--smoke", "--max-steps", "1", "--scenario-count", "2", "--seed", "87001",
        "--output", str(report_dir),
    ])

    assert cli.main() == 6
    assert observed == [[
        "--checkpoint", str(tmp_path / "model.pt"), "--smoke", "--max-steps", "1",
        "--scenario-count", "2", "--seed", "87001", "--device", "cpu",
        "--output", str((report_dir / "evaluation.json").resolve()),
    ]]


def test_pt_evaluation_defaults_to_full_existing_evaluator(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setitem(sys.modules, "scripts.evaluate_fast_stop_mastery", types.SimpleNamespace(
        main=lambda arguments: observed.append(arguments) or 6,
    ))
    monkeypatch.setitem(sys.modules, "isaac_field.env", types.SimpleNamespace(
        make_env=lambda *_args, **_kwargs: pytest.fail("RSL evaluation must rebuild from checkpoint task metadata"),
    ))
    output_dir = tmp_path / "full-evaluation"
    monkeypatch.setattr(sys, "argv", [
        "isaac_field", "evaluate", "--checkpoint", str(tmp_path / "model.pt"),
        "--output", str(output_dir),
    ])

    assert cli.main() == 6
    assert observed == [[
        "--checkpoint", str(tmp_path / "model.pt"), "--device", "cpu",
        "--output", str((output_dir / "evaluation.json").resolve()),
    ]]


def test_zip_evaluation_requires_explicit_legacy_switch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isaac_field", "evaluate", "--checkpoint", "old.zip"])
    with pytest.raises(SystemExit):
        cli.main()


def test_legacy_zip_evaluation_preserves_custom_seed(monkeypatch, tmp_path):
    def make_env(_cfg, _mode, _num_envs, seed, *_args, **_kwargs):
        assert seed == 87_001
        raise _BeforePhysics

    monkeypatch.setitem(sys.modules, "isaac_field.env", types.SimpleNamespace(
        make_env=make_env, SingleFieldAdapter=object,
    ))
    monkeypatch.setattr(sys, "argv", [
        "isaac_field", "evaluate", "--checkpoint", "old.zip", "--legacy-sb3",
        "--seed", "87001", "--plants", "0", "--flat", "--output", str(tmp_path / "legacy"),
    ])
    with pytest.raises(_BeforePhysics):
        cli.main()


def test_checkpoint_viewer_is_thin_forward_and_zip_requires_legacy(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setitem(sys.modules, "scripts.view_fast_stop_goal", types.SimpleNamespace(
        main=lambda arguments: observed.append(arguments) or 4,
    ))
    monkeypatch.setattr(sys, "argv", [
        "isaac_field", "viewer", "--checkpoint", str(tmp_path / "model.pt"),
        "--launch", "--output", str(tmp_path / "viewer"), "--port", "9000",
    ])

    assert cli.main() == 4
    assert observed == [[
        "--checkpoint", str(tmp_path / "model.pt"), "--launch", "--output",
        str(tmp_path / "viewer"), "--port", "9000",
    ]]
    monkeypatch.setattr(sys, "argv", ["isaac_field", "viewer", "--checkpoint", "old.zip"])
    with pytest.raises(SystemExit):
        cli.main()


def test_legacy_zip_viewer_keeps_generic_task_config_path(monkeypatch, tmp_path):
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA

    def make_env(cfg, *_args, **_kwargs):
        assert cfg.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
        raise _BeforePhysics

    monkeypatch.setitem(sys.modules, "isaac_field.env", types.SimpleNamespace(
        make_env=make_env, SingleFieldAdapter=object,
    ))
    monkeypatch.setattr(sys, "argv", [
        "isaac_field", "viewer", "--checkpoint", "old.zip", "--legacy-sb3",
        "--input-schema", ABSOLUTE_GOAL_INPUT_SCHEMA,
    ])
    with pytest.raises(_BeforePhysics):
        cli.main()
