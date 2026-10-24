"""Check real CLI schema propagation before any native simulation is opened."""
from __future__ import annotations

import builtins
import os
import sys
import types
from pathlib import Path

import pytest

from isaac_field import __main__ as cli
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA


class _BeforePhysics(Exception):
    pass


def _expected_kit_args(runtime_dir: Path, user_args: str = "") -> str:
    parts = []
    if os.name == "nt" and "--portable" not in user_args:
        parts.extend(("--portable-root", (runtime_dir / "kit").as_posix()))
    if user_args:
        parts.append(user_args)
    if "/app/settings/fabricDefaultStageFrameHistoryCount" not in user_args:
        parts.append("--/app/settings/fabricDefaultStageFrameHistoryCount=3")
    return " ".join(parts)


def _install_fake_official_entrypoint(monkeypatch, runner):
    entrypoint = types.ModuleType("isaaclab_rl.entrypoints")
    entrypoint.run_train_cli = runner
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints", entrypoint)
    # The forwarding tests use only the fake module above; allow normal
    # arguments while the parent process runs the native-runtime guard.
    monkeypatch.delenv("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME", raising=False)


@pytest.mark.parametrize("command", ["viewer", "evaluate"])
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


def test_new_train_delegates_to_official_rsl_cli_without_legacy_environment(monkeypatch, tmp_path):
    observed = []
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("ISAAC_FIELD_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    _install_fake_official_entrypoint(monkeypatch, lambda arguments: observed.append(arguments) or 7)
    monkeypatch.setattr(sys, "argv", ["isaac_field", "train", "--num_envs", "4", "--max_iterations", "384"])
    assert cli.main() == 7
    assert observed == [[
        "--rl_library", "rsl_rl", "--task", "Isaac-Field-FastStop-Rsl-Direct",
        "--num_envs", "4", "--max_iterations", "384",
        f"--kit_args={_expected_kit_args(runtime_dir)}",
    ]]
    assert os.environ["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_train_merges_both_kit_args_forms_and_preserves_user_overrides(monkeypatch, tmp_path):
    observed = []
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("ISAAC_FIELD_RUNTIME_DIR", str(runtime_dir))
    _install_fake_official_entrypoint(monkeypatch, lambda arguments: observed.append(arguments) or 0)
    monkeypatch.setattr(sys, "argv", [
        "isaac_field", "train", "--kit_args", "--portable-root C:/kit/custom",
        "--kit_args=--/app/settings/fabricDefaultStageFrameHistoryCount=7",
        "--num_envs", "2",
    ])

    assert cli.main() == 0
    forwarded = observed[0]
    kit_tokens = [token for token in forwarded if token == "--kit_args" or token.startswith("--kit_args=")]
    assert kit_tokens == ["--kit_args=--portable-root C:/kit/custom --/app/settings/fabricDefaultStageFrameHistoryCount=7"]
    assert forwarded[-3:] == ["--num_envs", "2", kit_tokens[0]]
    assert "--portable-root C:/kit/custom" in kit_tokens[0]
    assert "fabricDefaultStageFrameHistoryCount=7" in kit_tokens[0]
    assert "fabricDefaultStageFrameHistoryCount=3" not in kit_tokens[0]


def test_native_runtime_guard_rejects_non_help_before_importing_official_entrypoint(monkeypatch):
    monkeypatch.setenv("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME", "1")
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    original_import = builtins.__import__

    def reject_official_import(name, *args, **kwargs):
        if name == "isaaclab_rl.entrypoints":
            raise AssertionError("official entrypoint import reached under CPU-only guard")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_official_import)
    with pytest.raises(RuntimeError, match="CPU-only process"):
        cli._run_official_rsl_cli(["--num_envs", "2"])
    assert os.environ["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_help_forwarding_is_fake_only_and_accepts_eula_before_entrypoint_lookup(monkeypatch, tmp_path):
    observed = []
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME", "1")
    monkeypatch.setenv("ISAAC_FIELD_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    entrypoint = types.ModuleType("isaaclab_rl.entrypoints")

    def __getattr__(name):
        if name != "run_train_cli":
            raise AttributeError(name)
        assert os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES"
        return lambda arguments: observed.append(arguments) or 0

    entrypoint.__getattr__ = __getattr__
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints", entrypoint)

    assert cli._run_official_rsl_cli(["--help"]) == 0
    assert observed[0][:5] == ["--rl_library", "rsl_rl", "--task", "Isaac-Field-FastStop-Rsl-Direct", "--help"]
    kit_tokens = [token for token in observed[0] if token == "--kit_args" or token.startswith("--kit_args=")]
    assert kit_tokens == [f"--kit_args={_expected_kit_args(runtime_dir)}"]


def test_sb3_training_command_is_explicitly_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isaac_field", "train-sb3"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2


def test_resume_delegates_explicit_rsl_checkpoint_to_official_cli(monkeypatch, tmp_path):
    observed = []
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("ISAAC_FIELD_RUNTIME_DIR", str(runtime_dir))
    _install_fake_official_entrypoint(monkeypatch, lambda arguments: observed.append(arguments) or 0)
    monkeypatch.setattr(sys, "argv", ["isaac_field", "resume", "--checkpoint", "model_12.pt",
                                       "--max_iterations", "384"])

    assert cli.main() == 0
    assert observed == [[
        "--rl_library", "rsl_rl", "--task", "Isaac-Field-FastStop-Rsl-Direct",
        "--checkpoint", "model_12.pt", "--max_iterations", "384",
        f"--kit_args={_expected_kit_args(runtime_dir)}",
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
