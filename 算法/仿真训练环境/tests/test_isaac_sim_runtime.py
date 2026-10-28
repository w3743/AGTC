"""Runtime ownership tests: nested workflows must not close each other's app."""
import os
import sys
from types import ModuleType

import pytest

from isaac_field import runtime


@pytest.fixture
def launch_state(monkeypatch):
    # These tests install a fake launcher; no native runtime can start here.
    monkeypatch.delenv('ISAAC_FIELD_DISABLE_NATIVE_RUNTIME', raising=False)
    state = {"existing": False, "launched": [], "closed": [], "atexit": []}
    utils = ModuleType("isaaclab.utils")
    utils.has_kit = lambda: state["existing"]
    app = ModuleType("isaaclab.app")

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, *error):
            state["closed"].append(error)

    def launch(cfg, options):
        state["launched"].append((cfg, options))
        return Lease()

    app.launch_simulation = launch
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(runtime, "_runtime_options", None)
    monkeypatch.setattr(runtime, "_atexit_registered", False)
    monkeypatch.setattr(runtime.atexit, "register", state["atexit"].append)
    return state


def test_nested_curriculum_owns_one_app(launch_state):
    with runtime.launch_app():
        with runtime.launch_app():
            runtime.ensure_runtime()
        assert not launch_state["closed"]
    assert len(launch_state["launched"]) == 1
    assert launch_state["closed"] == [(None, None, None)]


def test_official_cli_app_is_not_closed_by_evaluator(launch_state):
    launch_state["existing"] = True
    with runtime.launch_app():
        runtime.ensure_runtime()
    assert launch_state["launched"] == []
    assert launch_state["closed"] == []


def test_failure_is_given_to_official_launcher(launch_state):
    with pytest.raises(ValueError, match="failed rollout"):
        with runtime.launch_app():
            raise ValueError("failed rollout")
    kind, error, traceback = launch_state["closed"][0]
    assert kind is ValueError and str(error) == "failed rollout" and traceback is not None


def test_second_stage_cannot_switch_device_silently(launch_state):
    with runtime.launch_app(device="cuda:0"):
        with pytest.raises(ValueError, match="switch"):
            runtime.ensure_runtime(device="cuda:1")
    assert len(launch_state["launched"]) == 1


def test_factory_preserves_rsl_environment_selection(monkeypatch):
    module = ModuleType("isaac_field.env")
    module.make_env = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, "isaac_field.env", module)
    args, kwargs = runtime.create_env("config", rsl=True, num_envs=4)
    assert args == ("config",) and kwargs == {"rsl": True, "num_envs": 4}


def test_cpu_process_guard_rejects_native_launch_before_import(monkeypatch):
    monkeypatch.setenv('ISAAC_FIELD_DISABLE_NATIVE_RUNTIME', '1')
    monkeypatch.setattr(runtime, '_runtime', None)
    with pytest.raises(RuntimeError, match='CPU-only process'):
        runtime.ensure_runtime()


def test_official_cli_kit_args_add_project_defaults_once(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("ISAAC_FIELD_RUNTIME_DIR", str(runtime_dir))
    source = ["--num_envs", "2"]

    prepared = runtime._prepare_official_rsl_cli_arguments(source)

    assert source == ["--num_envs", "2"]
    assert prepared[:-1] == source
    kit_tokens = [token for token in prepared if token == "--kit_args" or token.startswith("--kit_args=")]
    assert len(kit_tokens) == 1 and kit_tokens[0].startswith("--kit_args=")
    kit_args = kit_tokens[0].split("=", 1)[1]
    if os.name == "nt":
        assert f"--portable-root {(runtime_dir / 'kit').as_posix()}" in kit_args
    assert kit_args.count("/app/settings/fabricDefaultStageFrameHistoryCount") == 1
    assert "fabricDefaultStageFrameHistoryCount=3" in kit_args


def test_official_cli_kit_args_accept_both_forms_and_keep_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ISAAC_FIELD_RUNTIME_DIR", str(tmp_path / "runtime"))
    prepared = runtime._prepare_official_rsl_cli_arguments([
        "--kit_args", "--portable-root C:/custom/kit",
        "--kit_args=--/app/settings/fabricDefaultStageFrameHistoryCount=9",
        "--headless",
    ])

    assert prepared[:-1] == ["--headless"]
    kit_tokens = [token for token in prepared if token == "--kit_args" or token.startswith("--kit_args=")]
    assert kit_tokens == ["--kit_args=--portable-root C:/custom/kit --/app/settings/fabricDefaultStageFrameHistoryCount=9"]


def test_official_cli_kit_args_rejects_missing_value():
    with pytest.raises(ValueError, match="--kit_args requires a value"):
        runtime._prepare_official_rsl_cli_arguments(["--kit_args"])


def test_declared_field_camera_is_detected_by_official_static_launch_scan():
    from isaac_field.env import FieldEnvCfg
    from isaaclab.app.sim_launcher import validate_launch

    scan, kit_sources = validate_launch(FieldEnvCfg(), {})

    assert scan.has_kit_camera
    assert any("Kit-based renderer" in source for source in kit_sources)
