"""Check real CLI schema propagation before any native simulation is opened."""
from __future__ import annotations

import sys
import types

import pytest

from isaac_field import __main__ as cli
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA


class _BeforePhysics(Exception):
    pass


@pytest.mark.parametrize("command", ["viewer", "train", "evaluate"])
@pytest.mark.parametrize("schema", [INPUT_SCHEMA, ABSOLUTE_GOAL_INPUT_SCHEMA])
def test_cli_passes_selected_schema_to_native_environment(monkeypatch, tmp_path, command, schema):
    def make_env(cfg, *_args, **_kwargs):
        assert cfg.input_schema == schema
        raise _BeforePhysics

    monkeypatch.setitem(sys.modules, "isaac_field.env", types.SimpleNamespace(
        make_env=make_env, SingleFieldAdapter=object))
    arguments = ["isaac_field", command, "--plants", "0", "--flat",
                 "--output", str(tmp_path / command)]
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
