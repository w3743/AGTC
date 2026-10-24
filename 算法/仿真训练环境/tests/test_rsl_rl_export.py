from __future__ import annotations

import importlib.util
import math

import pytest
import torch


def _field_model():
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict

    from isaac_field.policy_contract import OBSERVATION_SHAPES
    from isaac_field.rsl_rl_policy import RslFieldRNNModel

    torch.manual_seed(731)
    obs = TensorDict(
        {
            key: torch.zeros((1, *shape), dtype=torch.float32)
            for key, shape in OBSERVATION_SHAPES.items()
        },
        batch_size=[1],
    )
    actor = RslFieldRNNModel(
        obs,
        {"actor": list(OBSERVATION_SHAPES)},
        "actor",
        4,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": math.exp(-2),
            "std_type": "log",
            "std_range": (1e-6, 1e6),
        },
        cache_inference=False,
    ).cpu().eval()
    return actor


def _sensor_step(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    result = {
        "lidar": torch.rand((1, 1, 400), generator=generator),
        "depth": torch.rand((1, 1, 160, 88), generator=generator),
        "detections": torch.rand((1, 16, 8), generator=generator),
        "state": torch.rand((1, 33), generator=generator),
        "feedback": torch.rand((1, 37), generator=generator),
    }
    # Exercise the existing positive-width-and-height presence rule, including
    # absent rows whose other detection fields remain nonzero.
    result["detections"][:, 0, 2] = 0
    result["detections"][:, 1, 3] = 0
    return result


def _actor_mean_and_state(actor, observation: dict[str, torch.Tensor]):
    from tensordict import TensorDict

    with torch.inference_mode():
        mean = actor(TensorDict(observation, batch_size=[1]))
        hidden, cell = actor.get_hidden_state()
    return mean, hidden, cell


def test_export_driver_sequence_verification_resets_scripted_model():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "export_rsl_policy.py"
    spec = importlib.util.spec_from_file_location("export_rsl_policy_driver_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FakeModel:
        def __init__(self):
            self.steps = 0

        def predict(self, observation, *, state, episode_start, deterministic):
            assert deterministic is True
            self.steps += 1
            return torch.zeros((1, 4)).numpy(), self.steps

    class FakeScripted:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def __call__(self, *inputs):
            assert len(inputs) == 5
            return torch.zeros((1, 4))

    model = FakeModel()
    scripted = FakeScripted()
    result = module._verify_sequence(model, scripted)

    assert result == {"sequence_steps": 4, "max_abs_error": 0.0}
    assert scripted.reset_count == 1
    assert model.steps == 4


def test_field_jit_export_scripts_saves_resets_and_matches_actor(tmp_path):
    from isaac_field.rsl_rl_export import field_model_as_jit

    actor = _field_model()
    eager_export = field_model_as_jit(actor)
    scripted = torch.jit.script(eager_export)
    path = tmp_path / "field_policy.pt"
    scripted.save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu").eval()

    actor.reset()
    loaded.reset()
    for seed in (11, 12, 13):
        observation = _sensor_step(seed)
        expected, hidden, cell = _actor_mean_and_state(actor, observation)
        with torch.inference_mode():
            actual = loaded(*(observation[key] for key in ("lidar", "depth", "detections", "state", "feedback")))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(loaded.hidden_state, hidden, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(loaded.cell_state, cell, rtol=1e-5, atol=1e-6)

    loaded.reset()
    actor.reset()
    assert torch.count_nonzero(loaded.hidden_state) == 0
    assert torch.count_nonzero(loaded.cell_state) == 0
    observation = _sensor_step(14)
    expected, hidden, cell = _actor_mean_and_state(actor, observation)
    with torch.inference_mode():
        actual = loaded(*(observation[key] for key in ("lidar", "depth", "detections", "state", "feedback")))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loaded.hidden_state, hidden, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loaded.cell_state, cell, rtol=1e-5, atol=1e-6)


def test_field_export_rejects_critic_model():
    from isaac_field.policy_contract import OBSERVATION_SHAPES
    from isaac_field.rsl_rl_export import field_model_as_jit
    from isaac_field.rsl_rl_policy import RslFieldRNNModel
    from tensordict import TensorDict

    actor = _field_model()
    obs = TensorDict(
        {key: torch.zeros((1, *shape)) for key, shape in OBSERVATION_SHAPES.items()},
        batch_size=[1],
    )
    critic = RslFieldRNNModel(
        obs,
        {"critic": list(OBSERVATION_SHAPES)},
        "critic",
        1,
        distribution_cfg=None,
        cnns=actor.cnns,
        cache_inference=False,
    )
    with pytest.raises(ValueError, match="actor only"):
        field_model_as_jit(critic)


def test_field_onnx_wrapper_matches_actor_and_declares_rsl_contract():
    from tensordict import TensorDict

    from isaac_field.rsl_rl_export import field_model_as_onnx

    actor = _field_model()
    wrapper = field_model_as_onnx(actor, verbose=False).cpu().eval()
    dummy = wrapper.get_dummy_inputs()
    assert tuple(wrapper.input_names) == (
        "lidar",
        "depth",
        "detections",
        "state",
        "feedback",
        "h_in",
        "c_in",
    )
    assert tuple(wrapper.output_names) == ("actions", "h_out", "c_out")
    assert [tuple(value.shape) for value in dummy] == [
        (1, 1, 400),
        (1, 1, 160, 88),
        (1, 16, 8),
        (1, 33),
        (1, 37),
        (1, 1, 512),
        (1, 1, 512),
    ]

    actor.reset()
    for seed in (21, 22, 23):
        observation = _sensor_step(seed)
        expected, hidden, cell = _actor_mean_and_state(actor, observation)
        explicit_state = (hidden.new_zeros(hidden.shape), cell.new_zeros(cell.shape)) if seed == 21 else last_state
        inputs = tuple(observation[key] for key in ("lidar", "depth", "detections", "state", "feedback"))
        with torch.inference_mode():
            mean, h_out, c_out = wrapper(*inputs, *explicit_state)
        torch.testing.assert_close(mean, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h_out, hidden, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(c_out, cell, rtol=1e-5, atol=1e-6)
        last_state = (h_out, c_out)


def test_field_onnx_wrapper_exports_when_export_dependencies_are_available(tmp_path):
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("ONNX export skipped: the preinstalled environment has no onnx package")
    if importlib.util.find_spec("onnxscript") is None:
        pytest.skip("ONNX export skipped: the preinstalled PyTorch exporter has no onnxscript package")

    from isaac_field.rsl_rl_export import field_model_as_onnx
    import onnx

    actor = _field_model()
    wrapper = field_model_as_onnx(actor, verbose=False).cpu().eval()
    output = tmp_path / "field_policy.onnx"
    torch.onnx.export(
        wrapper,
        wrapper.get_dummy_inputs(),
        str(output),
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
    )
    assert output.is_file() and output.stat().st_size > 0
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    assert [value.name for value in exported.graph.input] == wrapper.input_names
    assert [value.name for value in exported.graph.output] == wrapper.output_names
