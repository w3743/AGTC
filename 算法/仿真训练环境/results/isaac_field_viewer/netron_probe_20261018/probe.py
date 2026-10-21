"""CPU-only experiment exporting the current SB3 recurrent policy for Netron."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from sb3_contrib import RecurrentPPO


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (
    ROOT
    / "results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive"
    / "resume_8192/stage_04_turn90.zip"
)
OUT = Path(__file__).resolve().parent
MODEL_PATH = OUT / "stage_04_turn90_actor_critic.onnx"
VALIDATION_PATH = OUT / "validation.json"

class ActorCriticExport(torch.nn.Module):
    """Expose the real policy's single-step actor/critic recurrence as ONNX IO."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        low = torch.as_tensor(policy.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(policy.action_space.high, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def forward(self, lidar, depth, detections, state, actor_h, actor_c, critic_h, critic_c):
        observation = {"lidar": lidar, "depth": depth, "detections": detections, "state": state}
        features = self.policy.extract_features(observation)
        actor_sequence, (next_actor_h, next_actor_c) = self.policy.lstm_actor(
            features.unsqueeze(0), (actor_h, actor_c)
        )
        critic_sequence, (next_critic_h, next_critic_c) = self.policy.lstm_critic(
            features.unsqueeze(0), (critic_h, critic_c)
        )
        actor_latent = self.policy.mlp_extractor.forward_actor(actor_sequence[0])
        critic_latent = self.policy.mlp_extractor.forward_critic(critic_sequence[0])
        mean = self.policy.action_net(actor_latent)
        value = self.policy.value_net(critic_latent)
        clipped = torch.maximum(torch.minimum(mean, self.action_high), self.action_low)
        return (
            mean,
            clipped,
            value,
            next_actor_h,
            next_actor_c,
            next_critic_h,
            next_critic_c,
        )


def random_inputs(rng, hidden_size, num_layers):
    observation = {
        "lidar": rng.uniform(0.0, 1.0, size=(1, 2, 400)).astype(np.float32),
        "depth": rng.uniform(0.0, 1.0, size=(1, 2, 160, 88)).astype(np.float32),
        "detections": rng.uniform(-1.0, 1.0, size=(1, 16, 10)).astype(np.float32),
        "state": rng.uniform(-1.0, 1.0, size=(1, 44)).astype(np.float32),
    }
    observation["detections"][:, :, 8] = 0.0
    observation["detections"][:, [0, 3, 7, 11, 15], 8] = 1.0
    recurrent = tuple(
        rng.normal(0.0, 0.2, size=(num_layers, 1, hidden_size)).astype(np.float32)
        for _ in range(4)
    )
    return observation, recurrent


def torch_inputs(observation, recurrent):
    return tuple(torch.from_numpy(observation[key]) for key in ("lidar", "depth", "detections", "state")) + tuple(
        torch.from_numpy(value) for value in recurrent
    )


def compare_outputs(actual, expected, *, rtol=2e-5, atol=2e-6):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    model = RecurrentPPO.load(str(CHECKPOINT), device="cpu")
    policy = model.policy.eval()
    encoder = policy.features_extractor
    old_cache = getattr(encoder, "cache_inference", None)
    if old_cache is not None:
        encoder.cache_inference = False

    hidden_size = int(policy.lstm_actor.hidden_size)
    num_layers = int(policy.lstm_actor.num_layers)
    wrapper = ActorCriticExport(policy).eval()
    input_names = ["lidar", "depth", "detections", "state", "actor_h", "actor_c", "critic_h", "critic_c"]
    output_names = ["action_mean", "action_clipped", "value", "actor_h_out", "actor_c_out", "critic_h_out", "critic_c_out"]
    dynamic_batch_axes = {
        **{name: 0 for name in input_names[:4]},
        **{name: 1 for name in input_names[4:]},
        "action_mean": 0,
        "action_clipped": 0,
        "value": 0,
        "actor_h_out": 1,
        "actor_c_out": 1,
        "critic_h_out": 1,
        "critic_c_out": 1,
    }
    batch = torch.export.Dim("batch")
    dynamic_shapes = (
        {0: batch}, {0: batch}, {0: batch}, {0: batch},
        {1: batch}, {1: batch}, {1: batch}, {1: batch},
    )
    rng = np.random.default_rng(20261018)
    examples = [random_inputs(rng, hidden_size, num_layers) for _ in range(3)]
    sample = torch_inputs(*examples[0])

    export_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            wrapper,
            sample,
            str(MODEL_PATH),
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=17,
            dynamo=True,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
        export_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]

    graph = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(graph)
    op_types = sorted({node.op_type for node in graph.graph.node})

    max_torch_predict_error = 0.0
    max_reference_error = None
    reference_error = None
    reference_outputs = None
    try:
        evaluator = onnx.reference.ReferenceEvaluator(graph)
        reference_outputs = []
        max_reference_error = 0.0
        for observation, recurrent in examples:
            values = torch_inputs(observation, recurrent)
            with torch.no_grad():
                torch_output = wrapper(*values)
            feed = {name: value.detach().cpu().numpy() for name, value in zip(input_names, values)}
            onnx_output = evaluator.run(output_names, feed)
            reference_outputs.append(onnx_output)
            for actual, expected in zip(onnx_output, torch_output):
                compare_outputs(actual, expected.detach().cpu().numpy())
                max_reference_error = max(
                    max_reference_error,
                    float(np.max(np.abs(actual - expected.detach().cpu().numpy()))),
                )
    except Exception as exc:
        reference_error = f"{type(exc).__name__}: {exc}"
        reference_outputs = None

    predict_errors = []
    with torch.no_grad():
        for index, (observation, recurrent) in enumerate(examples):
            actor_state = (recurrent[0].copy(), recurrent[1].copy())
            critic_state = (torch.from_numpy(recurrent[2]), torch.from_numpy(recurrent[3]))
            values = torch_inputs(observation, recurrent)
            direct = wrapper(*values)

            predict_action, predict_state = model.predict(
                observation,
                state=actor_state,
                episode_start=np.array([False]),
                deterministic=True,
            )
            action_error = float(
                np.max(np.abs(direct[1].detach().cpu().numpy() - predict_action))
            )
            actor_h_error = float(np.max(np.abs(direct[3].detach().cpu().numpy() - predict_state[0])))
            actor_c_error = float(np.max(np.abs(direct[4].detach().cpu().numpy() - predict_state[1])))

            tensor_obs, _ = policy.obs_to_tensor(observation)
            predicted_value = policy.predict_values(
                tensor_obs,
                critic_state,
                torch.zeros(1, dtype=torch.float32),
            )
            features = policy.extract_features(tensor_obs)
            _critic_latent, predicted_critic = policy._process_sequence(
                features,
                critic_state,
                torch.zeros(1, dtype=torch.float32),
                policy.lstm_critic,
            )
            value_error = float(np.max(np.abs(direct[2].detach().cpu().numpy() - predicted_value.detach().cpu().numpy())))
            critic_h_error = float(np.max(np.abs(direct[5].detach().cpu().numpy() - predicted_critic[0].detach().cpu().numpy())))
            critic_c_error = float(np.max(np.abs(direct[6].detach().cpu().numpy() - predicted_critic[1].detach().cpu().numpy())))
            predict_errors.append(
                {
                    "case": index,
                    "max_abs_action_clip": action_error,
                    "max_abs_actor_h": actor_h_error,
                    "max_abs_actor_c": actor_c_error,
                    "max_abs_value": value_error,
                    "max_abs_critic_h": critic_h_error,
                    "max_abs_critic_c": critic_c_error,
                }
            )
            max_torch_predict_error = max(max_torch_predict_error, *predict_errors[-1].values())

    if old_cache is not None:
        encoder.cache_inference = old_cache

    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "device": "cpu",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_batch_axes": dynamic_batch_axes,
        "onnx_bytes": MODEL_PATH.stat().st_size,
        "onnx_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "onnx_checker": "passed",
        "op_types": op_types,
        "torch_predict_comparison": predict_errors,
        "torch_predict_max_abs_error": max_torch_predict_error,
        "reference_evaluator": "passed" if reference_outputs is not None and reference_error is None else "failed",
        "reference_evaluator_max_abs_error": max_reference_error,
        "reference_evaluator_error": reference_error,
        "export_warnings": export_warnings,
        "policy_forward_signature": str(inspect.signature(policy.forward)),
        "policy_forward_source": inspect.getsource(policy.forward),
        "source_files": {
            "probe": str(Path(__file__).resolve().relative_to(ROOT)),
            "python": sys.version,
        },
    }
    VALIDATION_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
