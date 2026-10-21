"""Trace/save the policy's real one-step modules for Netron inspection."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from probe import ActorCriticExport, CHECKPOINT, random_inputs, torch_inputs


OUT = Path(__file__).resolve().parent
MODEL_PATH = OUT / "stage_04_turn90_actor_critic.pt"
REPORT_PATH = OUT / "torchscript_validation.json"
INPUT_NAMES = ["lidar", "depth", "detections", "state", "actor_h", "actor_c", "critic_h", "critic_c"]
OUTPUT_NAMES = ["action_mean", "action_clipped", "value", "actor_h_out", "actor_c_out", "critic_h_out", "critic_c_out"]


def as_numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def main():
    torch.manual_seed(20261018)
    torch.set_num_threads(1)
    model = RecurrentPPO.load(str(CHECKPOINT), device="cpu")
    policy = model.policy.eval()
    encoder = policy.features_extractor
    cache_before = getattr(encoder, "cache_inference", None)
    if cache_before is not None:
        encoder.cache_inference = False

    hidden = policy.lstm_actor.hidden_size
    layers = policy.lstm_actor.num_layers
    wrapper = ActorCriticExport(policy).eval()
    rng = np.random.default_rng(20261018)
    initial_cases = [random_inputs(rng, hidden, layers) for _ in range(3)]
    trace_inputs = torch_inputs(*initial_cases[0])

    trace_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traced = torch.jit.trace(
            wrapper,
            trace_inputs,
            check_trace=True,
            check_inputs=[torch_inputs(*case) for case in initial_cases[1:]],
            check_tolerance=2e-5,
            strict=True,
        )
        trace_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]
    traced.save(str(MODEL_PATH))
    loaded = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()

    maximum_errors = {
        "action_mean_vs_wrapper": 0.0,
        "action_clipped_vs_predict": 0.0,
        "actor_h_vs_predict": 0.0,
        "actor_c_vs_predict": 0.0,
        "value_vs_predict_values": 0.0,
        "critic_h_vs_policy_lstm": 0.0,
        "critic_c_vs_policy_lstm": 0.0,
        "torchscript_vs_wrapper": 0.0,
    }
    comparisons = []
    with torch.no_grad():
        for case_index, (observation, recurrent) in enumerate(initial_cases):
            actor_state = (recurrent[0].copy(), recurrent[1].copy())
            critic_state = (torch.from_numpy(recurrent[2].copy()), torch.from_numpy(recurrent[3].copy()))
            for step_index in range(3):
                recurrent_numpy = (actor_state[0], actor_state[1], as_numpy(critic_state[0]), as_numpy(critic_state[1]))
                inputs = torch_inputs(observation, recurrent_numpy)
                wrapper_output = wrapper(*inputs)
                script_output = loaded(*inputs)

                step_errors = {}
                for name, actual, expected in zip(OUTPUT_NAMES, script_output, wrapper_output):
                    actual_np = as_numpy(actual)
                    expected_np = as_numpy(expected)
                    assert_close(actual_np, expected_np)
                    error = float(np.max(np.abs(actual_np - expected_np)))
                    maximum_errors["torchscript_vs_wrapper"] = max(
                        maximum_errors["torchscript_vs_wrapper"], error
                    )
                    step_errors[f"torchscript_vs_wrapper.{name}"] = error

                predict_action, next_actor = model.predict(
                    observation,
                    state=actor_state,
                    episode_start=np.array([False]),
                    deterministic=True,
                )
                for key, actual, expected in (
                    ("action_clipped_vs_predict", wrapper_output[1], predict_action),
                    ("actor_h_vs_predict", wrapper_output[3], next_actor[0]),
                    ("actor_c_vs_predict", wrapper_output[4], next_actor[1]),
                ):
                    actual_np = as_numpy(actual) if isinstance(actual, torch.Tensor) else np.asarray(actual)
                    error = float(np.max(np.abs(actual_np - expected)))
                    maximum_errors[key] = max(maximum_errors[key], error)
                    step_errors[key] = error
                    assert_close(actual_np, expected)

                tensor_obs, _ = policy.obs_to_tensor(observation)
                starts = torch.zeros(1, dtype=torch.float32)
                predicted_value = policy.predict_values(tensor_obs, critic_state, starts)
                features = policy.extract_features(tensor_obs)
                _critic_latent, next_critic = policy._process_sequence(
                    features, critic_state, starts, policy.lstm_critic
                )
                for key, actual, expected in (
                    ("value_vs_predict_values", wrapper_output[2], predicted_value),
                    ("critic_h_vs_policy_lstm", wrapper_output[5], next_critic[0]),
                    ("critic_c_vs_policy_lstm", wrapper_output[6], next_critic[1]),
                ):
                    actual_np = as_numpy(actual)
                    expected_np = as_numpy(expected)
                    error = float(np.max(np.abs(actual_np - expected_np)))
                    maximum_errors[key] = max(maximum_errors[key], error)
                    step_errors[key] = error
                    assert_close(actual_np, expected_np)

                comparisons.append({"case": case_index, "step": step_index, "max_abs_errors": step_errors})
                actor_state = (np.asarray(next_actor[0]).copy(), np.asarray(next_actor[1]).copy())
                critic_state = next_critic
                observation, _unused_next_state = random_inputs(rng, hidden, layers)

    if cache_before is not None:
        encoder.cache_inference = cache_before

    report = {
        "checkpoint": str(CHECKPOINT.relative_to(Path(__file__).resolve().parents[3])),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "format": "TorchScript trace (.pt), loaded with torch.jit.load",
        "torch": torch.__version__,
        "device": "cpu",
        "hidden_size": hidden,
        "num_layers": layers,
        "inputs": INPUT_NAMES,
        "outputs_in_order": OUTPUT_NAMES,
        "episode_start_input": "omitted; caller resets actor/critic h/c to zero at episode start",
        "wrapper": "actual loaded features_extractor, actor/critic LSTM, actor/critic MLP, action_net and value_net",
        "torchscript_bytes": MODEL_PATH.stat().st_size,
        "torchscript_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "torch_jit_trace_check": "passed",
        "torch_jit_trace_warnings": trace_warnings,
        "comparison_sequences": 3,
        "steps_per_sequence": 3,
        "nonzero_initial_recurrent_states": True,
        "max_abs_errors": maximum_errors,
        "all_compared_within_rtol_2e-5_atol_2e-6": True,
        "per_step_comparisons": comparisons,
        "note": "The action_mean output is the deterministic Gaussian mean before clipping; action_clipped matches SB3 predict for the checkpoint's [-1,1] action bounds.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
