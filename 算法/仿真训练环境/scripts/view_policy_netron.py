"""Export a CPU TorchScript view of an SB3 recurrent field policy for Netron."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
import warnings
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_field.action_contract import action_config
from isaac_field.policy import _validate_checkpoint_action_metadata, _validate_checkpoint_encoder_metadata

SCHEMA_VERSION = 3
INPUT_NAMES = ("lidar", "depth", "detections", "state", "feedback", "actor_h", "actor_c", "critic_h", "critic_c")
OUTPUT_NAMES = (
    "action_mean", "action_clipped", "action_std", "value",
    "actor_h_out", "actor_c_out", "critic_h_out", "critic_c_out",
)


class PolicyNetronWrapper(nn.Module):
    """Call the checkpoint's actual feature, recurrent, MLP, and output modules."""

    def __init__(self, policy):
        super().__init__()
        if not isinstance(policy.lstm_actor, nn.LSTM) or not isinstance(policy.lstm_critic, nn.LSTM):
            raise ValueError("export requires separate actor and critic nn.LSTM modules")
        if policy.shared_lstm or not policy.enable_critic_lstm:
            raise ValueError("export requires separate actor and critic LSTM state")
        if getattr(policy, "use_sde", False):
            raise ValueError("export currently expects the checkpoint's diagonal Gaussian action distribution")
        if bool(policy.squash_output):
            raise ValueError("export currently expects the checkpoint's unsquashed action distribution")
        if not isinstance(policy.action_space, gym.spaces.Box):
            raise ValueError("export requires a continuous Box action space")

        self.policy = policy
        self.register_buffer("action_low", torch.as_tensor(policy.action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(policy.action_space.high, dtype=torch.float32))

    def _features(self, observation):
        if self.policy.share_features_extractor:
            features = self.policy.extract_features(observation)
            return features, features
        actor_features = self.policy.extract_features(observation, self.policy.pi_features_extractor)
        critic_features = self.policy.extract_features(observation, self.policy.vf_features_extractor)
        return actor_features, critic_features

    def forward(self, lidar, depth, detections, state, feedback, actor_h, actor_c, critic_h, critic_c):
        observation = {"lidar": lidar, "depth": depth, "detections": detections, "state": state, "feedback": feedback}
        actor_features, critic_features = self._features(observation)
        actor_sequence, (next_actor_h, next_actor_c) = self.policy.lstm_actor(
            actor_features.unsqueeze(0), (actor_h, actor_c)
        )
        critic_sequence, (next_critic_h, next_critic_c) = self.policy.lstm_critic(
            critic_features.unsqueeze(0), (critic_h, critic_c)
        )
        actor_latent = self.policy.mlp_extractor.forward_actor(actor_sequence[0])
        critic_latent = self.policy.mlp_extractor.forward_critic(critic_sequence[0])
        mean = self.policy.action_net(actor_latent)
        action_std = self.policy.log_std.exp().expand_as(mean)
        clipped = torch.maximum(torch.minimum(mean, self.action_high), self.action_low)
        value = self.policy.value_net(critic_latent)
        return (
            mean, clipped, action_std, value,
            next_actor_h, next_actor_c, next_critic_h, next_critic_c,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_observation(rng):
    observation = {
        "lidar": rng.uniform(0.0, 1.0, size=(1, 1, 400)).astype(np.float32),
        "depth": rng.uniform(0.0, 1.0, size=(1, 1, 160, 88)).astype(np.float32),
        "detections": np.zeros((1, 16, 8), dtype=np.float32),
        "state": rng.uniform(-1.0, 1.0, size=(1, 33)).astype(np.float32),
        "feedback": rng.uniform(-1.0, 1.0, size=(1, 37)).astype(np.float32),
    }
    candidates = [0, 3, 7, 11, 15]
    observation["detections"][:, candidates, :] = rng.uniform(0.05, 0.9, size=(1, len(candidates), 8))
    observation["detections"][:, candidates, 4] = rng.integers(0, 2, size=(1, len(candidates)))
    observation["detections"][:, candidates, 7] = 1.0
    observation["state"][:, 0] = 1.0
    return observation


def _initial_state(rng, policy):
    shape = (policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
    return tuple(rng.normal(0.0, 0.2, size=shape).astype(np.float32) for _ in range(4))


def _torch_inputs(observation, states):
    sensors = tuple(torch.from_numpy(observation[key]) for key in ("lidar", "depth", "detections", "state", "feedback"))
    recurrent = tuple(torch.from_numpy(np.asarray(value).copy()) for value in states)
    return sensors + recurrent


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def _compare_runtime(model, policy, wrapper, traced, seed=20261018):
    rng = np.random.default_rng(seed)
    maxima = {name: 0.0 for name in (
        "torchscript_vs_wrapper", "mean_vs_policy_distribution", "std_vs_policy_distribution",
        "clip_vs_predict", "actor_h_vs_predict", "actor_c_vs_predict", "value_vs_predict_values",
        "critic_h_vs_lstm", "critic_c_vs_lstm",
    )}
    cases = []
    with torch.no_grad():
        for case_index in range(3):
            observation = _make_observation(rng)
            actor_h, actor_c, critic_h, critic_c = _initial_state(rng, policy)
            for step_index in range(3):
                states = (actor_h, actor_c, critic_h, critic_c)
                inputs = _torch_inputs(observation, states)
                actual = wrapper(*inputs)
                scripted = traced(*inputs)
                errors = {}
                for name, actual_value, scripted_value in zip(OUTPUT_NAMES, actual, scripted):
                    actual_np, scripted_np = _numpy(actual_value), _numpy(scripted_value)
                    _assert_close(actual_np, scripted_np)
                    error = float(np.max(np.abs(actual_np - scripted_np)))
                    maxima["torchscript_vs_wrapper"] = max(maxima["torchscript_vs_wrapper"], error)
                    errors[f"torchscript_vs_wrapper.{name}"] = error

                actor_state = (actor_h.copy(), actor_c.copy())
                action, next_actor = model.predict(
                    observation,
                    state=actor_state,
                    episode_start=np.array([False]),
                    deterministic=True,
                )
                tensor_obs, _ = policy.obs_to_tensor(observation)
                actor_state_t = (torch.from_numpy(actor_h), torch.from_numpy(actor_c))
                distribution, distribution_state = policy.get_distribution(
                    tensor_obs, actor_state_t, torch.zeros(1, dtype=torch.float32)
                )
                compare_pairs = (
                    ("mean_vs_policy_distribution", actual[0], distribution.distribution.mean),
                    ("std_vs_policy_distribution", actual[2], distribution.distribution.stddev),
                    ("clip_vs_predict", actual[1], action),
                    ("actor_h_vs_predict", actual[4], next_actor[0]),
                    ("actor_c_vs_predict", actual[5], next_actor[1]),
                )
                for key, actual_value, expected_value in compare_pairs:
                    actual_np = _numpy(actual_value) if isinstance(actual_value, torch.Tensor) else np.asarray(actual_value)
                    expected_np = _numpy(expected_value) if isinstance(expected_value, torch.Tensor) else np.asarray(expected_value)
                    _assert_close(actual_np, expected_np)
                    error = float(np.max(np.abs(actual_np - expected_np)))
                    maxima[key] = max(maxima[key], error)
                    errors[key] = error

                critic_state_t = (torch.from_numpy(critic_h), torch.from_numpy(critic_c))
                starts = torch.zeros(1, dtype=torch.float32)
                predicted_value = policy.predict_values(tensor_obs, critic_state_t, starts)
                if policy.share_features_extractor:
                    critic_features = policy.extract_features(tensor_obs)
                else:
                    critic_features = policy.extract_features(tensor_obs, policy.vf_features_extractor)
                _critic_latent, next_critic = policy._process_sequence(
                    critic_features, critic_state_t, starts, policy.lstm_critic
                )
                for key, actual_value, expected_value in (
                    ("value_vs_predict_values", actual[3], predicted_value),
                    ("critic_h_vs_lstm", actual[6], next_critic[0]),
                    ("critic_c_vs_lstm", actual[7], next_critic[1]),
                ):
                    actual_np, expected_np = _numpy(actual_value), _numpy(expected_value)
                    _assert_close(actual_np, expected_np)
                    error = float(np.max(np.abs(actual_np - expected_np)))
                    maxima[key] = max(maxima[key], error)
                    errors[key] = error

                _assert_close(_numpy(distribution_state[0]), next_actor[0])
                _assert_close(_numpy(distribution_state[1]), next_actor[1])
                cases.append({"case": case_index, "step": step_index, "max_abs_errors": errors})
                actor_h, actor_c = next_actor[0].copy(), next_actor[1].copy()
                critic_h, critic_c = _numpy(next_critic[0]), _numpy(next_critic[1])
                observation = _make_observation(rng)
    return maxima, cases


def _import_netron():
    isolated = ROOT / ".cache/netron"
    spec = importlib.util.find_spec("netron")
    origin = None if spec is None else spec.origin
    if isolated.is_dir() and (origin is None or not Path(origin).resolve().is_relative_to(isolated.resolve())):
        sys.path.insert(0, str(isolated))
    try:
        import netron
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Netron 9.3.2 is unavailable. Install it into the isolated target with: "
            "uv pip install --target .cache/netron netron"
        ) from exc
    if getattr(netron, "__version__", None) != "9.3.2":
        raise SystemExit(f"Expected isolated Netron 9.3.2, found {getattr(netron, '__version__', 'unknown')}")
    return netron


def _port_is_available(host: str, port: int) -> bool:
    if port == 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _default_output(checkpoint: Path, checkpoint_sha: str) -> Path:
    return ROOT / "results/isaac_field_viewer/netron_exports" / f"{checkpoint.stem}_{checkpoint_sha[:8]}.pt"


def _load_or_export(checkpoint: Path, output: Path, checkpoint_sha: str):
    _validate_checkpoint_action_metadata(checkpoint)
    _validate_checkpoint_encoder_metadata(checkpoint)
    metadata_path = output.with_suffix(".json")
    if output.exists():
        if not metadata_path.is_file():
            raise FileExistsError(f"Refusing to overwrite unverified artifact without sidecar metadata: {output}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("checkpoint_sha256") != checkpoint_sha:
            raise FileExistsError(f"Existing artifact metadata does not match this export request: {output}")
        if metadata.get("action_config") != action_config():
            raise FileExistsError(f"Existing artifact metadata has no matching action contract: {output}")
        if metadata.get("artifact_sha256") != _sha256(output):
            raise FileExistsError(f"Existing artifact hash does not match its sidecar metadata: {output}")
        torch.jit.load(str(output), map_location="cpu")
        return metadata, True

    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    model = RecurrentPPO.load(str(checkpoint), device="cpu")
    policy = model.policy.eval()
    if policy.device.type != "cpu":
        raise RuntimeError("policy did not load onto CPU")
    wrapper = PolicyNetronWrapper(policy).eval()
    cache_modes = [(module, module.cache_inference) for module in policy.modules() if hasattr(module, "cache_inference")]
    for module, _mode in cache_modes:
        module.cache_inference = False

    rng = np.random.default_rng(20261018)
    initial_cases = []
    for _ in range(3):
        observation = _make_observation(rng)
        states = _initial_state(rng, policy)
        initial_cases.append((observation, states))
    example = _torch_inputs(*initial_cases[0])
    check_inputs = [_torch_inputs(*case) for case in initial_cases[1:]]
    trace_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            traced = torch.jit.trace(
                wrapper,
                example,
                check_trace=True,
                check_inputs=check_inputs,
                check_tolerance=2e-5,
                strict=True,
            ).eval()
            trace_warnings = [
                {"warning": message, "count": count}
                for message, count in Counter(
                    f"{item.category.__name__}: {item.message}" for item in caught
                ).items()
            ]

        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(output.name + ".tmp")
        temporary_metadata = output.with_suffix(".json.tmp")
        if temporary_output.exists() or temporary_metadata.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete export files beside {output}")
        traced.save(str(temporary_output))
        loaded = torch.jit.load(str(temporary_output), map_location="cpu").eval()
        max_errors, comparisons = _compare_runtime(model, policy, wrapper, loaded)
        artifact_sha = _sha256(temporary_output)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "action_config": action_config(),
            "format": "TorchScript trace (.pt)",
            "device": "cpu",
            "torch_version": torch.__version__,
            "policy_hidden_size": policy.lstm_actor.hidden_size,
            "policy_num_layers": policy.lstm_actor.num_layers,
            "input_names": list(INPUT_NAMES),
            "output_names_in_order": list(OUTPUT_NAMES),
            "episode_start_input": "omitted; caller clears actor and critic h/c at episode start",
            "wrapper_uses": [
                "policy.features_extractor (or actual pi/vf feature extractors)",
                "policy.lstm_actor and policy.lstm_critic",
                "policy.mlp_extractor.forward_actor and forward_critic",
                "policy.action_net, policy.log_std, policy.value_net",
            ],
            "trace_check": "passed with 3 distinct nonzero actor/critic h/c samples",
            "trace_warnings": trace_warnings,
            "runtime_comparison": {
                "sequence_count": 3,
                "steps_per_sequence": 3,
                "max_abs_errors": max_errors,
                "tolerance": {"rtol": 2e-5, "atol": 2e-6},
                "per_step": comparisons,
            },
            "artifact_bytes": temporary_output.stat().st_size,
            "artifact_sha256": artifact_sha,
        }
        temporary_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        if output.exists() or output.with_suffix(".json").exists():
            raise FileExistsError(f"Output appeared during export; preserved existing files: {output}")
        temporary_output.replace(output)
        temporary_metadata.replace(output.with_suffix(".json"))
        return metadata, False
    finally:
        for module, mode in cache_modes:
            module.cache_inference = mode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="SB3-Contrib RecurrentPPO checkpoint .zip")
    parser.add_argument("--output", type=Path, help="TorchScript output path; default uses checkpoint stem and SHA-8")
    parser.add_argument("--export-only", action="store_true", help="Export and validate without starting Netron")
    parser.add_argument("--port", type=int, default=8081, help="Netron loopback port (default: 8081; 0 chooses a free port)")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    output = args.output.expanduser().resolve() if args.output else _default_output(checkpoint, checkpoint_sha).resolve()
    if output == checkpoint:
        parser.error("output must not overwrite the checkpoint")
    if output.suffix.lower() != ".pt":
        parser.error("output must use the .pt extension")
    if not (0 <= args.port <= 65535):
        parser.error("port must be between 0 and 65535")

    metadata, reused = _load_or_export(checkpoint, output, checkpoint_sha)
    print(f"TorchScript: {output}")
    print(f"Metadata: {output.with_suffix('.json')}")
    print(f"Checkpoint SHA-256: {checkpoint_sha}")
    print(f"Artifact SHA-256: {metadata['artifact_sha256']}")
    print(f"Trace check: {'reused verified artifact' if reused else metadata['trace_check']}")
    print(f"Max comparison errors: {metadata['runtime_comparison']['max_abs_errors']}")

    if args.export_only:
        return 0

    if not _port_is_available("127.0.0.1", args.port):
        raise SystemExit(f"127.0.0.1:{args.port} is already in use; choose another --port")
    netron = _import_netron()
    address = netron.start(str(output), address=("127.0.0.1", args.port), browse=False)
    print(f"Netron {netron.__version__} serving at http://{address[0]}:{address[1]} (browser not opened)", flush=True)
    netron.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
