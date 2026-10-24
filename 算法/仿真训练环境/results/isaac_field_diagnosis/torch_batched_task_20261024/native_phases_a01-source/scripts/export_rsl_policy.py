"""Export a native Isaac Sim PhysX field RSL policy as stateful TorchScript."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _ClippedFieldPolicy(nn.Module):
    """Keep the live inference action range and preserve an explicit reset."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        detections: torch.Tensor,
        state: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy(lidar, depth, detections, state, feedback).clamp(-1.0, 1.0)

    @torch.jit.export
    def reset(self) -> None:
        self.policy.reset()


def _dummy_observation(rng: np.random.Generator) -> dict[str, np.ndarray]:
    observation = {
        "lidar": rng.uniform(0.0, 1.0, size=(1, 1, 400)).astype(np.float32),
        "depth": rng.uniform(0.0, 1.0, size=(1, 1, 160, 88)).astype(np.float32),
        "detections": np.zeros((1, 16, 8), dtype=np.float32),
        "state": rng.uniform(-1.0, 1.0, size=(1, 33)).astype(np.float32),
        "feedback": rng.uniform(-1.0, 1.0, size=(1, 37)).astype(np.float32),
    }
    present = (0, 3, 7, 11, 15)
    observation["detections"][:, present, :] = rng.uniform(
        0.05, 0.9, size=(1, len(present), 8)
    )
    observation["detections"][:, present, 4] = rng.integers(0, 2, size=(1, len(present)))
    observation["detections"][:, present, 7] = 1.0
    return observation


def _tensor_inputs(observation: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(observation[key]) for key in (
        "lidar", "depth", "detections", "state", "feedback",
    ))


def _verify_sequence(model, scripted, *, seed: int = 20261024) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    # RslPredictionModel takes recurrent state explicitly; only the exported
    # stateful TorchScript wrapper owns hidden state that must be reset here.
    scripted.reset()
    recurrent_state = None
    max_abs_error = 0.0
    for step in range(4):
        observation = _dummy_observation(rng)
        expected, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=np.asarray([step == 0]),
            deterministic=True,
        )
        with torch.inference_mode():
            actual = scripted(*_tensor_inputs(observation)).cpu().numpy()
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=2e-5, atol=2e-6):
            raise RuntimeError(f"TorchScript action differs from RSL inference by {error:g}")
        max_abs_error = max(max_abs_error, error)
    return {"sequence_steps": 4, "max_abs_error": max_abs_error}


def export(checkpoint: Path, output: Path) -> dict:
    from isaac_field.policy_contract import BACKEND_IDENTITY
    from isaac_field.rsl_rl_inference import RslPredictionModel
    from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("RSL export requires a native .pt checkpoint; SB3 .zip is legacy-only")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.resolve() == checkpoint.resolve():
        raise ValueError("export output must not overwrite the source checkpoint")
    if output.suffix.lower() != ".pt":
        raise ValueError("TorchScript export output must use .pt")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite an existing export: {output}")

    source_sha = _sha256(checkpoint)
    payload = load_rsl_checkpoint_payload(checkpoint, map_location="cpu")
    contract = payload["field_policy_contract"]
    model = RslPredictionModel(
        checkpoint,
        device="cpu",
        input_schema=contract["input_schema"],
        expected_observation_config=contract["observation_config"],
        expected_backend_identity=BACKEND_IDENTITY,
    )
    eager = _ClippedFieldPolicy(model.actor.as_jit()).eval()
    scripted = torch.jit.script(eager).eval()
    verification = _verify_sequence(model, scripted)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary_metadata = output.with_suffix(".json.tmp")
    if temporary.exists() or temporary_metadata.exists():
        raise FileExistsError(f"refusing to overwrite incomplete export files beside {output}")
    # TorchScript's C++ path handling can misread non-ASCII Windows paths even
    # after Python created their directories. Use Python-managed file handles.
    with temporary.open("wb") as stream:
        torch.jit.save(scripted, stream)
    with temporary.open("rb") as stream:
        loaded = torch.jit.load(stream, map_location="cpu").eval()
    loaded_verification = _verify_sequence(model, loaded, seed=20261025)
    artifact_sha = _sha256(temporary)
    metadata = {
        "schema": "isaac-field-rsl-torchscript-export-v1",
        **BACKEND_IDENTITY,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_sha,
        "field_policy_contract": {
            key: contract[key] for key in (
                "format", "version", "input_schema", "lstm", "action_config",
                "observation_config", "simulator", "physics_backend", "learning_backend",
            ) if key in contract
        },
        "format": "stateful TorchScript .pt",
        "inputs": ["lidar", "depth", "detections", "state", "feedback"],
        "output": "four clipped normalized direct-track-speed-v1 actions",
        "reset_method": "reset() clears the single deployed environment's LSTM state",
        "torch_version": torch.__version__,
        "verification": {
            "tolerance": {"rtol": 2e-5, "atol": 2e-6},
            "eager_vs_scripted": verification,
            "saved_vs_scripted": loaded_verification,
        },
        "artifact_bytes": temporary.stat().st_size,
        "artifact_sha256": artifact_sha,
    }
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if _sha256(checkpoint) != source_sha:
        raise RuntimeError("source checkpoint changed during export")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"output appeared during export; preserved existing files: {output}")
    temporary.replace(output)
    temporary_metadata.replace(output.with_suffix(".json"))
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="native Isaac Sim PhysX RSL .pt")
    parser.add_argument("--output", type=Path, help="TorchScript output path; defaults to a SHA-named export")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    source_sha = _sha256(checkpoint)
    output = args.output.expanduser().resolve() if args.output else (
        ROOT / "results/isaac_field_exports" / f"{checkpoint.stem}_{source_sha[:8]}_rsl.pt"
    ).resolve()
    try:
        metadata = export(checkpoint, output)
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"TorchScript: {output}")
    print(f"Metadata: {output.with_suffix('.json')}")
    print(f"Source SHA-256: {source_sha}")
    print(f"Artifact SHA-256: {metadata['artifact_sha256']}")
    print(f"Verification: {metadata['verification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
