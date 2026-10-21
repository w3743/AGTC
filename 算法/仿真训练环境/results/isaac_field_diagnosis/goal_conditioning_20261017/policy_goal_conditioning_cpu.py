#!/usr/bin/env python3
"""CPU-only counterfactual analysis of recorded native goal observations.

Repeated-observation predictions are not physical trajectories.  The script
does not instantiate Isaac Lab or step an environment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

MODELS = {
    "source_53248": ROOT / "results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip",
    "gaussian_69632": ROOT / "results/isaac_field_improvement/exploration_gsde_20261017/gaussian/field_policy_after.zip",
    "gsde_69632": ROOT / "results/isaac_field_improvement/exploration_gsde_20261017/gsde/field_policy_after.zip",
}
INPUT = HERE / "goal_observations.npz"
MANIFEST = HERE / "manifest.json"
OUTPUT = HERE / "policy_goal_conditioning_analysis.json"
CHECKPOINT_STEPS = {"source_53248": 53248, "gaussian_69632": 69632, "gsde_69632": 69632}
BRANCHES = ("lidar", "depth", "detections", "state")


def read_observations() -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "captured" or not manifest.get("native_capture_only"):
        raise RuntimeError("native observation capture is not marked complete")
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if data["angles_deg"].shape != (7,) or data["obs_state"].shape != (7, 44):
        raise ValueError("unexpected goal observation dimensions")
    # The policy comparison is valid only if the recorded native capture varies
    # the two goal channels while holding all other sensor/state inputs fixed.
    for name in ("lidar", "depth", "detections"):
        values = data[f"obs_{name}"]
        if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
            raise ValueError(f"native {name} observations differ across goal angles")
    state = data["obs_state"]
    mask = np.ones(state.shape[1], dtype=bool)
    mask[2:4] = False
    if not np.array_equal(state[:, mask], np.broadcast_to(state[0, mask], state[:, mask].shape)):
        raise ValueError("native state channels outside state[2:4] differ across goal angles")
    if not np.allclose(state[:, 2:4], data["state_goal_xy"], atol=1e-7, rtol=0):
        raise ValueError("state_goal_xy does not match the captured state goal channels")
    return data, manifest


def common_observation(data: dict[str, np.ndarray], angle_index: int) -> dict[str, np.ndarray]:
    observation = {
        name: np.asarray(data[f"obs_{name}"][0], dtype=np.float32)[None, ...].copy()
        for name in BRANCHES
    }
    observation["state"][0, 2:4] = np.asarray(data["state_goal_xy"][angle_index], dtype=np.float32)
    return observation


def repeated_prediction(model: RecurrentPPO, observation: dict[str, np.ndarray]) -> dict[str, list[float]]:
    recurrent_state = None
    episode_start = np.ones(1, dtype=bool)
    selected = {}
    for step in range(1, 65):
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.zeros(1, dtype=bool)
        if step in (1, 20, 64):
            selected[f"step_{step}"] = np.asarray(action[0], dtype=np.float64).tolist()
    return selected


def distribution_mean_jacobian(model: RecurrentPPO, observation: dict[str, np.ndarray]):
    """Differentiate the ordinary recurrent policy distribution path."""
    policy = model.policy
    device = policy.device
    obs_tensor = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in observation.items()
    }
    state = obs_tensor["state"].clone().detach().requires_grad_(True)
    obs_tensor["state"] = state
    actor_lstm = policy.lstm_actor
    h = torch.zeros((actor_lstm.num_layers, 1, actor_lstm.hidden_size), device=device)
    c = torch.zeros_like(h)
    episode_starts = torch.ones((1,), dtype=torch.float32, device=device)
    distribution, _ = policy.get_distribution(obs_tensor, (h, c), episode_starts)
    mean = distribution.distribution.mean[0]
    rows = []
    for action_index in range(int(mean.shape[0])):
        gradient = torch.autograd.grad(mean[action_index], state, retain_graph=True)[0]
        rows.append(gradient[0, 2:4].detach().cpu().numpy().astype(np.float64))
    return np.stack(rows), mean.detach().cpu().numpy().astype(np.float64)


def run_model(name: str, checkpoint: Path, data: dict[str, np.ndarray]) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = RecurrentPPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)
    encoder = model.policy.features_extractor
    goal_gain = float(getattr(encoder, "goal_gain", float("nan")))
    angle_results = []
    jacobians = []
    first_jacobian = None
    first_mean = None
    for index, angle in enumerate(data["angles_deg"].tolist()):
        observation = common_observation(data, index)
        actions = repeated_prediction(model, observation)
        jacobian, mean = distribution_mean_jacobian(model, observation)
        angle_results.append({
            "angle_deg": float(angle),
            "raw_state_goal_xy": np.asarray(data["state_goal_xy"][index], dtype=np.float64).tolist(),
            "deterministic_action": actions,
            "distribution_mean_at_reset": mean.tolist(),
        })
        jacobians.append(jacobian)
        if first_jacobian is None:
            first_jacobian = jacobian
            first_mean = mean

    action_matrix = np.asarray([row["deterministic_action"]["step_1"] for row in angle_results])
    jacobian_array = np.asarray(jacobians)
    action_steer = action_matrix[:, 1]
    mean_delta = np.max(np.abs(action_matrix[:, None, :] - action_matrix[None, :, :]), axis=(0, 1))

    # Hook the actor's encoder once to record the actual state presented to the
    # proprioception branch after SensorEncoder's goal scaling.
    pi_encoder = getattr(model.policy, "pi_features_extractor", encoder)
    captured = []
    handle = pi_encoder.proprioception.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().cpu().numpy().copy())
    )
    try:
        distribution_mean_jacobian(model, common_observation(data, 0))
    finally:
        handle.remove()
    scaled_goal = None if not captured else captured[-1][0, 2:4].astype(np.float64).tolist()
    expected_scaled_goal = (np.asarray(data["state_goal_xy"][0], dtype=np.float64) * goal_gain).tolist()

    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "num_timesteps": int(model.num_timesteps),
        "checkpoint_step_label": CHECKPOINT_STEPS[name],
        "use_sde": bool(model.use_sde),
        "goal_gain": goal_gain,
        "policy_squash_output": bool(model.policy.squash_output),
        "actor_hidden_size": int(model.policy.lstm_actor.hidden_size),
        "repeated_observation_predictions": angle_results,
        "step1_action_component_ranges_over_goal_angles": mean_delta.tolist(),
        "step1_differential_action_by_angle": action_steer.tolist(),
        "step1_differential_sign_by_angle": [
            "positive" if value > 1e-6 else "negative" if value < -1e-6 else "near_zero"
            for value in action_steer
        ],
        "jacobian_raw_state_goal_xy_to_distribution_mean": {
            "shape": list(jacobian_array.shape),
            "per_angle_action_by_goal_channel": jacobian_array.tolist(),
            "per_angle_frobenius_norm": np.linalg.norm(jacobian_array, axis=(1, 2)).tolist(),
            "differential_action_row_by_goal_channel": jacobian_array[:, 1, :].tolist(),
        },
        "encoder_goal_scaling_probe": {
            "proprioception_input_state_goal_xy": scaled_goal,
            "expected_raw_state_goal_xy_times_goal_gain": expected_scaled_goal,
            "matches": bool(scaled_goal is not None and np.allclose(scaled_goal, expected_scaled_goal, atol=1e-6)),
        },
        "interpretation_limits": [
            "The 64 calls reuse one reset observation; they are recurrent-policy counterfactuals, not environment rollouts.",
            "The Jacobian establishes local input-to-mean sensitivity only; it does not establish useful steering or navigation skill.",
            "Actions are normalized policy outputs; physical yaw response is not evaluated here.",
        ],
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    data, manifest = read_observations()
    results = {
        name: run_model(name, checkpoint, data)
        for name, checkpoint in MODELS.items()
    }
    report = {
        "schema": "isaac-field-goal-conditioning-cpu-analysis-v1",
        "native_capture": str(INPUT.relative_to(ROOT)),
        "capture_manifest": str(MANIFEST.relative_to(ROOT)),
        "capture_seed": manifest["environment"]["episode_reset_seed"],
        "angles_deg": np.asarray(data["angles_deg"]).tolist(),
        "observation_invariance": {
            name: manifest["non_goal_observation_invariance"][name]["max_abs_diff_from_angle_-90"]
            for name in ("lidar", "depth", "detections", "state")
        },
        "models": results,
        "conclusion": (
            "The captured input pipeline is varied only through state[2:4] and its geometric encoding checks pass. "
            "Model response/Jacobian results measure goal conditioning at policy input, not closed-loop competence."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "models": {
            name: {
                "steps": value["num_timesteps"],
                "use_sde": value["use_sde"],
                "goal_gain": value["goal_gain"],
                "differential_action": value["step1_differential_action_by_angle"],
                "differential_jacobian": value["jacobian_raw_state_goal_xy_to_distribution_mean"]["differential_action_row_by_goal_channel"],
            }
            for name, value in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
