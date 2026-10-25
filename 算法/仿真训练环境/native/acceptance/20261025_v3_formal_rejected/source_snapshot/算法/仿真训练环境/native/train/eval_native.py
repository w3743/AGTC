"""Native-stack CPU/GPU policy evaluation: 60 s episodes per seed.

Success criterion follows the reference task semantics (`tensor_task.fast_stop_v2` /
`FieldTask`): final position within 0.4 m, world-XY speed < 0.05 m/s, full body angular
speed < 0.1 rad/s, held continuously for 0.5 s. Episodes end on success, tilt/overturn,
or the 1200-step (60 s @ 20 Hz) limit.

The evaluator disables env auto-reset (native_env(auto_reset=False)) so the terminal
state can be inspected; env reward/termination internals are ignored and the metrics
are recomputed from simulator truth. Output schema mirrors the existing
`evaluate_rsl_curriculum` summary/rows fields (seed/success/clean_success/reason/steps/
final_distance/goal_stop) and marks this run as native-domain with the registered
domain gap; `capability_metrics_allowed` stays false (protocol decision left to Lead).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(os.path.dirname(HERE), "proto")
if PROTO not in sys.path:
    sys.path.insert(0, PROTO)

from native_env import NativeFieldEnv  # noqa: E402
from policy_m0 import ActorCritic  # noqa: E402
from train_m0 import to_torch  # noqa: E402
from contract_identity import (validate_checkpoint_contract,
                               validate_checkpoint_preflight)  # noqa: E402

BANNER = "NATIVE_EVAL_CPU_SMOKE"

SUCCESS_POSITION_M = 0.4
SUCCESS_XY_SPEED = 0.05
SUCCESS_ANGULAR_SPEED = 0.1
SUCCESS_HOLD_S = 0.5
FLIP_UP_Z = math.cos(1.2)


def parse_seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in text.split(",") if v.strip()]


def infer_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def load_checkpoint_payload(checkpoint: str):
    """Load tensor-only checkpoint data on CPU before allocating a simulator."""
    if not checkpoint:
        return None
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("native checkpoint payload must be a dictionary")
    return payload


def load_policy(checkpoint: str, observation_space, payload=None):
    policy = ActorCritic(observation_space, action_dim=4)
    meta = {"checkpoint": None, "policy_loaded": False, "policy_type": "mlp",
            "note": "no checkpoint supplied: random policy, numbers are a harness smoke only"}
    if checkpoint:
        if payload is None:
            payload = load_checkpoint_payload(checkpoint)
        policy_type = payload.get("policy_type", "mlp")
        if policy_type == "lstm":
            from policy_lstm import LSTMActorCritic
            policy = LSTMActorCritic(observation_space, action_dim=4,
                                     hidden_size=int(payload.get("hidden_size", 512)))
        state = payload["policy"] if "policy" in payload else payload
        policy.load_state_dict(state)
        meta = {"checkpoint": checkpoint, "policy_loaded": True, "policy_type": policy_type}
    policy.eval()
    return policy, meta


def evaluate_seed(env, policy, seed: int, max_steps: int, device: torch.device,
                  deterministic: bool, zero_action: bool = False):
    obs, _ = env.reset(seed=seed)
    is_lstm = hasattr(policy, "init_hidden")
    hidden = policy.init_hidden(1, device) if is_lstm else None
    hold_s = 0.0
    steps = 0
    reason = "time_limit"
    success = False
    clean_success = False
    flipped_ever = False
    final = {}
    for step in range(max_steps):
        obs_t = to_torch(obs, device)
        with torch.no_grad():
            if is_lstm:
                mean, _value, hidden = policy.forward(obs_t, hidden)
            else:
                mean, _value = policy(obs_t)
        action = np.zeros((1, 4), np.float32) if zero_action else mean.cpu().numpy()
        try:
            obs, _reward, terminated, truncated, info = env.step(action)
        except Exception as exc:  # sensors/physics must never crash the whole evaluation
            steps = step + 1
            return {"seed": int(seed), "success": False, "clean_success": False,
                    "reason": "runtime_error", "steps": int(steps),
                    "final_distance": float("nan"), "goal_stop": False, "hold_s": 0.0,
                    "final_xy_speed": float("nan"), "final_angular_speed": float("nan"),
                    "flipped": bool(flipped_ever), "error": f"{type(exc).__name__}: {exc}"}
        if bool(np.any(info.get("nonfinite", False))):
            steps = step + 1
            read = env._last_read
            pos = read["pos"][0]
            return {"seed": int(seed), "success": False, "clean_success": False,
                    "reason": "non_finite_state", "steps": int(steps),
                    "final_distance": float(np.linalg.norm(env.goal_xy[0] - pos[:2])),
                    "goal_stop": False, "hold_s": 0.0,
                    "final_xy_speed": float("nan"), "final_angular_speed": float("nan"),
                    "flipped": bool(flipped_ever)}
        if not np.isfinite(_reward).all():
            steps = step + 1
            return {"seed": int(seed), "success": False, "clean_success": False,
                    "reason": "non_finite_reward", "steps": int(steps),
                    "final_distance": float("nan"), "goal_stop": False, "hold_s": 0.0,
                    "final_xy_speed": float("nan"), "final_angular_speed": float("nan"),
                    "flipped": bool(flipped_ever)}
        if is_lstm:
            done_t = torch.tensor([bool(terminated[0]) or bool(truncated[0])], device=device)
            hidden = policy.reset_hidden(hidden, done_t)
        steps = step + 1
        read = env._last_read
        pos = read["pos"][0]
        distance = float(np.linalg.norm(env.goal_xy[0] - pos[:2]))
        lin_speed = float(np.linalg.norm(read["lin_vel"][0, :2]))
        ang_speed = float(np.linalg.norm(read["ang_vel"][0]))
        up_z = 1.0 - 2.0 * (read["quat_wxyz"][0, 1] ** 2 + read["quat_wxyz"][0, 2] ** 2)
        flipped = bool(up_z < FLIP_UP_Z)
        flipped_ever = flipped_ever or flipped
        stopped = (lin_speed < SUCCESS_XY_SPEED) and (ang_speed < SUCCESS_ANGULAR_SPEED)
        within = distance <= SUCCESS_POSITION_M
        hold_s = hold_s + env.policy_dt if (within and stopped) else 0.0
        if hold_s >= SUCCESS_HOLD_S - 1e-9:
            success = True
            reason = "goal_stop"
            final = dict(distance=distance, lin_speed=lin_speed, ang_speed=ang_speed,
                         hold_s=hold_s, flipped=flipped)
            break
        if flipped:
            reason = "outside_or_overturned"
            final = dict(distance=distance, lin_speed=lin_speed, ang_speed=ang_speed,
                         hold_s=hold_s, flipped=flipped)
            break
        if truncated:
            reason = "time_limit"
            final = dict(distance=distance, lin_speed=lin_speed, ang_speed=ang_speed,
                         hold_s=hold_s, flipped=flipped)
            break
        final = dict(distance=distance, lin_speed=lin_speed, ang_speed=ang_speed,
                     hold_s=hold_s, flipped=flipped)
    clean_success = bool(success and not flipped_ever)
    return {
        "seed": int(seed),
        "success": bool(success),
        "clean_success": clean_success,
        "reason": reason,
        "steps": int(steps),
        "final_distance": round(float(final.get("distance", float("nan"))), 4),
        "goal_stop": bool(success),
        "hold_s": round(float(final.get("hold_s", 0.0)), 3),
        "final_xy_speed": round(float(final.get("lin_speed", float("nan"))), 4),
        "final_angular_speed": round(float(final.get("ang_speed", float("nan"))), 4),
        "flipped": bool(flipped_ever),
    }


def main():
    parser = argparse.ArgumentParser(description="native-stack evaluation (60 s fast_stop episodes)")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seeds", type=str, default="87010,87011")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--physics", type=str, default="mujoco_cpu", choices=("mujoco_cpu", "newton_xpbd"))
    parser.add_argument("--task-backend", type=str, default=None,
                        choices=("simplified", "fast_stop_v2"),
                        help="task implementation; defaults to simplified for compatibility")
    parser.add_argument("--allow-legacy-checkpoint-contract", action="store_true", default=False,
                        help="allow an unversioned checkpoint only for an explicitly selected task backend")
    parser.add_argument("--lidar", type=str, default="numpy", choices=("numpy", "warp"))
    parser.add_argument("--feedback", type=str, default="vqf", choices=("direct", "vqf"))
    parser.add_argument("--depth", type=str, default="constant", choices=("constant", "analytic"))
    parser.add_argument("--detections", type=str, default="constant", choices=("constant", "analytic"))
    parser.add_argument("--detections-warp", action="store_true", default=False)
    parser.add_argument("--sim-dt", type=float, default=1e-3,
                        help="physics substep dt; must satisfy sim_dt*decimation == 0.05 (match training)")
    parser.add_argument("--decimation", type=int, default=50)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--zero-action", action="store_true", default=False,
                        help="harness-validation mode: ignore the policy and command zero actions")
    args = parser.parse_args()
    task_backend = args.task_backend or "simplified"
    task_backend_explicit = args.task_backend is not None
    if abs(args.sim_dt * args.decimation - 0.05) > 1e-9:
        parser.error(f"--sim-dt*--decimation must equal 0.05 s, got {args.sim_dt * args.decimation}")

    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    checkpoint_contract_status = "no_checkpoint"
    if checkpoint_payload is not None:
        physics_device = ("cuda:0" if args.physics == "newton_xpbd" and args.device == "cuda"
                          else "cpu")
        checkpoint_contract_status = validate_checkpoint_preflight(
            checkpoint_payload,
            task_backend=task_backend,
            physics=args.physics,
            device=physics_device,
            sim_dt=args.sim_dt,
            decimation=args.decimation,
            allow_legacy=args.allow_legacy_checkpoint_contract,
            task_backend_explicit=task_backend_explicit,
        )

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch.set_num_threads(8)
    seeds = parse_seeds(args.seeds)
    out_dir = args.out or os.path.join(HERE, "runs", "eval_native_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{BANNER}: {len(seeds)} scenes, {args.max_steps} steps, device={args.device}")

    env = NativeFieldEnv(num_envs=1, seed=seeds[0], physics=args.physics,
                         task_backend=task_backend,
                         dt=args.sim_dt, decimation=args.decimation,
                         lidar_backend=args.lidar, device="cuda:0" if args.device == "cuda" else "cpu",
                         feedback_backend=args.feedback,
                         depth_backend=args.depth,
                         depth_device="cuda:0" if args.device == "cuda" else "cpu",
                         camera_backend=("analytic_warp" if args.detections_warp else args.detections),
                         camera_device="cuda:0" if args.device == "cuda" else "cpu",
                         auto_reset=False)
    from train_wsl import _contract_for
    contract = _contract_for(env, out_dir, phase="native_eval")
    if checkpoint_payload is not None:
        checkpoint_contract_status = validate_checkpoint_contract(
            checkpoint_payload, contract["task_identity_sha256"],
            allow_legacy=args.allow_legacy_checkpoint_contract,
        )
    policy, policy_meta = load_policy(args.checkpoint, env.single_observation_space,
                                      payload=checkpoint_payload)
    policy.to(device)

    t0 = time.perf_counter()
    rows = []
    for seed in seeds:
        rows.append(evaluate_seed(env, policy, seed, args.max_steps, device, deterministic=True,
                                  zero_action=args.zero_action))
    elapsed = time.perf_counter() - t0

    successes = sum(r["success"] for r in rows)
    reason_counts = {}
    for r in rows:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
    env_diagnostics = env.consume_nonfinite() if hasattr(env, "consume_nonfinite") else None
    result = {
        "schema_version": "native_eval_v1",
        "banner": BANNER,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_backend": task_backend,
        "task_identity_sha256": contract["task_identity_sha256"],
        "identity_schema_version": contract["identity_schema_version"],
        "checkpoint_contract_validation": checkpoint_contract_status,
        "native_domain": {
            "env": "proto/native_env.py (MuJoCo CPU / Newton XPBD)",
            "task_backend": task_backend,
            "physics": env._physics_identity(),
            "observation_backends": {k: contract["channels"][k].get("backend")
                                     for k in contract["channels"]},
            "note": "native WSL stack; not Isaac Sim / PhysX / RTX domain",
        },
        "domain_gap": {
            "depth": "analytic raycast vs Isaac RTX distance_to_image_plane (not pixel-compared)",
            "detections": "analytic leaf mask vs Isaac RTX instance segmentation (not pixel-compared)",
            "physics": "proxy MuJoCo/Newton model vs Isaac Lab PhysX; drive/COM fidelity gap documented in EQUIVALENCE_REPORT",
        },
        "capability_metrics_allowed": False,
        "protocol_note": "capability decisions are left to the Lead protocol; this run only reports facts",
        "env_diagnostics": env_diagnostics,
        "physics": {"dt_s": env.dt, "decimation": env.decimation, "policy_dt_s": env.policy_dt,
                    "experimental": bool(env.experimental_physics),
                    "note": ("5 ms / decimation 10 experimental configuration; keep separate from 1 ms/50 results"
                             if env.experimental_physics else "default 1 ms / decimation 50")},
        "success_criteria": {"position_m": SUCCESS_POSITION_M, "xy_speed_mps": SUCCESS_XY_SPEED,
                             "angular_speed_radps": SUCCESS_ANGULAR_SPEED, "hold_s": SUCCESS_HOLD_S,
                             "reasons": ["goal_stop", "time_limit", "outside_or_overturned"]},
        "policy": {**policy_meta, "observation_contract_sha256": contract["observation_contract_sha256"],
                   "task_identity_sha256": contract["task_identity_sha256"],
                   "identity_schema_version": contract["identity_schema_version"],
                   "checkpoint_contract_validation": checkpoint_contract_status,
                   "sha_scope": contract.get("sha_scope")},
        "config": {"seeds": seeds, "max_steps": args.max_steps, "device": args.device,
                   "physics": args.physics, "task_backend": task_backend,
                   "lidar": args.lidar, "feedback": args.feedback,
                   "depth": args.depth, "detections": args.detections,
                   "detections_warp": args.detections_warp, "zero_action": args.zero_action,
                   "sim_dt_s": args.sim_dt, "decimation": args.decimation,
                   "episode_length_s": env.episode_length_s, "policy_dt_s": env.policy_dt},
        "summary": {"scenes": len(rows), "successes": int(successes),
                    "success_rate": round(successes / max(1, len(rows)), 4),
                    "clean_successes": int(sum(r["clean_success"] for r in rows)),
                    "clean_success_rate": round(sum(r["clean_success"] for r in rows) / max(1, len(rows)), 4),
                    "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 1),
                    "mean_final_distance": round(float(np.mean([r["final_distance"] for r in rows])), 4),
                    "reasons": reason_counts,
                    "wall_s": round(elapsed, 2)},
        "rows": rows,
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("EVAL_SUMMARY " + json.dumps(result["summary"]))
    print("EVAL_ROWS " + json.dumps(rows))
    print("EVAL_OUT " + os.path.join(out_dir, "evaluation.json"))


if __name__ == "__main__":
    main()
