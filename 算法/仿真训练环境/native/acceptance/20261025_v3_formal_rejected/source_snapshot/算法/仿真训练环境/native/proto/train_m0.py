"""M0 throughput prototype: PPO training loop over the native MuJoCo CPU env.

Prints and records:
  * per-iteration JSONL: iter, collection_s, learning_s, wall_s, samples,
    samples_per_s, policy_steps_per_s, env_steps (physics), env_steps_per_s, rss_mb
  * observation_contract.json (real/placeholder channels; capability metrics disallowed)
  * identity.json (physics/obs/env identity for reproducibility)

The run banner starts with PROTOTYPE_THROUGHPUT_ONLY: no capability/闂傚倸鍊风欢锟犲礈濠靛纾块柤濮愬€愰崑?闂備浇顕ф绋匡耿闁秴纾婚柣妯荤ゴ閺€?conclusion may be drawn from this run.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from native_env import NativeFieldEnv  # noqa: E402
from policy_m0 import ActorCritic  # noqa: E402

BANNER = "PROTOTYPE_THROUGHPUT_ONLY"


def to_torch(obs: dict[str, np.ndarray], device="cpu") -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in obs.items()}


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    rewards = torch.stack(rewards) if isinstance(rewards, list) else rewards
    values = torch.stack(values) if isinstance(values, list) else values
    dones = torch.stack(dones) if isinstance(dones, list) else dones
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, obs_batch, actions, old_logp, advantages, returns,
               *, epochs: int = 4, minibatch: int = 256, clip: float = 0.2,
               value_coef: float = 0.5, entropy_coef: float = 0.0,
               max_grad_norm: float = 0.5):
    n = advantages.shape[0]
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    total = n
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch):
            idx = perm[start:start + minibatch]
            mb_obs = {k: v[idx] for k, v in obs_batch.items()}
            logp, entropy, value = policy.evaluate(mb_obs, actions[idx])
            ratio = torch.exp(logp - old_logp[idx])
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (value - returns[idx]).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
    return float(loss.detach().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--threads", type=int, default=8,
                    help="torch CPU threads (8 measured best: encoder conv + PPO backward)")
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"),
                    help="torch device for encoder/PPO (default cpu; cuda for the M3 GPU gate)")
    ap.add_argument("--physics", type=str, default="mujoco_cpu",
                    choices=("mujoco_cpu", "newton_xpbd"),
                    help="physics backend (newton_xpbd is the GPU backend; default CPU)")
    ap.add_argument("--lidar", type=str, default="numpy", choices=("numpy", "warp"),
                    help="lidar backend (warp = CUDA kernel when available)")
    ap.add_argument("--feedback", type=str, default="direct", choices=("direct", "vqf"),
                    help="feedback backend: direct analytic quaternion, or VQF via isaac_field SimFeedbackEstimator")
    ap.add_argument("--feedback-noise", action="store_true", default=True,
                    help="enable the reference estimator's documented sim sensor noise (vqf backend)")
    ap.add_argument("--feedback-no-noise", dest="feedback_noise", action="store_false",
                    help="disable sim sensor noise (used for equivalence verification)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if args.physics == "newton_xpbd" and args.device != "cuda":
        print("WARNING: newton_xpbd on CPU is a validation path, not the M3 GPU gate")
    device = torch.device(args.device)
    env_device = "cuda:0" if args.device == "cuda" else "cpu"
    if args.out:
        out_dir = args.out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(HERE, "runs", f"m0_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{BANNER}: native-stack M0 throughput prototype (no capability evaluation)")
    env = NativeFieldEnv(num_envs=args.num_envs, seed=args.seed, physics=args.physics,
                         lidar_backend=args.lidar, device=env_device,
                         feedback_backend=args.feedback, feedback_noise=args.feedback_noise)
    contract = env.observation_config()
    verification = {}
    verification_path = os.path.join(HERE, "m1_verification.json")
    if os.path.exists(verification_path):
        with open(verification_path, encoding="utf-8") as f:
            verification["m1"] = json.load(f)
    m2_verification_path = os.path.join(HERE, "m2_verification.json")
    if os.path.exists(m2_verification_path):
        with open(m2_verification_path, encoding="utf-8") as f:
            verification["m2"] = json.load(f)
    if verification:
        contract["verification"] = verification
    contract.update({"run_id": os.path.basename(out_dir), "phase": "prototype_throughput_only",
                     "capability_metrics_allowed": False,
                     "provider_contract_sha256": env.provider.contract_sha256(env.policy_dt),
                     "note": "depth/detections placeholders -> throughput only; see verification.m1/m2"})
    # SHA over the FULL contract (including the embedded M1/M2 verification evidence)
    import hashlib
    contract["observation_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    identity = {"banner": BANNER, "seed": args.seed, "num_envs": args.num_envs,
                "num_steps": args.num_steps, "iterations": args.iterations,
                "env": env.identity(), "torch": torch.__version__,
                "numpy": np.__version__, "python": sys.version.split()[0],
                "device": args.device, "physics": args.physics, "lidar": args.lidar}
    with open(os.path.join(out_dir, "observation_contract.json"), "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "identity.json"), "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2, ensure_ascii=False)
    print("IDENTITY " + json.dumps({k: identity[k] for k in ("seed", "num_envs", "num_steps", "iterations")}))
    print("OBS_CONTRACT_SHA " + contract["observation_contract_sha256"])
    print("PHYSICS " + json.dumps(env.backend.identity()))

    policy = ActorCritic(env.single_observation_space, action_dim=4).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    policy.train()
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_torch(obs, device)

    jsonl_path = os.path.join(out_dir, "m0_metrics.jsonl")
    records = []
    t_run = time.perf_counter()
    for it in range(1, args.iterations + 1):
        t_iter = time.perf_counter()
        buf_obs = {k: [] for k in obs_t}
        buf_actions, buf_logp, buf_values, buf_rewards, buf_dones = [], [], [], [], []

        t_collect = time.perf_counter()
        act_s = 0.0
        env_step_s = 0.0
        for _ in range(args.num_steps):
            t_act = time.perf_counter()
            action, logp, value = policy.act(obs_t)
            act_s += time.perf_counter() - t_act
            t_env = time.perf_counter()
            next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            env_step_s += time.perf_counter() - t_env
            done = np.logical_or(terminated, truncated)
            for k in buf_obs:
                buf_obs[k].append(obs_t[k])
            buf_actions.append(action)
            buf_logp.append(logp)
            buf_values.append(value)
            buf_rewards.append(torch.from_numpy(reward).to(device))
            buf_dones.append(torch.from_numpy(done.astype(np.float32)).to(device))
            obs_t = to_torch(next_obs, device)
        collection_s = time.perf_counter() - t_collect

        with torch.no_grad():
            _, last_value = policy.forward(obs_t)
        advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, last_value)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        obs_batch = {k: torch.cat(v, dim=0) for k, v in buf_obs.items()}

        t_learn = time.perf_counter()
        loss = ppo_update(policy, optimizer, obs_batch, torch.cat(buf_actions),
                          torch.cat(buf_logp), advantages, returns, epochs=args.epochs)
        learning_s = time.perf_counter() - t_learn

        wall_s = time.perf_counter() - t_iter
        samples = args.num_envs * args.num_steps
        env_steps = samples * env.decimation
        record = {
            "iter": it,
            "collection_s": round(collection_s, 4),
            "policy_act_s": round(act_s, 4),
            "env_step_s": round(env_step_s, 4),
            "env_step_per_policy_step_s": round(env_step_s / args.num_steps, 4),
            "learning_s": round(learning_s, 4),
            "wall_s": round(wall_s, 4),
            "samples": samples,
            "samples_per_s": round(samples / collection_s, 2),
            "policy_steps_per_s": round(samples / collection_s, 2),
            "env_steps": env_steps,
            "env_steps_per_s": round(env_steps / collection_s, 1),
            "end_to_end_policy_steps_per_s": round(samples / wall_s, 2),
            "end_to_end_env_steps_per_s": round(env_steps / wall_s, 1),
            "rss_mb": round(rss_mb(), 1),
            "mean_reward": round(float(torch.stack(buf_rewards).mean().item()), 4),
            "ppo_loss": round(loss, 4),
            "obs_contract_sha256": contract["observation_contract_sha256"],
            "phase": "prototype_throughput_only",
        }
        records.append(record)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("ITER " + json.dumps(record))

    total_s = time.perf_counter() - t_run
    totals = {
        "banner": BANNER,
        "iterations": args.iterations,
        "total_wall_s": round(total_s, 4),
        "total_samples": args.num_envs * args.num_steps * args.iterations,
        "total_env_steps": args.num_envs * args.num_steps * args.iterations * env.decimation,
        "mean_collection_s": round(float(np.mean([r["collection_s"] for r in records])), 4),
        "mean_learning_s": round(float(np.mean([r["learning_s"] for r in records])), 4),
        "mean_env_steps_per_s": round(float(np.mean([r["env_steps_per_s"] for r in records])), 1),
        "mean_end_to_end_policy_steps_per_s": round(float(np.mean([r["end_to_end_policy_steps_per_s"] for r in records])), 2),
        "mean_end_to_end_env_steps_per_s": round(float(np.mean([r["end_to_end_env_steps_per_s"] for r in records])), 1),
        "rss_mb": round(rss_mb(), 1),
        "obs_contract_sha256": contract["observation_contract_sha256"],
        "capability_metrics_allowed": False,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2, ensure_ascii=False)
    print("SUMMARY " + json.dumps(totals))
    print(f"OUT_DIR {out_dir}")
    print(BANNER)


if __name__ == "__main__":
    main()
