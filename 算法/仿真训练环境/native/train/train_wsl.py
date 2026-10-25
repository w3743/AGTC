"""WSL native training harness (prep phase).

Runs PPO over the native WSL env (proto/native_env.py), reusing the prototype's
encoder/policy, GAE and PPO update. Unlike the M0 throughput prototype this
entry adds:

  * checkpoint save every --save-interval iterations (policy, optimizer,
    iteration, contract SHA);
  * --resume from a checkpoint with a strict observation-contract check;
  * JSONL metrics + summary.json;
  * explicit phase banner: native-domain capability acceptance remains disabled
    while the registered sensor/physics domain gaps are unresolved.

This is preparation infrastructure, not a capability-accepted training run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
ROBOT_USD = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\native\viz\field_robot_19.usda"
TERRAIN_USD = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field\assets\010a1c5428ffe08e\terrain.usda"


def _usd_name(native: str) -> str:
    """Map a Newton body label to the USD robot prim name used by the Viser viewer."""
    raw = str(native).split("_/", 1)[-1]
    if raw == "base":
        return "base_link"
    if raw == "gimbal_yaw":
        return "gimbal_yaw_link"
    if raw == "gimbal_pitch":
        return "gimbal_pitch_link"
    if raw.startswith("left_wheel_") or raw.startswith("right_wheel_"):
        return raw.replace("_link", "_wheel_link")
    return raw


def _make_live_exporter(path: str, body_names: list[str], run_id: str):
    """Load the repository LiveStateExporter lazily (read-only; no repo edits)."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location(
        "isaac_field_live_state", os.path.join(REPO, "isaac_field", "live_state.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exporter = module.LiveStateExporter(path, robot_usd_path=ROBOT_USD,
                                        terrain_usd_path=TERRAIN_USD, body_names=body_names)
    exporter.run_id = run_id
    return exporter


def _write_live_state(exporter, env, origins, sim_time: float) -> bool:
    """Write one schema-v1 live-state packet for the first 4 envs (newton XPBD state)."""
    try:
        backend = env.backend
        body_q = backend.state_0.body_q.numpy()  # (nworld*bodies, 7) px py pz qx qy qz qw
        labels = [str(v) for v in backend.model.body_label]
        bodies_per_world = len(labels) // env.num_envs
        base_local = int(backend.base_body[0]) % bodies_per_world
        body_link_pose_w = np.zeros((4, bodies_per_world, 7))
        for e in range(4):
            first = int(backend.base_body[e]) - base_local
            ids = np.arange(first, first + bodies_per_world)
            pose = body_q[ids].copy()
            pose[:, 0] += origins[e, 0]
            pose[:, 1] += origins[e, 1]
            body_link_pose_w[e] = pose
        root_rows = body_link_pose_w[:, base_local, :]
        goals = np.stack([env.goal_xy[e] - origins[e, :2] for e in range(4)])
        exporter.write(root_link_pose_w=root_rows, body_link_pose_w=body_link_pose_w,
                       env_origins_w=origins, goals_local_xy=goals,
                       task_times_s=np.full(4, sim_time), sim_time_s=sim_time)
        return True
    except Exception as exc:  # viewer must never break training
        print(f"LIVE_STATE_DISABLED reason={type(exc).__name__}: {exc}")
        return False

from native_env import NativeFieldEnv  # noqa: E402
from policy_m0 import ActorCritic  # noqa: E402
from train_m0 import compute_gae, ppo_update, rss_mb, to_torch  # noqa: E402
from contract_identity import (IDENTITY_SCHEMA_VERSION, attach_contract_identity,
                               validate_checkpoint_contract)  # noqa: E402

BANNER = "WSL_NATIVE_TRAINING_PREP"


def _contract_for(env, out_dir: str, phase: str) -> dict:
    contract = env.observation_config()
    verification = {}
    for name in ("m1_verification.json", "m2_verification.json", "depth_verification.json",
                 "detections_verification.json", "detections_warp_verification.json"):
        path = os.path.join(PROTO, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                key = {"m1_verification.json": "m1", "m2_verification.json": "m2",
                       "depth_verification.json": "depth",
                       "detections_verification.json": "detections",
                       "detections_warp_verification.json": "detections_warp"}[name]
                verification[key] = json.load(stream)
    if verification:
        contract["verification"] = verification
    depth_ok = contract.get("channels", {}).get("depth", {}).get("verified_vs") is not None
    det_ok = contract.get("channels", {}).get("detections", {}).get("verified_vs") is not None
    if depth_ok and det_ok:
        blockers = ["RTX domain gap not pixel-compared (depth, detections)"]
    elif depth_ok:
        blockers = ["detections placeholder (16x8 zeros)",
                    "depth domain gap vs Isaac RTX not pixel-compared"]
    else:
        blockers = ["depth placeholder", "detections placeholder"]
    contract.update(
        run_id=os.path.basename(out_dir),
        phase=phase,
        capability_metrics_allowed=False,
        capability_blockers=blockers,
        provider_contract_sha256=env.provider.contract_sha256(env.policy_dt),
        note="WSL training-prep harness; capability metrics disabled until depth/detections are verified",
        sha_scope=("observation channels, physics/backend identity, success/reward config and embedded "
                   "verification only; runtime metadata (run_id, phase, note) is excluded from the hash"),
    )
    return attach_contract_identity(contract)



def _save_checkpoint(path: str, policy, optimizer, iteration: int, contract: dict, identity: dict) -> None:
    temporary = path + ".tmp"
    payload = {
        "banner": BANNER,
        "iteration": int(iteration),
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "observation_contract_sha256": contract["observation_contract_sha256"],
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "task_backend": contract["task_backend"],
        "task_identity": contract["task_identity"],
        "task_identity_sha256": contract["task_identity_sha256"],
        "identity": identity,
        "policy_type": identity.get("policy", {}).get("policy_type", "mlp"),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if "hidden_size" in identity.get("policy", {}):
        payload["hidden_size"] = identity["policy"]["hidden_size"]
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--physics", type=str, default="newton_xpbd",
                        choices=("mujoco_cpu", "newton_xpbd", "newton_mujoco_warp"))
    parser.add_argument("--physics-cuda-graph", action="store_true", default=False,
                        help="experimental capture of 50 MuJoCo-Warp substeps; excluded from formal V2")
    parser.add_argument("--task-backend", type=str, default="simplified", choices=("simplified", "fast_stop_v2"))
    parser.add_argument("--lidar", type=str, default="warp", choices=("numpy", "warp"))
    parser.add_argument("--feedback", type=str, default="vqf", choices=("direct", "vqf"))
    parser.add_argument("--feedback-noise", action="store_true", default=True)
    parser.add_argument("--feedback-no-noise", dest="feedback_noise", action="store_false")
    parser.add_argument("--depth", type=str, default="constant", choices=("constant", "analytic"),
                        help="depth channel: constant placeholder or analytic raycast (verified CPU-side)")
    parser.add_argument("--depth-noise", action="store_true", default=True)
    parser.add_argument("--depth-no-noise", dest="depth_noise", action="store_false")
    parser.add_argument("--detections", type=str, default="constant", choices=("constant", "analytic"),
                        help="detections channel: zeros placeholder or analytic mask->bbox (verified CPU-side)")
    parser.add_argument("--detections-noise", action="store_true", default=True)
    parser.add_argument("--detections-no-noise", dest="camera_noise", action="store_false")
    parser.add_argument("--detections-warp", action="store_true", default=False,
                        help="render the 64x64 detector mask with the warp kernel (CUDA when --device cuda)")
    parser.add_argument("--sim-dt", type=float, default=1e-3,
                        help="physics substep dt; must satisfy sim_dt*decimation == 0.05 (default 1 ms)")
    parser.add_argument("--decimation", type=int, default=50,
                        help="physics substeps per policy step; must satisfy sim_dt*decimation == 0.05")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="accumulate per-sensor timings and write sensor_times_s into the JSONL")
    parser.add_argument("--nonfinite-limit", type=int, default=64,
                        help="abort with failure.json after this many non-finite env recoveries")
    parser.add_argument("--live-state-file", type=str, default="",
                        help="write a schema-v1 live-state packet for the Viser viewer (newton_xpbd only)")
    parser.add_argument("--live-state-interval", type=int, default=5,
                        help="write the live-state packet every N policy steps")
    parser.add_argument("--live-state-spacing", type=float, default=2.5,
                        help="world spacing used to place the 4 virtual viewer environments")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--policy", type=str, default="mlp", choices=("mlp", "lstm"),
                        help="policy architecture: prototype MLP (default) or LSTM-512")
    parser.add_argument("--lstm-hidden", type=int, default=512)
    args = parser.parse_args()
    if args.physics_cuda_graph and args.physics != "newton_mujoco_warp":
        parser.error("--physics-cuda-graph requires --physics newton_mujoco_warp")

    if args.iterations < 1 or args.save_interval < 1:
        parser.error("iterations and save-interval must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)

    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(HERE, "runs", "prep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    if abs(args.sim_dt * args.decimation - 0.05) > 1e-9:
        parser.error(f"--sim-dt*--decimation must equal 0.05 s, got {args.sim_dt * args.decimation}")
    print(f"{BANNER}: native-stack training harness (capability metrics disabled)")
    env = NativeFieldEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        physics=args.physics,
        physics_cuda_graph=args.physics_cuda_graph,
        task_backend=args.task_backend,
        lidar_backend=args.lidar,
        device="cuda:0" if args.device == "cuda" else "cpu",
        dt=args.sim_dt,
        decimation=args.decimation,
        feedback_backend=args.feedback,
        feedback_noise=args.feedback_noise,
        depth_backend=args.depth,
        depth_device="cuda:0" if args.device == "cuda" else "cpu",
        depth_noise=args.depth_noise,
        camera_backend=("analytic_warp" if args.detections_warp else args.detections),
        camera_noise=args.camera_noise,
        camera_device="cuda:0" if args.device == "cuda" else "cpu",
        profile=args.profile,
    )
    contract = _contract_for(env, out_dir, phase="wsl_training_prep")
    is_lstm = args.policy == "lstm"
    if is_lstm:
        from policy_lstm import LSTMActorCritic
        policy = LSTMActorCritic(env.single_observation_space, action_dim=4,
                                 hidden_size=args.lstm_hidden).to(device)
        policy_meta = policy.policy_meta()
    else:
        policy = ActorCritic(env.single_observation_space, action_dim=4).to(device)
        policy_meta = {"policy_type": "mlp",
                       "params_total": int(sum(p.numel() for p in policy.parameters()))}
    identity = {
        "banner": BANNER,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "device": args.device,
        "physics": args.physics,
        "physics_cuda_graph": args.physics_cuda_graph,
        "lidar": args.lidar,
        "task_backend": args.task_backend,
        "feedback": args.feedback,
        "policy": policy_meta,
        "env": env.identity(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "task_identity": contract["task_identity"],
        "task_identity_sha256": contract["task_identity_sha256"],
    }
    identity["env"]["task_backend"] = args.task_backend
    identity["env"]["task_identity_sha256"] = contract["task_identity_sha256"]
    identity["env"]["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    with open(os.path.join(out_dir, "observation_contract.json"), "w", encoding="utf-8") as stream:
        json.dump(contract, stream, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "identity.json"), "w", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, ensure_ascii=False)
    print("IDENTITY " + json.dumps({"seed": identity["seed"], "num_envs": identity["num_envs"],
                                    "num_steps": identity["num_steps"], "policy": policy_meta}))
    print("OBS_CONTRACT_SHA " + contract["observation_contract_sha256"])
    print("PHYSICS " + json.dumps(env.backend.identity()))

    # optional Viser live-state writer (read-only integration; never touches the contract SHA)
    live_exporter = None
    live_origins = None
    if args.live_state_file:
        try:
            backend = env.backend
            labels = [str(v) for v in backend.model.body_label]
            bodies_per_world = len(labels) // env.num_envs if env.num_envs else 0
            body_names = [_usd_name(name) for name in labels[:bodies_per_world]]
            if args.physics not in ("newton_xpbd", "newton_mujoco_warp"):
                raise RuntimeError("live-state writer needs a Newton GPU backend (state_0)")
            if env.num_envs < 4:
                raise RuntimeError("live-state viewer needs at least 4 envs")
            if len(body_names) != 19:
                raise RuntimeError(f"expected 19 viewer bodies, got {len(body_names)}")
            live_path = args.live_state_file
            live_dir = os.path.dirname(os.path.abspath(live_path))
            os.makedirs(live_dir, exist_ok=True)
            live_exporter = _make_live_exporter(live_path, body_names, os.path.basename(out_dir))
            spacing = float(args.live_state_spacing)
            live_origins = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0],
                                     [0.0, spacing, 0.0], [spacing, spacing, 0.0]])
            print("LIVE_STATE " + json.dumps({"file": live_path, "interval": args.live_state_interval,
                                              "bodies": len(body_names),
                                              "run_id": live_exporter.run_id}))
        except Exception as exc:
            print(f"LIVE_STATE_DISABLED reason={type(exc).__name__}: {exc}")
            live_exporter = None

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        try:
            validate_checkpoint_contract(checkpoint, contract["task_identity_sha256"])
        except ValueError as exc:
            raise RuntimeError(f"checkpoint task contract rejected: {exc}") from exc
        ck_type = checkpoint.get("policy_type", "mlp")
        if ck_type != args.policy:
            raise RuntimeError(f"checkpoint policy_type={ck_type!r} does not match --policy {args.policy!r}")
        if is_lstm and int(checkpoint.get("hidden_size", -1)) != args.lstm_hidden:
            raise RuntimeError(f"checkpoint hidden_size={checkpoint.get('hidden_size')} does not match "
                               f"--lstm-hidden {args.lstm_hidden}")
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"])
        policy.to(device)
        print(f"RESUMED iteration={start_iteration} from {args.resume} (policy={ck_type})")

    policy.train()
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_torch(obs, device)
    jsonl_path = os.path.join(out_dir, "metrics.jsonl")
    failure_path = os.path.join(out_dir, "failure.json")
    nonfinite_jsonl = os.path.join(out_dir, "nonfinite_events.jsonl")
    nf_total = 0

    def fail(reason: str, detail: dict, iteration: int):
        """Write failure.json, keep the last checkpoint and exit safely (no silent crash)."""
        ckpt = os.path.join(out_dir, "latest.pt")
        try:
            _save_checkpoint(ckpt, policy, optimizer, iteration, contract, identity)
            saved = ckpt
        except Exception as exc:  # never let failure reporting itself crash
            saved = f"save_failed: {exc}"
        payload = {
            "banner": BANNER,
            "reason": reason,
            "detail": detail,
            "iteration": int(iteration),
            "last_checkpoint": saved,
            "observation_contract_sha256": contract["observation_contract_sha256"],
            "nonfinite_total": int(nf_total),
            "nonfinite": env.consume_nonfinite() if hasattr(env, "consume_nonfinite") else None,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with open(failure_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        print("FAILURE " + json.dumps({k: payload[k] for k in
                                       ("reason", "iteration", "last_checkpoint", "nonfinite_total")}))
        raise SystemExit(3)

    end_iteration = start_iteration + args.iterations
    t_run = time.perf_counter()
    live_counter = 0
    for name, param in policy.named_parameters():  # catch NaN weights resumed from a bad checkpoint
        if not bool(torch.isfinite(param).all()):
            fail("non_finite_parameter", {"phase": "resume", "param": name}, start_iteration)
    for iteration in range(start_iteration + 1, end_iteration + 1):
        t_iter = time.perf_counter()
        buf_obs = {key: [] for key in obs_t}
        buf_actions, buf_logp, buf_values, buf_rewards, buf_dones = [], [], [], [], []
        buf_done_masks = []
        nf_iter = 0
        nf_action_iter = 0
        solver_diag_totals = None
        hidden = policy.init_hidden(args.num_envs, device) if is_lstm else None
        hidden0 = None
        if is_lstm:
            hidden0 = (hidden[0].clone(), hidden[1].clone())

        t_collect = time.perf_counter()
        act_s = env_step_s = 0.0
        for _ in range(args.num_steps):
            t_act = time.perf_counter()
            try:
                if is_lstm:
                    action, logp, value, hidden = policy.act(obs_t, hidden)
                else:
                    action, logp, value = policy.act(obs_t)
            except Exception as exc:  # e.g. torch Normal rejects NaN loc
                fail("policy_forward_error",
                     {"iter": iteration, "step": len(buf_actions), "error": f"{type(exc).__name__}: {exc}"},
                     iteration)
            act_s += time.perf_counter() - t_act
            if not bool(torch.isfinite(action).all()):
                fail("non_finite_action", {"iter": iteration, "step": len(buf_actions),
                                           "action_absmax": float(action.abs().nan_to_num(0).max())}, iteration)
            t_env = time.perf_counter()
            next_obs, reward, terminated, truncated, _info = env.step(action.detach().cpu().numpy())
            env_step_s += time.perf_counter() - t_env
            step_diagnostics = _info.get("physics_solver_diagnostics")
            if isinstance(step_diagnostics, dict) and step_diagnostics.get("policy_step_sampled"):
                if solver_diag_totals is None:
                    solver_diag_totals = {"world_substeps": 0, "capacity_overflow_substeps": 0,
                                          "outer_iteration_limit_substeps": 0, "line_search_limit_substeps": 0}
                solver_diag_totals["world_substeps"] += args.num_envs * args.decimation
                for name in ("capacity_overflow", "outer_iteration_limit", "line_search_limit"):
                    values = step_diagnostics.get(name + "_substeps_by_world")
                    if (not isinstance(values, list) or len(values) != args.num_envs
                            or any(type(value) is not int or value < 0 for value in values)):
                        raise RuntimeError(f"malformed physics solver diagnostics: {name}")
                    solver_diag_totals[name + "_substeps"] += sum(values)
            step_nf = int(np.sum(_info.get("nonfinite", False)))
            step_nf_action = int(np.sum(_info.get("action_nonfinite", False)))
            nf_iter += step_nf
            nf_action_iter += step_nf_action
            if step_nf or step_nf_action:
                nf_total += step_nf + step_nf_action
                with open(nonfinite_jsonl, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"iter": iteration, "step": len(buf_actions),
                                             "env_nonfinite": step_nf,
                                             "action_nonfinite": step_nf_action},
                                            ensure_ascii=False) + "\n")
                if nf_total > args.nonfinite_limit:
                    fail("nonfinite_state_budget_exceeded",
                         {"iter": iteration, "step": len(buf_actions), "nf_total": nf_total,
                          "limit": args.nonfinite_limit}, iteration)
            if not np.isfinite(reward).all():
                fail("non_finite_reward", {"iter": iteration, "step": len(buf_actions)}, iteration)
            for channel, channel_value in next_obs.items():
                if not np.isfinite(channel_value).all():
                    fail("non_finite_observation",
                         {"iter": iteration, "step": len(buf_actions), "channel": channel}, iteration)
            done = np.logical_or(terminated, truncated)
            for key in buf_obs:
                buf_obs[key].append(obs_t[key])
            buf_actions.append(action)
            buf_logp.append(logp)
            buf_values.append(value)
            buf_rewards.append(torch.from_numpy(reward).to(device))
            buf_dones.append(torch.from_numpy(done.astype(np.float32)).to(device))
            if is_lstm:
                done_t = torch.from_numpy(done.astype(bool)).to(device)
                buf_done_masks.append(done_t)
                hidden = policy.reset_hidden(hidden, done_t)
            obs_t = to_torch(next_obs, device)
            live_counter += 1
            if (live_exporter is not None and live_origins is not None
                    and live_counter % max(1, args.live_state_interval) == 0):
                sim_time = (iteration - start_iteration - 1) * args.num_steps * env.policy_dt \
                    + (len(buf_actions)) * env.policy_dt
                if not _write_live_state(live_exporter, env, live_origins, sim_time):
                    live_exporter = None  # disable after a write failure, keep training
        collection_s = time.perf_counter() - t_collect

        with torch.no_grad():
            if is_lstm:
                _, last_value, _ = policy.forward(obs_t, hidden)
            else:
                _, last_value = policy.forward(obs_t)
        advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, last_value)
        obs_batch = {key: torch.cat(value, dim=0) for key, value in buf_obs.items()}

        t_learn = time.perf_counter()
        if is_lstm:
            from policy_lstm import ppo_update_lstm
            loss = ppo_update_lstm(
                policy, optimizer,
                [{key: buf_obs[key][t] for key in buf_obs} for t in range(args.num_steps)],
                torch.stack(buf_actions), torch.stack(buf_logp), advantages, returns,
                hidden0, torch.stack(buf_done_masks), epochs=args.epochs,
            )
        else:
            loss = ppo_update(
                policy, optimizer, obs_batch,
                torch.cat(buf_actions), torch.cat(buf_logp),
                advantages.reshape(-1), returns.reshape(-1), epochs=args.epochs,
            )
        learning_s = time.perf_counter() - t_learn
        wall_s = time.perf_counter() - t_iter
        samples = args.num_envs * args.num_steps
        record = {
            "iter": iteration,
            "wall_s": round(wall_s, 4),
            "collection_s": round(collection_s, 4),
            "policy_act_s": round(act_s, 4),
            "env_step_s": round(env_step_s, 4),
            "learning_s": round(learning_s, 4),
            "samples": samples,
            "samples_per_s": round(samples / max(collection_s, 1e-9), 2),
            "end_to_end_samples_per_s": round(samples / max(wall_s, 1e-9), 2),
            "rss_mb": round(rss_mb(), 1),
            "mean_reward": round(float(torch.stack(buf_rewards).mean().item()), 5),
            "ppo_loss": round(loss, 4),
            "observation_contract_sha256": contract["observation_contract_sha256"],
            "phase": "wsl_training_prep",
            "policy_type": args.policy,
            "capability_metrics_allowed": False,
            "nonfinite_env_resets": int(nf_iter),
            "nonfinite_actions": int(nf_action_iter),
        }
        if solver_diag_totals is not None:
            record["physics_solver_events"] = solver_diag_totals
        # parameter/loss finiteness guards: a NaN update must not silently poison checkpoints
        if not np.isfinite(loss):
            fail("non_finite_loss", {"iter": iteration, "loss": float(loss)}, iteration)
        for name, param in policy.named_parameters():
            if not bool(torch.isfinite(param).all()):
                fail("non_finite_parameter", {"iter": iteration, "param": name}, iteration)
        events = env.consume_nonfinite() if hasattr(env, "consume_nonfinite") else None
        if events and events.get("events"):
            with open(nonfinite_jsonl, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"iter": iteration, **events}, ensure_ascii=False) + "\n")
        sensor_times = env.consume_sensor_times()
        if sensor_times is not None:
            record["sensor_times_s"] = sensor_times
        with open(jsonl_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("ITER " + json.dumps(record))
        if iteration % args.save_interval == 0:
            _save_checkpoint(os.path.join(out_dir, f"checkpoint_{iteration}.pt"),
                             policy, optimizer, iteration, contract, identity)
            _save_checkpoint(os.path.join(out_dir, "latest.pt"),
                             policy, optimizer, iteration, contract, identity)
            print(f"SAVED checkpoint_{iteration}.pt")

    _save_checkpoint(os.path.join(out_dir, f"checkpoint_{end_iteration}.pt"),
                     policy, optimizer, end_iteration, contract, identity)
    _save_checkpoint(os.path.join(out_dir, "latest.pt"), policy, optimizer,
                     end_iteration, contract, identity)
    summary = {
        "banner": BANNER,
        "phase": "wsl_training_prep",
        "capability_metrics_allowed": False,
        "capability_blockers": contract["capability_blockers"],
        "iterations": args.iterations,
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "total_wall_s": round(time.perf_counter() - t_run, 4),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "device": args.device,
        "physics": args.physics,
        "physics_cuda_graph": args.physics_cuda_graph,
        "observation_contract_sha256": contract["observation_contract_sha256"],
        "final_checkpoint": os.path.join(out_dir, "latest.pt"),
        "rss_mb": round(rss_mb(), 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print("SUMMARY " + json.dumps(summary))
    print(f"OUT_DIR {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
