#!/usr/bin/env python3
"""Small Gymnasium Robotics smoke/train/view entry point for Fetch and Hand."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_IDS = {"arm": "FetchReachDense-v4", "hand": "HandReachDense-v3"}
SEED = 7


def task_id(task: str) -> str:
    """Require the environment version validated for this deployment."""
    try:
        import gymnasium as gym
        import gymnasium_robotics
    except ImportError as exc:
        raise RuntimeError(
            "缺少 gymnasium。请在已安装 robotics 依赖的虚拟环境中运行此脚本。"
        ) from exc

    gym.register_envs(gymnasium_robotics)
    env_id = TASK_IDS[task]
    if env_id in gym.registry:
        return env_id
    raise RuntimeError(f"缺少锁定环境 {env_id}，请重新运行安装程序。")


def json_value(value: Any) -> Any:
    """Convert common NumPy values to JSON-compatible values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary['mode']}_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    return path


def smoke(task: str, env_id: str, output_dir: Path) -> None:
    import gymnasium as gym

    summary: dict[str, Any] = {
        "task": task,
        "mode": "smoke",
        "environment_id": env_id,
        "seed": SEED,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_success_claim": False,
    }
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=SEED)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        frame = env.render()
        if frame is None:
            raise RuntimeError(f"{env_id} 的 rgb_array 渲染没有返回图像。")

        image_path = output_dir / "smoke_render.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("保存渲染截图需要 Pillow；请检查 Robotics 环境依赖。") from exc
        Image.fromarray(frame).save(image_path)

        summary.update(
            {
                "reset_ok": True,
                "step_ok": True,
                "observation_keys": list(obs) if isinstance(obs, dict) else None,
                "action_shape": list(getattr(action, "shape", ())),
                "next_observation_keys": list(next_obs) if isinstance(next_obs, dict) else None,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reset_info_keys": sorted(info.keys()),
                "step_info_keys": sorted(step_info.keys()),
                "render_file": str(image_path.resolve()),
                "render_shape": list(frame.shape),
            }
        )
    finally:
        env.close()

    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        summary["check_env"] = "skipped: stable-baselines3 unavailable"
    else:
        checked_env = gym.make(env_id)
        try:
            check_env(checked_env.unwrapped, warn=True)
        finally:
            checked_env.close()
        summary["check_env"] = "passed"

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path = save_summary(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_value))
    print(f"摘要已保存：{summary_path}")


def train(task: str, env_id: str, steps: int, output_dir: Path) -> None:
    import gymnasium as gym
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(4)
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id)
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=128,
            batch_size=64,
            device="cpu",
            seed=SEED,
            verbose=1,
        )
        model.learn(total_timesteps=steps)
        checkpoint_path = output_dir / "checkpoint.zip"
        model.save(str(checkpoint_path.with_suffix("")))
    finally:
        env.close()

    metadata = {
        "task": task,
        "mode": "train",
        "environment_id": env_id,
        "requested_timesteps": steps,
        "algorithm": "PPO",
        "policy": "MultiInputPolicy",
        "device": "cpu",
        "n_steps": 128,
        "batch_size": 64,
        "seed": SEED,
        "stable_baselines3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "checkpoint": str(checkpoint_path.resolve()),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_success_claim": False,
        "status": "steps_completed_success_not_evaluated",
        "note": "完成指定训练步数不代表策略已经学会任务；本脚本不做成功率评估。",
    }
    summary_path = save_summary(output_dir, metadata)
    print(f"训练步数已完成；这不代表任务学会。模型：{checkpoint_path}")
    print(f"运行元数据：{summary_path}")


def find_checkpoint(output_arg: Path, task: str) -> Path | None:
    if output_arg.is_file():
        return output_arg
    candidates = [
        output_arg / task / "checkpoint.zip",
        output_arg / "checkpoint.zip",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def view(task: str, env_id: str, steps: int, output_dir: Path, checkpoint: Path | None) -> None:
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    model = None
    if checkpoint is not None:
        from stable_baselines3 import PPO

        model = PPO.load(str(checkpoint), device="cpu")
        print(f"查看来源：已训练 checkpoint ({checkpoint.resolve()})")
    else:
        print("查看来源：未找到 checkpoint，使用随机动作（不是训练策略）")

    env = gym.make(env_id, render_mode="human")
    try:
        obs, _ = env.reset(seed=SEED)
        for _ in range(steps):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            time.sleep(0.02)
    finally:
        env.close()

    save_summary(
        output_dir,
        {
            "task": task,
            "mode": "view",
            "environment_id": env_id,
            "steps": steps,
            "policy_source": str(checkpoint.resolve()) if checkpoint else "random actions",
            "training_success_claim": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 手臂或 Shadow Hand 的 Gymnasium Robotics 环境入门入口。"
    )
    parser.add_argument("--task", choices=("arm", "hand"), required=True, help="arm=Fetch 手臂，hand=灵巧手")
    parser.add_argument("--mode", choices=("smoke", "train", "view"), default="smoke")
    parser.add_argument("--steps", type=int, default=4096, help="train/view 的步数；smoke 固定只走一步")
    parser.add_argument("--output", type=Path, default=Path("training_runs"), help="日志、截图和模型输出目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("--steps 必须是正整数。", file=sys.stderr)
        return 2
    try:
        env_id = task_id(args.task)
        output_dir = args.output / args.task
        print(f"任务：{args.task}；锁定环境：{env_id}")
        if args.mode == "smoke":
            smoke(args.task, env_id, output_dir)
        elif args.mode == "train":
            train(args.task, env_id, args.steps, output_dir)
        else:
            checkpoint = find_checkpoint(args.output, args.task)
            view(args.task, env_id, args.steps, output_dir, checkpoint)
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
