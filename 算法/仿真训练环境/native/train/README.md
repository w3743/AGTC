# WSL 原生训练准备（wsl_train）

目的：用户已决定**放弃 Isaac 训练、以后以 WSL 原生栈为主**。本目录是 WSL 侧的训练骨架，复用 `../proto/` 的物理/观测/策略原型，并补上真实训练需要的 checkpoint / resume / 指标落盘。

## 状态（2026-10-25 11:20）

- 环境：`Ubuntu-24.04-Arm64`（WSL2），GPU 可见（RTX Spark N1X / 98304 MiB），torch 2.14.1+cu130、warp 1.18、newton 1.7、mujoco 3.17、ovphysx 0.7 全部可用；根盘约 945 GB 可用。
- `train_wsl.py` 已冒烟验证：
  - 2 iterations：iter2 collection 1.63 s、**628 samples/s**（64 env × 16 steps = 1024 samples/iter）、端到端 563 samples/s；
  - `--save-interval 2` 生成 `checkpoint_2.pt` / `latest.pt`；
  - `--resume` 从 checkpoint_2 继续到 iter 3–4，observation contract SHA 校验通过，iteration 连续。
- **depth 已就绪**：`observation_provider.depth_backend="analytic"`（Warp），与独立 numpy caster 对照 mask 不一致 0、max 2.06e-4 m；`train_wsl.py --depth analytic` 冒烟 iter2 collection 2.06 s / 496 samples/s。
- **仍未就绪**：`detections 16×8` 仍是占位，`capability_metrics_allowed=false`；depth 与 Isaac RTX 的域差未逐像素对照。

## 用法

```bash
# 新训练（GPU，64 env × 16 steps/iter）
/opt/venv/bin/python /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/train_wsl.py \
  --num-envs 64 --num-steps 16 --iterations 100 --save-interval 10 \
  --device cuda --physics newton_xpbd --lidar warp --feedback vqf \
  --out /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/<trial>

# 从 checkpoint 继续
/opt/venv/bin/python /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/train_wsl.py \
  ... --out <同一目录> --resume <同一目录>/latest.pt
```

- 产物：`metrics.jsonl`（每 iter 吞吐/奖励/loss）、`observation_contract.json`、`identity.json`、`checkpoint_{iter}.pt`、`latest.pt`、`summary.json`。
- 约束：checkpoint 的 observation contract SHA 必须与当前环境一致，否则拒绝 resume；draft 阶段 `capability_metrics_allowed=false`，**不得据此宣称能力/避苗/实车结论**。

## 进入正式训练的门槛

1. depth/detections 与 reference（以及登记 RTX 域差）验证通过，contract 中所有通道 `real`；
2. 用真观测重跑 `train_wsl.py` 2 iterations，确认吞吐不退化（预计 VQF+depth 成本仍有明显余量）；
3. 定义并冻结本地的 native trial 协议（观测/奖励/终止/种子/验收口径），再开始长训练。
