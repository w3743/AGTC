# Native 训练栈（WSL2 / Linux ARM64 与 x86_64 Linux 通用）

本目录把原先放在仓库外的 WSL 原生栈代码纳入版本管理，便于换机、复现和远程执行。
Isaac Sim/Kit 训练已停止（见 `../PERF_PLATFORM.md` §20）；这里是 WSL/Linux 原生训练路径。

## 布局

```
native/
├── proto/     # 物理/观测/策略原型：native_env、newton_backend、observation_provider、
│              # depth_caster、detections_caster、reward_fast_stop、M0/M1/M2 验证与基准脚本
├── train/     # 正式训练与评估入口：train_wsl、eval_native、policy_lstm、
│              # PROTOCOL_NATIVE_V1.1（冻结协议）、trial 启动脚本、协议 SHA 校验
├── viz/       # 复用原有 Viser 查看器的 native 状态写入器（native_live.py）、2D 回放生成器
└── remote/    # 换机/远程执行（SSH、Tailscale、端口转发、结果拉取）模板
```

`runs/`、`*.pt`、日志和实时状态文件不纳入版本库（见 `.gitignore`）。

## 环境

```bash
python >= 3.12
torch (CUDA), numpy, warp-lang, newton, mujoco, ovphysx（可选：RTX 对照时才需要 Isaac）
```

WSL 参考环境：Ubuntu-24.04-Arm64、torch 2.14.1+cu130、warp 1.18、newton 1.7、mujoco 3.17、ovphysx 0.7；
x86_64 Linux 同样适用（wheel 更全，性能通常更好）。

## 运行

```bash
# 正式 trial（1 ms/50，协议 V1.1；在有 GPU 的机器上执行）
SIM_DT=0.001 DECIMATION=50 bash train/run_native_v1_trial.sh pilot01 \
    --live-state-file /tmp/native_trial_live.json --live-state-interval 5

# 单独跑一次评估（4 seeds）
python train/eval_native.py --checkpoint runs/native_v1_1_pilot01_*/latest.pt \
    --seeds 87010-87013 --max-steps 1200 --device cuda --physics newton_xpbd \
    --lidar warp --feedback vqf --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.001 --decimation 50 --out runs/eval_pilot01

# 复用原有 Viser 查看器（中文界面）
python train/../viz/native_live.py --state-file /tmp/native_trial_live.json \
    --checkpoint runs/native_v1_1_pilot01_*/latest.pt --max-steps 40000 --real-time
# Windows 侧另起（或经 SSH 隧道）：
#   scripts/view_training_state_web.py --state-file <同一文件> --output <状态目录> --port 8792 --language zh
```

## 协议

- `train/PROTOCOL_NATIVE_V1.1.md` 是正式协议（冻结 SHA 见 `.sha256`）；V1 原文保留；
- 5 ms/10 已降级为 experimental-only（周期数值失稳证据见协议 §17）；
- `capability_metrics_allowed=false` 在协议明确授权前保持不变。

## 换机 / 远程

见 `remote/README.md`：SSH + Git + rsync + SSH 端口转发即可，**不需要 A2A**。
