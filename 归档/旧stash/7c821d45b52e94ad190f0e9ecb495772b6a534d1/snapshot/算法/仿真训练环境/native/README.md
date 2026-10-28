# Native 训练栈（WSL2 / Linux ARM64 与 x86_64 Linux 通用）

> 本机当前入口：[迁移与验收说明](MIGRATION_ACCEPTANCE.md)。本轮正式训练显式使用 `fast_stop_v2`、五路真实观测及 1 ms/50；旧 pilot01 的简化任务数据不能作为正式试验。下列其他平台 wheel/示例说明不代表本机已实测 Windows ARM64 原生训练。

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

## Windows x86_64（如 Ultra 9 + RTX 5070 Ti）原生运行

已验证 Windows wheel 齐全（`pip download --platform win_amd64 --python-version 312`）：
- `newton 1.7.0`（py3-none-any）
- `warp-lang 1.18.0`（win_amd64）
- `ovphysx 0.7.0`（win_amd64）
- `mujoco 3.17.0`（cp312-win_amd64）
- PyTorch 用官方 Windows CUDA wheel（Blackwell/5070 Ti 需要 CUDA 12.8+，装 cu128 及以上的 torch）

因此**第二台 Windows 机器不需要 WSL**，可直接原生运行：

```powershell
py -3.12 -m venv C:\venvs\agtc-native
C:\venvs\agtc-native\Scripts\pip install --upgrade pip
C:\venvs\agtc-native\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
C:\venvs\agtc-native\Scripts\pip install warp-lang==1.18.0 newton==1.7.0 mujoco==3.17.0 ovphysx==0.7.0 numpy
# 查看器依赖（可选，本机看可视化时）
C:\venvs\agtc-native\Scripts\pip install viser==1.0.30 usd-core numpy

git clone https://github.com/w3743/AGTC C:\AGTC
cd C:\AGTC\算法\仿真训练环境\native

# 冒烟（2 iter，确认 CUDA/物理/观测链路）
C:\venvs\agtc-native\Scripts\python train\train_wsl.py --policy lstm --num-envs 64 --num-steps 32 --iterations 2 ^
  --device cuda --physics newton_xpbd --lidar warp --feedback vqf --depth analytic --detections analytic --detections-warp ^
  --sim-dt 0.001 --decimation 50 --out runs\smoke_win

# 正式 trial（Windows 版 runner）
powershell -NoProfile -ExecutionPolicy Bypass -File train\run_native_v1_trial.ps1 -Tag pilot02 ^
  -Python C:\venvs\agtc-native\Scripts\python.exe -LiveStateFile C:\tmp\native_live.json

# 局域网 worker（无需管理员；首次运行允许防火墙专用网络）
C:\venvs\agtc-native\Scripts\python remote\worker_agent.py --host 0.0.0.0 --port 8790 --token <随机串>
```

注意：
- 5070 Ti 是 Blackwell（sm_120），需要 CUDA 12.8+ 的 torch；若 `torch.cuda.is_available()` 为 False，先升级驱动/换 cu129/cu130 wheel；
- 该机器 GPU 显存 16 GB，当前配方（64 env）约 3 GB，后续可提高 `--num-envs`；
- Windows 上跑 `run_native_v1_trial.ps1` 与 Linux 的 `.sh` 等价（协议 SHA 校验、trial 目录、manifest 一致）。
