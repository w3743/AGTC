# 训练平台性能改造记录（2026-10-25）

本文件只记录实际执行过的命令与实测数字；未执行的检查不写成通过，未测到的倍数不作声明。

## 1. 机器与基线

- 主机：NVIDIA RTX Spark N1X（ARM64 Windows 11，20 逻辑核，32 GiB 物理内存，GPU 可见预算约 96 GiB）。
- 基线：a07（4 env，`dt=0.001`，`decimation=50`，`torch_batched`，Isaac Sim 6.2 + Kit + PhysX GPU）。
  - 实测 collection 约 219–273 s/iteration（128 步 × 4 env = 512 samples），约 **2.18 samples/s**。
  - 现场 py-spy：MainThread 88.1% 时间在 `physx_manager.py:569 physx_sim.fetch_results()`；`simulate` 1.7%。
  - a07 在 2026-10-25 02:30:23（本地）被外部中断，`run_status.json` 记为 `failed`（exit 4294967295），最后完整 checkpoint 为 `model_80.pt`。

## 2. 探针结果（fresh training，seed 69041，各 2 iterations）

| 配置 | collection time | 每轮 samples | samples/s | 相对基线 |
|---|---|---|---|---|
| 4 env, dt=1 ms, dec=50（a07 历史） | ~235 s | 512 | 2.18 | 1× |
| 32 env, dt=1 ms, dec=50 | 365.6 s / 374.8 s | 4096 | 11.0 | ~5.0× |
| 32 env, dt=5 ms, dec=10 | 98.3 s / 93.8 s | 4096 | 41.7 | ~19× |

- `dt=5 ms, dec=10` 时每 policy step 的 PhysX `fetch_results()` 次数从 50 降到 10；每步耗时从约 1.78 s 降到约 0.76 s（32 env）。
- 奖励量级对比（fresh 随机策略，仅说明量级）：dt=1 ms 探针 `Train/mean_reward` = -17.165/-17.559；dt=5 ms 探针 = -17.035/-17.530。**这不构成物理等价性结论**，需做固定动作轨迹保真对比。
- 未改变：`position_iterations=40`、`velocity_iterations=1`、solver type、传感器规格、观测/动作维度。

## 3. 代码改动

`isaac_field/env.py` 的 `_validate_task_backend` 增加显式性能开关：

- 默认（不设置 `ISAAC_FIELD_PERF_PHYSICS`）：仍强制 `sim.dt=0.001`、`decimation=50`、`render_interval=50`、求解器迭代等协议值，行为不变。
- 设置 `ISAAC_FIELD_PERF_PHYSICS=1`：允许任意 `sim.dt * decimation == task.dt`（0.05 s）的组合，其余物理/传感器协议检查保持；不一致仍抛错。

## 4. 正式训练命令（本次运行）

```powershell
# 环境：$env:ISAAC_FIELD_PERF_PHYSICS='1'; $env:OMNI_KIT_ACCEPT_EULA='YES'
# 运行目录：results/isaac_field_improvement/perf_20261025_n32_dt5
C:\pioneer\isaac-training\.venv\Scripts\python.exe -B -u -m isaac_field train `
  --num_envs 32 --max_iterations 96 --device cuda:0 --seed 69041 `
  --logger tensorboard --experiment_name perf_20261025_n32_dt5 --run_timestamp train `
  env.task.size=12.0 env.scene.env_spacing=22.0 env.task_backend=torch_batched `
  env.sim.dt=0.005 env.decimation=10
```

- 设计预算：96 iterations × 4096 = 393216 transitions（基线为 196608）。
- 预计 ~107–113 s/iteration；`launch.json` 记录启动时间、命令与环境变量。

## 5. WSL2 ARM64 原生栈（2026-10-25，已就绪，GPU 基准待训练空闲）

- 发行版：`Ubuntu-24.04-Arm64`，用官方 rootfs（341,871,845 B）`wsl --import` 导入，无提权、未 `wsl --install`、未重启。
- WSL 内网络直连可用（apt/pypi/pytorch 4–10 MB/s）；GPU 可见：`/usr/lib/wsl/lib/nvidia-smi` 报 RTX Spark N1X 98304 MiB / CUDA UMD 13.4。
- 已装 aarch64 官方 wheel：python 3.12.3、torch 2.14.1+cu130、triton 3.8.0、warp-lang 1.18.0、mujoco 3.17.0 + mujoco-warp 3.17.0、newton 1.7.0、ovphysx 0.7.0、ovstage 0.3.0；CPU-only import 全部通过。
- CPU-only 基准（GPU 隐藏）：torch CPU fp32 2048³ 294.7 ms ≈ 0.058 TFLOPS；MuJoCo CPU 64 env × 18 joints **28,668 env-steps/s**（0.112 s/50 步，约 57× 实时）；mujoco-warp CPU 回退 4,744 env-steps/s（仅对照）。
- 接入评估结论（详见 `C:/pioneer/wsl-native/INTEGRATION_ASSESSMENT.md`）：
  - 传感器编码器、PPO 接口、驱动控制、batched 奖励/终止在纯 torch 层，可近乎零改动复用；
  - 轮系/前导轮 mimic 在 `newton_drive.py` 已有实现可复用；
  - **主要风险是五路传感器**：现有 reference 后端是解析 raycast（`task.py:431–499`），可移植为 Warp kernel；CPU 实测约 1.4–1.55 M rays/s，64 env 的 depth 901k rays/次在 CPU 不可行，必须 GPU；GPU 数字待实测；
  - 接触/摩擦不能照抄 PhysX（40 pos-iter/μ0.8）到 MuJoCo/Newton，需重标定；
  - 关节口径：14 轮 + 2 云台 + 2 前导 = 18 actuated joints，另有 6-DOF 浮动基座（“19-DOF”是 body 数误称）。
- 工作量粗估：奖励/编码器/PPO 接口 1–2 人日；物理底座/接触标定/mimic 重接 3–5 人日；LiDAR+depth+detections 的 Warp 重建与逐元素对照 1–2 周。

## 6. 待验证

- dt=5 ms 与 dt=1 ms 的固定动作轨迹/接触保真对比（未做，不做等价性声明）。
- 训练后策略在 dt=1 ms（默认协议）下的评估与基线对比。
- WSL2 原生栈的 GPU 基准（matmul、Warp raycast、64 env × 18 joints 仿真 steps/s、GPU util/功耗），需在 Isaac 训练暂停后执行。
- 原生栈最小训练闭环的原型吞吐（解析 raycast 替代 RTX 传感器），用于决定是否迁移。
