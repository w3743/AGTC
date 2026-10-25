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

### 正式运行结果（2026-10-25 05:37 自然结束）

- 实际训练时间 **9218.8 s（2:33:19）**，96/96 iterations，无异常退出；最终 checkpoint `model_95.pt`（36,858,339 B，iter 95，393,216 timesteps，Adam step 1920）。
- `model_95.pt` CPU 审计：actor 29 tensors / 1,587,016 参数、critic 8 tensors / 1,478,785 参数，全部有限；SHA256 `710ccb9432b8f27b8d2de6d24f2fba87ffbb54e695bdc50231e3d290f5cf03d9`；contract `num_envs=32`、`num_steps_per_env=128`、backend identity `isaac_sim/physx/rsl_rl`。
- 每轮：collection 中位 **88.49 s**（80.68–103.55），learning 中位 **6.07 s**（5.05–17.92）。
- 学习信号：`Train/mean_reward` -17.03 → **-15.49**；`Train/mean_episode_length` 73.8 → **1135.2**（单回合上限 1200 步）。
- 实测吞吐：`Perf/total_fps` 34 → 45（中位 43 samples/s）；全跑平均 393216/9218.8 = **42.65 samples/s**，相对 a07 的 2.18 samples/s 约 **19.6×**。
- 结论边界：以上是训练吞吐与训练期指标，**不是控制能力验收**；仍需在默认 1 ms 物理协议下做 16 场景评估后才能讨论策略效果。

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

## 7. 策略评估（2026-10-25 06:34）

- 严格评估器（`scripts/evaluate_rsl_curriculum.py` + a07 adapter）要求冻结的 **100 m** 全图 task profile；本 trial 用 12 m 场地（a07 口径），因此评估前对 `size` 只做“正数”检查，其余 fast-stop 语义仍校验；`drive_config.physics_contact` 因评估物理与训练不同而允许不同（本次改为同 5 ms，差异仅 env 数量等元数据）。
- 默认 1 ms 物理的单环境评估实测过慢：一次 16 场景评估在 20 分钟内未完成一个回合的输出，已中止并保留 `eval_final_attempt*/`。改用与训练相同的 `dt=0.005/decimation=10` 物理做快速评估（`run_eval_fast.py`，通过临时替换 `FieldEnvCfg` 实现）。
- 2 场景快速评估（seed 87010/87011，60 s 回合）：**0/2 成功**；1 个 `outside_or_overturned`（355 步）、1 个 `time_limit`（1200 步）；平均最终距离 8.94 m、无损伤/碰撞；**不具备 strict-pass 资格，不作能力结论**。证据：`eval_final_fast/evaluation.json`、`eval_summary_2scenarios.json`。
- 16 场景完整评估若仍用 5 ms 物理，按实测约 6.5 分钟/场景，需约 1.7 小时；尚未运行。

## 8. 续训（2026-10-25 06:42 起）

- 由于 96 iterations 的策略尚不能到点（见 §7），从 `model_95.pt` 继续训练：`perf_20261025_n32_dt5_cont`，`--max_iterations 192`，其余配置不变（32 env、dt=5 ms、dec=10、seed 69041、resume optimizer state），预计约 4.5–5 小时。
- 训练日志：`cont_train.log`；checkpoint 在 `logs/rsl_rl/perf_20261025_n32_dt5_cont/cont/`。

## 13. 策略评估更新（2026-10-25 08:17，5 ms 物理、12 m 场地）

| checkpoint | 场景数 | 成功 | 关键结果 |
|---|---|---|---|
| model_95（393,216 transitions） | 2 | 0 | 1 个 outside_or_overturned（355 步）、1 个 time_limit（1200 步），平均最终距离 8.94 m |
| model_128（536,576 transitions） | 4 | 0 | 2 个 outside_or_overturned（284/756 步）、2 个 time_limit（1200 步），平均最终距离 6.59 m |

- 两次均 `strictpass_eligible=false`，**不构成能力结论**；但趋势说明：平台吞吐问题已解决，策略质量仍是任务/算法层面的问题（536k transitions 仍未学会到点，距离从 8.94 m 降到 6.59 m）。
- 结论口径：**本次交付的是“训练平台性能改造”，不是“策略学会导航”**；若继续提升策略，需要更长训练（百万级 transitions）或任务/奖励/课程调整，不能把吞吐提升等同于能力提升。

## 14. 续训状态（2026-10-25 08:25 起）

- 从 `model_128.pt` 续训 `perf_20261025_n32_dt5_cont3`，`--max_iterations 159`（目标 287），配置不变；预计约 4 小时。
- 暂停/恢复记录：`cont_stop_record.json`、`cont2_stop_record.json`（评估窗口用）。
- 评估产物：`eval_final_fast/`（model_95，2 场景）、`eval_model_128_fast/`（model_128，4 场景）、对应 `eval_summary_*.json`。

## 15. Git / PR 状态（2026-10-25 08:45）

- 分支：`perf/max-throughput`（已推送，基于 `chore/physx-pointmove-training`）。
- 草稿 PR：https://github.com/w3743/AGTC/pull/24（标题见 PR，内容含性能开关、19.6× 实测、评估边界、原生栈基准与限制）。
- 说明：PR 为 draft，等待最终评估/原生栈 depth/detections 决策后再转 ready 或补充。

## 16. 失败模式分析（model_128，4 场景，5 ms 物理）

| seed | 终止原因 | 步数 | 最终距离 | 说明 |
|---|---|---|---|---|
| 87010 | outside_or_overturned（boundary） | 284 | 9.85 m | 14.2 s 越出 12 m 场地边界，朝远离目标方向行驶 |
| 87011 | outside_or_overturned（boundary） | 756 | 10.38 m | 37.8 s 越界，姿态稳定（tilt 0.002 rad） |
| 87012 | time_limit | 1200 | 4.08 m | 接近目标但未进入 0.4 m，末速 0.14 m/s |
| 87013 | time_limit | 1200 | 2.05 m | 最近但仍未进入 0.4 m，末速 0.16 m/s |

- 结论：策略**不是静止/悬空**，而是在行驶；问题是不朝目标收敛（2 例越界）或接近后不满足停稳条件（2 例 0.4 m/0.05 m/s 未达成）。body stability 全程正常（max tilt < 0.08 rad）。
- 这说明限制在**长时程导航/停止策略学习**，不是平台吞吐；需要更多 transitions、奖励/课程调整或更好的探索策略。不能据此归因于原生栈或性能改造。

## 17. 学习诊断（2026-10-25 09:25）

从 TensorBoard 记录的优化量看：
- 学习率（adaptive KL 调度）在可用区间内波动：首轮 train 1e-5–1.3e-3、cont2 1.7e-4–1.3e-3、cont4 1.1e-4–8.7e-4；**没有卡在下界**。
- `Policy/mean_std` 始终在 **0.049–0.051**（init_std=0.05），`Loss/entropy` 约 -6.3 且几乎不变。
- 结合失败模式（越界/接近后不停稳），更像**探索不足 + 长时程停止奖励难以发现**，而不是优化崩溃或平台问题。
- 可选的下一步（需新 trial，不能覆盖现有协议）：提高 init_std/熵系数（例如 0.1–0.15）、加强目标接近/停稳奖励塑形或课程；在 19.6× 的 perf 平台上这些实验成本已大幅降低。


## 18. 系统休眠对训练计时的干扰（2026-10-25 10:46）

- `cont4` 的 iteration 170 记录 collection=4316.0 s（约 72 分钟），原因是宿主在 09:34–10:46 进入休眠，训练进程被挂起；恢复后 GPU 时钟回到 2.1–2.4 GHz，后续 iteration 恢复 ~85 s。
- 已启动用户级 keep-awake 辅助进程（`C:/pioneer/isaac-training/tools/keep_awake.ps1`，SetThreadExecutionState），后续长训练不再受空闲休眠影响。
- 结论：该 4316 s 是休眠墙钟，不是训练变慢；不得当作性能退化或吞吐数据。
