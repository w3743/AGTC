# M0 原生栈最小吞吐原型 — 报告（PROTOTYPE_THROUGHPUT_ONLY）

> **本 run 只能用于吞吐/Go-No-Go 判断，不能用于能力、避苗、损伤、实车结论。**
> `depth` 与 `detections` 是占位通道；`lidar/state/feedback` 是真实计算但尚未与 Isaac 逐元素对照。
> 每次 run 都写 `observation_contract.json`（`capability_metrics_allowed=false`），stdout 首行打印 `PROTOTYPE_THROUGHPUT_ONLY`。
> 本报告所有 M0 数字都是 **CPU-only（MuJoCo CPU + NumPy/PyTorch CPU）**；GPU 未使用（Isaac 续训占用中）。

---

## 1. 环境与版本（实测）

| 项 | 值 |
|---|---|
| 宿主 | Windows 11 ARM64（RTX Spark N1X），Isaac 续训并行运行中 |
| WSL | `Ubuntu-24.04-Arm64`，kernel `6.18.33.1-microsoft-standard-WSL2`，aarch64 |
| Python | 3.12.3（`/opt/venv`） |
| PyTorch | 2.14.1+cu130（M0 仅用 CPU；`torch.set_num_threads(8)`） |
| MuJoCo | **3.17.0**（直连 CPU 后端；3.15 版本约束只针对 newton 的 `SolverMuJoCo`，本原型不用该路径） |
| gymnasium / scipy / numpy | 1.4.0 / 1.18.1 / 2.5.4 |
| newton / warp（M3 后端，M0 未用） | 1.7.0 / 1.18.0 |
| 复用的仓库实现（只读） | `isaac_field/sensor_encoder.py`（纯 torch 编码器，feature_dim=192）、`task.py` 的 STATE_FIELDS/ABSOLUTE_GOAL 布局、`feedback.py` 的 37 维 feedback 布局、`task.py:431-499` 的解析 raycast 思路 |

## 2. 代码与命令

```
C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\native\proto\
  physics_backend.py       # MuJoCo CPU：19 bodies / 19 joints / 24 DOF + 14 mimic，1ms×50，2Nm/轮，PI kp.04/ki.1，μ0.8
  newton_backend.py        # M3 GPU 后端：同模型，newton XPBD（默认不启用）
  observation_provider.py  # lidar 解析 raycast（numpy 或 warp kernel）+ state 33 + feedback 37 + depth/detections 占位 + contract
  native_env.py            # gymnasium Env，obs 键/形状严格等于 policy_contract，动作 4 维，20Hz，64 env 批量
  policy_m0.py             # 复用 isaac_field SensorEncoderCore + 最小 Gaussian actor/critic（192→64→4 / 192→64→1）
  train_m0.py              # PPO（GAE γ.99 λ.95，clip .2，4 epochs，minibatch 256），JSONL + contract + summary
  run_m0_acceptance.sh     # M0 验收：64 env × 16 policy steps × 3 iter，同 seed 跑两次
  run_m3_gpu_acceptance.sh # M3 GPU 门禁：M3a（CPU 物理 + GPU torch/lidar）与 M3b（GPU 物理 + GPU torch/lidar），各 2 iter + nvidia-smi 采样
  smoke_physics.py / smoke_newton.py / profile_*.py  # 分层计时与诊断
```

运行（WSL）：

```bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto
export CUDA_VISIBLE_DEVICES=''          # M0：CPU-only
bash run_m0_acceptance.sh
```

M3（GPU 空闲时）：

```bash
bash run_m3_gpu_acceptance.sh           # 打印 GPU identity、M3a/M3b 各 2 iterations、每项 100ms 采样
```

## 3. 原型范围与偏差（必须连同数字一起读）

- **物理**：19 bodies / 19 joints / **24 DOF**（6-DOF base + 14 轮 + 2 云台 + 2 前导）+ **14 条 mimic equality**；dt=1ms、decimation=50（20Hz 策略步）、每轮 2Nm、PI kp=0.04/ki=0.1（条件抗积分饱和）、μ=0.8。
  - 偏差：μ 值/接触参数未与 PhysX 标定；地面为无限平面（无地形起伏，lidar 用独立解析地形）；力矩按策略步保持、每物理子步用当前 qd 重算 P 项（积分每策略步更新一次）。
- **动作**：`[左履带速度, 右履带速度, 云台yaw角速度, 云台pitch角速度]`，[-1,1]，加速度斜坡 1.5 m/s²/步，与 `env.py` 语义一致。
- **奖励**：简化的 goto 任务（progress×0.15 − 0.01/s + 成功+10/失败−10，goal 0.4m、hold 0.3s、倾覆 1.2rad、60s 截断）。**不是**训练任务，不用于能力结论。
- **观测**：见下表；`state` 用 true pose 当 estimated pose（未模拟估计误差）。
- **并行**：64 个独立 `mjData` 共享 1 个 `MjModel`，**顺序 stepping**（MuJoCo CPU 无跨 world 并行）；不是 GPU 版语义。

| 通道 | 形状 | 模式 | 说明 |
|---|---|---|---|
| lidar | (1,400) | **real** | 解析 raycast（茎圆柱+冠层椭球，8m/0.12m 量程、10Hz、相对噪声 5%、一步延迟队列）；未与 Isaac 逐线对照 |
| depth | (1,160,88) | **placeholder** | 常数 0.5；不等价于 RTX 深度相机 |
| detections | (16,8) | **placeholder** | 全零；不等价于 instance segmentation |
| state | (33) | **real** | ABSOLUTE_GOAL 布局，由仿真状态计算；布局一致、数值未与 Isaac 对照 |
| feedback | (37) | **real** | 21 值+8 有效+8 年龄；由仿真 IMU/云台/电机状态计算（含文档化噪声），未做 VQF 融合 |

`observation_contract.json`（M0 两个 run 相同）：SHA-256 = `7392ceef801853531ab98431112b6201ec5d2fd30391c87a054b4c272d73e4b9`。
M3 起 contract 内含 physics/lidar backend 身份，换后端 SHA 会变。

## 4. 验收结果（64 env × 16 policy steps × 3 iterations，seed 0，重复 2 次）

输出目录（Windows 可直接打开）：
- `C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\native\proto\runs\m0_rep_a\`（`m0_metrics.jsonl`、`observation_contract.json`、`identity.json`、`summary.json`）
- `C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\native\proto\runs\m0_rep_b\`（同上）

### 4.1 每次 iteration（实测）

| run | iter | collection_s | 其中 env_step_s（物理+观测） | 其中 policy_act_s | learning_s | wall_s | env_steps/s（collection） | 端到端 samples/s |
|---|---|---|---|---|---|---|---|---|
| rep_a | 1 | 7.4881 | 5.9952 | 1.4908 | 4.9125 | 12.5483 | 6837.5 | 81.60 |
| rep_a | 2 | 7.5102 | 6.0952 | 1.4114 | 4.6237 | 12.3240 | 6817.4 | 83.09 |
| rep_a | 3 | 7.3597 | 5.7602 | 1.5967 | 5.2162 | 12.7387 | 6956.8 | 80.38 |
| rep_b | 1 | 8.7162 | 6.7682 | 1.9453 | 5.0290 | 13.9383 | 5874.1 | 73.47 |
| rep_b | 2 | 7.5258 | 6.0220 | 1.5012 | 4.5619 | 12.2970 | 6803.3 | 83.27 |
| rep_b | 3 | 7.5289 | 6.0336 | 1.4913 | 4.9323 | 12.7064 | 6800.5 | 80.59 |

**汇总（6 个 iteration）**
- collection：**7.45–8.72 s/iter**（首个 iter 有 warmup 偏差；稳定后 ≈7.5 s）
- learning（PPO 4 epochs × 1024 样本）：**4.56–5.22 s/iter**
- **端到端：≈12.3–12.7 s/iter → 80–83 samples/s（64 env）**
- 流程：51,200 物理子步/iter、1024 样本/iter
- 进程内存（RSS）：**1.08 GB**（宿主 32 GiB、WSL 15 GiB，无压力）
- 复现性：两次 run 的 collection_s 均值差 **6.3%**（7.4527 vs 7.9236），env_steps/s 差 **5.8%**；`observation_contract` SHA 完全相同

### 4.2 与 MuJoCo CPU 物理基准的同口径对比

| 口径 | 值 | 说明 |
|---|---|---|
| `bench_sim.py` 纯物理基准（恒定 0.5 动作、无转向） | **17,271 env-steps/s** | 训练结束后独占 CPU 时测得 |
| 本原型 physics-only（smoke，恒定动作） | 15,996–16,290 env-steps/s | 含每子步显式力矩计算 |
| 本原型 **physics-only + 随机转向动作** | **10,687 env-steps/s** | 接触/求解活跃度上升：`ncon` 0→1.7、`solver_niter` 1→1.9；训练时真实工况 |
| 本原型 **env_step（物理+观测）** | ≈8,300–8,900 env-steps/s | 加 lidar/state/feedback 生成 |
| 本原型 **collection（+策略前向）** | **6,500–6,960 env-steps/s** | 端到端吞吐的分子 |
| 本原型 **端到端（含 PPO 更新）** | **≈4,020–4,160 env-steps/s** | 12.3–12.7 s/iter |

> **结论**：端到端约为纯物理基准的 **24%**；物理仍是大头（≈6 s/12.5 s），PPO 更新 ≈4.9 s（38%），策略编码器前向 ≈1.5 s（12%）。
> 物理在“随机转向”工况下比恒定动作慢约 1.6×，这是 MuJoCo 接触/求解成本，不是 Python 开销。

## 5. 与 Isaac perf 同口径对比（samples/s，Lead 指定口径）

**口径定义（统一）**：1 sample = 1 个 env 走 1 个 policy step；`samples/iteration = env 数 × num_steps_per_env`；
**端到端 samples/s = samples/iteration ÷ iteration_wall_s（含 PPO/编码器）**；collection-only 只除 collection_s。

| 项 | Isaac perf 基线（Lead 提供） | M0 CPU 原型（本报告） |
|---|---|---|
| env 数 | 32 | **64** |
| num_steps_per_env（每 iter） | 128 | **16** |
| samples / iteration | 4096 | **1024** |
| iteration_wall_s | 96.0 s（注：按 42.65 samples/s 反推 4096/42.65≈96.0 s；若按 88.5 s 则应为 46.3 samples/s） | **12.30–12.74 s**（rep_a/b 稳态，见 §4.1） |
| **端到端 samples/s（含 PPO/编码器）** | **42.65**（Lead 给定） | **80.4–83.1**（6 iter：81.60/83.09/80.38/73.47/83.27/80.59；稳态 80–83） |
| collection-only samples/s（不含 PPO） | — | **117.5–139.1**（6 iter；稳态约 136） |
| 每 iter 物理子步（env-steps） | 204,800 | 51,200 |

> “80–83 policy-steps/s”不是 batch-step 速率，而是 **samples/s**（= 1024 samples ÷ 12.3–12.7 s）。
> 它等于“每环境 policy-step × env 数”的聚合：**64 env × 1.28 batch-step/s = 81.9 samples/s**；
> 换言之 batch 级 policy-step 速率只有 **1.28 batch-steps/s**（每个 batch step 含 64 env 的 50 个物理子步）。

**一句话结论：M0 CPU 端到端 = 81.7 / 42.65 ≈ 1.9× Isaac（collection-only ≈ 3.1×），未达到 5× 门禁。**
M0 的观测（depth/detections 占位）、任务（简化 goto）与 Isaac 不等价，该倍数只代表“CPU 原型相对 Isaac perf 的吞吐量级”，
**不能**等价为训练速度或能力结论。要接近 5×，瓶颈在 CPU 物理（6.0 s/iter）与 CPU PPO（4.9 s/iter），必须依赖 M3 的 GPU 物理 + GPU 编码器/PPO。

## 5b. 瓶颈拆解与已做的 CPU 调优

1. **物理（最大项）**：0.38 s/policy-step（64 env × 50 子步 ≈ 3200 次 `mj_step`）。随机转向动作把 `ncon`/`solver_niter` 拉高 → 10.7k vs 18.2k env-steps/s。
2. **PPO 更新（第二大项）**：CPU 上 1024 样本 × 4 epochs = 4.6–5.2 s；实测 thread 调优：**8 线程最快（4.2–4.9 s）**，20 线程反而 15.6 s（小张量线程争用）。`OMP_WAIT_POLICY=PASSIVE` 改善 collection 但使 learning 恶化到 14.5 s，不采用。
3. **策略编码器前向**：≈0.10 s/policy-step（64 env）。分支实测：lidar 分支 27 ms、depth 分支 33 ms（aarch64 torch 卷积很慢，单线程反而比 8 线程快）。真实栈里这部分在 GPU 上。
4. **观测生成**：≈21 ms/policy-step（lidar 每 2 步一次解析 raycast ≈22 ms/帧，其余可忽略）。
5. **多线程物理不可行**：实测 MuJoCo Python 绑定无法有效并行（2 线程 1.20×，4 线程 1.02×，8/16 线程 0.55×）。

## 6. 失败点 / 已知问题（均保留现场）

- 首次 PPO 实现两个 bug：`wheel_target_qd` 广播错误（(2,) vs (14,)）、`advantages` 未展平（(T,B) 当 1-D 用）；已修复并重跑。
- 两次早期运行因 shell 变量展开问题写进同一个 `runs/m0_ 2` 目录（已删除）；验收用 `runs/m0_rep_a/b` 是干净重跑。
- **物理疑点**：恒定动作、从原点出发时 `ncon=0`（轮子与地面无接触），机器人在“滑行”；随机位姿/动作后才出现接触 → spawn 高度/间隙需校准（M0 不影响吞吐结论，但影响物理正确性评估）。
- **numpy lidar 方向 bug（M0 之后发现并修复）**：把 `cos(az)` 又喂给 `cos()` 一次，导致 M0 run 的 lidar 几何错误；吞吐数字不受影响，但 lidar 通道原本就不等价于 Isaac，仍只能用于吞吐。
- lidar 未与 Isaac 的 `Scene.cast`/PhysX raycast 逐线对照；numpy 与 warp 两条解析实现在掠射光线上仍有 float32 命中差异（mean |diff| ≈0.48 m）→ 保持 `usable_for_capability=false`；state/feedback 只有布局一致、无数值对照。
- 本机 CPU 与 Isaac 续训共享（load average 2.4–2.8），吞吐数字有 ±6% 波动；物理基准 17,271 是训练结束后的独占值。

## 7. M3 GPU 门禁（代码就绪，等 GPU；未运行）

- `physics_backend.py` 默认 `mujoco_cpu`；`newton_backend.py` 提供 `NewtonXPBDGBackend`（GPU XPBD，同 19 bodies/19 joints/24 DOF/14 mimic）。
- `observation_provider.py` 新增 `WarpLidarCaster`（Warp CUDA kernel，`--lidar warp` 启用，默认 numpy）。
- `train_m0.py` 新增 `--device cpu|cuda`、`--physics mujoco_cpu|newton_xpbd`、`--lidar numpy|warp`；**默认路径仍为 CPU/numpy，行为不变**。
- `run_m3_gpu_acceptance.sh`：M3a（GPU 编码器/PPO + GPU lidar + CPU 物理）与 M3b（GPU 物理 + GPU 编码器/PPO + GPU lidar）各 2 iterations，每项独立 100 ms nvidia-smi 采样。

M3 前修复的两个关键问题（CPU 验证通过）：
1) **newton backend NaN**：Newton 对 density=0 的 shape 重算惯性导致不稳定。改为 `add_link` 显式给惯性矩阵（按 `isaac_field/assets.py` 公式：车身盒、圆柱轮、云台盒）后，零控制/速度目标 50 子步均 FINITE。
2) **target 索引布局**：`joint_target_qd` 按 **DOF**（`joint_qd_start`）索引，`joint_target_q` 按 **coord**（`joint_target_q_start`）索引；混用会把轮速目标写进云台/轮位置槽导致发散。已修正并 CPU 验证（2 world）。
- 后端偏差：newton 用速度目标（`joint_target_qd`，target_kd=0.04）代替 CPU 的显式每轮力矩 + 侧 PI 前馈；gimbal 用位置目标。

M3 预测（待实测）：M3a 仍受 CPU 物理 6 s 限制（约 2–2.5×）；M3b（GPU 物理 33k env-steps/s + GPU 编码器/PPO）有望把单 iter（51,200 物理步、1024 samples）压到 ≈1.5–3 s → 350–680 samples/s，即 Isaac 42.65 的 8–16×。

## 8. M3 GPU 实测结果（2026-10-25 07:09–07:15，Isaac 续训已暂停，GPU 独占）

设备：`NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)`，sm_121（warp 自报 110 GiB）。
运行：`bash run_m3_gpu_acceptance.sh`（M3a/M3b 各 2 iterations，64 env × 16 steps = 1024 samples/iter）。
输出：`runs/m3_a_20261025_070929/`、`runs/m3_b_20261025_070929/`；日志 `runs/m3_a.log`、`runs/m3_b.log`；采样 `runs/gpu_samples_m3_a.csv`、`runs/gpu_samples_m3_b.csv`、`runs/gpu_samples_m3_clock.csv`。

### 8.1 结果（samples/s 同口径 = samples/iteration ÷ 端到端 iteration 时间）

| 配置 | iter1 wall | iter2 wall（稳态） | 稳态 samples/s | collection-only samples/s | 2-iter 平均 samples/s |
|---|---|---|---|---|---|
| **M3a**：GPU torch（编码器/PPO）+ GPU warp lidar + **CPU 物理** | 24.65 s（含一次性 CUDA/cuDNN warmup） | **4.154 s** | **246.5** | 281.8 | 71.0（2048/28.84） |
| **M3b**：GPU torch + GPU warp lidar + **GPU 物理（newton XPBD）** | 19.70 s（同上） | **3.417 s** | **299.6** | 353.5 | 88.4（2048/23.17） |
| M3b 复测（带 SM 时钟采样，`run_m3_clock.sh`） | 19.44 s | 3.243 s | **315.7** | 378.0 | 90.2 |

- 稳态 **M3b = 299.6–315.7 samples/s**，相对 Isaac perf 42.65 → **7.0–7.4×**；**M3a = 246.5 → 5.8×**。
- 门禁（213 samples/s = 5×）在**稳态**下 **M3a/M3b 均通过**；但**首次 iteration 含一次性 CUDA context/cuDNN/内核编译 warmup**（19–25 s），
  把 2-iteration 平均拉到 71/88 samples/s（1.7×/2.1×）。对 175-iteration 续训而言 warmup 摊销后可忽略（<0.1 s/iter），
  因此**门禁判定以稳态为准 = 通过**，但报告同时保留保守的 2-iter 平均。
- 分项（M3b 稳态 3.243 s/iter）：env_step 2.556 s（含 64×50 物理子步 + 观测）、policy_act 0.094 s、learning 0.523 s；
  GPU 物理吞吐 ≈ **18,900 env-steps/s**（含观测/read，纯物理基准 33,244）；M3a 的 env_step 0.214 s（CPU 物理）比 M0 的 0.38 s 更快，
  因为 torch 移到 GPU 后不再与 CPU 物理争线程。
- 内存：进程 RSS ≈2.8 GB；GPU 显存峰值 ≈4.9 GB。

### 8.2 GPU 采样（util / 时钟 / 功耗）

| 运行 | 采样 | util 均值/峰值 | SM 时钟 | 功耗均值/峰值 | 显存峰值/温度 |
|---|---|---|---|---|---|
| M3a | 341 | 7.2% / 74% | 未采（见下） | 1.94 / — W | 4,884 MiB / 45 °C |
| M3b | 404 | 8.9% / 77% | 未采（见下） | 2.15 / — W | 4,925 MiB / 45 °C |
| M3b 时钟复测 | 266 | 10.6% / 77% | **均值 963 MHz，峰值 2236 MHz，最低 234 MHz** | 2.29 / 4.57 W | 4,925 MiB / 45 °C |

> util 均值低（7–11%）是因为工作负载是“短 kernel + host 往返”的突发型（单个 iter 3–4 s），峰值 74–77% 表明 kernel 确实在跑；
> **功耗读数仍不可信**（2–5 W）；SM 时钟在突发负载下可短暂 boost 到 2.2 GHz，但均值只有 ~1 GHz，符合此前 matmul 复核发现的“时钟/功耗受限”特征。

### 8.3 失败点与限制

- M3 仅 2 iterations：newton XPBD 的长期稳定性（>100 iter）与 NaN 风险尚未验证；物理动力学与 CPU/Isaac 不等价（速度目标代替显式力矩 + PI）。
- warp lidar 与 numpy 在掠射光线仍有 float32 命中差异，均未与 Isaac 逐线对照。
- M3a 的 env_step 仍是 CPU 物理（0.21 s/policy step），因此 5.8× 已接近 CPU 物理天花板；M3b 的 7–7.4× 才体现 GPU 物理增益。
- 一次性 warmup 未从官方 run 中剔除；建议正式基准用“第 2 个 iteration 起”的稳态值，并在脚本里加 `--warmup-iters 1`。

