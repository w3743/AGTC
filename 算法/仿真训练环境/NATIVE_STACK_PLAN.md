# 原生 ARM64 训练栈迁移计划（2026-10-25）

目标：把训练从「Isaac Sim + Kit + x64 模拟」迁移到「WSL2 Ubuntu 24.04 ARM64 原生 + 无 Kit 的物理/传感器栈」，以消除当前每 policy step 50 次 PhysX `fetch_results()` 阻塞回读的平台瓶颈。本文只记录已实测事实与可执行计划；未执行项不写成通过。

## 1. 当前状态

| 项 | 状态 |
|---|---|
| Isaac 侧性能改造 | 已实测 19.6×（32 env、dt=5 ms、dec=10、42.65 samples/s），96 iterations / 393,216 transitions 用 2h33m 跑完 |
| Isaac 策略 | `model_95.pt` 已审计（权重有限、SHA `710ccb94…`）；2 场景快速评估 0/2，尚不能到点 |
| Isaac 续训 | `perf_20261025_n32_dt5_cont` 从 model_95 续训 192 iterations，运行中 |
| WSL2 环境 | Ubuntu-24.04-Arm64 已导入；GPU 可见（96 GiB、sm_121）；torch 2.14.1+cu130、warp 1.18、newton 1.7、mujoco 3.17、ovphysx 0.7 全部就绪 |
| 原生基准 | 见 §2 |
| 迁移评估 | 见 `C:/pioneer/wsl-native/INTEGRATION_ASSESSMENT.md` §7（接口、观测等价登记、WBS、M0–M4 退出标准） |

## 2. 原生栈实测基准（WSL2 内，训练结束后独占 GPU）

- torch CUDA matmul 4096³：fp32 1.65–2.94 TFLOPS（前/后段波动）、fp16 8.01、bf16 12.33–13.83；CPU 同尺寸 0.083/0.309。
- Warp：4.19M 粒子 417 G updates/s；mesh BVH raycast 25.6k/262k/901k 射线 = 3.13/9.90/34.77 ms（8.2–26.5 M rays/s）。
- 64 env × 18 joints（14 轮 + 2 云台 + 2 前导，6-DOF base，14 mimic）物理：newton XPBD GPU **33,244 env-steps/s**；MuJoCo CPU **17,271 env-steps/s**；mujoco-warp GPU 8,028（WSL 小批量 launch-bound）；ovphysx 64 clones 62,968（样例场景，非本车）。
- 解析 raycast CUDA（400 线 LiDAR + 160×88 depth × 64 env）：LiDAR 0.157 ms/帧、depth 1.536 ms/帧，20 Hz 摊销 **0.271 ms/策略步**；同场景 mesh BVH 约 5.9 ms/步。
- **平台上限**：96% 负载下 SM 时钟固定在约 663–690 MHz（最大 3003 MHz、P0、无 throttle reason、`power.limit=N/A`、功耗 ~5 W），疑似 vGPU/功耗墙；1.65 TFLOPS fp32 是该平台真实可复现数字，不能外推到裸机。

## 3. 迁移架构（M0 目标）

```
WSL2 Ubuntu-24.04-Arm64（原生）
├─ physics_backend：MuJoCo CPU（先）/ newton XPBD GPU（后）
│    · 18 actuated joints：14 驱动轮 + 2 云台 + 2 前导轮；6-DOF 浮动基座；14 条 mimic
├─ observation_provider：LiDAR 400（解析 raycast 或 Warp kernel）、state 33、feedback 37 必须真实；
│    depth 160×88 / detections 16×8 允许占位（登记在 observation_contract.json）
├─ native_env：gym 接口，obs 键/形状与 policy_contract 一致，20 Hz 策略步，64 env 并行
├─ train_m0：rsl_rl PPO，JSONL 记录 iter/collection/learning/samples-per-s
└─ 评估器：只在五路观测全部 real+verified 时输出能力指标；占位 run 只能标 PROTOTYPE_THROUGHPUT_ONLY
```

## 4. 里程碑与退出标准

| 里程碑 | 内容 | 退出标准 |
|---|---|---|
| M0 | 最小吞吐闭环（LiDAR/state/feedback 真观测，depth/detections 占位） | 64 env 跑通 ≥2 iteration、无 NaN；端到端 policy-step/s ≥ 当前 Isaac 42.65 samples/s 的 5×（目标 ≥200/s）；重复 2 次可复现 |
| M1 | 物理/接触标定 | 与 Isaac 同初态固定动作轨迹对比误差在约定门槛内；μ/求解器参数登记 |
| M2 | 传感器等价 | LiDAR 逐射线 ≤1e-5、depth/detections 与参考实现逐元素对照；五路全部 real 后才允许能力评估 |
| M3 | 训练规模 | 与 Isaac perf 配置同等或更多 transitions；wall-clock/iteration 同口径对比 |
| M4 | 验收 | 默认协议下 16 场景评估（或明确替代口径）+ 实车边界说明；Git 记录协议/SHA/模型 |

## 5. 决策门

- M0 若端到端达不到 5× 当前吞吐，则保留 Isaac perf 配置（已 19.6×），不继续投入原生栈。
- M0 若达标但 M1/M2 保真不可接受，则原生栈仅用于快速实验/超参搜索，正式训练仍走 Isaac 或 DGX Spark。
- 任何能力/避苗/实车声明必须等 M2 完成；占位观测 run 不出能力结论。

## 6. 复现命令

```bash
# 导入发行版（Windows，无管理员）
wsl --import Ubuntu-24.04-Arm64 C:\pioneer\wsl-native\Ubuntu2404 <rootfs.tar.gz> --version 2
# WSL 内环境（官方 aarch64 wheel）
python3.12 -m venv /opt/venv && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
/opt/venv/bin/pip install warp-lang mujoco==3.15 newton ovphysx ovstage numpy
```

- GPU 基准脚本：`C:/pioneer/wsl-native/scripts/gpu_bench.sh`（日志 `/var/log/wsl-native/gpu_bench.log`）。
- 原型代码位置（仓库外）：`C:/pioneer/wsl-native/proto/`，完成后按项目规则整理再决定是否入库。

## 7. 风险登记

1. 观测不等价（depth/detections 占位、解析 LiDAR 与 RTX 差异）→ 只做吞吐结论，能力验收另立。
2. 接触/摩擦不能照抄 PhysX（40 pos-iter、μ0.8）→ 需重标定与轨迹对照。
3. `newton 1.7.0` 与 `mujoco 3.17` 版本不匹配（SolverMuJoCo 多 world 崩）→ 原型用 XPBD 或 pin mujoco 3.15。
4. WSL2 小批量 GPU launch 开销（mujoco-warp 8,028 env-steps/s）→ 原型默认 newton XPBD 或 CPU。
5. 宿主物理内存 32 GiB、WSL 默认上限 15 GiB；64 env 原型需监控内存，必要时调 `.wslconfig`（需用户确认）。
6. GPU 时钟/功耗墙（约 690 MHz）→ 任何原生栈也无法突破该上限，只能减少同步与 Kit 开销。

## 8. M0/M3 实测结果（2026-10-25，WSL2 原型，非等价观测）

同口径定义：1 sample = 一个 env 的一个 policy step；M0 原型 iteration = 64 env × 16 steps = 1024 samples。

| 配置 | 端到端 iteration | samples/s（稳态） | 相对 Isaac 42.65 samples/s |
|---|---|---|---|
| M0 CPU（mujoco_cpu + numpy lidar + CPU 编码器/PPO） | 12.3–12.7 s | 80.4–83.1 | 1.9× |
| M3a（CPU 物理 + GPU 编码器/PPO + warp lidar） | 4.154 s | 246.5 | 5.8× |
| M3b（newton XPBD GPU 物理 + GPU 编码器/PPO + warp lidar） | 3.417 s（复测 3.243 s） | 299.6（复测 315.7） | 7.0–7.4× |

- 首 iteration 含 19–25 s 一次性 CUDA warmup；2-iteration 平均口径为 1.7–2.1×，不达 5×。**门禁按稳态判定通过（M3a 5.8×、M3b 7.0–7.4×）**，报告同时给出两种口径。
- M3b 分项：env_step 2.556 s（≈18,900 env-steps/s，纯物理基准 33,244）、policy_act 0.094 s、learning 0.523 s；RSS 2.8 GB。
- 观测等价性：**未完成**——lidar、state/feedback 未与 Isaac 逐元素对照，depth/detections 为占位；`observation_contract.json` 5 通道 `usable_for_capability=false`，stdout 打印 `PROTOTYPE_THROUGHPUT_ONLY`。
- 已知问题：newton XPBD 只跑 2 iteration，长期稳定性未验证；物理/观测均不等价；`spawn` 高度与接触（ncon=0 滑行）疑点；M3a 已接近 CPU 物理天花板。
- 产物：`C:/pioneer/wsl-native/proto/REPORT.md` §5/§8、`proto/runs/m0_rep_{a,b}/`、`proto/runs/m3_{a,b}_20261025_070929/`、`runs/gpu_samples_m3_{a,b,clock}.csv`。

## 9. 决策建议（待用户确认）

- 吞吐门禁已过：原生栈稳态 7.0–7.4×（M3b），但仅限**吞吐**；能力评估不可用。
- 继续迁移需要 M1（观测等价 1.5–2 人日）+ M2（任务/reset/奖励等价 1–1.5 人日），共 2.5–3.5 人日；完成后需用真观测重跑 M3 确认吞吐不退化。
- 在那之前，正式训练仍建议留在已验证的 Isaac perf 配置（19.6×）；原生栈只用于吞吐实验/超参搜索。
