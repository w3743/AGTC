# 原生 ARM64 训练栈迁移计划（2026-10-25）

> 当前实现已入库 native/，五路真实观测与显式 fast_stop_v2 训练路径已接入。本文件保留历史阶段计划；当前配方、实测与边界以 [迁移验收说明](native/MIGRATION_ACCEPTANCE.md) 为准。

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
- **平台上限**：高负载（~96%）时 SM 时钟会降到约 663–690 MHz、功耗读数仅 ~5 W（`power.limit=N/A`，读数不可信）；但负载 30–50% 时实测 2.1–2.4 GHz（窗口采样 max 2385 MHz），**不是固定锁频**，更像功率/热预算限制。原生栈负载峰值 2236 MHz；软件无法突破该预算；1.65 TFLOPS fp32 是该平台真实可复现数字，不能外推到裸机。

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

## 10. M1 观测等价进展（2026-10-25 07:21，CPU-only，RSS 213 MB）

对照实现全部用仓库 reference 代码（只读）：lidar = `task.py` 的 `Scene.cast`，state = `FieldTask._observation`，feedback = `SimFeedbackEstimator(noise_enabled=False)`（为跑 VQF 在 WSL 装了 `vqf==2.1.2`）。

| 通道 | 结果 |
|---|---|
| lidar（numpy） | 固定 scene（400 株/3200 叶/seed 77001）、16 位姿 × 400 线 = 6400 射线：**hit-id 不一致 0，距离 max diff 0.0 m**，100% ≤1e-5；terrain 分支命中 0（水平射线 0.55 m vs 地形幅值 0.35 m，已验证） |
| lidar（warp） | 同数据：hit-id 不一致 0，距离 max diff **3.35e-4 m**（6400 中 2 条掠射光线超 1e-5，均值 6.6e-8）→ 可用，需容忍 float32 掠射差异 |
| state 33 | 8 组随机输入逐元素对照：**max abs error 4.05e-7**（float32 舍入；age/动作/云台 delta 字段全部对齐） |
| feedback 37 | 直通姿态模式：**37 维 max abs error 2.9e-8**；VQF 模式（Isaac 真实路径）：16 个非姿态值仍 2.9e-8，但 **attitude quaternion 最大差 0.977** → feedback 未完全等价 |
| depth / detections | 仍占位，能力指标继续禁止 |

- 修复：provider 的 state 语义（last_action 用 slew 后命令、云台 delta 置 0）对齐 reference；warp lidar origin 广播 bug（曾导致全部 miss）。
- 产物：`proto/m1_lidar.py`、`proto/m1_state_feedback.py`、`proto/run_m1_verify.py`、`proto/m1_verification.json`；`observation_contract.json` 新增 `verification` 字段，train 时自动嵌入。
- 剩余阻塞：VQF 姿态未移植（需评估用 Python `vqf` 包旁路或重写数值版本）；M2（spawn 接触、reset 语义、reward/termination、timeout bootstrap）进行中。

## 11. M2 任务等价与 VQF 旁路（2026-10-25 07:30，CPU-only）

- **spawn 接触**：修正此前“ncon=0 滑行”的误判——零动作下 `ncon` 均值 28、z 稳定 0.149957 m；原 ncon=0 是 2 Nm/轮 + action 0.5 的激进驱动把代理车抬翻（z→1.38 m）。环境已改为 Isaac 式 +1 cm clearance，1 s 漂移 <1e-5。
- **reset**：复用 reference `sample_episode_start_goal`；纯地形场景 3/3 `pose_clear+connected` 通过；400 株场景 pose_clear 5/5 通过，但场景本身 202/965,849 格不可达（场景属性，不是 reset bug）。已加 `reset_with_poses()`。
- **奖励/终止**：`FastStopV2Adapter` 与 `tensor_task.fast_stop_v2_step` **逐位相等**（rewards + 12 分量）；env 集成 reward 与同状态重算全等。
- **60 s 超时**：reference 语义为 terminal failure（`done/terminated=True`、`truncated=False`、`reason=3`、失败+剩余时间惩罚）→ **无需 bootstrap**，现有 GAE done 掩码正确。
- **VQF**：可直接复用 WSL 内 python `vqf==2.1.2`（与 reference 同包）；两条独立 estimator 路径残差 **0.0**，relative 直通姿态误差 0.977 → 0。成本 **1.264 ms/env/update**（64 env × 20 Hz ≈ 1.62 CPU 核；32 env ≈ 0.81 核）。接入 0.5 人日；全速 torch 批处理移植 2–4 人日；折中降频 5–10 Hz 可到 0.4–0.8 核（需重新标注姿态延迟）。

### 等价性总表（详见 `C:/pioneer/wsl-native/proto/EQUIVALENCE_REPORT.md`）

| 通道 | 状态 |
|---|---|
| lidar numpy | 逐位精确（6400 射线，0 差异） |
| lidar Warp CUDA | hit-id 一致；2/6400 掠射光线差 3.35e-4 m |
| state 33 | max 4.05e-7 |
| feedback 37（直通姿态） | max 2.9e-8 |
| feedback 37（VQF） | 未接入原生栈，姿态差 0.976967（旁路已验证 0 残差，接入 ≈0.5 人日） |
| depth 160×88 / detections 16×8 | **仍占位** |

→ `capability_metrics_allowed=false`；M0/M3 吞吐结论不受影响。

### 能力验收剩余缺口（按优先级）

1. feedback VQF 旁路接入（0.5 人日，CPU 成本 ~1.6 核@64env）；
2. depth 160×88：Warp raycast 替代 RTX，需与 Isaac RTX 做域差登记；
3. detections 16×8：从解析场景几何推导，需与 reference 逐元素对照。

## 12. VQF 旁路接入与能力缺口设计（2026-10-25 07:34，CPU-only）

- **VQF 旁路已接入**：`ObservationProvider(feedback_backend="vqf")` 使用 `isaac_field.feedback.SimFeedbackEstimator` + python `vqf==2.1.2`，逐 env 喂 engine-truth 信号、按 episode 时间单调推进、reset 同步 estimator.reset；`direct` 仍为默认。
- **验证**：与独立 reference estimator（noise off）对比，**attitude quaternion max abs error = 0.0，full 37 维 = 0.0**（此前直通姿态差 0.976967）。
- **成本**：0.456 ms/env/update（noise off）/ 0.758 ms（noise on 实际路径）；64 env ≈ 48.5 ms/policy step、0.78 s/16-step iter（python 串行）。
- **吞吐影响**（CPU 同机 64env×8step）：direct 65.9 samples/s → vqf **59.4 samples/s（-9.9%）**；推算 M0 稳态 16-step ≈ **75 samples/s（-8%）**。不建议降到 32 env（VQF 串行 + learning 固定成本占比上升，估 50–55 samples/s）。
- **能力缺口设计**（`C:/pioneer/wsl-native/proto/CAPABILITY_GAPS.md`）：
  - depth 160×88：Warp 解析 raycast 可行，CUDA 估 4–6 ms/帧（0.5–0.8 ms/step 摊销），CPU 1–2.5 s/帧；主要风险是光栅化 vs raycast 的边缘/遮挡域差与轴向 Z vs 径向距离换算；实现 2–3 人日 + RTX 对照 1–2 人日。
  - detections 16×8：从解析 mask→bbox 推导，CUDA 0.4–0.6 ms/帧；语义依赖 mask→bbox/confidence 规则，需先确认 `task.py`；实现 2–3 人日 + 对照 1 人日/轮。
  - 域差登记：contract/verification 新增 `sensor_domain_ap` 字段（isaac_source / native_source / expected_differences / edge_policy / usable_for_capability）。
  - 合计 **6–9 人日**，建议 depth 先行。
- 收口：`proto/` 已含吞吐原型、等价性报告、迁移评估、缺口设计；剩余 blockers 只剩 depth/detections 占位 + state 用真值位姿的 ~0.25 人日 follow-up；`capability_metrics_allowed=false` 不变。

## 13. M3b-4096：与 Isaac 同样本量的 wall-clock 对比（2026-10-25 08:52，GPU 窗口 55 s）

- 配置：`train_m0.py --num-envs 64 --num-steps 64 --iterations 3 --device cuda --physics newton_xpbd --lidar warp`，**4096 samples/iter**（与 Isaac 32 env × 128 steps 同样本量）。
- 结果（iter2–3 稳态）：wall **12.48 s/iter**、collection 10.28 s、learning 2.18 s；**328.2 samples/s**（collection-only 398.4）；物理吞吐 ≈19,900 env-steps/s。
- iter1 含一次性 CUDA/cuDNN warmup：29.696 s（policy_act 首次 11.8 s、learning 7.2 s），已单列，不计入倍数。
- 对比 Isaac（4096 samples/iter，88–96 s）：**同样本量下原生栈约 7.1–7.7× 更快**（iter wall 12.5 vs 88–96 s）。
- 口径限制：原生 64 env vs Isaac 32 env；Isaac 侧含 Isaac Sim/Kit 渲染与逐步 PhysX 同步开销；不是逐项受控的等工作量基准，不能外推为“训练总时长必然缩短同倍数”。
- GPU 采样：util mean 18.4% / p95 71% / max 79%；SM clock mean 1181 MHz / max 2236 MHz（仍低于标称 3003）；mem 峰值 5396 MiB；power 读数不可信。
- 产物：`C:/pioneer/wsl-native/proto/runs/m3b_4096/`（run.log、m0_metrics.jsonl、gpu_samples.csv、summary/identity/contract，contract SHA `452357d9…`）。
