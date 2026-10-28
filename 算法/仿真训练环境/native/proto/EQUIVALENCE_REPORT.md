# 原生栈观测/任务等价性报告（M1 + M2，CPU-only）

> 本报告只登记**等价性证据**与**仍然存在的差距**；不改变「原型吞吐 run 不能用于能力/避苗/实车结论」这一约束。
> 所有对照都使用仓库 reference 实现（只读 import），没有修改仓库；全程 CPU、GPU 不碰；RSS 峰值：M1 **213 MB**、M2 **573 MB**（要求 ≤1.5 GB）。
> 数据文件：`m1_verification.json`、`m2_verification.json`；脚本：`m1_lidar.py`、`m1_state_feedback.py`、`run_m1_verify.py`、`reward_fast_stop.py`、`m2_equivalence.py`。

---

## 1. 结论速览

| 项 | 参考实现 | 结果 | 可用于能力验收 |
|---|---|---|---|
| lidar（numpy 解析） | `task.py Scene.cast` (431-499) | 固定 scene（400 株/3200 叶/seed 77001）、16 位姿×400 线=6400 射线：**hit-id 不一致 0，距离 max diff 0.0 m（逐位精确）**；100% ≤1e-5 | 是（几何等价） |
| lidar（Warp kernel） | 同上 | **hit-id 不一致 0**；距离 max diff **3.35e-4 m**（2/6400 掠射光线超 1e-5，均值 6.6e-8） | 是（容许浮点掠射差异） |
| state 33 | `task.py FieldTask._observation` (1537-1619) | 8 组随机输入逐元素：**max abs error 4.05e-7**（纯 float32 舍入） | 是（布局+数值等价） |
| feedback 37（直通姿态） | `feedback.py SimFeedbackEstimator(noise_enabled=False)` | **37 维 max abs error 2.9e-8**，valid/age 全 0 误差 | 是（缩放/布局等价） |
| feedback 37（VQF 路径） | 同上（不供外部姿态，走 VQF 2.1.2） | 16 个非姿态值仍 2.9e-8；**attitude quaternion 最大差 0.976967** | **否**（未移植 VQF） |
| depth 160×88 | RTX Camera | 占位常数 0.5 | 否 |
| detections 16×8 | instance segmentation | 占位全零 | 否 |
| spawn 接触 | Isaac `env.py` spawn 约定 | 零动作 20 步 ncon 均值 **28**、静止 z 稳定 0.149957 m；+1 cm clearance spawn 同样稳定、1 s 漂移 <1e-5 m | — |
| reset 语义 | `sample_episode_start_goal` + `pose_clear`/`validate_connectivity` | 纯地形场景 3/3 通过；带 400 株场景 pose_clear 5/5 通过，但场景本身有 202/965,849 不可达格子（连通性检查为保守全连通，不作为单次 reset 失败） | — |
| reward/termination | `tensor_task.fast_stop_v2_step` (torch CPU) | adapter vs 直接调用：**rewards/12 个分量逐位相等**；环境集成：`fast_stop_v2` task backend 的 env reward 与该状态重算的 reference reward **全部相等** | 是（奖励/终止等价） |
| 60 s 超时 | fast_stop_v2 语义 | `done=True`、`terminated=True`、`truncated=False`、reason=3（time limit）、失败奖励+剩余时间惩罚；**无需 bootstrap** | — |
| VQF 旁路 | `SimFeedbackEstimator` + python `vqf==2.1.2` | 同一 estimator 两条独立路径 **残差 0.0**，相对直通姿态误差从 0.977 → 0；稳态成本 **1.264 ms/env/update**（≈1.62 CPU 核 @64 env×20 Hz） | 旁路可用（需接入） |

**当前 capability_blockers（写进 `m1_verification.json`）**：depth 占位、detections 占位、feedback VQF 姿态未移植（0.977 差；已有旁路）。

## 2. M1 细节

### 2.1 lidar

- reference：`Scene.cast`（地形 raycast + 叶片椭圆面片 + 茎圆柱 + 岩石）。
- 候选 1（numpy，`m1_lidar.candidate_cast`）：按 reference 同一数学、同一 float32 dtype 的顺序实现，逐位相等（max diff **0.0**）。
  - 关键坑：reference 把 origin/directions 转 float32；候选若用 float64，掠射光线 disc≈0 处会翻转 hit/miss（曾出现 1/6400 不一致，max 0.63 m）。
- 候选 2（Warp kernel，`candidate_cast_warp`）：6400 射线、3200 叶/400 株；hit-id 0 不一致，2 条掠射光线距离差 3.35e-4 m。
  - 关键坑：origin 按 env 广播（曾只有 n 个 origin，kernel 读越界 → 全部 miss）。
- 地形分支：水平射线 z=0.55 m、地形幅值 0.35 m，reference 的 terrain 命中数为 **0**（已记录）；因此候选未实现地形求交，验证覆盖叶片/茎。
- 未覆盖：岩石（`rock_density=0`）、非水平射线、`lidar_relative_error` 噪声（噪声在 `_lidar_noise` 单独做，未纳入等价性对照）。

### 2.2 state 33

- 参考：`FieldTask._observation()` 的 state 段（ABSOLUTE_GOAL schema）。
- 输入：随机 pos/yaw/gimbal/goal/action/time、三种传感器的 capture 时间与帧位姿（量化到 policy 步，和原型一致）。
- 结果：max abs error **4.05e-7**；此前发现的语义差异已修正：last_action 用**原始动作**（不是 slew 后 track_command）、三路传感器的 gimbal delta 不再置零、最终 vector clip 到 [-1,1]。

### 2.3 feedback 37

- 直通姿态模式：37 维全对齐（max 2.9e-8）。
- VQF 模式：非姿态 16 个值完全一致；attitude quaternion 差 0.976967（VQF 融合 vs 直接四元数）→ **feedback 未完全等价**。

## 3. M2 细节

### 3.1 spawn 接触（此前 "ncon=0 滑行" 结论修正）

- 零动作：ncon 均值 **28**（14 轮×2 接触点）、静止 z=0.149957 m、无非物理漂移。
- Isaac 风格 +1 cm clearance spawn：同样 settle 到 0.149957 m，1 s 水平漂移 <1e-5 m。
- 原 "ncon=0" 是**激进驱动**（2 Nm/轮 + action 0.5）下代理车被反扭矩抬翻（z 升到 ~1.38 m）导致的读数，不是接触缺失；这是 drive/COM 物理保真度差距，已记录为 M2 发现。
- 环境已改为 +1 cm spawn clearance（与 `isaac_field/env.py` 一致）。

### 3.2 reset 语义

- 用 reference `sample_episode_start_goal` 采 5 组 start/goal；`scene.pose_clear(include_small_rocks=True)` 5/5 通过，`validate_connectivity` 因场景本身 202 个格子不可达返回 False（属场景属性，不是采样失败）。
- 纯地形（plants=0）场景 3/3 `pose_clear + connected` 全通过 → sampler/validator 语义对齐。
- `NativeFieldEnv.reset_with_poses(poses, goals)` 已实现，可直接吃 reference 采样位姿。

### 3.3 reward/termination（fast_stop_v2）

- `reward_fast_stop.FastStopV2Adapter` 是对 `tensor_task.fast_stop_v2_step` 的 numpy↔torch 边界；adapter 输出与该函数直接调用**逐位相等**（rewards + 12 个分量）。
- `NativeFieldEnv(task_backend="fast_stop_v2")` 已接入；同状态重算 reward 与 env 输出**全部相等**（4 env 首次转移）。
- 60 s time limit：reference 语义为 **terminal failure**（done/terminated=True、truncated=False、reason=3、failure+remaining-time 惩罚）；因此 fast_stop_v2 口径下超时**不需要 bootstrap**。训练 GAE 现有实现用 `done=terminated|truncated`，在改用 fast_stop_v2 后 `truncated` 恒 False，不会引入错误 bootstrap。

### 3.4 VQF 旁路实验

- 直接用 python `vqf==2.1.2`（与 reference 相同的 estimator）两条独立路径残差 **0.0**；相对当前直接姿态实现，quaternion 误差从 0.977 → 0。
- 稳态成本 **1.264 ms/env/update**；64 env × 20 Hz ≈ **1.62 CPU 核**（32 env ≈0.81 核）。参考实现（Isaac env）本身也在 CPU 上按 env 运行 VQF，所以这是原生 CPU 栈的真实成本。
- 工量估计：
  - **接入 python VQF 旁路**：约 **0.5 人日**（provider 增加 per-env estimator、state/feedback 联动、重跑 M1 验证）。
  - 若要在 64 env 全速下压成本：**移植 torch 批处理 VQF** 约 **2–4 人日**（含数值等价验证）；warps 版更久，不建议先做。
  - 折中：VQF 降到 5–10 Hz 更新（每 2–4 个 policy step）可把成本降到 0.4–0.8 核，但姿态延迟需重新标注。

## 4. 对能力验收的含义

- 仍**不能**做能力验收：`capability_metrics_allowed=false`，因为 depth/detections 是占位、feedback VQF 未接入（旁路已验证但未集成）。
- 若完成「depth/detections 真实化 + VQF 旁路接入」，则 5 通道中 lidar/state/feedback 可标 usable；depth/detections 的误差门槛需另行定义（本报告未覆盖）。
- 本报告所有对照均为 CPU、小幅数据；GPU 路径（warp lidar）只做 CPU 一致性验证，未在 GPU 上复测。

## 5. 复现命令

```bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto
export CUDA_VISIBLE_DEVICES=''          # CPU-only
/opt/venv/bin/python run_m1_verify.py   # -> m1_verification.json ; RSS ~213 MB
/opt/venv/bin/python m2_equivalence.py  # -> m2_verification.json ; RSS ~573 MB
/opt/venv/bin/python m1_vqf_verify.py   # -> m1_vqf_verification.json ; RSS ~65 MB
```

## 6. VQF 旁路接入与验证（新增，2026-10-25 07:2x）

**实现**：`ObservationProvider(feedback_backend="vqf")` 使用 `isaac_field.feedback.SimFeedbackEstimator` + python **vqf 2.1.2**，
逐 env 喂入 engine-truth 的 accel/gyro/mag/gimbal q/qd/motor Iq/RPM，按 episode 时间单调推进；env reset 时同步 `estimator.reset()`。
`feedback_backend="direct"` 仍为默认（旧行为不变）。

**验证**（独立 reference estimator，noise 关闭）：
- **attitude quaternion max abs error = 0.0**，full 37 维 max abs error = 0.0 → 目标 ≤1e-6 **达成**（此前直通姿态路径差 0.977）。
- 数据：`m1_vqf_verification.json`；已并入 `m1_verification.json` 的 `feedback_vqf_bypass` 字段。

**CPU 成本**（WSL，aarch64，python 循环，串行）：
- 0.456 ms/env/update（noise 关闭）、**0.758 ms/env/update（noise 开启，实际路径）**；
- 64 env ≈ **48.5 ms/policy step**、**0.78 s / 16-step iteration**（串行时间直接加到墙钟，不能跨核并行）。

**吞吐影响（CPU 同机对比，64 env × 8 steps）**：

| feedback | collection_s | end-to-end_s | samples/s（端到端） |
|---|---|---|---|
| direct | 4.83 | 7.77 | **65.9** |
| vqf | 5.49 | 8.62 | **59.4**（-9.9%） |

推算到 M0 稳态 16-step iteration（collection 7.5 s）：VQF 增加 ≈16×83 ms ≈ 1.33 s → collection ≈8.83 s、端到端 ≈13.7 s →
**≈75 samples/s（相对 81.7 下降 ~8%）**。

**32 env 建议**：不建议为了 VQF 降 env 数。VQF 串行时间随 env 线性减半，但 samples 也减半，而 learning（≈4.8 ms/sample）与固定开销占比上升；
估算 32 env 端到端约 50–55 samples/s，低于 64 env 的 ~75。**建议保持 64 env**；若要全速再考虑 torch 批处理 VQF（2–4 人日）。

**contract/blockers 更新**：feedback channel 增加 `backend: vqf_reference|direct_analytic`；`capability_blockers` 只剩
depth 占位、detections 占位、以及「state 仍用真值 heading/gimbal（reference 用 VQF 估计位姿，差异未量化，~0.25 人日 follow-up）」。

## 7. 剩余能力缺口设计

见 `CAPABILITY_GAPS.md`（depth 160×88 的 Warp raycast 可行性/域差、detections 16×8 的解析推导、以及域差登记字段与验收口径；
CPU 估算合计 **6–9 人日**，depth 先行）。
