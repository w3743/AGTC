# WSL 训练迁移准备（2026-10-25）

## 决策

- 用户 2026-10-25 决定：**放弃 Isaac 训练，以后以 WSL 原生栈为主训练环境**。
- Isaac 训练已停止（`cont4` 在 iter ~183 被终止，最后完整 checkpoint `model_176`，741,376 timesteps）；不再启动新的 Isaac 训练。
- Isaac 仅保留为对照/验收工具；正式验收结论仍以真实运行与实测数据为准。

## 已完成

1. WSL 环境就绪：`Ubuntu-24.04-Arm64`、GPU 可见（RTX Spark N1X, 98304 MiB）、torch 2.14.1+cu130 / warp 1.18 / newton 1.7 / mujoco 3.17 / ovphysx 0.7；根盘 ~945 GB 可用。
2. 训练骨架 `C:/pioneer/wsl-native/wsl_train/train_wsl.py`：
   - 64 env、newton_xpbd GPU 物理、Warp lidar、VQF feedback、PPO；
   - JSONL 指标、`observation_contract.json`、`identity.json`、`checkpoint_{iter}.pt`/`latest.pt`；
   - `--save-interval` 与 `--resume` 已冒烟验证：iter2 628 samples/s、resume 从 checkpoint_2 到 iter 3–4、contract SHA 校验通过。
3. 训练冒烟结论（仅吞吐，非能力）：64 env × 16 steps，稳态 ~600–650 samples/s；相比 Isaac perf 的 42.65 samples/s 约 14–15×（不同 batch 下固定开销差异，正式对比用 4096 samples/iter 口径：12.48 s vs 88–96 s = 7.1–7.7×）。

## 尚未就绪（训练前必须关闭）

1. **depth 160×88**：✅ 已完成（2026-10-25 11:20）：Warp 解析深度与独立 numpy caster 对照 6 位姿 × 14,080 像素，**mask 不一致 0、valid ratio 0.4044/0.4044、距离 max 2.06e-4 m、100% ≤1e-3 m**；provider 集成对照 max diff 0.0；`observation_provider` 新增 `depth_backend="analytic"`，`train_wsl.py --depth analytic` 已冒烟（iter2 collection 2.06 s、496 samples/s，contract depth=real）。剩余：与 Isaac RTX 的域差登记（未逐像素对照）。
2. **detections 16×8**：✅ 已完成（2026-10-25 11:35）：`proto/detections_caster.py` 按 env.py 语义实现，与 `FieldTask._observation` 同输入对照 **max abs error 0.0**（7 位姿、34 条检测；字段顺序/缩放/range 分支/排序/截断全对齐），provider 集成 max diff 0.0；`train_wsl.py` 已加 `--detections analytic`。CPU 成本：mask 渲染 ≈6.0 ms/帧（64×64），需 GPU 版本进训练循环。
3. 五通道现在全部 `mode=real`（lidar/depth/detections analytic、state 4.05e-7、feedback VQF 0.0）；`capability_metrics_allowed=false` 的剩余 blocker 只有 **RTX 域差未逐像素对照**（depth/detections 的 `sensor_domain_gap` 已登记验收建议）。

## 进入正式 WSL 训练的验收门

- depth/detections 与 reference 逐元素对照通过（或明确的域差登记），contract 五通道全部 `real`；
- 用真观测重跑 `train_wsl.py` 2 iterations，吞吐不明显退化；
- 冻结本地 native 试验协议（观测/奖励/终止/种子/评估口径）并记录 SHA；
- 训练产出的 checkpoint 能保存/恢复，评估流程明确（native 自评 + 必要时 Isaac 跨环境抽检）。

## 风险

- 原生栈物理/传感器与 Isaac RTX 存在域差（depth 为几何射线、detections 为几何推导），策略可迁移性需实车/Isaac 对照验证；
- WSL2 非项目官方支持环境，长期稳定性需观察；
- 训练规模化后注意宿主 32 GB 内存与 WSL 默认内存上限。

## 2026-10-25 11:30 更新：五通道全 real + WSL pilot 训练

- depth 已完成并接入（`depth_backend="analytic"`）：6 位姿 × 14,080 像素对照 mask 不一致 0、距离 max 2.06e-4 m；GPU 批渲染实测 **1.013 ms/帧**（比 4–6 ms 目标更快）。
- detections 已完成并接入（按 env.py 语义）：与 `FieldTask._observation` 同输入 **max/mean abs error 0.0**；provider 集成 max diff 0.0。
- 真观测 smoke（64 env × 16 steps）：稳态 wall **2.897 s/iter**、**353.5 samples/s** 端到端；contract 五通道全 real（lidar/depth/detections analytic、state、feedback vqf）；`capability_metrics_allowed=false`，唯一 blocker 是 **RTX 域差未逐像素对照**。
- 已启动 WSL pilot 训练（`wsl_train/runs/pilot_20261025`）：64 env × 32 steps × 300 iterations、newton_xpbd + warp lidar + vqf + analytic depth/detections，每 20 iter 存 checkpoint（训练使用原型 MLP 策略，仅验证 WSL 训练闭环/吞吐/学习信号，**不作能力结论**）。
- 后续：detections Warp mask kernel（预期把稳态吞吐从 353 提回 550–650 samples/s）、native 评估脚本（60 s 场景/固定种子/停稳判据）、真实 LSTM 策略接入与本地协议冻结。

## 2026-10-25 11:35 更新：warp detections / eval_native / 成本定位

- warp detections mask GPU：64 位姿 × 4096 射线 = **0.658 ms/帧**（比 numpy 5.6 ms 快 ~8.5×），但 64 pose 逐次 kernel launch 使批成本 42 ms（未达 ≤30 ms 目标）。
- `--detections-warp` 端到端 smoke：2.948 s/iter、347.3 samples/s；与 analytic CPU detections（2.897 s、353.5）**无实质差异**。修正：此前“detections ≈1.5 s/iter 是主成本”是由微基准外推，非实测；本配置下 **physics 主导（~2.6 s/iter）**。
- `eval_native.py` 链路已验证：pilot `checkpoint_40` 2 场景均可加载/评估，但该 40-iter 原型策略在 native 域翻车（49/182 步、flipped），**不构成能力结论**；evaluation.json 已生成。
- 下一步（CPU-only）：provider 加 per-sensor 计时；native 物理加可配置 `--sim-dt/--decimation`（默认 1 ms/50 不变），实验 `dt=0.005/dec=10`（与 Isaac perf 同 5 ms 物理，预期 physics 成本降 ~5×）；warp detections 批量化单 launch。

## 2026-10-25 11:40 现状与下一步排期

- 已完成：五通道 real；depth/detections 与 reference 对照通过；`eval_native.py` 链路可用；WSL pilot（原型 MLP）49 iter 验证了训练/checkpoint/恢复；warp detections mask 单帧 0.658 ms 但逐 pose launch 42 ms/批、端到端无收益。
- 成本定位修正：16-step 配置下 **physics 主导（~2.6 s/iter）**，不是 detections。
- 下一步排期（队友执行，CPU 优先）：
  1. provider per-sensor 计时（`--profile`，默认零开销）；
  2. native 物理可配置 `--sim-dt/--decimation`（默认 1 ms/50 不变），实验 5 ms/10（预期 physics 成本 ~5× 降低）；
  3. warp detections mask 单 launch 批量化（目标 ≤30 ms/批）；
  4. **真实 LSTM-512 策略接入**（`wsl_train/policy_lstm.py` + `--policy lstm`，含序列化 PPO 更新）；
  5. 之后做 GPU A/B（1 ms/50 vs 5 ms/10），冻结 native 试验协议，启动正式 WSL 训练。

## 2026-10-25 11:45 A/B 与成本再定位

- **batch warp detections mask**：64 poses 从 per-pose launch 52.3 ms/批 降到**单 launch 5.66 ms/批（0.0885 ms/帧，11,305 帧/s）**，≤30 ms 目标达成；与逐 pose 逐像素一致。
- **A/B（64 env×16 steps×2 iter，全 real 观测）**：
  | 配置 | wall | collection | env_step | samples/s |
  |---|---|---|---|---|
  | 1 ms/50 | 3.031 s | 2.828 s | 2.704 s | 337.9 |
  | 5 ms/10 | 2.211 s | 1.938 s | 1.848 s | **463.1** |
  - 5ms 相对 1ms 仅 **1.37×**（不是物理子步数的 5×），因为 observation 合计 ~1.5 s/iter 不随物理 dt 缩放。
- **新瓶颈**：detections 的 ~1.0 s/iter 来自 `cast_center_distance` 的 numpy 逐 pose 地形 marching + 组装（mask 已批量到 5.66 ms/4 帧 ≈0.02 s），不是 mask 本身。
- 下一步：detections 中心射线/组装批量化（预计 1.0 s → ~0.1 s），再复测 1ms vs 5ms；之后接 LSTM-512 策略；5ms 暂不作训练默认，复测后再定（标 experimental、与 1ms 结果分开记录）。

## 2026-10-25 11:50 优化结果与 LSTM pilot 启动

- detections 批量化后：**1.016 → 0.064 s/iter（16×）**；observation 合计 0.583 s/iter（lidar 0.248 / depth 0.051 / detections 0.064 / state 0.003 / feedback 0.216）。
- A/B 复测（64 env×16 steps，iter2 稳态）：
  | 配置 | wall | env_step | samples/s |
  |---|---|---|---|
  | 1 ms/50 | 2.000 s | 1.696 s | 512.1 |
  | **5 ms/10（experimental）** | **1.156 s** | 0.895 s | **885.6** |
- LSTM-512 已接入：参数量 1,619,913；GPU smoke iter2 0.909 s（351.2 samples/s），checkpoint/resume 正常。
- contract SHA 已改为不覆盖 `run_id/phase/note`（新增 `sha_scope`），跨目录 resume 可用；旧 pilot/smoke checkpoint 因此不兼容（已记录）。
- 决策：5ms/10 作为 WSL **实验默认**（CLI 默认仍 1ms/50；正式训练显式传参并标 `physics_experimental=true`）。
- 已启动 LSTM pilot：`wsl_train/runs/lstm_pilot_5ms`，64 env×32 steps×300 iter、5ms/10、全 real 观测、`--profile`；预计 13–18 分钟，跑完用 `eval_native` 评估。

## 2026-10-25 12:00 LSTM pilot 崩溃与根因假设

- LSTM pilot（`wsl_train/runs/lstm_pilot_5ms`）在 iter ~180 崩溃：`ValueError: xy must be finite`（detections → terrain height），随后评估同 checkpoint 在 5ms 也崩于 VQF `gyro_native must be finite`。
- 对照证据：
  1) **zero-action + 5ms/10 全长度 400 步稳定**（无 NaN、未翻车）；
  2) **checkpoint_160 权重全部有限**（LSTM-512、33 tensors、SHA `7355264f…`）；
  3) **同 checkpoint 在 1ms/50 评估不崩、无 NaN**：2 场景约 590 步后 `outside_or_overturned`（flipped=true）。
- 根因假设：策略会主动把车开翻；**翻车/激进动作下 5ms XPBD 数值发散**，先出现 non-finite 再在传感器/地形处抛异常。不是传感器算法错误。
- 修复要求（队友进行中）：backend 每步 finite 检查并终止/reset 失效 env；sensor 遇 non-finite 返回无效观测；train/eval 加 NaN 保护并写 `failure.json` 后安全退出；修完复跑 300-iter pilot；若 5ms 仍频繁发散，正式训练回退 1ms 或调 solver。
- 本 pilot 为 pre-protocol 验证 run，不改变已冻结的 `PROTOCOL_NATIVE_V1`（SHA `dc5e4a00…`）。

## 2026-10-25 12:10 5ms 数值发散：根因与加固

- 根因（证据链）：**5 ms/10 XPBD 在激进/翻车动态接触下出现稀有数值发散**，先产生非有限 env 状态，再在 sensor（terrain marching）处抛异常；不是策略 NaN、不是传感器 bug。
  - iter 179 正常（2.712 s/755 samples/s）→ iter 180 首次 non-finite env 状态 → detections 路径抛 `xy must be finite`；
  - checkpoint_160 权重有限；CPU 重放动作 |a|max 0.117–0.286、无饱和；
  - 同 ckpt 1ms GPU 评估无 NaN、干净翻车终止；zero-action 5ms 400 步稳定；
  - CPU 复现（MuJoCo 5/1ms 各 500 步、newton XPBD 5/1ms 各 120 步）0 次 non-finite → 稀有/GPU 特定，约 5.7 万物理子步后出现一次。
- 加固（已实现 + CPU 测试通过）：
  1. env 每步 finite 检查（pos/quat/lin_vel/ang_vel/gimbal_q/wheel_qd/action）；训练坏 env 立即 reset，eval 标 `non_finite_state`；`nonfinite_events.jsonl` + ITER 计数；
  2. sensors finite mask：坏 env 保留上帧并计数，state/feedback 置 0，VQF 不污染；
  3. `train_wsl.py`：action/obs/reward/loss/参数 non-finite 或前向异常 → `failure.json` + latest + exit 3；`--nonfinite-limit 64` 预算保护；
  4. `eval_native.py`：单 seed 记录 `non_finite_state/reward/runtime_error` 并继续其余 seed；加 `env_diagnostics`；
  5. 顺带修复 LSTM PPO 变量遮蔽 bug。
- 测试：`proto/test_nonfinite_guards.py` 全通过；报告 `proto/DEPTH_DETECTIONS_REPORT.md` §16。
- 正在 GPU 复跑 300-iter LSTM pilot（5ms/10，`--nonfinite-limit 64`），验证是否不再崩溃及 non-finite 频率；结果决定正式 trial 用 5ms 还是回退 1ms。
