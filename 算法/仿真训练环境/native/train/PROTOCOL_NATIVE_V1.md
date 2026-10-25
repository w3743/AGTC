# PROTOCOL_NATIVE_V1 —— WSL 原生栈正式试验协议（FROZEN）

> protocol_sha256: dc5e4a00eab44b1fd8a4e00ca7fe9aed7597b1c6b9f3b508a6471553af90a962
> 状态：**FROZEN**（冻结日期 2026-10-25；Lead 已确认 §10 决策）。冻结后任何修改都必须新开版本（V2），不得原地改写。
> 适用范围：WSL2 Ubuntu-24.04-Arm64 原生栈（MuJoCo CPU / Newton XPBD + warp 传感器）的 native 训练与评估。
> 关联文档：`proto/EQUIVALENCE_REPORT.md`、`proto/DEPTH_DETECTIONS_REPORT.md`、`proto/INTEGRATION_ASSESSMENT.md`、`proto/CAPABILITY_GAPS.md`。
> 历史草案：`wsl_train/PROTOCOL_NATIVE_DRAFT.md`（已被本文件取代，仅留档）。

## 1. 范围与域

- 主环境：WSL2 原生栈（MuJoCo CPU / Newton XPBD），观测由 warp/numpy 解析传感器产生。
- Isaac Sim/PhysX/RTX 仅作对照与历史参考；**不得**把 native 成绩与 Isaac RTX 成绩直接比较或混算。
- 本协议只定义 native 试验；不改变 Isaac 冻结记录与字节。

## 2. 观测契约

| 通道 | 实现 | 验证状态 |
|---|---|---|
| lidar (1,400) | 解析 raycast（warp kernel / numpy） | M1：与 reference `Scene.cast` 逐位一致 |
| depth (1,160,88) | 解析 raycast（leaf/stem/terrain、轴向 Z、0.3–5 m、`where(valid,v,0)/5`） | 算法等价：mask 不一致 0、100% ≤1e-3 m、平面轴向-Z 7.8e-5 m |
| detections (16,8) | 解析 64×64 instance mask → env.py 规则 → task.py 打包 | 与 reference 打包逐字段 0.0；批量管线与逐 pose 0.0 |
| state (33) | ABSOLUTE_GOAL schema | max abs error 4.05e-7 |
| feedback (37) | VQF 旁路（python vqf 2.1.2） | 与 reference `SimFeedbackEstimator` 0.0；~0.76 ms/env/update |

- `capability_metrics_allowed = false`（默认；开启条件见 §8）。
- `observation_contract_sha256` 只覆盖观测/物理/verification 身份字段（`sha_scope` 字段明示）；
  `run_id/phase/note` 等运行元数据写入 contract 但不参与 SHA。

## 3. 物理配置

- **native 正式实验默认**：`--sim-dt 0.005 --decimation 10`（5 ms 物理）；`policy_dt` 恒为 0.05 s。
  contract 标 `physics_experimental=true`，结果与 1 ms/50 结果**分开记录**。
- **CLI 默认不变**：`train_wsl.py` 不传 `--sim-dt/--decimation` 时仍为 1 ms/50；正式 trial 必须显式传参。
- 约束：`sim_dt*decimation == 0.05`，违反报错。
- 记录：contract 写 `physics_dt_s/decimation/policy_dt_s/physics_experimental/physics_note`；
  `evaluation.json` 写 `physics.{dt_s,decimation,policy_dt_s,experimental}`。
- **评估必须与训练同物理与同 backends**；eval 的 `policy.observation_contract_sha256`
  必须与训练 checkpoint 的 `observation_contract_sha256` 一致（同一 contract 构造/SHA scope）。

## 4. 训练配方（LSTM-512）

- 策略：`SensorEncoderCore`(192) → 单层 LSTM(512) → actor `Linear(512→64)+Tanh+Linear(64→4)`、
  critic `Linear(512→64)+Tanh+Linear(64→1)`、`log_std=log(0.05)`；参数量 **1,619,913**
  （LSTM 1,445,888 + encoder 108,032 + heads 65,989 + 4）。
- PPO：64 env × 32 steps × 300 iter = **614,400 samples**；epochs=4、lr=3e-4、clip=0.2、
  value_coef=0.5、entropy_coef=0.01、grad clip 0.5。
- Rollout 记录 hidden，done 后 reset；更新用 `ppo_update_lstm` 从 hidden0 重放序列（含 done 重置），
  不得按单步独立样本近似。
- 记录：JSONL 每 iter（wall/collection/env_step/learning/samples_per_s/end_to_end_samples_per_s/
  ppo_loss/mean_reward/policy_type；`--profile` 时 `sensor_times_s`）；checkpoint 含
  `policy_type/hidden_size/iteration/observation_contract_sha256`；resume 校验 policy 类型与 contract SHA。
- 训练 seed 0；场景布局由 env seed 决定。

## 5. 奖励/终止（reference 对齐）

- 复用 `isaac_field.tensor_task.fast_stop_v2_step`（torch CPU）：位置容差 0.4 m、`goal_hold=0.3 s`、
  时间上限 60 s；**60 s 为 terminal failure**（done/terminated=True、truncated=False、reason=3），不需要 bootstrap。
- **评估成功判据**（独立于训练奖励）：位置 ≤0.4 m ∧ XY 速度 <0.05 m/s ∧ 全角速度 <0.1 rad/s，
  连续保持 **0.5 s**。
- 报告必须区分训练任务口径（0.3 s hold）与评估口径（0.5 s），不得混算。

## 6. 评估（eval_native）

- 工具：`wsl_train/eval_native.py`；每 seed 一场景，`auto_reset=False`，1200 步 @20 Hz，
  动作取均值（deterministic）。
- **开发/pilot 评估**：先 4 seeds **87010–87013**。
- **正式 trial 终评**：16 seeds **87010–87025**。
- 830xx 为保留未观察数据，不做 held-out 使用；禁止用训练/评估滚动窗口冒充 held-out。
- 命令（4 seeds，pilot）：
  ```bash
  bash wsl_train/run_eval_pilot_4seeds.sh <checkpoint.pt> runs/eval_native_<tag>
  ```
- 命令（16 seeds，正式终评，加 `--formal` 标签由调用方记录）：
  ```bash
  bash wsl_train/run_eval_16seeds.sh <checkpoint.pt> runs/eval_native_<tag> 87010-87025
  ```
- 失败原因：`goal_stop` / `time_limit` / `outside_or_overturned`（up-z < cos(1.2)）。
- `evaluation.json` 字段：`schema_version, banner, generated_at, native_domain, domain_gap,
  capability_metrics_allowed, protocol_note, physics, success_criteria, policy, config, summary, rows`
  （细则见 §13）。
- `capability_metrics_allowed=false` 与 `protocol_note` 在输出中保持显式未定，§8 授权后才可改。

## 7. 域差登记（accepted）

- **决策（Lead 已确认）**：接受「仅算法等价 + 登记 RTX 域差、不做逐像素 RTX 对照」作为 WSL-native 路径。
- depth/detections 与 Isaac RTX 的域差：光栅化 vs raycast、mask 边缘、相机光轴约定、材质/TAA；
  `domain_equivalence.status=not_run`。
- 物理为代理模型（MuJoCo/Newton 盒体+圆柱轮 vs Isaac PhysX），drive/COM 保真度差异已记录。
- **能力指标只在 native 域内自评；禁止与 Isaac RTX 成绩直接比较或混算。**
- 若未来需要 RTX 对照，须新开协议版本（V2），不得回填本协议数据。

## 8. capability_metrics_allowed 策略

默认 **false**。只有同时满足以下条件，并由**用户最终书面同意**后，才由 Lead 授权改为 true：
1) 本协议冻结且 `protocol_sha256` 记录在 trial 中；
2) depth/detections 域差按 §7 登记（已完成）；
3) 正式 trial 完成 16-seed native 评估（含配对 baseline，按 §6/§9 口径）；
4) 用户最终书面同意。

在此之前：“训练跑通”“smoke 通过”“samples/s”“pilot 4-seed”都不是能力证据。

## 9. Trial 命名、SHA 与产物

- 正式 trial 目录：`wsl_train/runs/native_v1_<tag>_<YYYYmmdd_HHMMSS>/`。
- 每个正式 trial 必须包含 `trial_manifest.json`：
  `{protocol_sha256, git_or_workspace_state, cli_args, observation_contract_sha256, checkpoint_files,
  eval_json_paths, created_at}`；`protocol_sha256` 必须等于本文件冻结值。
- **本 LSTM pilot 为协议冻结前的验证 run（pre-protocol）**，不命名 native_v1、不作为正式 trial；
  其 checkpoint 可用于链路/吞吐验证与 4-seed 开发评估，但结果只能标 `pre_protocol`。
- 正式 trial 在协议冻结后启动，命名/SHA 按本节；旧 pre-protocol 产物不得重命名为正式 trial。
- 产物保留：检查点、JSONL、identity/contract、evaluation.json、本协议副本。

## 10. 决策记录（Lead 确认，2026-10-25）

| # | 问题 | 决定 |
|---|---|---|
| 1 | 5ms/10 是否正式实验默认 | **是**（`physics_experimental=true`；CLI 默认仍 1ms/50，正式 trial 显式传参） |
| 2 | 评估种子数 | **pilot/开发 4 个（87010–87013）；正式终评 16 个（87010–87025）** |
| 3 | RTX 域差处理 | **接受“仅算法等价 + 登记域差、不做逐像素 RTX 对照”**；能力指标仅 native 域内自评，禁止与 Isaac 成绩比较/混算 |
| 4 | 是否初期跑 1ms/50 配对 | **不要求**；1ms 结果仅作 legacy 历史对照，需要时单独跑 |
| 5 | 当前 LSTM pilot 定位 | **pre-protocol 验证 run**，不是正式 trial；正式 trial 协议冻结后启动，命名/SHA 按 §9 |
| 6 | capability 开启条件 | **按 §8**：冻结 SHA + 域差登记 + 16-seed native 评估（含配对 baseline）+ 用户最终书面同意；此前一律 false |

## 11. 冻结与 SHA

- `protocol_sha256` 的计算：对本文件内容中 `> protocol_sha256: ...` 这一行做归一化（置空值）后，
  取 UTF-8 字节的 SHA-256；脚本 `wsl_train/protocol_sha.py` 负责计算/校验/嵌入，并生成
  `PROTOCOL_NATIVE_V1.sha256`（含算法与范围说明）。
- 任何对协议正文的修改都会改变 SHA；冻结后若需修改必须新建 V2，不得原地改。
- trial 中记录协议 SHA 与协议文件副本。

## 12. 复现命令

- 训练（5ms/10，LSTM，300 iter，正式 trial 模板）：
  ```bash
  /opt/venv/bin/python wsl_train/train_wsl.py --policy lstm \
    --num-envs 64 --num-steps 32 --iterations 300 --seed 0 --threads 8 \
    --device cuda --physics newton_xpbd --lidar warp --feedback vqf \
    --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.005 --decimation 10 --profile --save-interval 10 \
    --out wsl_train/runs/native_v1_<tag>_<timestamp>
  ```
- 评估（pilot 4 seeds）：`bash wsl_train/run_eval_pilot_4seeds.sh <checkpoint.pt> <out_dir>`；
  正式 16 seeds：`bash wsl_train/run_eval_16seeds.sh <checkpoint.pt> <out_dir> 87010-87025`。
- 验证数据：`proto/m1_verification.json`、`m2_verification.json`、`depth_verification.json`、
  `detections_verification.json`、`detections_warp_verification.json`、`detections_batch_verification.json`。

## 13. evaluation.json 字段（正式终评口径）

- 顶层：`schema_version, banner, generated_at, native_domain, domain_gap, capability_metrics_allowed,
  protocol_note, physics, success_criteria, policy, config, summary, rows`
- `physics`：`dt_s, decimation, policy_dt_s, experimental, note`
- `policy`：`checkpoint, policy_loaded, policy_type, observation_contract_sha256, sha_scope`
- `config`：`seeds, max_steps, device, physics, lidar, feedback, depth, detections, detections_warp,
  zero_action, sim_dt_s, decimation, episode_length_s, policy_dt_s`
- `summary`：`scenes, successes, success_rate, clean_successes, clean_success_rate, mean_steps,
  mean_final_distance, reasons, wall_s`
- `rows[]`：`seed, success, clean_success, reason, steps, final_distance, goal_stop, hold_s,
  final_xy_speed, final_angular_speed, flipped`
- 正式 trial 的 evaluation.json 必须附带 `protocol_sha256` 与 `trial_manifest.json` 路径。
