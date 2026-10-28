# NATIVE（WSL 原生栈）试验协议草案 —— 待 Lead 确认后冻结

> 状态：**DRAFT / 未冻结**。本文件只供 Lead 审阅；在冻结并由 Lead 明确授权前，
> 任何 native 训练/评估结果都只能作为**吞吐或冒烟证据**，不得用于能力结论。
> 关联文档：`proto/EQUIVALENCE_REPORT.md`、`proto/DEPTH_DETECTIONS_REPORT.md`、
> `proto/INTEGRATION_ASSESSMENT.md`、`proto/CAPABILITY_GAPS.md`。

## 1. 范围与域

- **主环境**：WSL2 Ubuntu-24.04-Arm64 原生栈（MuJoCo CPU 或 Newton XPBD + warp 传感器）。
- **参考/对照**：Isaac Sim/PhysX/RTX 只作对照与验收工具（当前已停止训练）。
- **本协议只定义 native 试验**；不改变 Isaac 历史记录与冻结字节。

## 2. 观测契约

| 通道 | 实现 | 状态 |
|---|---|---|
| lidar (1,400) | 解析 raycast（warp kernel/numpy） | 已验证（M1：与 reference Scene.cast 逐位一致） |
| depth (1,160,88) | 解析 raycast（leaf/stem/terrain、轴向 Z、0.3–5 m、归一化 /5） | 算法等价已验证（mask 0 差异、100% ≤1e-3 m；RTX 域差未逐像素对照） |
| detections (16,8) | 解析 64×64 instance mask → env.py 规则 → task.py 打包 | 打包与 reference **逐字段 0.0**；RTX 域差未对照 |
| state (33) | ABSOLUTE_GOAL schema | max abs error 4.05e-7 |
| feedback (37) | VQF 旁路（python vqf 2.1.2） | 与 reference estimator **0.0**；成本 ~0.76 ms/env/update |

- `capability_metrics_allowed = false`（默认，直到本协议冻结 + Lead 明确授权）。
- 观测/物理/verification 身份由 `observation_contract_sha256` 覆盖；`run_id/phase/note`
  等运行元数据写入 contract 但不参与 SHA（`sha_scope` 字段说明）。

## 3. 物理配置

- **native 训练/评估实验默认**：`--sim-dt 0.005 --decimation 10`（5 ms 物理，policy_dt 保持 0.05 s），
  contract `physics_experimental=true`，结果**必须**与 1 ms/50 结果分开记录。
- **legacy 对照**：`--sim-dt 0.001 --decimation 50`（默认 CLI 行为不变）。
- 约束：`sim_dt*decimation == 0.05`（约 50 ms），违反直接报错。
- **评估必须使用与训练相同的 physics 参数**；evaluation.json 记录
  `physics.dt_s/decimation/policy_dt_s/experimental` 且 `policy.observation_contract_sha256`
  应与训练 checkpoint 的 contract SHA **完全一致**（已验证方法：eval_native 内部用与 train_wsl 相同的 contract 构造）。

## 4. 训练配方（LSTM-512）

- 策略：`SensorEncoderCore`(192) → LSTM(512, 1 层) → actor/critic heads，`log_std=log(0.05)`；
  参数量 **1,619,913**（LSTM 1,445,888 + encoder 108,032 + heads 65,989 + 4）。
- PPO：64 env × 32 steps × 300 iter = **614,400 samples**；epochs=4、lr=3e-4、clip=0.2、
  value_coef=0.5、entropy_coef=0.01、grad clip 0.5；rollout 记录 hidden，done 后 reset；
  更新用 `ppo_update_lstm` 从 hidden0 重放序列（含 done 重置）。
- 记录：JSONL 每 iter（wall/collection/env_step/learning/samples_per_s/end_to_end_samples_per_s/
  ppo_loss/mean_reward/policy_type；`--profile` 时加 `sensor_times_s`）；checkpoint 每 `--save-interval`，
  含 `policy_type/hidden_size/iteration/observation_contract_sha256`；resume 校验类型+contract SHA。
- 种子：训练 seed 0；场景由 env seed 决定。

## 5. 奖励/终止（与 reference 对齐）

- 奖励/终止复用 `isaac_field.tensor_task.fast_stop_v2_step`（torch CPU）：位置容差 0.4 m、
  `goal_hold=0.3 s`（reference 任务配置）、时间上限 60 s；**时间上限是 terminal failure**
  （done/terminated=True、truncated=False、reason=3），因此不需要 bootstrap。
- **评估成功判据**（独立于训练奖励，按 Lead 口径）：位置 ≤0.4 m ∧ XY 速度 <0.05 m/s ∧
  全角速度 <0.1 rad/s，连续保持 **0.5 s**。
- 报告必须区分“训练任务成功口径（0.3 s hold）”与“评估成功口径（0.5 s）”，不得混算。

## 6. 评估（eval_native）

- 工具：`train/eval_native.py`；每条 seed 一个场景，`auto_reset=False` 检查终态，1200 步 @20 Hz，
  策略取动作均值（deterministic）。
- 种子：开发池 **87010–87025**（先 4 个：87010–87013）；830xx 保留未观察，不做 held-out 使用。
- 命令（16 seeds，5ms/10，全 real 观测）：
  ```bash
  bash train/run_eval_16seeds.sh <checkpoint.pt> runs/eval_native_<tag> 87010-87025
  ```
- 失败原因：`goal_stop`（成功）/`time_limit`（1200 步）/`outside_or_overturned`（flip，up-z < cos(1.2)）。
- `evaluation.json` 字段：
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
- `capability_metrics_allowed=false` 与 `protocol_note` 在评估输出中保持显式未定，冻结后统一修改。

## 7. 域差声明（domain_gap）

- depth/detections 与 Isaac RTX 的域差（光栅化 vs raycast、mask 边缘、相机光轴约定、材质/TAA）
  仅做**算法等价**验证，未做逐像素 RTX 对照；`domain_equivalence.status=not_run`。
- 物理为代理模型（MuJoCo/Newton 盒体+圆柱轮 vs Isaac PhysX），drive/COM 保真度差异已记录。
- 验收建议（待 Lead 定稿）：depth 平面区 ≤1e-3 m、边缘 1px 排除、p95 ≤2e-3 m；
  detections box ≤2 px、中心实例 range ≤0.02 m；physics 以同初态配对评估代替逐项对照。

## 8. capability_metrics_allowed 策略

- 默认 **false**；只有同时满足以下条件才由 Lead 授权改为 true：
  1) 本协议冻结并记录冻结 SHA；2) depth/detections 域差按 §7 接受或明确不参与能力指标；
  3) 16-seed 评估按 §6 跑完且结果与配对 baseline 一致；4) Lead 书面授权。
- 在此之前：“训练跑通”“smoke 通过”“samples/s”都不是能力证据。

## 9. 复现与产物

- 训练（5ms/10，LSTM，300 iter，示例）：
  ```bash
  /opt/venv/bin/python train/train_wsl.py --policy lstm \
    --num-envs 64 --num-steps 32 --iterations 300 --seed 0 --threads 8 \
    --device cuda --physics newton_xpbd --lidar warp --feedback vqf \
    --depth analytic --detections analytic --detections-warp \
    --sim-dt 0.005 --decimation 10 --profile --save-interval 10 \
    --out train/runs/<trial>
  ```
- 评估：§6 命令；产物 `evaluation.json` + `eval.log`。
- 对照/验证数据：`proto/m1_verification.json`、`m2_verification.json`、`depth_verification.json`、
  `detections_verification.json`、`detections_warp_verification.json`、`detections_batch_verification.json`。
- SHA 规则：`observation_contract_sha256` 只覆盖观测/物理/verification 身份字段（`sha_scope` 明示）。

## 10. 待 Lead 确认的问题（冻结清单）

1) 5ms/10 是否作为 native 试验的**正式实验默认**（本协议 §3 暂按批准执行）？
2) 评估种子数：先 4（87010–87013）还是直接 16（87010–87025）？
3) depth/detections 的 RTX 域差是否接受“仅算法等价 + 登记域差、不逐像素对照”，还是要求离线 RTX 采样对照？
4) 是否需要同时跑 1ms/50 作为配对 legacy 对照，还是只留历史记录？
5) 本 LSTM pilot（300 iter）是否作为**协议内首个正式试验**记录，命名/trial 路径与冻结字段由谁定？
6) `capability_metrics_allowed` 的开启条件是否按 §8 执行？
