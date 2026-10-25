# depth / detections 等价实现报告（持续更新）

> 范围：`C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\native\proto\` 内实现；不改仓库。所有数字来自实际运行（CPU-only，GPU 仅在明确窗口内使用）。
> 状态：depth **算法等价已验证并接入 provider**（RTX 域差已登记、未逐像素对照）；detections 未完成；`capability_metrics_allowed=false` 不变。

## 0. 当前状态

| 通道 | reference 语义来源 | 现状 | usable_for_capability |
|---|---|---|---|
| depth 160×88 | `task.py FieldTask.depth_intrinsics` + env.py RTX camera/depth/noise | 解析 raycast（warp kernel + numpy 独立实现）已验证 0 mask 差异、100% ≤1e-3 m；已接入 `ObservationProvider(depth_backend="analytic")` | 否（RTX 域差未对照） |
| detections 16×8 | env.py Detector mask→box/confidence/class + task.py 打包 | 待实现 | 否 |
| lidar / state / feedback | 见 `EQUIVALENCE_REPORT.md` | 已验证（VQF 旁路 0.0） | 部分（反馈 VQF 路径已验证） |

## 1. depth 实现

- 文件：`proto/depth_caster.py`（numpy 独立实现 + warp kernel）、`proto/m1_depth_verify.py`（对照/几何/性能）、`proto/m1_depth_integration.py`（provider 集成对照）。
- 语义（对齐 `isaac_field`）：
  - 内参直接读取 `FieldTask.depth_intrinsics`（160×88，缩放自原生 352×640），像素射线 `(x, y, 1)`，**轴向 Z = 射线参数 t**；
  - 相机位姿 = body yaw × gimbal yaw × gimbal pitch，原点 = `root_z + camera_height(0.4)`；
  - 几何 = leaf patch（椭圆面片）+ stem（垂直圆柱）+ terrain（4 波正弦高度场，解析求交）；岩石密度 0；
  - clip `[0.3, 5.0] m`，归一化 `where(valid, v, 0)/5.0`，invalid→0；输出 `(n,1,160,88)` float32；
  - 延迟：depth 300 ms（6 policy steps）队列，2.5 Hz 周期（每 8 policy step 一帧）。
- 与 provider 集成：`ObservationProvider(depth_backend="analytic", depth_device="cpu"|"cuda")`；`NativeFieldEnv`/`train_wsl.py` 已透传（`--depth analytic --depth-noise/--depth-no-noise`）。

## 2. depth 对照结果（实际运行）

场景：`build_reference_scene(seed=77001, 400 plants, terrain amplitude 0.35, rocks 0)`，
6 个固定在植物旁 3 m 的相机位姿，160×88 = 14,080 像素/帧。

| 指标 | 值 |
|---|---|
| 射线总数 | 84,480 |
| 有效像素比例（numpy / warp） | **0.4044 / 0.4044** |
| valid mask 不一致 | **0** |
| 距离 max abs diff | **2.06e-4 m** |
| 距离 mean abs diff | 0.0 |
| p95（共同有效像素） | 0.0 |
| ≤1e-3 m 比例 | **1.0（100%）** |

几何自检（纯平面 z=0、无植物、pitch −0.3 rad）：
- 有效像素 8,624、mask 不一致 0；
- 解析轴向 Z 期望 vs 实测：max abs error **7.8e-5 m**、mean 4e-5 m；中心像素 1.895898 vs 1.895937 m。

provider 集成（2 env、MuJoCo CPU、noise off、第一帧 depth）：
- 输出通道与直接调用 caster **max abs diff 0.0**、mask 不一致 0、valid ratio 0.2078（姿态不同植物可见比例不同）；
- `observation_contract` depth channel 更新为 `mode=real`、`backend=analytic_warp`，`verified_vs=m1_depth_verification.json`。

## 3. 失败点与修复记录（保留原始过程）

1. 初版 warp kernel 未实现 terrain → valid ratio 0.0045 vs 0.4044，33787 mask 不一致；补上 4 波解析高度场 marching + 8 次二分后修复。
2. warp kernel 每 pose launch 但 origin/基向量仍按 `env=0` 索引 → 所有 pose 用同一相机，max diff 3.8 m；改为每 pose 传单元素数组后修复。
3. numpy 参考实现用 float64、kernel 用 float32 → 掠射/叶面边缘 2/84480 像素翻转（max 0.52 m）；numpy 叶子/茎数学改为 float32 后 max diff 降到 2.06e-4 m。
4. 几何自检初版把"地平线以上（无命中）"像素也计入误差 → 402 m 假误差；限制在 valid ∧ expected∈[0.3,5] 后为 7.8e-5 m。

## 4. 与 Isaac RTX 的域差登记（`depth_verification.json.sensor_domain_gap`）

- Isaac：RTX `distance_to_image_plane` 光栅化 + Z-buffer + 材质/TAA/抗锯齿 + 加性 range noise；
- 原生：解析 ray 求交，无栅格化/抗锯齿；
- 预期差异：遮挡/边缘像素、薄叶片亚像素覆盖、无材质与 TAA；
- 算法等价（解析 vs 独立 numpy）已验；**域等价（vs RTX）尚未逐像素对照**（Isaac 已停，需一次离线 RTX 采样），登记 `domain_equivalence.status=not_run`，验收口径建议：平面区 ≤1e-3 m、边缘 1px 排除、p95 ≤2e-3 m。

## 5. 性能（CPU 实测；GPU 待窗口）

- numpy 独立实现：**≈0.08–0.11 s / 帧**（CPU，单 pose，多线程未开）；
- warp kernel（CPU device）：**≈0.10–0.16 s / 帧**，主要成本是 terrain marching（250 步 × 14k 射线 × 4 sin）；
- 结论：CPU 上 depth 无法进训练循环（64 env × 2.5 Hz → 160 帧/s，需 16 s/s）；GPU 版本目标 **4–6 ms/帧**（按解析 raycast 已有实测推算），需一次 3–5 min GPU 窗口实测 64 env 批渲染吞吐；
- 若 GPU 达标：64 env × 2.5 Hz ≈ 160×5 ms = 0.8 s/s GPU，需考虑每 pose 一次 kernel launch 的开销（后续可批量化/多 pose 合并）。

## 6. detections 16×8（已完成算法等价并接入 provider）

实现：`proto/detections_caster.py`（解析 64×64 instance mask → env.py 规则 → task.py 打包）、`proto/m1_detections_verify.py`（与 reference 打包逐元素对照）、`proto/m1_detections_integration.py`（provider 集成对照）。

参考语义（env.py 实际代码）：
- 64×64 detector 相机（f=55.4256、cx=cy=31.5），圆形 FOV `(u−31.5)²+(v−31.5)² ≤ 32²`；
- instance segmentation mask → 每个 leaf instance（路径 `/Corn_i/Leaf_j`；茎不被 reference 正则匹配，故不产生检测）；
- 过滤 `len(pixels) < 2`、`detection_drop=0.08`；
- `confidence = clip(0.70 + 0.28·min(1, n_pix/20) + N(0, 0.015), 0, 0.99)`；
- class = `leaf_dry ^ (rand < class_error=0.015)`；
- range：仅当该 instance 是中心像素 `image[32,32]` 且 0.12 ≤ rangefinder 距离 ≤ camera_range(7.0) 时给出 `distance + N(0, range_noise=0.01)`，否则 None；
- box = `[(xmin+xmax+1)/128, (ymin+ymax+1)/128, (xmax−xmin+1)/64, (ymax−ymin+1)/64]`（cx, cy, w, h）；
- 按 confidence 降序，取前 `max_detections=16`；
- task.py 打包：`[cx, cy, w, h, class_id, confidence, min(1, range/5), valid_range]`，无检测行补 0。

对照结果（7 个固定位姿：6 个植物旁 3 m + 1 个"中心像素锁定叶片"位姿）：
- **打包逐元素误差（vs `FieldTask._observation`）max 0.0 / mean 0.0** → 字段顺序、缩放、range/valid 处理、排序、截断全部一致；
- 活跃检测行 34 条（每帧 7/5/3/3/9/1/6），`range` 分支被实际触发 1 次（中心实例），无 range 时为 0 且 valid=0；
- box 检查：w/h 全部 >0、全部 ∈[0,1]，min w 0.0312、min h 0.0156、max w 0.3125；
- mask 命中率 0.0105（近水平相机、叶片目标小，符合几何预期），mask 中 70 个不同 leaf instance；
- 性能（CPU）：**mask 渲染 ≈6.0 ms/帧**（64×64）+ 组装 0.2 ms/帧 → 64 env×5 Hz 约 1.9 s/s（CPU 不可进训练循环；如需要再写 warp mask kernel）；
- provider 集成（2 env、plant-facing spawn、noise off）：**max abs diff 0.0、active rows 8/6**，detections 通道 `(2,16,8)`、观测形状 `(16,8)` 一致。

域差登记（`detections_verification.json.sensor_domain_gap`）：
- Isaac 用 RTX instance-seg mask + 1-ray rangefinder；原生用解析 mask → 同一 env.py 规则 → 同一 task.py 打包；
- 预期差异：mask 边缘 1–2 px 的 box 差异；detector 相机光轴与 depth 相机共用 mount basis（**该光轴约定尚未与 RTX 逐像素核对**，是主要域差风险）；置信度/类别/range 噪声分布相同但 RNG 流独立；
- 验收建议：box 中心/边缘 ≤2 px、中心实例 range 差 ≤0.02 m；域等价 `status=not_run`。

## 7. 当前 contract 状态（CPU 检查）

- 五个通道全部 `mode=real`：lidar（analytic）、depth（analytic_warp，已验证）、detections（analytic_detector，已验证）、state（m1 4.05e-7）、feedback（VQF 旁路 0.0）；
- `capability_metrics_allowed=false`，唯一 blocker：`RTX domain gap not pixel-compared (depth, detections)`；
- `capability_blockers` 仅在 depth/detections 未 analytic 时回退为 placeholder 列表。

## 8. GPU 窗口结果（2026-10-25 11:24，约 22 s）

**depth GPU 性能**（`proto/depth_gpu_perf.py` → `proto/runs/depth_gpu_perf.json`，`proto/runs/gpu_analytic_20261025_112406/`）：
- 64 个相机位姿批渲染（901,120 射线，160×88，axial-Z，含地形/叶片/茎）；
- 3 次重复：0.0818 / 0.0815 / **0.0648 s**（best）→ **1.013 ms/帧**、986.9 帧/s、0.072 ms/1000 射线；
- JIT/编译 warmup 0.994 s；valid ratio 0.426；
- **目标 4–6 ms/帧，实测约 1.0 ms/帧（比目标更快）**：GPU 解析 raycast 成本可忽略；注意这是每 pose 单独 kernel launch 的结果，批量化后只会更快。
- 解释：`meets_target=false` 是因为实测低于目标下界，不是性能不足。

**`train_wsl.py` 真观测 smoke**（`train/runs/smoke_analytic_20261025_112406/`，64 env × 16 steps × 2 iter，newton_xpbd + cuda + warp lidar + VQF + analytic depth/detections，`--save-interval 1`）：
- iter1：wall 6.509 s（policy_act 2.052 s 含首次 CUDA 编译、env_step 2.564 s、learning 1.776 s）；
- iter2（稳态）：wall **2.897 s**、collection 2.722 s（env_step 2.595 s、policy_act 0.062 s）、learning 0.167 s；
- **samples/s：稳态端到端 353.5**（1024/2.897）、collection-only 376.2；RSS 2791 MB；
- checkpoint 正常写出（checkpoint_1.pt / checkpoint_2.pt / latest.pt），exit 0；
- 对比无真观测基线 smoke（≈600–650 samples/s）：analytic depth+detections+VQF 约 -40%；主要成本是 **detections 的 CPU mask（4 帧/iter × 64 env × 6 ms ≈ 1.5 s/iter）**，depth GPU 仅约 0.13 s/iter，VQF 约 0.4 s/iter。若要把 detections 成本也 GPU 化，可加 warp mask kernel（估 0.2–0.5 ms/帧，约 0.1 s/iter）。

**contract（smoke 输出）**：
- `{"lidar": real/warp_kernel, "depth": real/analytic_warp, "detections": real/analytic_detector, "state": real, "feedback": real/vqf_reference}`；
- `capability_metrics_allowed=false`，blockers 仅 `RTX domain gap not pixel-compared (depth, detections)`；
- 本次 smoke 的 `verification` 只含 m1/m2/depth（detections 校验文件是在 run 之后才加入 `_contract_for`）；已修补并 CPU 复检，当前 contract 的 `verification` 键为 **m1/m2/depth/detections**。

**GPU 采样**（194 样本，100 ms；含 init/idle）：util mean 10.2%、p95 36%、max 67%；SM clock mean 1408 MHz、max 2470 MHz；mem 峰值 5334 MiB；temp max 53 °C。窗口结束 GPU 已释放（6%/4891 MiB），无残留进程。

## 9. detections warp mask kernel（任务 A，CPU 正确性已完成；GPU 待窗口）

- 实现：`detections_caster.cast_mask_warp(poses, device=...)`（每 pose 一次 kernel，64×64=4,096 射线，host 侧按距离裁剪叶 patch，与 numpy 路径同一数学/dtype）。
- 对照（CPU，8 位姿 × 4,096 像素 = 32,768 像素；`detections_warp_verification.json`）：
  - **mask 不一致 0 像素**；leaf 命中率 0.017（numpy）；
  - **检测行逐字段 max abs error 0.0**（cx/cy/w/h/class_id/confidence/range_norm/range_valid 全 0）；active rows 20 = 20；
  - provider 集成（`camera_backend="analytic_warp"`，CPU device）：**max abs diff 0.0**，active rows 8/6。
- 计时（CPU device）：numpy mask 5.6 ms/帧；warp CPU 216 ms/帧（CPU 上 warp 线程开销反而更差；GPU 成本待窗口实测，估 0.2–0.5 ms/帧）。
- 开关：`train_wsl.py --detections-warp`（默认仍 analytic CPU 路径，不改变现有行为）；`NativeFieldEnv(camera_backend="analytic_warp", camera_device=...)`；contract backend 变为 `analytic_detector_warp`，`verification` 增加 `detections_warp` 键。

## 10. native 评估入口（任务 B，CPU smoke 通过）

- `train/eval_native.py`：每 seed 一个场景（默认 `--seeds 87010,87011`），60 s = 1200 步 @20 Hz；`auto_reset=False` 以便检查终态；成功判据 = 位置 ≤0.4 m ∧ XY 速度 <0.05 m/s ∧ 全角速度 <0.1 rad/s ∧ 连续保持 0.5 s；失败原因 = `time_limit` / `outside_or_overturned`。
- 输出 `evaluation.json`：`schema_version=native_eval_v1`、`banner=NATIVE_EVAL_CPU_SMOKE`、`native_domain`、`domain_gap`、`capability_metrics_allowed=false`、`success_criteria`、`policy`（含 contract SHA）、`config`、`summary`（scenes/successes/success_rate/clean_successes/clean_success_rate/mean_steps/mean_final_distance/reasons/wall_s）、`rows`（seed/success/clean_success/reason/steps/final_distance/goal_stop/hold_s/final_xy_speed/final_angular_speed/flipped）。
- CPU smoke ①（2 场景 × 1200 步，2-iter smoke checkpoint `smoke_analytic_20261025_112406/latest.pt`）：2 场景均很快 `outside_or_overturned`（该 checkpoint 只训练了 2 iter，属预期；仅验证评估链路可跑）。
- CPU smoke ②（`--zero-action`，2 场景 × 1200 步）：**wall 8.16 s**、均 `time_limit`、final_xy/angular speed 0、未翻车 → 全长度路径与 schema 通过；产物 `train/runs/eval_cpu_smoke_zero/evaluation.json`。
- 未宣称能力；`capability_metrics_allowed` 保持 false，协议字段由 Lead 后续填写。

## 11. 第二个 GPU 窗口（2026-10-25 11:31–11:32）：warp detections 实测 + pilot checkpoint 评估

**① warp detector mask GPU 成本**（`proto/detections_gpu_perf.py` → `proto/runs/detections_gpu_perf.json`）：
- 64 位姿 × 4096 像素 = 262,144 射线；3 次 **42.1 / 43.1 / 45.3 ms** → **0.658 ms/帧**（1520.6 帧/s）；
- Lead 目标"≤30 ms/批"**未达标（42 ms）**：每 pose 一次 kernel launch 的 host 侧开销占比高；若把 64 位姿合并成一次 launch（环境索引数组，类似 depth 的批量化）预估可到 10–20 ms/批；
- 与 numpy 单帧 5.6 ms 相比，单帧快 ~8.5×；leaf 命中率 0.0037、495 个不同 instance。

**② `--detections-warp` 训练 smoke**（`train/runs/smoke_warp_20261025_113119/`，64 env×16 steps×2 iter）：
- iter2 稳态：wall **2.948 s**、collection 2.744 s（env_step 2.637 s、policy_act 0.058 s）、**347.3 samples/s 端到端**；
- 与上一窗口 analytic CPU detections（wall 2.897 s、353.5 samples/s）相比**没有可测收益**（差异在噪声内）；
- **诚实修正**：上一轮报告里"detections ≈1.5 s/iter 是主要成本"是微基准外推，不是实测；本窗口 smoke 显示 env_step 几乎没变（2.595→2.637 s），说明该 16-step 配置下 **physics 主导（~2.6 s）**，detections 渲染并未主导墙钟。需要下一步在 provider 内加 per-sensor 计时才能给出真实拆分（不能继续用外推）。
- contract：channels `lidar=warp_kernel, depth=analytic_warp, detections=analytic_detector_warp, feedback=vqf_reference`；`capability_metrics_allowed=false`；blockers 仍仅 RTX 域差；`verification` 键为 **m1/m2/depth/detections/detections_warp**。

**③ `eval_native` 链路（pilot checkpoint_40）**：
- `--checkpoint train/runs/pilot_20261025/checkpoint_40.pt --seeds 87010,87011 --max-steps 1200 --device cuda --physics newton_xpbd --lidar warp --feedback vqf --depth analytic --detections analytic --detections-warp`
- 结果：**wall 16.07 s**；2 场景均 `outside_or_overturned`（49 / 182 步，终态角速度 6.61 / 1.51 rad/s、flipped=true，final distance 34.75 / 11.84 m）；
- 含义：checkpoint 加载、native 观测、评估链路全部跑通；该 pilot checkpoint（iter 40、64×32）在 native 域不稳定，**不构成能力结论**；产物 `train/runs/eval_native_20261025_113119/evaluation.json`。

**GPU 采样**（360 样本，100 ms）：util mean 17.1%、p95 35%、max 47%；SM clock mean 1655 MHz、max 2340 MHz；mem 峰值 5937 MiB；temp max 54 °C。窗口结束 GPU 已释放（4%/5191 MiB），无残留进程。

## 12. 任务 C：per-sensor 计时 + dt/decimation + detections 批量管线（CPU）

**per-sensor 计时**：`ObservationProvider(profile=True)` 累计 lidar/depth/detections/state/feedback；
`env.consume_sensor_times()` 写入 `train_wsl.py` ITER 的 **`sensor_times_s`**；`--profile` 开关，默认关闭（零计时开销）。
CPU 示例（4 env×4 steps，iter2）：lidar 1.71ms / depth 0.11ms / detections 0.02ms / state 0.42ms / feedback 0.67ms（VQF）/ total 2.93ms。

**dt/decimation**：`native_env` 强制 `sim_dt*decimation==0.05`（否则 ValueError）；`train_wsl.py --sim-dt/--decimation`（默认 1e-3/50 行为不变）；
contract 写 `physics_dt_s/decimation/policy_dt_s/physics_experimental/physics_note`（默认 false；5ms 标 true）。CPU 验证 5ms/10 可跑、非法组合被拒。

**detections 批量管线**（`detections_caster`）：
- `cast_mask_warp_batch`：64 pose 单 launch；GPU 42.1 ms（per-pose 64 launch）→ **5.66 ms/批（0.0885 ms/帧）**，与逐 pose mask 完全一致；
- `cast_center_distance_batch`：地形一次 march 全部中心射线；CPU **3.719 → 0.461 ms/pose（8.1×）**，与逐 pose 最大误差 **0.0**、inf 不一致 0；
- `assemble_batch`：扁平分组的批量 mask→box；CPU **0.173 → 0.039 ms/pose（4.4×）**，rows max abs error **0.0**；
- provider 已切换到批量版本；预计 detections 从 ~1.0 s/iter 降到 **~0.15 s/iter**（256 poses：GPU mask 22.6ms + center 118ms + assembly 10ms），待 GPU 窗口复测确认。
- 数据：`proto/detections_batch_verification.json`。

## 13. 任务 D：真实 LSTM-512 策略接入（CPU smoke 通过；GPU 2-iter smoke 待窗口）

- `train/policy_lstm.py`：`SensorEncoderCore`(192) → 单层 **LSTM(512)** → actor `Linear(512→64)+Tanh+Linear(64→4)`、critic `Linear(512→64)+Tanh+Linear(64→1)`、`log_std=log(0.05)`。
- 结构化验证：act/forward 输出 `[B,4]/[B]/[B]`，hidden `[1,B,512]`×2；`evaluate_sequence(T=4,B=3)` 输出 `[T,B]` 三个张量、全 finite；done 位置重置生效。
- **参数量**：总计 **1,619,913**（LSTM 1,445,888 + encoder 108,032 + heads 65,989 + log_std 4）。仓库 actor ~1.58M，差异来自 encoder/head 实现（+~40k），可解释。
- `train_wsl.py --policy lstm`（默认 mlp 不变）：rollout 记录 hidden、done 后 reset；更新用 **`ppo_update_lstm`** 从存储的 hidden0 重放序列重算 logp/value（含 done 重置），不是按单步独立样本；JSONL 增加 `policy_type`；checkpoint 记录 `policy_type/hidden_size`，resume 校验类型一致。
- CPU smoke：2 iter 正常（ITER `policy_type=lstm`，loss 有限）；checkpoint `policy_type=lstm`；同 out 目录 resume 成功（`RESUMED iteration=2 (policy=lstm)`）；`eval_native` 已支持 LSTM checkpoint（自动识别 + hidden 维护），1 场景 CPU 链路跑通。
- 观察（未改语义）：`contract` 的 SHA 包含 `run_id`（out 目录名），因此跨 out 目录 resume 会因 SHA 不同失败（同目录 resume 正常）；是否把 run_id 移出 SHA 由 Lead 决定。

## 14. 第三个 GPU 窗口（2026-10-25 11:40:58–11:41:54）：批量管线复测 + A/B + LSTM GPU smoke

**① detections 成本复测（iter2，`--profile`）**：
- **detections 1.016 s → 0.064 s/iter（约 16×）**，与批量中心射线/组装预期一致；observation 合计 0.583 s/iter
  （lidar 0.248 / depth 0.051 / **detections 0.064** / state 0.003 / feedback 0.216）

**② 1ms/50 vs 5ms/10 A/B（64 env×16 steps，iter2 稳态）**
| 配置 | wall | collection | env_step | **samples/s（端到端）** | sensor_times 合计 |
|---|---|---|---|---|---|
| 1ms/50 | 2.000 s | 1.788 s | 1.696 s | **512.1** | 0.583 s |
| 5ms/10 | **1.156 s** | 0.985 s | 0.895 s | **885.6** | 0.604 s |

- 5ms 相对 1ms：**wall 1.73×、samples/s 1.73×**；physics env_step 1.90×（1.696→0.895 s）
- 对比批量前：1ms 从 337.9 → **512.1 samples/s**；5ms 从 463.1 → **885.6 samples/s**
- 结论：5ms/10 已达 **~1.16 s/iter、886 samples/s**（远超 500 目标），可考虑作为 WSL 训练**实验默认**（仍标 experimental、与 1ms 结果分开记录；是否切换由 Lead 决定）

**③ LSTM-512 GPU smoke（32 env × 8 steps × 2 iter + resume）**：
- iter1 wall 2.917 s（首次编译 policy_act 0.966 s）；**iter2 wall 0.909 s**（collection 0.729、policy_act 0.027、learning 0.175），256 samples/iter → **351.2 samples/s**
- checkpoint `policy_type=lstm, hidden_size=512, iteration=2, params_total=1,619,913`；**resume 成功**（`RESUMED iteration=2 (policy=lstm)`，iter3 wall 2.403 s 含首次 hidden/序列更新开销）
- 结论：LSTM 训练链路 GPU 可用，hidden/序列化 PPO 更新正常，checkpoint/resume 正常

**GPU 采样**（498 样本，100 ms）：util mean 15.7%、p95 44%、max 69%；SM clock mean 1757 MHz、max 2515 MHz；mem 峰值 5839 MiB；temp max 59 °C。窗口结束 GPU 已释放（2%/5087 MiB），无残留进程。

**contract SHA 范围调整（窗口后）**：`run_id/phase/note` 等运行元数据**仍写入 contract 文件**，但不再进入 `observation_contract_sha256`；新增 `sha_scope` 字段说明覆盖范围（观测通道/物理后端/成功奖励配置/embedded verification）。CPU 验证：两个不同 out 目录得到**相同 SHA**，且跨目录 resume 成功；代价是旧 pilot/smoke checkpoint（旧 SHA 覆盖元数据）不再兼容，需重新 prepare（当前只有 pilot/smoke，可接受）。

## 15. eval_native 与训练 contract 对齐 + 16-seed 运行准备（CPU）

- `eval_native.py` 新增 `--sim-dt/--decimation`（强制 `sim_dt*decimation==0.05`），并在 `evaluation.json` 写：
  - `physics: {dt_s, decimation, policy_dt_s, experimental, note}`；
  - `config: {sim_dt_s, decimation, ...}`；
  - `policy.observation_contract_sha256`（与 train_wsl 相同的 contract 构造与 SHA scope）。
- **SHA 对齐验证**：同一配置（5ms/10、mujoco_cpu、numpy lidar、direct feedback、constant depth/detections）
  下 `train_wsl` 的 `OBS_CONTRACT_SHA` 与 `eval_native` 的 `policy.observation_contract_sha256`
  **完全一致**（`23f61709…`）；即评估可证明与训练使用同一观测/物理身份。若训练用了 analytic depth/detections
  或 warp lidar，评估必须传相同 backends 才能 SHA 相等。
- 5ms/10 eval 实测 `physics.experimental=true`、`dt_s=0.005`、`decimation=10`，与训练参数一致（CPU 冒烟）。
- 16-seed 运行脚本：`train/run_eval_16seeds.sh <checkpoint.pt> <out_dir> 87010-87025`
  （newton_xpbd + cuda + warp lidar + vqf + analytic depth/detections + detections-warp + 5ms/10 + 1200 步）。
- 协议草案：`train/PROTOCOL_NATIVE_DRAFT.md`（待 Lead 冻结；含 §10 待确认清单）。
- 字段清单见协议 §6；`capability_metrics_allowed=false` / `protocol_note` 在 eval 输出中保持显式未定。

## 16. 5ms/10 LSTM pilot（iter ~180）non-finite 崩溃：根因、修复与测试

**现象（实际日志，`train/runs/lstm_pilot_5ms`）**：iter 179 正常（wall 2.712 s、755.16 samples/s、loss 0.0515），iter 180 collection 内崩溃：
`env.step → provider.observe → detections_caster.cast_center_distance_batch → depth_caster.terrain_march → terrain_height → isaac_field/terrain.py _xy_array: ValueError: xy must be finite`。
最后一个完整 checkpoint 为 `checkpoint_160.pt`；`metrics.jsonl` 最后一行 iter=179。

**证据链（区分物理发散 / 策略 NaN / 传感器 bug）**
1. **不是策略 NaN**：checkpoint_160 权重全部有限（Lead 审计：33 tensors、SHA `7355264f…`）；CPU 用该 LSTM 重放 seed 87010/87011：动作 |a|max 0.117–0.286、均值绝对值 0.003–0.03、饱和比例 0；LSTM 输出没有发散。
2. **不是传感器本身**：non-finite 先出现在 env 状态（pos/quat/…），随后才被 sensors 用到；同一 checkpoint 在 **1ms/50 GPU 评估** 下不崩、无 NaN，两场景约 590 步后 `outside_or_overturned`（干净终止）。
3. **不是 5ms 零动作问题**：zero-action + 5ms/10 全长度 400 步稳定（Lead 只读测试：2 场景 time_limit、无 NaN）。
4. **CPU 无法复现**：MuJoCo 5ms/10、1ms/50 各 500 步 + newton XPBD 5ms/10、1ms/50 各 120 步（同一 checkpoint、auto_reset=True）均 0 次 non-finite（`proto/diag_5ms_divergence.json`）。
5. GPU pilot 在 ~180 iter × 32 steps × 10 substeps ≈ **5.7 万物理子步**后才出现一次，属低概率事件。

**结论（根因）**：**5ms/10 XPBD 在激进/翻车类动态接触下出现稀有数值发散**，先产生非有限 env 状态，再在传感器（terrain marching/VQF）处触发异常；不是策略/传感器代码 bug。CPU 物理无法复现该稀有事件，需 GPU 复跑统计发生率。

**修复**
1. `native_env.py`：每步对 `pos/quat_wxyz/lin_vel/ang_vel/gimbal_q/wheel_qd` + action 做 finite 检查；action non-finite → 置 0 并记录；state non-finite 时训练（auto_reset=True）**立即 reset 该 env 恢复训练**，eval（auto_reset=False）标记 `non_finite_state`；事件写入 `nonfinite_events`（含 env id/字段/qpos 样本）并计数 `nonfinite_count`；`info` 带 `nonfinite`/`action_nonfinite`。
2. `observation_provider.py`：每帧先算 finite mask；**只对 finite envs 更新 lidar/depth/detections**，坏 env 保留上一帧并计数 `nonfinite_frames`；state/feedback 对坏 env 置 0，VQF estimator 跳过不污染。
3. `train_wsl.py`：action/obs/reward/loss/参数 non-finite 与 `policy.act` 异常 → 写 **`failure.json`**（iter/step/reason/last_checkpoint/nonfinite 事件）+ 保存 latest checkpoint + **exit 3**，不再无声崩溃；`--nonfinite-limit`（默认 64）超预算即安全退出；ITER JSONL 增加 `nonfinite_env_resets`/`nonfinite_actions`，事件写 `nonfinite_events.jsonl`。
4. `eval_native.py`：非有限状态/奖励/运行异常 → 单 seed 记 `reason=non_finite_state|non_finite_reward|runtime_error` 并继续其余 seed，`evaluation.json` 增加 `env_diagnostics`。
5. 顺带修复测试中暴露的 LSTM PPO 隐藏 bug：obs finite 检查循环变量 `value` 覆盖了策略 value 张量（导致 `torch.stack(values)` 类型错误），已改名。

**验证（CPU，`proto/test_nonfinite_guards.py`，`nonfinite_guard_tests.json`）**
- 注入单 env NaN pos：env 记录事件、自动 reset、后续 obs 全部 finite；provider 通道 finite、`nonfinite_frames=1`；
- NaN 权重 checkpoint resume：训练写出 `failure.json`（reason=`non_finite_parameter`，last_checkpoint 存在）并 **exit 3**，无静默崩溃；
- 全部测试 `pass=true`；eval_native CPU smoke 正常（time_limit 分支、env_diagnostics 写入）。

**待 GPU 复跑（下一步）**：300-iter LSTM pilot（5ms/10、同配置、`--nonfinite-limit`）统计每 iter nonfinite 次数；若频率高则按协议回退 1ms/50 或调 XPBD solver（iterations/substep），结论需数据支撑；在此之前不把 5ms 当成已验证的正式训练默认。

## 17. 5ms/10 LSTM pilot 复跑（GPU，带防护）结果：5ms 不可作该配方的实验默认

**运行**：`train/runs/lstm_pilot_5ms_fixed/`（12:08:47–12:13:51，约 5 分钟到 iter 98；64 env×32 steps、5ms/10、全 real 观测、detections-warp、`--profile --nonfinite-limit 64`、`--save-interval 20`）。
**退出**：`reason=nonfinite_state_budget_exceeded`，**exit 3**（防护正常，非崩溃）；`failure.json` + `latest.pt` 保留；checkpoint_20/40/60/80 均存在。

| 指标 | 数值 |
|---|---|
| 完成 iter | 97（iter 98 触发预算退出） |
| nonfinite 事件 | iter 76：64（8 envs × 8 个连续 step，auto-recovered）；iter 98 step 702：**全部 64 envs 同时** pos/lin_vel/ang_vel/wheel_qd 非有限（total 128 触发 limit） |
| 每 iter nonfinite（均值/最大） | 0.66 / 64（仅 2 个 iter 出现，突发式） |
| 稳态 samples/s（iter 2–75） | **730.1**（wall 2.808 s、collection 1.965 s、env_step 1.727 s） |
| 首次事件后（iter 76–97） | 619.9 samples/s（wall 3.374 s；GPU 采样 util mean 24%、max 82%、SM max 2528 MHz，疑似外部负载/复位开销叠加） |
| RSS | ≤2791.5 MB；GPU 采样：util mean 24%、max 82%、mem max 6366 MiB、temp max 61 °C |
| 进程/GPU | 无残留进程；窗口结束 GPU 由 Lead 使用（36%/4345 MiB） |

**结论**
1. 防护有效：**没有再崩溃**，non-finite 被记录、env 自动 reset、失败时写 `failure.json` 并保存 `latest.pt`，exit 3 安全退出。
2. **5ms/10 XPBD 确实会周期性数值发散**：98 iter 内 2 次（约每 50 iter），且第 2 次是**全部 64 env 同时**发散（不是个别 env），说明是求解器级别的失稳，而不是个别接触/策略动作。
3. **建议：正式训练回退 1ms/50**（同一 checkpoint 1ms GPU 评估无 NaN、干净翻车终止；1ms 稳态 samples/s 约 512，仍可接受）。若坚持 5ms，需先调 XPBD solver（提高 iterations 10→20/30、或减小 substep 到 2ms/25）并重跑本测试，达标前不进入正式 trial。
4. 协议 V1 冻结不改；是否把正式 trial 的默认物理从 5ms 改为 1ms 由 Lead 决定（建议记为 V1 的 explicit deviation 或开 V2）。
5. `--profile` 下 5ms 稳态观测成本：lidar 0.46 / depth 0.07 / detections 0.18 / feedback 0.44 / state 0.005 s/iter（5ms 下仍是 observation 主导，非 physics）。

