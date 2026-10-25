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
2. **detections 16×8**：仍为占位，实现中（按 env.py 的 64×64 detector/cone/drop/confidence/box 语义）。
3. 因此 `capability_metrics_allowed=false`：**当前 WSL 训练只能叫 throughput/learning 冒烟，不能做能力结论**。

## 进入正式 WSL 训练的验收门

- depth/detections 与 reference 逐元素对照通过（或明确的域差登记），contract 五通道全部 `real`；
- 用真观测重跑 `train_wsl.py` 2 iterations，吞吐不明显退化；
- 冻结本地 native 试验协议（观测/奖励/终止/种子/评估口径）并记录 SHA；
- 训练产出的 checkpoint 能保存/恢复，评估流程明确（native 自评 + 必要时 Isaac 跨环境抽检）。

## 风险

- 原生栈物理/传感器与 Isaac RTX 存在域差（depth 为几何射线、detections 为几何推导），策略可迁移性需实车/Isaac 对照验证；
- WSL2 非项目官方支持环境，长期稳定性需观察；
- 训练规模化后注意宿主 32 GB 内存与 WSL 默认内存上限。
