# 训练平台性能改造：当前入口

当前主训练路径已转向 WSL2 ARM64 原生栈。实现已纳入 `native/`，无需启动 Isaac Sim/Kit。本轮配置、测量口径、原始证据与限制以 [迁移与验收说明](native/MIGRATION_ACCEPTANCE.md) 为准。

## 当前实现

- Newton XPBD GPU 物理；Warp LiDAR、解析 depth/detections、状态与 VQF 五路真实观测。
- 五路编码器、LSTM-512、PPO、checkpoint/resume 和 native 评估。
- 正式配置为 1 ms × 50；5 ms × 10 有数值失稳记录，不能用于正式训练。
- 本轮审计发现旧 pilot01 实际使用简化任务，不能作为 V1.1 fast-stop 试验。新运行显式选择 `fast_stop_v2` 并校验任务/物理契约。
- 解析传感器与 RTX、代理物理与完整 Isaac 场景仍有域差；`capability_metrics_allowed=false`。吞吐验收与导航、实车能力验收分别处理。

## 历史性能参照

| 运行 | 记录 | 使用边界 |
|---|---|---|
| Isaac a07，4 env、1 ms/50 | collection 约 2.18 samples/s | 历史采集速率；不是端到端速率 |
| Isaac perf，32 env、5 ms/10 | 393216 samples / 9218.8 s = 42.65 samples/s | 改了并行数与物理步长；非物理等价对照 |
| 原生早期 M3b，64 env、1 ms/50 | 4096 samples/iter，稳态约 328.2 samples/s | 早期短原型，depth/detections 占位；不可替代完整栈新测量 |

不把以上数字连乘为迁移倍数，也不把采集速率与完整训练速率混算。完整历史记录保留在 [PERF_PLATFORM.md](PERF_PLATFORM.md) 和 [NATIVE_STACK_PLAN.md](NATIVE_STACK_PLAN.md)，其中较早“尚未实现/仓库外”的描述只属于当时状态。

## Git

延续 `perf/max-throughput` 和 [PR #24](https://github.com/w3743/AGTC/pull/24)。最新提交、检查与本轮测量见 PR 和迁移验收说明；旧文档中的“CI 已绿/mergeable”仅是历史观察。
