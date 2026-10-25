# 训练平台性能改造 · 执行摘要（2026-10-25）

本文件是 [PERF_PLATFORM.md](PERF_PLATFORM.md) 与 [NATIVE_STACK_PLAN.md](NATIVE_STACK_PLAN.md) 的索引式摘要；所有数字均为本机实测，未外推。

## 结论

| 目标 | 结果 |
|---|---|
| Isaac Sim/Kit 配置优化（本机可直接用） | **19.6×**：4 env/1 ms 基线 2.18 samples/s → 32 env/5 ms/dec=10 **42.65 samples/s**；96 iter / 393,216 transitions 用时 2h33m |
| 原生 ARM64/WSL2 训练栈（同一 4096 samples/iter 口径） | 稳态 **12.48 s/iter** vs Isaac 88–96 s → **7.1–7.7×**；328.2 samples/s（steaady） |
| 观测/任务等价性 | lidar（numpy 逐位一致；Warp 仅 2/6400 掠射差 3.35e-4 m）、state 4.05e-7、feedback 直通 2.9e-8、VQF 旁路 0.0；**depth/detections 仍占位** |
| 策略能力 | **未达成**：model_95 0/2、model_128 0/4（非 strict-pass，仅开发口径）；失败模式为越界或接近后不满足 0.4 m/0.05 m/s 停稳条件 |
| Git/PR | 分支 `perf/max-throughput` 已推送；草稿 PR **#24**（CI 绿灯、mergeable） |

## 关键改动

- `isaac_field/env.py`：新增 `ISAAC_FIELD_PERF_PHYSICS=1` 开关，允许 `sim.dt*decimation==task.dt`；**默认路径仍强制 1 ms/50、solver 40/1、render_interval=50**，行为不变。
- 评估/训练记录、checkpoint 审计、fast eval 脚本保存在 `results/isaac_field_improvement/perf_20261025_n32_dt5/`。
- 原生栈原型与基准在仓库外 `C:/pioneer/wsl-native/`（WSL2 Ubuntu 24.04 ARM64），报告见该目录 `REPORT.md`、`EQUIVALENCE_REPORT.md`、`CAPABILITY_GAPS.md`。

## 平台上限

- 高负载（~96%）时 SM 时钟会降到约 663–690 MHz、功耗读数仅 ~5 W（`power.limit=N/A`，读数不可信）；但负载 30–50% 时实测 2.1–2.4 GHz（窗口采样 max 2385 MHz），**不是固定锁频**，更像功率/热预算限制。原生栈负载峰值 2236 MHz；软件无法突破该预算；1.65 TFLOPS fp32 为可复现平台数字，不能外推到裸机。

## 下一步建议（需用户决定）

1. **策略能力**：继续训练到百万级 transitions 或调整奖励/课程；当前 606k transitions 仍未收敛。
2. **原生栈**：完成 depth/detections（估 6–9 人日）后才可替代正式训练；当前只能用于吞吐实验。
3. **PR**：可转 ready 合入（仅含性能开关与文档；不改变默认训练协议），或保持 draft 等待后续。
