# 训练平台性能改造：当前入口

当前主训练路径为WSL2 ARM64的Newton XPBD + Warp + PyTorch，无需启动Isaac Sim/Kit。旧pilot06存在缺接触、world0索引复用与实际摩擦不一致，已拒绝其330.2 samples/s和7.7倍结论；旧4-seed失败也不能直接归因于策略学习。

v3已修复接触、每环境状态/控制/reset隔离、请求device与实际材料身份。正确物理探针通过，两个完整栈短测已完成；新的300轮验收在iter58出现env11非有限物理状态，exit3、formal_eligible=false；正式验收未通过，继续定位长期稳定性。配置、证据、复现命令及最终状态统一见[迁移与验收说明](native/MIGRATION_ACCEPTANCE.md)。

| v3实验完整栈（正式未通过），64 env × 32 steps，1 ms/50，LSTM-512 | rep01 | rep02 |
|---|---:|---:|
| Collection samples/s | 317.599 | 191.149 |
| Iteration window（采集+PPO）samples/s | 278.538 | 167.671 |
| 全部样本/完整进程墙钟 samples/s | 196.598 | 142.085 |

两轮各2轮预热+5轮测量，五路均real、异常计数0，checkpoint策略与优化器全部finite。后者迭代吞吐低39.8%，原因尚未定位，当前不能宣称稳定提速倍数。iteration window不含后置有限性检查/模型保存，完整进程墙钟包含这些开销。

历史Isaac为32 env、5 ms/10，393,216 samples / 9218.8 s = 42.65 samples/s；M3b约328.2 samples/s为占位通道短原型。这些配置/物理不同，不能充作严格等价对照或混入新栈验收。完整历史保留在[PERF_PLATFORM.md](PERF_PLATFORM.md)和[NATIVE_STACK_PLAN.md](NATIVE_STACK_PLAN.md)，其中早期实现状态与CI观察只属于当时记录。

当前仍为代理车辆、平面物理碰撞和速度目标驱动；解析传感器与RTX有域差，`capability_metrics_allowed=false`。平台吞吐不代表导航或实车能力。

延续`perf/max-throughput`与[PR #24](https://github.com/w3743/AGTC/pull/24)，冻结协议、旧检查点与成绩保留原字节。最新提交、CI与正式训练状态以PR和迁移说明为准。
