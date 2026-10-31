# 训练平台性能改造：当前入口与证据

截至2026-10-28，原始`results/`训练结果与运行日志已按用户要求删除。下文保留当时的实验与状态记录，其中引用的`results/...`路径仅作历史溯源，不表示文件或进程仍可核验；删除范围与当前可核实状态见[训练结果汇总](训练结果汇总.md)。

## 当前正式状态

当前主线为 Windows 11 ARM64 / RTX Spark N1X 主机上的 Ubuntu 24.04 ARM64 WSL2、Newton MuJoCo Warp V6、普通 eager GPU、NOT_GUARANTEED。Native V2 协议已冻结，protocol SHA256 为 83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e；当前 task SHA256 为 209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2。源码 commit 7d8a167f，raw source manifest 6026336cad5b961e2fe95885d6010781ae75e59e792d1715988bda45d35e8938。

V6 1/3/64-world preregistered physics gate 已通过；同身份 resume 与 64→1 strict-evaluation smoke 已通过。正式 run native_v2_formal_v6_live_20261025_224005 于 2026-10-25 22:40:05 北京时间启动，当时记录为 active，最终 300 轮门尚无通过记录。后续工作区缺少该运行目录，当前运行状态不能由此历史记录确认；见[训练结果汇总](训练结果汇总.md)。23:06:51 仅记录 CPU viewer 切换，不是 iteration 采样。

完整配方、依赖限制和历史失败见[迁移与验收说明](native/MIGRATION_ACCEPTANCE.md)。固定入口为 native/train/run_native_v2_trial.sh；评估使用 native/train/run_eval_v2.sh。禁止把 V1/XPBD 命令或模型当作当前正式 V2。

## 前置门与 checkpoint 链路

1、3、64 worlds 的冻结输入门各运行 20 policy steps / 1,000 policy physics substeps，并通过逐 world 控制/状态、接触、partial reset 与 reset 后 step 检查；非有限/capacity 门及已登记的 clearance、mimic、torque 上限均通过。证据索引见 [V2 freeze validation](native/acceptance/20261025_v2_freeze_validation/README.md)。

同身份 resume 从 iteration 7 到 8，Adam step 从 28 到 32，contract SHA 不变；单环境 evaluation 标记 strict_match 并匹配相同 task SHA。证据见 [resume summary](native/acceptance/20261025_mjw_quiet_resume/run/summary.json) 与 [evaluation record](native/acceptance/20261025_mjw_quiet_eval/evaluation.json)。这只证明 checkpoint 与评估链路，不是策略能力或成功率验收。

## 已完成的 isolated 7-iteration 短基线

每个独立子进程为 2 warmup + 5 measured、每轮 2,048 samples。iteration window 为 10,240 measured samples 除 rollout+PPO 时间；full-launch 为全部 14,336 samples 除子进程完整 wall。每组有两个独立进程，加权速率按总样本/总时间计算。

| Warning-enabled baseline | rep01 | rep02 | 加权 |
|---|---:|---:|---:|
| Iteration window samples/s | 112.036 | 89.540 | 99.533 |
| 全部样本 / 完整进程 wall samples/s | 84.178 | 78.313 | 81.139 |

来源 commit fce4ba90ab3f5bce15cf869cad9623334087579a，raw source manifest 5641d9ae5bc77c2d533a3b504fe01a5798c374e5b494565cedba84c4d284fabf。两次 exit 0，非有限 reset/action、capacity overflow、outer iteration-limit 均为 0；line-search-limit 事件保留。原始数据：[rep01](native/acceptance/20261025_mjw_eager_rep01/benchmark_result.json)、[rep02](native/acceptance/20261025_mjw_eager_rep02/benchmark_result.json)。

| Quiet baseline | rep01 | rep02 | 加权 |
|---|---:|---:|---:|
| Iteration window samples/s | 170.946 | 175.633 | 173.258 |
| 全部样本 / 完整进程 wall samples/s | 136.085 | 143.581 | 139.733 |

来源 commit 8903e4e91fddfce63875ae91ddc6ba765f2dadbf，raw source manifest 57b0d3bffe311f7633474726d2a816d9d4fa9d899bdedda1743d7bbdb686f861。每次 7 轮共 716,800 world-substeps；fresh line-search-limit 事件分别 76,971 与 76,974。quiet 只改变 warning 输出，不改变 task SHA 或物理 revision。两组速率差异不能单独归因于日志静音。原始数据：[quiet rep01](native/acceptance/20261025_mjw_quiet_rep01/benchmark_result.json)、[quiet rep02](native/acceptance/20261025_mjw_quiet_rep02/benchmark_result.json)。

这四个短测没有本次 formal 长测的 viewer、Git 与 CPU 几何导出活动；不以它们估算或替代正式 run wall / throughput。它们不是 300 轮稳定性、导航能力或实车成绩。

## Formal run 与实时可视化

固定训练为 64 env × 32 steps × 300 iterations，seed 0，LSTM-512、fast_stop_v2、1 ms × 50、普通 eager GPU、零非有限预算。正式 trial 启动前，当前 V6 1/3/64 物理 gate 和 resume/eval chain 已通过。正式 run 必须等待 300 轮完整 metrics、真实 exit、身份和 checkpoint CPU audit；active 不能写成 pass。

当前 viewer 每 5 个 policy steps 消费训练进程的 raw state、目标与每环境任务时间；静态 19-body primitives 和 local transforms 来自冻结 CPU 模型导出，地面为实际 flat plane。旧 Isaac USD 缺少两个云台和两个前导轮共 4 个 body 的几何，旧 viewer 方块 fallback 不再用于 Native V2；旧 USD 不再参与车辆或地面绘制。viewer 不显示传感器图像，也没有 GPU shape arrays 逐字节比对声明。训练过程中的 UI、Git、CPU geometry exporter 与桌面活动都必须按实际情况保留在 long-run 负载记录中。

## 历史结果与边界

Newton XPBD v3 的两次短测、正式训练 iter58 非有限失败、pilot06 invalid run / 330.2 samples/s / 7.7 倍拒绝结论均保留在迁移说明历史节及原始 acceptance 目录，不与 V2 合并。旧接触缓冲与 world 映射问题也不代表当前 V6 行为。

早期 Isaac 32-env / 5 ms 约 42.65 samples/s、M3b 占位通道约 328.2 samples/s，以及旧 v4b 与 CUDA Graph 探针，都不是当前相同任务/物理/通道的对照。当前平台仍用代理物理与解析传感器；RTX 感知域差未逐像素核验，capability_metrics_allowed=false。吞吐不代表导航或实车能力。
