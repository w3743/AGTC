# Newton MuJoCo-Warp 训练复现与链路证据

本目录汇总四次短训练复测及其恢复、评估 smoke 证据。原始文件保留在各自 run 目录，本汇总不复制检查点。逐文件的原始路径、大小、SHA-256、Git 忽略状态和 LFS 属性见 [manifest/repeats_comparison.json](manifest/repeats_comparison.json)；供主协调者逐项 stage 判断的外部清单位于仓库外的 C:\pioneer\agtc-native-acceptance-backup\repro_acceptance\training_repro_stage_list.txt。

## 运行范围与身份

四次训练均为同一子进程内 2 次 warmup + 5 次 measured iteration；每 iteration 64 个环境 × 32 步 = 2,048 samples。配置为 seed 0、LSTM hidden size 512、Newton MuJoCo-Warp eager、fast_stop_v2、physics dt 1 ms、decimation 50；lidar、depth、detections、state、feedback 均为 real mode。CUDA Graph 未请求且未启用，模式为 eager。求解器身份报告 NOT_GUARANTEED，不能把这些结果解释为逐位确定性。

四次运行的 task identity SHA-256 相同：209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2。传感器 backend 及各字段形状记录在每个 run 的 identity.json 和 manifest 中。

警告输出开启组：
- [eager rep01 原始目录](../20261025_mjw_eager_rep01)：commit fce4ba90ab3f5bce15cf869cad9623334087579a；raw source manifest SHA-256 5641d9ae5bc77c2d533a3b504fe01a5798c374e5b494565cedba84c4d284fabf。
- [eager rep02 原始目录](../20261025_mjw_eager_rep02)：相同 commit 和 source manifest SHA。

LS_ITERATIONS 控制台警告屏蔽组：
- [quiet rep01 原始目录](../20261025_mjw_quiet_rep01)：commit 8903e4e91fddfce63875ae91ddc6ba765f2dadbf；raw source manifest SHA-256 57b0d3bffe311f7633474726d2a816d9d4fa9d899bdedda1743d7bbdb686f861。
- [quiet rep02 原始目录](../20261025_mjw_quiet_rep02)：相同 commit 和 source manifest SHA。

两组 source commit 与 source manifest 不同。组间吞吐差异只能作描述性观察，不能归因于 warning mask 单一变量。每个 manifest 均报告其纳入的 raw source 字节与 HEAD 匹配、运行结束时仍稳定；该 dirty 范围不等价于全工作区 Git clean。主要依赖版本为 Python 3.12.3、torch 2.14.1+cu130、NumPy 2.5.4、Warp 1.18.0、Newton 1.7.0、MuJoCo/MJWarp 3.15.0、ovphysx 0.7.0.74035321、vqf 2.1.2。

## 吞吐与计时口径

样本速率分母来自训练 metrics 的 host wall-clock 记录，不是 CUDA events。Collection 和 PPO 分别以 measured samples 除以 measured 行的 collection_s 或 learning_s 总和；iteration-window 以 measured samples 除以 metrics wall_s 总和，包含 collection 与 PPO，但训练器在后续 finite guard 和 checkpoint 序列化之前记录该 wall_s。Full launch wall 从外层启动到子进程结束，包含启动、两个 warmup、五个 measured iteration 及进程内其他开销。Full-launch measured rate 的分子不含 warmup；full-launch all-samples rate 的分子含 warmup。每 iteration 均为 2,048 samples。

| Run | Collection samples/s | PPO update samples/s | Iteration-window samples/s | Full launch wall (s) | Measured samples / full launch (samples/s) | All 14,336 samples / full launch (samples/s) |
|---|---:|---:|---:|---:|---:|---:|
| eager rep01 | 121.348 | 1,466.565 | 112.036 | 170.307 | 60.127 | 84.178 |
| eager rep02 | 95.714 | 1,395.837 | 89.540 | 183.060 | 55.938 | 78.313 |
| quiet rep01 | 182.055 | 2,820.781 | 170.946 | 105.346 | 97.204 | 136.085 |
| quiet rep02 | 186.643 | 2,998.799 | 175.633 | 99.846 | 102.558 | 143.581 |

按各组总样本数除以对应时间总和计算的加权吞吐：

| Cohort | Collection samples/s | PPO update samples/s | Iteration-window samples/s | Full launch wall total (s) | Measured samples / full launch (samples/s) | All samples / full launch (samples/s) |
|---|---:|---:|---:|---:|---:|---:|
| 警告输出开启（2 runs） | 107.018 | 1,430.327 | 99.533 | 353.367 | 57.957 | 81.139 |
| LS_ITERATIONS 控制台屏蔽（2 runs） | 184.321 | 2,907.068 | 173.258 | 205.192 | 99.809 | 139.733 |

同组 rep02 相对 rep01：警告输出开启组的 collection、PPO、iteration-window、full-launch measured rate 分别变化 -21.12%、-4.82%、-20.08%、-6.97%；屏蔽组分别变化 +2.52%、+6.31%、+2.74%、+5.51%。两组源版本不同，组间速度差不构成 warning mask 的因果结论。

## 求解器事件与 GPU 采样

四次训练的每个 iteration metrics 行均记录 2,048 samples、102,400 world substeps；capacity overflow、outer iteration limit、nonfinite environment reset、nonfinite action 均为 0。按每次运行的 7 行（含 warmup）汇总，LS_ITERATIONS line-search-limit substeps 分别为 77,102、76,988、76,971、76,974；其中五个 measured iteration 分别为 53,411、53,356、53,321、53,351。警告开启组控制台匹配数分别为 99,028 和 98,915；屏蔽组控制台匹配数为 0，实际 warning mask 为 3071，结构化 line-search overflow 标志和计数仍保留。具体逐 iteration 计数见 manifest 的 metrics_audit。

GPU 采样器目标间隔为 1 秒，记录的是 nvidia-smi 查询结果，包含查询开销，不能当作严格单调的 1 Hz trace：

| Run | 样本数 | 观测跨度 (s) | 平均间隔 (s) | 中位间隔 (s) | 最大间隔 (s) |
|---|---:|---:|---:|---:|---:|
| eager rep01 | 155 | 170.040 | 1.104 | 1.095 | 2.302 |
| eager rep02 | 165 | 182.660 | 1.114 | 1.106 | 3.136 |
| quiet rep01 | 98 | 104.998 | 1.082 | 1.081 | 1.158 |
| quiet rep02 | 93 | 101.171 | 1.100 | 1.077 | 2.854 |

eager rep02 的一次采样墙钟时间倒退 0.343 秒，因此该轨迹不是 monotonic clock。功率读数不可靠。每个原始目录保留 nvidia_smi_samples.jsonl 及启动前后 GPU/进程快照。资源声明为：无其他已知训练/推理任务；桌面 GPU 活动与 aria2 下载仍在；这不是整机隔离。

## 检查点与恢复 smoke

四个训练 run 的 iteration 7 检查点均做了 CPU safe-load 有限性审计：policy 子树 33 个 tensor、optimizer 子树 99 个 tensor、整个 checkpoint 共 132 个 tensor，全部 finite。每个 run 的 checkpoint_7.pt 与 latest.pt 字节一致；原始文件及 SHA-256 列在 manifest 中，均未复制到本目录。

[quiet resume smoke](../20261025_mjw_quiet_resume) 从 quiet rep02 的 iteration 7 latest.pt 恢复并完成 iteration 8；task SHA 保持相同。CPU 审计确认 policy 33 个 tensor 更新、Adam 33 个状态条目延续，Adam step 从 28 推进到 32；policy、optimizer、整个 132 tensor checkpoint 均 finite。checkpoint 未保存 RNG state，因此这验证的是 payload/optimizer 状态延续与 iteration 7→8，不是 bitwise 或随机流连续性。resume 输出 checkpoint_8.pt 与 latest.pt 字节一致。

[quiet eval smoke](../20261025_mjw_quiet_eval) 从 resume checkpoint 以 seed 87010、1 个环境、最多 32 步运行，并严格匹配 task SHA。策略加载成功，运行到 32 步 time_limit，未成功到达目标。它只验证加载与执行链路，不是能力评估；manifest 将 capability_metrics_allowed 标为 false。

## 证据与归档边界

manifest 包含四个训练目录、resume/eval 目录及同级 Windows/WSL sidecar 的逐文件路径、字节数和 SHA-256；本次重新核验 75 个文件，忽略状态为 53 个未忽略、22 个忽略，其中 12 个日志、10 个检查点。检查点匹配 Git LFS filter。外部 stage 清单也列出这些文件供主协调者决定归档；本目录不包含旧 v3、candidate probe、pilot 或 formal run 数据，也不复制重复的巨大检查点。

这是一组短吞吐和恢复链路证据，不是长程训练稳定性、策略能力或整机隔离验收。
