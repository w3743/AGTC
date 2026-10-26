# native_trace_a02 只读审查

## 结论

- `native_trace_a02.json` 有实际 PyTorch CPU operator trace：1,850 个 `cat=cpu_op` span，另有 1 个 `Trace` span 和 metadata/instant 事件。Trace 头部记录了 RTX Spark、CUDA driver/runtime 和 CUPTI 版本，但它们只是设备/运行时元数据，不代表捕获了 GPU 活动。
- CUDA 活动没有被捕获。`native_trace_a02.log` 明确记录 `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED (39)`，并说 CUDA profiler activities will be missing；a01 log 也有同样的 CUPTI 错误。JSON 没有 kernel、CUDA runtime/driver、memcpy 或同步事件。因此 kernel 总执行时间、GPU 时间区间、memcpy、同步 API 都是**未知**，不能填成 0，也不能用 CPU span 推算。
- 这不是完整 CPU 线程/调度 trace；它是单线程上的 PyTorch operator spans。1850 个 span 的包含式时长和为 995.066 ms，按区间并集为 968.257 ms；直接累加会重复计入嵌套区间。最长事件是顶层 `aten::mean` 915.761 ms；项目里 `drive_control.py` 的 side_mean 和 `env.py` 的 command mean 都是候选调用点，但 trace 没有 callsite/tensor-device 信息，不能判定对应哪一处，也不能静态断言它造成了等待或 GPU 慢。

## PhysX 主机阶段

这些是报告里的 host-wall `inclusive_seconds`，不是 GPU kernel time。`fetch_results` 高度集中地占据 `physics_step` 时长，值得优先查等待/同步边界；由于区间是 inclusive，不能将子阶段再次相加作为总耗时。

| 记录 | fetch calls | fetch 总时长 / 均值 | physics_step 总时长 / 均值 | simulate 总时长 / 均值 | fetch / physics_step |
|---|---:|---:|---:|---:|---:|
| a02，1 次 correctness step | 50 | 1.243352 s / 24.867 ms | 1.269134 s / 25.383 ms | 0.015130 s / 0.303 ms | 97.97% |
| a01，4 次 correctness steps | 200 | 6.848619 s / 34.243 ms | 6.969058 s / 34.845 ms | 0.068233 s / 0.341 ms | 98.27% |

`fetch_results` 是 host 侧调用耗时；这里可以称为 fetch 边界的大块主机等待/耗时，但无法拆成 PhysX GPU 计算、设备同步、线程调度或调用自身开销。两份结果是不同长度的正确性检查，不是可比的吞吐基准。

后续`native_drive_final_a01.json`通过4环境×16步correctness检查，记录800次物理调用的host inclusive `physx_fetch_results`为35.8614712 s、`physx_simulate`为0.2453846 s。该结果仍是主机侧嵌套计时，不提供GPU kernel时间；这次correctness运行也没有生成CUDA activity trace。

## 具体排查线索

- CPU trace 记录了 22 次 `aten::item`、22 次 `aten::is_nonzero`、22 次 `aten::_local_scalar_dense`，以及 173 次 `aten::to`、103 次 `aten::copy_`、94 次 `aten::_to_copy`。如果这些调用接收 CUDA tensor，标量取值/布尔判断可能形成同步点；trace 没有输入设备或 memcpy 事件，所以目前只是需要回到调用点核对的线索，不能断言发生了传输或同步。
- 优先追踪 915.761 ms 的 `aten::mean` 调用对应的 Python 调用点和 tensor device。确认它属于本次物理步热路径后再决定是否有可保持数值语义的改写；不要仅凭 CPU op 名称改算法。
- 下一次受控诊断应让 CUPTI 只有一个活动订阅者，确认 CUDA activity 成功后再汇总 kernel 时长、区间并集、memcpy 和 sync；同时分段记录 submit/fetch host-wall。维持 `dt=0.001`、`decimation=50`、`position_iterations=40` 和现有传感器/停稳合同。当前结果不支持 FPS 或提速倍数结论。

## Kit profiler 静态审查

- 本次实际启动日志 `kit_20261024_201349.log` 使用 `isaaclab.python.headless.rendering.kit`。该体验配置设置 `app.profilerBackend = "nvtx"`；它会启用 NVTX backend/标记，但单凭此设置不能证明启动了 GPU 活动采集或是 CUPTI 冲突方。
- 同一实际 Kit 日志显示 `carb.profiler.gpu.plugin` 已初始化，并有 `omni.rtx` 的 CUPTI 调用报 `CUPTI_ERROR_INVALID_DEVICE`，随后 `cuptiActivityDisable` 也报错。这证明 Kit/RTX GPU profiler 路径参与了启动并访问 CUPTI，是需要隔离验证的候选冲突来源；现有文本日志不能确定哪个组件持有 CUPTI subscriber。
- `omni.kit.profiler.window` 在本次日志中仅 `registered`，没有 `startup`；它属于 `isaacsim.exp.full.kit`，不是本次 Isaac Lab headless-rendering 体验的活动窗口扩展。没有证据支持把它认作这次错误的直接来源。未发现 `omni.physx` 专用 profiler 启动或 CUPTI 订阅的日志证据。
- 已安装的 `SimulationApp` 源码默认 `profiler_backend=[]`；只有明确选择 Tracy/NVTX backend 时才额外加入 `--/app/profileFromStart=true`、`--/profiler/enabled=true`。本次实际启动参数中没有这两项。Profiler Window 源码将 GPU Profiler 绑定到 `/profiler/enabled`，并将 `/profiler/perfsdkRealtimeMetrics/enabled` 与 `/profiler/perfsdkReportGenerator/enabled` 默认设为 `false`。
- 候选隔离设置 `--/app/profilerBackend=""`、`--/profiler/enabled=false` 和两个 PerfSDK 开关 `false` 已用于下文所述的单次启动尝试。环境初始化异常发生在运行时设置读取和首个环境步之前；没有 settings readback，也没有 CUDA trace，因而不能判定 CUPTI 订阅是否解除。

## partial_reset_checked 字段校正

- 旧 a02 报告记录 `steps=1, partial_reset_checked=true`，a01 phases 报告记录 `steps=4, partial_reset_checked=true`；对应 smoke 源码只在零基 `step == 7` 后调用 partial reset，因此这两个短运行的布尔字段不能作为实际执行证据。旧 JSON 未改写。
- `native_a03.json` 记录 16 步，配套 `native_smoke_a03.py` 在 `step == 7` 执行 reset，满足这个路径条件。准备的 `native_smoke_profiler_off_a01.py` 配置 16 步、reference observation 比对、camera pose check、首个 `env.step()` 收 trace、运行时读取 4 个 profiler 设置，并用 `partial_reset_count` 派生 `partial_reset_checked`。该副本已执行一次；Kit 在构造环境时因 `instance_segmentation_fast` annotator 未注册而退出，未到达步进、trace 导出或报告写入。

## profiler-off 单次启动结果

- `run_native_check.py` 记录子进程实际 return code `1`、耗时 `29.407 s`、`timed_out=false`；日志和 process receipt 已保存。日志显示 Kit 启动至环境创建阶段，`cuda:0`、4 个环境和物理场景已初始化。
- `IsaacFieldRslEnv` 初始化触发 `SimulationContext.reset` 时，Camera 初始化请求 Replicator annotator `instance_segmentation_fast`，但 `omni.replicator.core` 的 Python extension startup 失败，随后报 `AnnotatorRegistryError: No annotator of name instance_segmentation_fast registered`。这是本次退出的直接原因。
- 报告和 trace 均未生成。因此无法核验 camera pose、partial reset、16 步 observation 对照，无法读回四个 profiler 设置，也无法判断 CUPTI 冲突是否解除、CUDA 活动是否捕获或 fetch 等待是否变化。没有重启第二个 Kit。
- profiler-off 命令行 flags 仅用于该次失败隔离尝试，未写入项目默认配置，也未作为训练恢复启动参数；因退出发生在首个物理 step 前，不能称为“profiler关闭后的”运行结果。
- 六个纳入冻结的项目源文件在启动前后 SHA-256 一致；已执行脚本 SHA-256 为 `4cf22d7143d6006b399f65ec33d66b8cbf3f53cd7694c04c79bf03a39727cb37`。原始 a01/a02 trace、报告和日志未改。

## 记录范围

a02 报告为 4 环境、1 次 correctness step；训练、策略加载、优化器、FPS benchmark 均未启动。物理设置为 `dt=0.001`、50 子步、40 次 position iterations，torch_batched 与 GPU dynamics。报告中的 reference observation 最大误差为 LiDAR 0、depth `5.96e-8`、detections 0、state `1.19e-7`、feedback 0；这支持该次观测合同检查，不单独证明整体物理等价。a01/a02 原始 trace、日志和报告均未改。本次另按授权启动一次非训练 headless Kit；环境构造阶段异常退出，未开始 correctness steps、trace capture 或 FPS benchmark，详见上节。未启动训练、策略或优化器，未改生产或训练源。
