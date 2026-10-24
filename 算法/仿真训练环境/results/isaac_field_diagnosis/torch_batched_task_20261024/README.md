# GPU批处理、轻量观察端与驱动控制器诊断（2026-10-24；a06训练运行中）

## 当前结论

GPU批处理已完成后，用户最新选择是用轻量窗口实时显示训练车辆状态、优先训练效率，并授权从26112 transitions GPU检查点恢复。a06恢复训练已实际启动，当前run status为running；其协议与冻结源、启动参数和checkpoint进度详见“a06续训与实时窗口”。`env.task_backend`默认`reference`；显式`torch_batched`仅支持flat12、plants=0、rocks=0及上述停稳配置，其它组合会被拒绝。

批处理实现和轻量观察端验收完成，第二十一批已由3/5达到5/5并完成范围整理。随后完成驱动控制器拓扑缓存去重，作为第二十二批第1项；a06 GPU恢复入口与配对评估wrapper为第2项，当前2/5，训练运行和日志不另计功能。新a06 trial正在训练；配对评估尚未开始。a06真实浏览器已验收Connected LIVE：深色主题下车辆和地面清晰可见，source序列84→91、截图sequence95、旋转交互有效。原始packet样本57→114→121显示位姿变化。间隔超过既有5s阈值时状态会暂时转为STALE，packet恢复后自动回LIVE；阈值未改，不报告持续高FPS。a05恢复准备因现生产源与reference恢复wrapper不匹配而deferred，没有生成新冻结协议或启动恢复。

## a06续训与实时窗口（运行中）

用户已授权从a04重启后审计通过的checkpoint恢复，checkpoint为26112 transitions、Adam step 1020、SHA `26c55ba2c1fc9e3da09ed174ec01b20b08c78186ee390657b400dde6152a9025`。a06于2026-10-24 21:00:52+08启动controller launcher PID36144/controller PID27560；native PID34280于21:00:57.560910+08启动，训练launcher PID37044。当前[`run_status.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/run_status.json)为`running`/`resume_training_cli`，实际参数使用`torch_batched`、`cuda:0`、4环境、333 updates、`--viz none`和a06 `live_state.json`。

a06 protocol SHA为`6e2860e77bd84e6dc42466793719fd15361d835de35674c6169d42a8f8aabd72`，冻结97个源文件；prepare/preflight、继承checkpoint CPU审计通过，新增恢复接口与配对评估wrapper CPU专项10 passed（[`tests/test_isaac_field_gpu_recovery_pipeline.py`](../../../tests/test_isaac_field_gpu_recovery_pipeline.py)）。首个续训checkpoint已通过CPU审计：26624 transitions、Adam1040，37个Adam optimizer state条目完整且有限；actor/critic权重条目共33个变化（actor 25、critic 8），策略合同与物理/传感器配置一致，见[`first_resumed_checkpoint_cpu.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/first_resumed_checkpoint_cpu.json)。baseline/final配对评估均指定`torch_batched`并使用seed87010–87025。计划目标为196608 transitions、final payload iter380、Adam step7680；配对评估尚未开始，因此没有训练终局或能力结果。

最近a06 viewer status已验证服务读取a06 live state；browser verification确认Connected LIVE、source PID34280与run id匹配、序列84→91且截图sequence95，车辆地面、深色主题与旋转操作通过。source packet样本sequence57→114→121显示位姿变化；证据见a06的[`browser_verification.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/diagnosis/live_viewer_validation/browser_verification.json)、[`validation.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/diagnosis/live_viewer_validation/validation.json)及[实时截图](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/diagnosis/live_viewer_validation/browser_live.png)。packet/PPO间隔有时超过既有5s阈值，viewer会短暂显示STALE并在packet恢复后自动回LIVE；阈值不变。首rollout新增512 transitions，Collection 415.597s、Learning 9.861s，等待仍未解决；没有训练提速或持续FPS结论。a06完成唤醒器已按此前授权部署到外部单实例state-dir `C:/pioneer/isaac-training/watchers/pointmove-gpu-a06/`，launcher PID11808、source SHA `e0c26ef3...315c9`；已核验native watcher PID10728、parent PID27560、`parent_identity=match`，最近lastpoll `13:15:36Z`、`delivery.events={}`。部署/状态/投递快照见a06的[`deployment.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/diagnosis/completion_watch/deployment.json)、[`watch_status_snapshot.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/diagnosis/completion_watch/watch_status_snapshot.json)及[`delivery_snapshot.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/diagnosis/completion_watch/delivery_snapshot.json)；既有PAUSED automations未改。协议/运行身份/准备/首checkpoint/live/viewer证据分别见a06的[`protocol.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/protocol.json)、[`controller_launch.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/controller_launch.json)、[`preflight.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/preflight.json)、[`first_resumed_checkpoint_cpu.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/first_resumed_checkpoint_cpu.json)、[`live_state.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/live_state.json)与[`viewer_status.json`](../../isaac_field_improvement/physx_pointmove_flat12_gpu_20261024_a06/live_viewer/viewer_status.json)。

仅配置字段示例（默认值仍为`reference`）：

```python
task_backend = "torch_batched"
```

## 数据路径与约束

- 奖励、终止和五路观测批量在设备上计算；depth/LiDAR原始大数组和规范化结果留在GPU侧。RL步不会把大数组拉回CPU，也不会先执行CPU `finish` 再重复任务计算。
- CPU仍负责小型VQF 2.1.2反馈估计、各环境NumPy RNG、传感器绝对时钟以及latency/new/age队列。噪声按原每环境随机序列生成后合并上传；这不是端到端GPU实现。
- PhysX真值root COM水平速度和完整车体角速度继续用于严格停稳奖励/终止；策略观测仍使用噪声UWB/VQF姿态估计。
- dt=.001、decimation=50、位置求解40、五路观测shape/频率/延迟及停稳判据保持不变。未改相机AOV、传感器规格、环境数量或PhysX缓存容量。
- reference后端保留作逐步比较和其它任务兼容。空场camera继续推进frame clock并返回空detections。

## 验证证据

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| CPU门禁 | 最终49模块；546 passed、4 skipped、47 warnings、10 subtests，61.60秒，exit 0。530 passed的旧门禁保留为发现`packet_age_seconds`运行期类型错误之前的历史记录，不再作为最终门禁 | [cpu_gate_after_drive.json](cpu_gate_after_drive_20261024T1233Z/cpu_gate_after_drive.json)、[旧cpu_gate.json](cpu_gate.json) |
| CUDA任务数学 | 小批量CUDA实现与真实`FieldTask.finish/_finish_step`比较通过；不包含物理模拟或训练 | [cuda_math_final.json](cuda_math_final.json) |
| native_a01 | 暴露`DirectRLEnv.device`字符串与`torch.device`直接比较的缺陷，exit 1；实现及CPU用例已修复，原日志保留 | [native_a01.log](native_a01.log) |
| native_a02 | 4环境×16步功能检查exit 0；`Kit.close`时没有落结构化JSON，原始日志和source snapshot保留 | [native_a02.log](native_a02.log) |
| native_a03 | 4环境×16步、`torch_batched`、GPU physics；partial reset和旧observation不可变检查通过，exit 0，生产源hash前后相同 | [native_a03.json](native_a03.json)、[native_a03-process.json](native_a03-process.json) |
| native drive final | 4环境×16步、process exit 0；两次camera pose检查、partial reset、old observation immutability、五路设备输出shape及CPU reference比较通过；六个源文件与source snapshot哈希匹配，物理dt/decimation/solver不变 | [native_drive_final_a01.json](native_drive_final_a01.json)、[native_drive_final_a01-process.json](native_drive_final_a01-process.json) |
| profiler-off isolation attempt | exit 1 after 29.407s；Replicator启动缺少`instance_segmentation_fast` annotator，未走step、无trace/readback，flags未采用 | [native_smoke_profiler_off_a01-process.json](native_smoke_profiler_off_a01-process.json)、[native_smoke_profiler_off_a01.log](native_smoke_profiler_off_a01.log) |
| native相位 | 200个物理子步、GPU physics=true、`rateLimitEnabled=false`；source hash稳定 | [native_phases_a01.json](native_phases_a01.json) |
| camera pose | 两次相机位姿检查通过，exit 0 | [native_pose_a01.json](native_pose_a01.json) |
| Fabric实验 | 将`fabricUpdateTransformations`和`fabricUpdateVelocities`关闭后，相机位姿最大差异9.1671m，exit 1；拒绝采用该设置 | [native_fabric_a02.log](native_fabric_a02.log) |

`native_a03`的physics参数为dt=.001、decimation=50、位置求解40。相对CPU reference，LiDAR、detections、feedback最大绝对误差为0，depth为5.96e-8，state为1.49e-7；shape分别保持`(4,1,400)`、`(4,1,160,88)`、`(4,16,8)`、`(4,33)`、`(4,37)`。这支持该配置下的功能/数值对齐，不代表其它任务配置均已支持。

新增`native_drive_final_a01.json`的4环境×16步检查实际exit 0，`camera_positions_checked=2`、partial reset和old observation immutability均通过，dt=.001、decimation=50、position iterations=40保持不变。五路输出在设备侧保留；shape按LiDAR/depth/detections/state/feedback顺序为`(4,1,400)`、`(4,1,160,88)`、`(4,16,8)`、`(4,33)`、`(4,37)`，对CPU reference的最大误差为0、5.96e-8、0、1.49e-7、0。六个选定源文件的SHA与source snapshot匹配，运行前后SHA不变。该项验证没有加载策略/优化器或训练。

## 等待与性能边界

200子步相位采样中，主机侧`physx_fetch_results`累计等待6.848619秒，`physx_simulate`记录0.0682333秒；`physics_step`为包含调用的计时。`native_drive_final_a01`另记录800次物理调用的host inclusive `physx_fetch_results` 35.8614712秒、`physx_simulate` 0.2453846秒。两项均是主机侧包含/嵌套时长；它们显示fetch边界仍是当前可见的主要等待，但不能分解为GPU kernel时间。样本包含启动/reset影响，不是持续吞吐、FPS或训练速度基准；目前没有GPU加速倍数或持续FPS数据。

`runtime_architecture.json`记录Windows ARM64 host，以及AMD64 Python/Torch扩展/Isaac组件。该RTX Spark Windows工作流采用x86仿真安装路径属于官方推荐方式，不能据此认定安装错误，也不能单独断定它是性能瓶颈。静态inventory没有验证运行时Kineto/CUPTI采集。

## 轻量webviewer最终验收

新观察端[`scripts/view_training_state_web.py`](../../../scripts/view_training_state_web.py)只读最新真实位姿并离线解析USD，不加载策略或构建训练环境。首次启动的`packet_age_seconds`类型问题已修；原失败日志保留，最终专项17 passed。`web_view_validation/verification_final.json`记录HTTP 200、`no_simulation=true`、`forbidden_modules_loaded=[]`，并确认没有加载Isaac Sim、SimulationApp、SimulationContext、训练环境或Torch；深色主题、4.02m车辆初始相机以及1Hz面板刷新均通过检查。

真实浏览器中车辆、地面、目标和`STALE · holding last pose`标签可见，滚轮缩放和拖动旋转均改变视图，刷新恢复初始车辆相机；证据为[`web_view_validation/browser_verification.json`](web_view_validation/browser_verification.json)和[`web_view_validation/browser_final.png`](web_view_validation/browser_final.png)。画面使用a04在系统重启前落盘的历史状态，run标记`interrupted/system_reboot`，不是实时训练展示。browser与HTTP证据都没有测量FPS。

只读profiler审查见[`trace_review.md`](trace_review.md)。`native_trace_a02.log`记录`CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`及CUDA profiler activities缺失；CPU operator trace不含kernel、CUDA runtime/driver、memcpy或同步事件，所以GPU kernel时长及其它GPU耗时均未知，不能以0或CPU span替代。`native_trace_a02_report.json`里的passed只表示功能断言通过，不是可用CUDA trace。更早的[`trace_export_a01_review.json`](trace_export_a01_review.json)记录Kineto窄字符路径无法导出至Unicode仓库路径。一次profiler-off隔离启动实际exit 1、耗时29.407秒，Replicator启动失败并缺少`instance_segmentation_fast` annotator，未走step、无settings readback或CUDA trace；这些flags仅用于失败尝试，未被采用为项目配置。不能据此判定CUPTI冲突已解除或GPU kernel时长，也不据此给出端到端耗时分解或GPU提速倍数。

## 驱动控制器固定拓扑去重（第二十二批1/5）

`SideIntegralDriveController`在初始化时验证并克隆固定`side_ids`，缓存左右wheel indices；update内侧均值和feedforward直接使用`index_select`，避免公开helper每次重复验证。内部积分路径使用纯tensor helper；公开`side_mean`、`broadcast_side_values`和`conditional_integral_step`仍保留原有严格校验。finite tensor检查合并成单次host bool，Python scalar参数仍保留校验。

focused CPU专项实际输出为28 passed in 3.27s；CUDA小型等价检查输出`CUDA equivalent: device=cuda:0, steps=18, partial_reset=true, saturation=true, topology_clone=true`。原测试和CUDA脚本只通过工具执行/输出，未分别保存原始log或进程记录；[`drive_validation.json`](drive_validation.json)与本段说明是事后照实际工具结果转录，数值没有重算或补造。可复现代码保存在[`drive_cuda_equivalence_repro.py`](drive_cuda_equivalence_repro.py)，它是从当时检查逻辑重建的脚本，本次文档整理没有执行它。

静态路径计数显示update中的host `bool()`转换预期从11降到1，feedforward读取从3降到0；这是静态代码路径分析，不是计时或实测同步次数。最终49模块CPU门禁报告在[`cpu_gate_after_drive.json`](cpu_gate_after_drive_20261024T1233Z/cpu_gate_after_drive.json)：546 passed、4 skipped、47 warnings、10 subtests，61.60秒，实际exit 0。旧530门禁继续保留但仅作历史记录。该优化没有GPU性能倍率或FPS基准。

## 文件保留

此目录中的原始失败、process记录、logs、CUDA核验、native source snapshots及其哈希均保留。本文仅汇总已存在的证据，不改写原始JSON、日志或实验源码快照。
