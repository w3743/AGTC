# Native 原型与诊断工具

本目录保留 M0–M3 原生栈原型、语义对照/集成检查、性能剖析和诊断脚本。它们帮助说明开发过程中的实现与证据，不是 Native V2 正式训练入口；V2 使用 [train/README.md](../train/README.md)。本目录的 JSON 与报告是对应阶段生成的记录，需结合文件内时间、配置和结论阅读，不能替代当前状态或正式能力验收。正式 trial 快照见上级 acceptance/ 索引。

## 源码、脚本与记录

| 文件 | 用途 |
| --- | --- |
| CAPABILITY_GAPS.md | 记录原型传感器通道的能力缺口、实现建议和估算。 |
| check_contract.sh | 在 CPU 构造带解析 depth/detector 的原型环境，并打印观测通道模式、provider 身份和能力阻塞项。 |
| debug_m1_lidar.py | 在 M1 LiDAR 对照中定位差异最大的单条光线及命中 ID。 |
| debug_m1_warp.py | 用已知命中射线逐阶段检查 Warp LiDAR kernel 与 NumPy 路径。 |
| debug_m1_warp2.py | 分步检查 Warp LiDAR kernel 的射线求交条件。 |
| debug_newton_isolation.py | 改变驱动、mimic、子步数和 solver 迭代次数，定位 Newton XPBD 非有限状态。 |
| debug_newton_nan.py | 逐步打印 Newton XPBD 初始化、控制与仿真状态，查找 NaN 出现位置。 |
| debug_newton_sizes.py | 打印 Newton 原型机器人与多 world 模型的关节/刚体规模和标签。 |
| debug_state_len.py | 检查 Isaac 参考任务状态字段数量及其输入 schema，定位 39/33 维差异。 |
| depth_caster.py | 从参考场景几何生成 160×88 轴向 Z 深度；包含 NumPy 与 Warp 实现。 |
| DEPTH_DETECTIONS_REPORT.md | 汇总原型 depth 与 detections 实现、对照和未关闭的传感器域差。 |
| depth_gpu_perf.py | 构造相机位姿并测量 64 个相机的解析 depth raycast 性能。 |
| depth_integration.json | 保存原型环境 depth caster 接入检查的输出。 |
| depth_verification.json | 保存解析 depth 语义、数值、性能和 provider 集成检查的汇总结果。 |
| detections_batch_verification.json | 保存批量 detector 路径对照及计时记录。 |
| detections_caster.py | 从解析场景生成 detector mask，并装配、排序和打包 16×8 检测观测。 |
| detections_gpu_perf.py | 测量 Warp detector mask 路径的批量 GPU 性能。 |
| detections_integration.json | 保存 detector 输出接入原型环境后的形状与逐元素比较结果。 |
| detections_verification.json | 保存解析 mask、检测框语义、字段打包及 provider 检查记录。 |
| detections_warp_verification.json | 保存 Warp mask/batch kernel 与 NumPy 路径的比较和计时结果。 |
| diag_5ms_divergence.json | 保存不同 timestep/后端诊断案例的输出记录。 |
| diag_5ms_divergence.py | 使用 pilot checkpoint 在 CPU 上比较 5 ms/10 与 1 ms/50 条件下的状态有限性。 |
| EQUIVALENCE_REPORT.md | 记录 M1/M2 阶段对观测、反馈、接触、重置和任务计算的原型对照结果及限制。 |
| inspect_crashed_pilot.sh | 查看指定 5 ms pilot run 的目录、身份和末尾训练指标。 |
| inspect_crashed_pilot2.sh | 查看指定 5 ms pilot run 的文件、身份及观测契约摘要。 |
| m1_depth_integration.py | 在 MuJoCo CPU 环境中检查 provider depth channel 与独立解析 caster 输出是否一致。 |
| m1_depth_verification.json | 保存 M1 depth 算法与语义检查结果。 |
| m1_depth_verify.py | 对照 NumPy/Warp depth caster，并检查参考内参、深度归一化与解析平面场景。 |
| m1_detections_batch_verify.py | 比较逐位姿与批量 detector 的中心距离、观测装配及耗时。 |
| m1_detections_integration.py | 在 MuJoCo CPU 环境中检查 provider detector channel 与同位姿独立输出。 |
| m1_detections_verify.py | 将解析 detector 结果按 Isaac Field 观测字段顺序打包并对照。 |
| m1_detections_warp_verify.py | 比较 Warp detector mask kernel、NumPy mask 和检测行打包结果。 |
| m1_lidar.py | 构建参考场景，比较 NumPy/Warp 解析 LiDAR 与 Isaac Field scene cast 的量程及命中结果。 |
| m1_state_feedback.py | 对照 Isaac Field 参考实现计算的 33 维 state 和 37 维 feedback。 |
| m1_verification.json | 保存 M1 观测等价检查、能力阻塞项及合并后的反馈对照记录。 |
| m1_vqf_verification.json | 保存 VQF feedback 旁路的数值与 CPU 成本结果。 |
| m1_vqf_verify.py | 对照 VQF feedback 路径的姿态/观测结果并测量 CPU 更新成本。 |
| m2_contact_probe.py | 逐步记录单个 MuJoCo CPU world 的接触数、车体高度、速度和 solver 迭代。 |
| m2_equivalence.py | 检查原型 spawn/reset、reward/termination、timeout 与环境适配等 M2 对照项。 |
| m2_spawn_probe.py | 观察初始状态接触数，并检查轮地间隙、模型质量和落地过程。 |
| m2_verification.json | 保存 M2 spawn、reset、reward、timeout 和环境接入对照结果。 |
| merge_vqf_into_m1.py | 将独立 VQF 检查结果写入 m1_verification.json，并更新其中的 capability blockers。 |
| native_env.py | 实现 M0 用的 Gymnasium 向量化原生环境及 reset/step 接口。 |
| newton_backend.py | 实现各 world 独立状态的 Newton XPBD 原型后端。 |
| newton_mujoco_backend.py | 实现继承 Newton 原型接口的官方 MuJoCo-Warp solver 后端。 |
| nonfinite_guard_tests.json | 保存非有限状态保护相关的测试结果。 |
| observation_provider.py | 为 M0 原型计算批量 LiDAR、state、feedback 和标记为占位的 depth/detections 通道。 |
| physics_backend.py | 实现 M0 MuJoCo CPU 车辆模型、驱动和批量仿真后端。 |
| policy_m0.py | 实现吞吐原型使用的轻量 actor-critic 与 PPO 策略模块。 |
| profile_encoder.py | 单独测量 CPU 上观测编码器前向耗时。 |
| profile_m0.py | 分段测量 M0 环境采集、观测、策略和优化等耗时。 |
| profile_ppo_threads.py | 比较不同 PyTorch CPU 线程数下的 PPO update 耗时。 |
| profile_ppo.py | 单独测量 M0 原型 PPO update 的 CPU 耗时。 |
| profile_solver.py | 比较固定与随机状态/动作下的物理 step 耗时及 solver 迭代信息。 |
| profile_step.py | 分段分析 M0 向量环境 step 的时间消耗。 |
| profile_threading.py | 对比 MuJoCo data 分片并行 step 的线程耗时。 |
| REPORT.md | 记录 M0 吞吐原型的配置、结果和明确限制。 |
| reward_fast_stop.py | 调用 Isaac Field fast_stop_v2 张量逻辑的原型 reward/termination 适配器。 |
| run_depth_verify.sh | 以 CPU 运行 depth NumPy/Warp、解析平面和 provider 验证脚本。 |
| run_detections_verify.sh | 以 CPU 运行 detector mask、字段语义和打包验证脚本。 |
| run_detections_warp_verify.sh | 以 CPU Warp device 运行 detector kernel 与 NumPy 对照脚本。 |
| run_lstm_pilot_fixed.sh | 启动旧的受保护 LSTM pilot 脚本；源码配置为实验性 5 ms/10 路径，不是 V2 runner。 |
| run_m0_acceptance.sh | 执行 M0 的短 CPU 重复性/吞吐试跑，并写入该原型的 run 目录。 |
| run_m1_verify.py | 顺序运行 M1 观测等价检查并写出 m1_verification.json。 |
| run_m3_clock.sh | 在 M3b 采样窗口内记录 GPU 使用率、时钟、功耗和温度。 |
| run_m3_gpu_acceptance.sh | 执行 M3a/M3b 短 GPU gate；脚本注释要求仅在明确的 GPU 空闲窗口运行。 |
| run_m3b_4096.sh | 执行 M3b 的 64×64 sample 短测，并写入测量输出。 |
| run_vqf_throughput.sh | 在 CPU 上比较 64 环境 direct 与 VQF feedback 的短测耗时。 |
| smoke_newton.py | 以 CPU 初始化并短步运行 Newton XPBD 原型后端。 |
| smoke_physics.py | 不采集观测，仅检查 MuJoCo CPU 物理后端身份和 step 行为。 |
| summarize_fixed_pilot.sh | 打印旧 fixed 5 ms pilot 的失败文件和指标摘要。 |
| summarize_m3b_4096.sh | 汇总 M3b-4096 GPU 采样数据并检查进程/GPU 释放状态。 |
| test_newton_backend_isolation.py | 提供 Newton XPBD world 隔离 CPU 检查及可选的有界 GPU 物理验收入口。 |
| test_newton_mujoco_backend.py | 提供 Newton MuJoCo-Warp 后端和 masked reset 的 CPU 回归用例。 |
| test_nonfinite_guards.py | 检查环境、provider 和训练失败路径的非有限状态保护。 |
| train_m0.py | 运行 M0 MuJoCo CPU 向量环境上的简化 PPO 吞吐训练循环。 |

目录内部分 shell 脚本会直接启动计算或采样 GPU；本 README 只说明其作用。运行前先核对脚本配置和上级迁移说明。当前正式训练入口在 train/，不由这些原型脚本替代。
