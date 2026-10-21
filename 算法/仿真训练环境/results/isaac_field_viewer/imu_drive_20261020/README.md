# IMU/驱动反馈试验：二维查看器与当前状态

本记录汇总2026-10-20新feedback-v2合同下的网络UI验收、仿真反馈来源和训练结果。训练源码快照已冻结，trial已完成；该次UI验收仍只覆盖8,192步checkpoint，不代表final checkpoint完成了GUI验收，也不构成田间能力结论。项目总览及训练细节另见 [ISAAC_LAB.md](../../../ISAAC_LAB.md) 和 [使用说明.md](../../../使用说明.md)。

## 页面与启动

车辆交互页在 `http://127.0.0.1:8769/`，独立网络二维页在 `http://127.0.0.1:8770/`。从项目根目录的Ubuntu-24.04 WSL shell启动暂停的正式0步baseline预览：

```bash
bash results/isaac_field_viewer/imu_drive_20261020/run_viewer_frozen_baseline.sh
```

它先以 `results/isaac_field_improvement/fresh_imu_drive_20261020/untrained_baseline.zip` 绑定viewer；已训练检查点可在viewer中选择。2026-10-20约02:24实际加载8,192步near检查点时，viewer API报告 `num_timesteps=8192`、仿真时间0、暂停、60层且 `errors=[]`。这只是checkpoint载入与二维显示检查，并非最终模型评估。训练入口 `launch_training.sh` 已用于创建当前试验；它要求参数 `--after-root-source-freeze`，并会拒绝复用已有trial目录。本trial已完成，不要重复prepare或launch。

页面固定深色，没有主题切换；网络始终保留完整二维层图，不提供整体/overview模式。当前feedback-v2结构实际展示60个神经网络层和71条真实数据边。分区导航可收窄，选择章节后聚焦对应区域但不删层或边；默认详情折叠，点击卡片才显示该层详情。选中层会突出显示真实直接父/子边，其余边仍保留但降低对比度。连续曲线中的交叉桥只标出路径交越，不创建额外神经元或伪造连边。

主代理实际浏览器检查通过的截图：

- [宽屏完整层图](ui_dark_full.png)
- [反馈分支详情](ui_dark_feedback.png)
- [窄屏布局](ui_dark_narrow.png)
- [窄屏导航抽屉展开](ui_dark_narrow_nav.png)
- [车辆深色查看页](vehicle_dark.png)

1280 px宽屏下顶栏约130 px；745×871 px窄屏下顶栏111 px、画布高760 px。窄屏导航抽屉能展开，点击动作区后自动收起，未见页面水平溢出。选层导航不改变完整网络结构。

## 实际运行时数据核验

在未训练UI演示viewer中，以同一checkpoint记录输入时间0.00 s和1.80 s两帧；1.80 s的网络输入对应动作步后车辆反馈时间1.85 s。原始数据比较通过：两帧都是60层、parents引用完整，全部张量值有限；41/60层的raw张量统计改变，其余恒定层允许存在。全局色标在一帧所有层与参数之间共享；选中权重矩阵不会触发重定标。`viewBox`前后完全一致，重置后序列、时间和当前帧清零，暂停时重新读取不推进仿真。

另行加载新trial的8,192步near checkpoint后，实际API显示步数8,192、暂停时间0且60层正常；root对该checkpoint做了0→1.70 s runtime检查，41层raw激活统计改变且 `viewBox` 不变，记录位于 `runtime_8192/`。这部分确认新checkpoint与UI实际连通；依然只是激活/显示验收，不是任务表现评估。

参数面板读取反馈分支权重 `32×41=1,312` 个数值。可复核的原始工件为：

- `runtime_final_ui/comparison.json`
- `runtime_final_ui/browser_state.json`
- `runtime_final_ui/frame_before.json`、`frame_after.json`
- `runtime_final_ui/tensors_before.json`、`tensors_after.json`
- `capture_final_before.log`、`capture_final_after.log`

截图/激活变化只证明网络前向和UI数据刷新有效，不表示权重在这次查看中发生学习，也不证明到点或田间导航能力。暂停preview不施加车辆动作；实际步进回读时间比输入观测晚一个仿真dt（0.05 s）。

## 新trial快照

trial于2026-10-20 02:45:20北京时间结束：runner PID31873和exec98623真实exit0，累计196,608步、29个checkpoint，最终 `stage_04_turn90.zip` SHA-256为 `0a9b310865e854108221cb1983a8804ddbde75c0d77a535e246502d7e83fefd6`。protocol SHA-256仍为 `7f42fb76117bd9620a47a4c8e031694f3d04c20a17e952e071cb794e0a6a7a39`，baseline SHA-256为 `afb53b326b8fadc7928363a34a63ea5585ff07b8d746c3909ea0784f23751129`。独立只读完结验证通过：40个源快照与当前源哈希匹配、final checkpoint的metadata和SHA有效、policy权重及Adam状态有限且更新。机器可读记录：[process_exit.json](../../isaac_field_improvement/fresh_imu_drive_20261020/process_exit.json)、[comparison_summary.json](../../isaac_field_improvement/fresh_imu_drive_20261020/comparison_summary.json)、[completion_validation.json](../../isaac_field_improvement/fresh_imu_drive_20261020/completion_validation.json) 与 [completion_model_validation.json](../../isaac_field_improvement/fresh_imu_drive_20261020/completion_model_validation.json)。

此前进行中状态保留为时间快照：02:21:59为8,192步near，02:23:18为16,384步near，02:26报告完成near 32,768步并转入steer30。02:24–02:26 GUI实际验收使用8,192步near checkpoint，暂停时间0后运行到1.70 s；这是旧于final的preview。当前UI仍未载入最终196,608步权重，因此没有final checkpoint的GUI实测记录。

确定性配对开发评估中，free场景0株、4 seeds无损成功0/4→2/4，平均损伤与碰撞均0，符合预先设定的安全门禁；16株field场景4 seeds无损成功0/4→0/4，平均末距1.683→1.601 m，summary判定 `no_improvement`。旧direct-track-v4 trial结果仅属于其旧观测合同，不与本次比较混算；820xx为开发种子，830xx没有评估或观察。这不是盲测、架构消融或整田可靠性结论。

## 训练设置与可解释范围

当前输入合同 `sensor-imu-drive-v5` 含LiDAR `[2,400]`、深度 `[1,160,88]`、检测 `[16,10]`、state39和feedback41；五路编码为32+64+32+32+32=192维，RecurrentPPO LSTM隐藏宽度128，actor输出左/右履带目标速度及云台偏航/俯仰角速度4个归一化动作。当前编码器维数/网络拓扑是预设的，优化算法学习权重而非架构。

fresh trial为4环境、每环境rollout128步（共512 transition）、PPO minibatch128、5个优化epoch，learning rate 3e-4、γ0.995、GAE λ0.95、target KL0.03、初始log-std−2、`use_sde=false`，每8,192步checkpoint。训练动作由高斯策略采样；确定性评估与viewer推理使用 `predict(... deterministic=True)` 取策略分布中心。episode起点以 `episode_start` 信号复位对应actor LSTM记忆；episode内部跨步延续记忆。每个独立评估episode以新环境reset及初始记忆开始。网络图中的价值支路是独立LSTM记忆的额外只读诊断forward，不控制车辆；实际RecurrentPPO训练仍有其自身critic。

四阶段goto课程为near 32,768步（0.6–1.2 m、目标角±15°）、steer30 32,768步（1–2 m、±30°）、steer60 65,536步（1–3 m、±60°）、turn90 65,536步（1–3 m、±90°），总计196,608步。各阶段固定平地、plants=0、episode上限20 s。进度预算5、到点+5、每株新增受损植物−20、岩石碰撞−0.1、时间−0.01/秒，另有边界惩罚；`distance_cost=0`、`goal_gain=1`。成功定义为距离≤0.4 m、车速绝对值<0.2 m/s、偏航角速度绝对值<0.3 rad/s并保持0.3 s，不要求最终朝向。

因此目前训练直接覆盖的是无植物平地的基础goto操控。它不提供植物识别、绕行、避苗或障碍场地导航的训练经验。当前早期training-near日志的滚动成功数是无植物训练中的统计，不能当作16株field表现。这样的训练场景提供了“尚无田间学习证据”的直接限制，但不能据此断言网络绝无迁移可能；对当前field能力仍须等正式配对开发评估。820xx已观察，830xx未观察，但只有后者可作为之后独立验证候选；不应用于训练决策或调参。

## 仿真反馈与实机边界

41维feedback由25个原始/估计测量值、8个有效位和8个年龄值组成，含云台IMU加速度/角速度/磁场原始计数/温度/Euler/四元数、云台两轴q/qd以及左右电机Iq/RPM。策略接收的是仿真实现，不是实机传感数据：VQF及传感噪声/偏置来自 `SIM_SENSOR_MODEL` 假设；仿真Iq按代理轮 `joint_f`、`Kt=0.035 Nm/A`及效率1假设换算，RPM按仿真代理轮速、传动比和几何换算。它们不是真实力传感器、实测电流或实测电机RPM。

用户确认实车10轴ATK-IMU901装在云台，随云台偏航/俯仰运动；左右驱动器可回读电流和实际转速，云台两轴可回读角度与转速。项目FOC接口存在96项host check（0 fail），Python telemetry和feedback专项分别10与8项通过；这些检查没有打开串口/CAN，也没有实机/台架闭环联调。IMU安装外参、磁场物理单位、各通道噪声、驱动器遥测协议与实物固件版本、电流标定都仍需实测确认。ATK云台IMU不能简单等同车体IMU；当前仿真更不能用云台目标动作冒充实测速度。传感器实现细节见项目 [ISAAC_LAB.md](../../../ISAAC_LAB.md) 和硬件代码/协议。

## 验证记录与未清事项

训练源码/UI冻结前的最终CPU复验日志 `final_cpu_verified.log` 显示222 passed、11 skipped、1项既有NumPy float32极端值std溢出warning，pytest exit0。`final_cpu.log`保留了较早shell变量转义问题造成的真实exit1；复验由另一条修正命令执行，不覆盖失败记录。fresh runner专项日志 `fresh_runner_validation.log` 为16 passed及compileall exit0。未在这里重跑训练或CPU测试。

UI证据通过 `network_2d.py/html` 当前代码生成；LLM Visualization的上游来源、MIT许可全文及修改说明记录在 `isaac_field/visualization/THIRD_PARTY_NOTICES.md`。新版IAB tab14在本次 `dev.logs({levels:['error','warn'],limit:12})` 检查返回空列表；旧tab11在viewer重启期间成为connection-refused错误页，不能把它当作新版页面错误或tab14的console error。
