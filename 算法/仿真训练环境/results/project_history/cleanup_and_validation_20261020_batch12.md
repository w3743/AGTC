# 第十二批整理与验证记录

日期：2026-10-20。本批前三项完成后，IMU/云台测量反馈输入为第4项，显式驱动与电机负载反馈及协议适配为第5项；本次按五项规则整理并同步说明。批次计数由根代理统一维护，本记录不改 `AGENTS.md`。

## 当前输入、动作和驱动合同

当前策略输入为 `sensor-imu-drive-v5`：state 39维，feedback `gimbal-imu-drive-feedback-v1` 41维；策略编码器为 `depth64-lidar-circular1d-feedback-v2`，五路分支 LiDAR32、depth64、detection32、state32、feedback32，合并192维，RecurrentPPO LSTM为128维。动作保持 `direct-track-speed-v1` 四维 `[-1,1]`，顺序为左履带速度、右履带速度、云台偏航角速度、云台俯仰角速度；驱动为 `independent_left_right_track_speed_explicit_v4`。当前加载器在SB3读取前校验输入、encoder、动作合同；旧模型缺字段或不匹配时会拒绝，不允许只重贴metadata。旧模型、协议快照、训练记录和失败结果仍保留。

feedback前25个值按组排列：加速度3、角速度3、磁场raw signed counts 3+温度1、Euler3+四元数xyzw4、云台yaw q/qd2、云台pitch q/qd2、左Iq/RPM2、右Iq/RPM2；随后8个valid和8个age。groups为 `accel, gyro, mag_temperature, attitude, gimbal_yaw, gimbal_pitch, motor_left, motor_right`。硬件ID1 Euler和ID2 xyzw四元数分别保留板载字段；仿真attitude来自VQF，缺失硬件值不由truth补齐。输入只应用配置内单位缩放，不裁剪有效值；invalid/stale组置零并清valid，age最多2 s。每组按自己的采样时间判断新鲜度；任一侧的Iq或RPM无效会将该侧电机组整体置invalid。磁力计保留厂家原始signed counts，不假定µT比例；压力/海拔ID5丢弃。加速度代表传感器proper acceleration（含重力），不能与线性加速度混称。

用户已确认ATK-IMU901安装在云台，随偏航和俯仰转动；两轴q/qd与左右驱动器Iq和机械RPM可回读。策略姿态使用带噪云台IMU姿态与实际gimbal q反算出的车体姿态估计，不用truth quaternion、truth body yaw/roll/pitch或真值车速。goal-relative方向与3D传感器运动补偿共用该估计；位置来自模拟UWB noisy sample，且每个唯一采样时刻有独立白噪声。物理真值位姿仅用于渲染等旁路；同一physics time重复读取观测不推进过滤器或随机数。

仿真反馈模型的VQF 2.1.2按20 Hz更新，使用带噪原生单位加速度/角速度与合成磁场；过程噪声、偏置和温度噪声均为 `SIM_SENSOR_MODEL` 中的仿真假设，不表示ATK固件融合算法、实际固件采样率或硬件噪声标定。驱动显式向Newton写入限幅后的 `joint_f`；仿真Iq按每侧7个代理滚轮的实际已施加effort、`Kt=0.035 Nm/A`与效率1假设换算，RPM由代理滚轮求解角速度、45 mm代理滚轮、30 mm主动链轮和4:1齿比换算。这些量只是模型等效电流/转速，非实测值、转矩测量或履带接地力传感器。速度伺服gain 0.04与每轮2 Nm effort cap也是数值参数，尚未实车标定；代理仍是14滚轮与地面接触，不是连续履带模型。

Python decoder/assembler位于 `isaac_field/telemetry.py`，只负责将有时间戳与质量状态的协议数据组装为sample，不自行打开串口或CAN。FOC主机codec源文件位于 `电力/FOC驱动模块/软件/速度反馈接口.c/.h`、`速度反馈接口测试.c`；CAN协议和 `golden_vectors.json` 在同目录，PowerShell runner为 `运行主机测试.ps1`。runner以严格C11和GCC编译到临时目录；96 checks、0 failures、exit 0。它验证主机codec、固定向量和边界逻辑，不验证G431目标构建、HAL、CAN电气收发、FOC固件实际发送状态帧、实机驱动环或台架响应。项目中当前尚无可构建的G431 FOC/HAL状态遥测实现。Python telemetry10项和feedback8项专项通过。协议磁场映射已记录为ATK原始有符号counts，未伪造µT。

本地手册 `docs/hardware/atk_imu901/ATK-IMU901_User_Guide_v1.2.pdf` 文件名写v1.2，但PDF封面标注V1.0，是下载的镜像副本；尚未证明它与安装硬件版本或最新固件一致。IMU磁场比例、安装外参、噪声、真实电流标定及整条实机遥测链路仍需实机核验。

## 训练和界面状态

独立trial `results/isaac_field_improvement/fresh_imu_drive_20261020/` 当前尚未创建、未prepare、未启动，没有feedback-v2正式随机初始化0步基线或训练checkpoint。`results/isaac_field_viewer/imu_drive_20261020/untrained_imu_drive_preview.zip` 是单独UI演示随机权重，不是训练基线或checkpoint。trial需在根代理冻结源文件并确认后，使用带 `--after-root-source-freeze` 限制的launcher启动。

拟议训练总计196,608步、4环境、阶段seed69001–69004，20 s/episode、plants=0、平地goto、`damage_penalty=20`、`goal_progress_budget=5`、`distance_cost=0`、`goal_gain=1`：near 32,768步（0.6–1.2 m、±15°）、steer30 32,768步（1–2 m、±30°）、steer60 65,536步（1–3 m、±60°）、turn90 65,536步（1–3 m、±90°）。到点为≤0.4 m且低速保持0.3 s，不要求终态朝向。方案拟保留随机初始化0步基线和配对开发评估：free无植物与16株field场景各4个seed；field仅是开发检查。830xx仍为未观察集。此项计划尚未执行，不宣称策略学会作物避让、搜索或整田导航。

此前完成的 `fresh_direct_tracks_20261020` 使用旧 `sensor-depth-direct-tracks-v4`/43 state/160融合，动作相同但输入和encoder不兼容；当前loader拒绝它。旧trial196,608步、exit0、final checkpoint SHA `342237b1b2cacfb08135535891f9aaf1d4eb5ef957a68d85168bfe2f43e21329`。free开发配对无损成功0/4→1/4，损伤/碰撞为0；16株field无损成功0/4→0/4，平均末距1.797→2.811 m。只显示free一例无损到达、field无提升，不据此说总体能力改善；旧源、权重、protocol、评估和模型均保留。

当前二维网络页仍是Graphviz连续曲线原型，几何统计为卡片重叠0、连线穿卡0、共线边0、边交叉16处；没有浏览器目视验收。后续界面计划复用仓库内MIT许可 `external/llm-viz` 重构LLMVisualization并固定深色主题。此前57节点/68边布局的浏览器验收属于旧viewer，不能当作当前feedback-v2新界面的验收。车辆三维页8769另已目视/截图核对深色面板与背景、无主题切换且物理地面保留材质，截图为 `results/isaac_field_viewer/imu_drive_20261020/vehicle_dark.png`；该检查不验收二维网络页。

## 验证记录

本轮完整CPU suite由Ubuntu-24.04 WSL中的项目Python 3.12运行：`scripts/isaaclab_python.sh -m pytest -q`，日志最终记录 **218 passed、11 skipped、1 warning，明确exit 0**，详见 [CPU清理日志](../isaac_field_viewer/imu_drive_20261020/cpu_cleanup.log)。warning为极端float32测试中NumPy对平方计算的overflow warning，suite通过。日志保留了早期输出状态为空的调用记录；末尾重跑写出 `WSL_pytest_exit=0`。第一次Windows Python 3.11尝试因缺SB3收集失败，另一个中间 `policy_contract_validation.log` 在测试源码并发写入时记录了SyntaxError；两份均保留为失败历史，不代表后续最终WSL全量回归。

静态验证见 [static清理日志](../isaac_field_viewer/imu_drive_20261020/static_cleanup.log)：`compileall -q isaac_field scripts tests` exit0；`bash -n scripts/install_isaaclab.sh scripts/isaaclab_python.sh` exit0；Node解析正式 `network_2d.html` 内嵌JS，共1个脚本块、exit0。之后fresh runner增加SciPy版本与Newton runtime USD路径/哈希记录；当前 `tests/test_fresh_foundation.py` 专项为16 passed，源代码 `compileall` exit0，日志见 [fresh runner验证](../isaac_field_viewer/imu_drive_20261020/fresh_runner_validation.log)。focused pytest覆盖prepare/preflight、动态snapshot集合、schema/action拒绝、checkpoint metadata和配对条件；它不启动仿真，未实采Newton runtime USD哈希。此次未对新增元数据代码重跑完整CPU suite。命令输出含WSL localhost代理提示，但真实退出码为0。

Newton native测试按不同gain分开报告。显式gain 0.04的运动/姿态反馈、低速制动和转向检查为3 passed、39 warnings，日志 `native_explicit_gain_validation.log`；gain 0.04的双环境自动reset及左右独立命令/限幅/加速度测试为2 passed、38 warnings，日志 `native_batch_reset_validation.log`。早期gain 0.5配置日志 `native_initial.log` 保留1失败、1通过、38 warnings，其中驱动不稳定测试失败；该失败促成将gain调低，不能将其与gain 0.04的通过合并成一次全通过。初始运行另含官方wrapper检查通过的结果。FOC硬件codec主机测试日志为 [foc_host_validation.log](../isaac_field_viewer/imu_drive_20261020/foc_host_validation.log)，96 checks、0 failures；它只在主机通过GCC运行，不是实机验证。native检查也是仿真物理回归，不是实机测试。以上日志均在 `results/isaac_field_viewer/imu_drive_20261020/`。

## 整理范围

本次文档整理同步了 [ISAAC_LAB.md](../../ISAAC_LAB.md) 与 [使用说明.md](../../使用说明.md)，并新增本记录；本整理没有改Isaac源码、测试、权重、训练protocol或viewer静态文件，也没有启动训练、GPU长测或FPS基准。旧模型、协议、训练输出与失败日志均保留。当前项目原始数据与结论必须区分：真值仅用于物理仿真/渲染或诊断，不等于策略传感器；无标定的effort→Iq换算和合成传感器噪声只称为仿真模型假设。
