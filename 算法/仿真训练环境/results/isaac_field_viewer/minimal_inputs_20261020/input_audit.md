# 策略输入必要性审计与精简决定

日期：2026-10-20。用户要求核对全部输入并尽量缩小；进一步确认保留160×88深度图、保留磁场，其余按独立信息和重复性精简。本文记录设计取舍；运行、训练与验证结果另见本目录验收记录，不能把下表的设计目标当成已经通过的能力测试。

## 数量与结构

以下均不含batch维、LSTM循环状态；输入元素数不等同于独立信息量、模型参数量或计算耗时。

| 分支 | v5输入 | v6输入 | 变化 |
|---|---|---|---|
| LiDAR | 2×400，800个值 | 1×400，400个值 | 全部400束保留；0表示无效，有效距离仍除以8m |
| 深度 | 1×160×88，14080个值 | 保持 | 用户选择保留细小障碍分辨能力 |
| 检测 | 16×10，160个值 | 16×8，128个值 | 去固定0列及可由正框尺寸恢复的存在列；保留测距有效位 |
| state | 39 | 33 | 模式2→1；删重复云台角编码3、固定dt和容差2 |
| feedback | 41 | 37 | 删温度1、Euler3；保留磁场、四元数和全部8组质量/年龄 |
| 总计 | 15120 | 14678 | 减442，约2.92%；非深度1040→598，减42.5% |

五路编码仍为32/64/32/32/32，融合192、actor/critic LSTM128、四维动作和物理驱动保持。本次不缩减深度、雷达束数或检测槽数，不附带主干容量调整。

## 全部输入的必要性判断

索引为v5的零基索引。

| 分支/旧字段 | 决定 | 理由与限制 |
|---|---|---|
| LiDAR距离400束 | 保留 | 障碍几何信息；不降低角分辨率 |
| LiDAR有效性400项 | 合入单通道 | 用最终valid决定无效输出0；有效最小测距>0，仍能区分无效和被噪声裁到8m的有效回波。内部原始mask保留；不从range<8重新猜mask。无效样本的伪距离不作为可靠测距保留 |
| 深度所有14080像素 | 保留 | 用户选择；缩小分辨率会损失细茎、窄叶、遮挡缝细节，不能算无损精简 |
| 检测框cx/cy/w/h，0–3 | 保留 | 目标在视野中的位置及尺寸 |
| 类别4、置信度5 | 保留 | 分类和可见性质量是独立信息；0植物训练没有覆盖它们不等于以后无用 |
| 检测距离6、距离有效7 | 都保留 | 缺失距离为0，但当前带噪测距理论上也可出现0或负值，不能严格从数值恢复有效位 |
| 检测存在8 | 从正w/h恢复 | 空槽全0，真实框正宽高；将正框尺寸写入并校验v6输入合同，不能允许非法回调框静默冒充padding |
| 检测保留9 | 删除 | 恒为0，无独立信息 |
| state模式0–1 | 单个二值量 | 两种合法任务search/goto，只需一个值0/1；不删除搜索任务语义 |
| state目标局部xy，2–3 | 保留 | 当前到点任务的目标条件；搜索模式仍置0 |
| state估计绝对xy，4–5 | 保留 | 全局坐标锚点，对用户后续全局记忆/搜索方向有价值；不把任务压成仅局部goto |
| state车体航向sin/cos，6–7 | 保留 | 已完成云台IMU到车体的坐标变换，且逐传感器位移使用世界XY分量；直接删除会把这项推导转移给网络 |
| state云台yaw sin/cos、pitch，8–10 | 删除 | 已有feedback实测yaw/pitch关节角。删去便利性编码可能改变学习难度，仍需新训练验证；不声称策略数值输出不变 |
| state上一动作11–14 | 保留 | 指令历史与实测速度不同；当前履带加速度限制及云台积分目标有控制状态记忆 |
| state时间比例15 | 保留 | 有限时限任务的剩余时间条件 |
| state dt16、容差/场地尺寸17 | 删除 | 本轮固定配置常量；配置仍随实验与模型元数据记录。若以后随机化这些参数，需重新评估是否作为条件输入 |
| state三类传感器帧龄、新帧标记 | 保留 | LiDAR/深度/检测相机周期、延迟不同；新帧状态和年龄作用不同 |
| state三类传感器各自的位移/航向/云台角变化 | 保留 | 对应三个不同捕获位姿，不能共享一组；为延迟观测提供运动背景 |
| feedback加速度0–2 | 保留 | 传感器proper acceleration含动态与重力响应，独立于融合姿态 |
| feedback角速度3–5 | 保留 | 直接角运动测量，独立于姿态值 |
| feedback磁场6–8 | 保留 | 用户明确保留；原始磁场可补充融合结果以外的信息，单位为raw counts |
| feedback温度9 | 删除策略输入 | 当前温度不参与VQF，仿真为25℃附近的噪声。串口解码/诊断可继续保留，不能据此宣称实机温漂不重要 |
| feedback Euler10–12 | 删除策略输入 | 表达与四元数相同的姿态；原始固件Euler可继续诊断，策略不重复输入两套姿态表示 |
| feedback四元数13–16 | 保留 | 云台IMU姿态；硬件路径以有效四元数为姿态源，不再要求Euler包同时到达才能有效；仿真仍用VQF，不用真值补姿态 |
| feedback云台q/qd，17–20 | 保留 | q用于区分车体与云台姿态，qd是独立实测运动；不拿目标命令冒充测量 |
| feedback左右Iq/RPM，21–24 | 保留 | 负载与转速互补；当前仿真Iq是未标定模型等价值，实车链路尚未验证 |
| feedback 8组valid，25–32 | 保留 | 区分真实零值与无效零填充；显式无效数据可能仍很新鲜，不能从age推导 |
| feedback 8组age，33–40 | 保留 | 不同源可能异步、丢包；不能把当前仿真中偶尔同龄当成实机恒相同 |
| LSTM h/c与episode_start | 保留 | 循环状态及回合重置控制，不属于上述15120/14678个外部观测 |

## v6字段顺序

state33：mode0；目标局部xy1:3；估计绝对xy3:5；车体航向sin/cos5:7；上一动作7:11；时间比例11；LiDAR元数据12:19、depth19:26、camera26:33。三个元数据块均为age,new,dx,dy,body-yaw-delta,gimbal-yaw-delta,gimbal-pitch-delta。

feedback37：accel0:3、gyro3:6、mag6:9、quat9:13、yaw q/qd13:15、pitch15:17、left Iq/RPM17:19、right19:21；valid21:29；age29:37。组顺序为accel,gyro,mag,attitude,gimbal_yaw,gimbal_pitch,motor_left,motor_right。只做已记录的单位尺度变换，有效测量不新增裁剪。

detections每行8项：cx,cy,w,h,class,confidence,range/depth_range,range_valid。存在mask由正w/h计算后参与真实模型的padding排除，查看器也应复用同一个计算函数。

新合同为`sensor-minimal-v6`、`gimbal-imu-drive-feedback-v2`、`depth64-lidar-single1d-minimal-v3`。旧v5权重和实验快照保留，不能给旧权重重贴新metadata；输入变动需要新初始化训练。即使某些字段可推导，也不等于新策略训练后的性能必然不下降。

## 实际验证与运行状态

- 官方Ubuntu-24.04环境CPU全套231 passed/11 skipped；加强单通道LiDAR尾部非空断言后专项171 passed/11 skipped，均pytest exit0。首次测试迁移失败及修复日志完整保留。8项warning来自TorchScript tracing/deprecation及既有NumPy极端值std溢出。
- 真实Newton原生回归6 passed/44 upstream warnings；pytest与独立probe均记录exit0。探针确认五路形状、磁场valid始终1、age随10Hz采样在0/.05秒交替。外层状态回显曾因PowerShell变量展开错误exit1，该外层错误保留，不称整条调用exit0。
- 真实Netron export-only含五路传感器和四个循环状态输入；3序列×3步九项比较最大绝对误差均0。未启动Netron服务。详见[验证摘要](validation_summary.md)。
- 训练前的独立0步随机预览曾在8769/8770运行：仿真0→10秒后暂停，60层全有限、41层raw值变化、两项age变化，viewBox稳定、60卡片无重叠，console/errors为空。它只验证新结构的随机初始化预览；final策略部署验收已更新到下一条。原始证据见[runtime_ui/raw_comparison.json](runtime_ui/raw_comparison.json)、[浏览器记录](runtime_ui/browser_acceptance.json)与[预览截图](ui_minimal_feedback.png)。
- 预览ZIP SHA-256 `03cc62b601ff007bbf7886dcd833d129b3ba04fddbb4ec94d7592c39df09f4b8`，num_timesteps=0、field_preview_only=true；不是正式训练基线。v5 final在SB3加载前被新合同拒绝，SHA前后一致。
- 当前训练前0步预览已由trial final checkpoint替换；8769/8770实际载入196608步final并运行0→8.4秒后暂停，60层有限、50层raw张量变化、2项age变化、viewBox稳定、dev.logs为空。raw与截图见[runtime_final/raw_comparison.json](runtime_final/raw_comparison.json)、[browser_acceptance.json](runtime_final/browser_acceptance.json)、[network_final.png](runtime_final/network_final.png)和[vehicle_final.png](runtime_final/vehicle_final.png)。
- 新初始化trial `fresh_minimal_inputs_20261020` 已完成196608步；runner PID155311于2026-10-20 07:12:30北京时间真实exit0，launcher exec86650真实exit0。protocol SHA-256 `b5e8e9c3b0561bc73733528aa0c5be3120bb5aad3a54a1fb3418ad84f30a6d65`，final SHA-256 `2c03cc130977cc005adf56527bc10bc4719a8a9d9489fa82c5a1b41f9d3966e5`，0步baseline SHA-256 `443b9fc73676bc5dfe6d42450bf2112089852d27c373d12c4b2443f484e43ca9`。40项快照与当前源码哈希匹配、共29个checkpoint；completion_model_validation检查Adam37项有限且step=7531，权重有限，LiDAR/深度/state/feedback/LSTM/action均变化。检测框MLP有限但未变化，符合plants=0训练协议。
- 配对开发评估均为no_improvement：free 87001–87004成功0/4→0/4，平均回报−0.339→−2.363、末距1.680→2.321m；field 82001–82004成功0/4→0/4，平均回报−0.320→−2.037、末距1.837→2.448m；损伤和碰撞均0。见[completion validation](../../isaac_field_improvement/fresh_minimal_inputs_20261020/completion_validation.json)、[模型校验](../../isaac_field_improvement/fresh_minimal_inputs_20261020/completion_model_validation.json)及[comparison summary](../../isaac_field_improvement/fresh_minimal_inputs_20261020/comparison_summary.json)。这是开发数据，未评估或观察830xx，不称能力提升或整田可靠性，不做FPS测量。
