# 农田场景与 Gazebo Harmonic

本入口把 `field_sim` 的既有场景真值导出为独立的 Gazebo Harmonic / `gz-sim 8` world，用来检查静态几何、履带底盘、云台、传感器和仿真步频。它保留农田仿真作为当前训练环境；Gazebo runner 不调用 Gym `step()`，也没有把检测、植物损伤奖励或策略闭环移入 Gazebo，因此它报告的是场景/传感器 smoke 与 free-running physics RTF，不是 Gym 训练吞吐。

## 启动

双击仓库根目录的 `启动农田Gazebo.cmd` 会以固定 seed 7 启动可视化 world，并保持履带静止，避免长时间观察时机器人驶出有限地形。需要长时间手动观察时，可在 PowerShell 中运行：

```powershell
.\scripts\field_gazebo.ps1 -Gui -Seconds 3600 -Seed 7 -NoMotion
```

`-Seconds` 是传感器预热后的仿真秒数。GUI 静止检视时，可在 Entity Tree 中选择 `field_robot` 并使用 `Move To` 定位视角；不需要自动驾驶控制器。无 GUI smoke 默认会发送温和运动命令，使用 `-NoMotion` 可关闭。按 Ctrl+C 会结束本 runner 自己启动的 Gazebo 进程、保存当时已收到的 capture，并在 `acceptance.json` 中标记 `interrupted`。使用已有的配对场景/状态快照时，两项必须同时提供：

```powershell
.\scripts\field_gazebo.ps1 -Gui -Seconds 3600 `
  -SceneJson .\results\gazebo_field_source\current_scene.json `
  -StateJson .\results\gazebo_field_source\current_state.json
```

无 GUI 的 Linux/WSL smoke 命令（先前已导出的 world 可直接复用）：

```bash
bash scripts/field_gazebo.sh \
  --world results/gazebo_field/current/world.sdf \
  --output results/field_gazebo/manual-smoke \
  --seconds 10
```

默认隔离设置为独立 `GZ_PARTITION`、`ROS_DOMAIN_ID=73`，Gazebo 使用独立进程组，CPU 数学库线程限制为 1。默认不启用 ROS bridge、RTAB-Map 或 UWB 插件，不连接旧 mapping 会话，也不占用现场服务端口 `8767`；`-WithUwb` 会加载本机已构建的 UWB 插件，并按 manifest 中的锚点/link ID 订阅全部距离 pair topic，不属于基准场景负载。插件 `gz.msgs.Double` 原始值单位为 cm，`uwb.csv` 同时写入原始 cm 和换算后的 m。当前预设将 NLOS dropout 关闭、噪声标准差固定为 5 cm、距离丢包率固定为 2%，以免 heightmap AABB 被插件误认为约 50 m 厚的墙；超过 75 m 仍会触发插件的硬丢包。该兼容性设置不模拟真实 NLOS 或遮挡，锚点位置也未按实车场地标定，UWB smoke 至少需要 3 个不同锚点 pair 收到有限距离才通过。`-Unlimited` 会在 world 文件旁创建临时 SDF 副本并只改其中的 RTF 设置，结束后移除副本；原 world 和相对 assets URI 不变。Headless 是 runner 默认模式，只有显式传 `-Gui` / `--gui` 才打开窗口。

## 场景与资产映射

Windows 入口用 `.venv-field` 生成 `scene_payload` 与匹配的 `state_payload`，再由 `simulation.field_world` 写入每次运行独立的 `results/gazebo_field_runs/<run-id>/world/`。可复用的输出包括 `world.sdf`、`manifest.json`、`world.topics.json` 和 `assets/`：

- 513×513 灰度 PNG 高度图从现有四个正弦地形分量采样，像素到高度的映射和量化误差记录在 manifest。Probe 08 的局部地面反投影检查中，114,175 个有效深度点有 110,274 个（96.58%）落在原四波地面的 ±8 cm 内，残差中位数为 6.38 mm；这验证了该次相机 Z 深度、内参、云台和地面高度的局部对齐，不代表全场误差已标定。碰撞高度图的XY覆盖范围每侧比名义边界缩进约9.75 cm；Ogre2还会裁掉最后一行和一列，边缘覆盖仍受此限制。
- 健康叶片、枯叶和茎分别合并为三个静态 OBJ 视觉网格，减少数千个叶片实体的管理成本。原来的 plant/leaf ID 及叶片状态映射保存在 `manifest.json`。叶片和茎没有碰撞体，不支持弯折、压坏或单叶刚体交互；后续奖励/检测适配器可使用这些 ID，但不能把通过网格本身视作已计算植物损伤。
- 岩石使用可复用的 OBJ 网格，并按场景位置和尺寸放置独立碰撞体；阻挡标签保存在 manifest。表面接触仍取决于 Gazebo 几何、DART/Bullet 碰撞设置及实机参数标定。
- 底盘总质量默认 30 kg，重心高度 0.15 m 是未测量的代理假设。质量分配、车体/履带和云台惯量写在 manifest；履带接触和摩擦采用 Gazebo 物理接触模型，尚未用实车力矩、滑移或地面试验标定。不得把该参数视为实车质心或摩擦系数测量值。
- 场景 XY 原点平移到农田中心，单位仍为米，Z 保留。manifest 记录 scene ID、原始 JSON 哈希、资产统计和坐标约定；场景与状态 ID 不配对时 exporter 会拒绝导出。

机器人通过 Harmonic `TrackedVehicle` 和左右 `TrackController` 插件驱动；`/field/cmd_vel` 接收 Gazebo `Twist`，云台 yaw/pitch 使用 `Double` 位置目标。runner 以不超过 0.3 m/s 的前进速度和小幅转向/云台波形做传感器烟测，不直接改车体位姿，也不伪造 IMU 姿态。

## 传感器、观测契约与边界

导出的深度相机是 88×160、`R_FLOAT32`、相机 Z 深度，配置为 0.3–5 m、2.5 Hz；GPU LiDAR 是 400 个水平射线、0.12–8 m、10 Hz。IMU 与磁力计也从 Gazebo 消息读取。runner 将深度帧写入 `depth.npy` 和伪彩预览，LiDAR 原始 ranges 写入 `lidar.npz`，世界位姿从 SceneBroadcaster 动态 pose 写入 `world_trajectory.csv`；TrackedVehicle 的 odometry 单独写入 `odom_trajectory.csv`，因为它可能由命令积分，不能单独证明实体实际移动。`joint_states.csv`、`imu.csv` 和可选 `uwb.csv` 保存可解析样本。UWB enabled 时，验收还要求至少 3 个不同锚点 pair 的有限 range 消息。

当前 `field_sim` 的策略 observation 形状是 LiDAR `(2, 400)`、深度 `(2, 160, 88)`（归一化值和有效掩码）、检测 `(max_detections, 10)` 与状态 `(44,)`；action 仍是 4 维。Gazebo smoke 只收集原始传感器与运动数据，不生成检测，不组装这四项 observation，不计算 reward、终止条件、损伤或时间对齐后的新帧掩码。将来接训练器时应保留相同形状、归一化、有效掩码、帧时间/新鲜度字段和四维 action 语义；奖惩与植物损伤评估可作为独立任务层后接。

## 结果与性能解释

默认每次运行写入独立的 `results/gazebo_field_runs/<run-id>/telemetry/`（Linux runner 默认使用 `results/field_gazebo/`）。重点查看：

- `acceptance.json`：实际收到的传感器数、测量窗口内的新时间戳 Hz、保存尾窗数量与丢弃旧样本数、量程有效比例、physics iterations、报告及实测 RTF、物理 world 轨迹，以及 lidar/depth 目标频率是否达到。headless/unlimited 的物理步频可能超过渲染和传感器帧产生速率；RTF 不能替代新鲜观测 Hz，更不能作为 Gym policy steps/s。
- `gazebo.log` 与新鲜度通过检查后的 `renderer.log`：诊断 Gazebo 载入、渲染插件和传感器问题。若 `renderer_evidence.status` 是 `stale` 或 `missing`，不要将旧 Ogre 日志当作本次显卡证据。
- `depth.npy`、`depth_preview.png`、`lidar.npz`、`world_trajectory.csv` 和其他 CSV：用于看实际数据是否持续变化、尺寸是否符合契约、是否存在 NaN/无效量程或机器人未移动。

是否达到预期帧率必须以该次 `acceptance.json` 的 measured sensor Hz 和 fresh frame 数判定；SDF schema 检查、Gazebo 进程启动或单帧出现都不能代替运行性能验证。场景 smoke 通过也不等于 Gazebo 训练器或实车标定已经完成。

## 本机实测：2026-10-17

比较的是当前 `field_sim` 与 Gazebo 8.15.0，场景均为 seed 7、100×100 m、400株/3200片叶、105颗石头，雷达400束@10 Hz、深度88×160@2.5 Hz。两套分别运行，预热后静止测量60仿真秒，不计启动、导出、GUI和策略推理；数值库线程限制为1。FieldSim运行在Windows CPU，Gazebo运行在WSL，渲染日志确认D3D12 / NVIDIA GeForce RTX 5070 Ti Laptop GPU。Gazebo自身仍有物理、渲染和消息线程，不能称为整个引擎单线程比较。

| 指标 | 当前轻量环境 | Gazebo场景与传感器 |
| --- | --- | --- |
| 仿真秒 / 墙钟秒 | 12.47–12.72，中位12.69（三次） | 12.96、13.13（两次） |
| Gym环境步 / 墙钟秒 | 249–254，中位253.75 | 未接Gym循环，不报告 |
| 每仿真秒雷达新帧 | 10.00 | 9.98、10.00左右 |
| 每仿真秒深度新帧 | 2.50 | 2.48、2.50 |

原始结果分别在 `results/gazebo_field_runs/field_stationary_isolated.json`、`benchmark_unlimited_02/acceptance.json` 和 `benchmark_unlimited_03/acceptance.json`。20秒先导结果另存 `field_stationary_comparison.json` 和 `benchmark_unlimited_01/acceptance.json`，不混入上述60秒统计。

当前负载下没有证据表明Gazebo明显更慢，也不能把这几项数值当成等功能训练排名。FieldSim的计时包含模拟检测、观测组装和reward，Gazebo计时包含刚体接触、真实渲染传感器及Transport采集，却尚无这些任务层。下一次训练吞吐比较应在Gazebo接好相同的四维动作、四项观测、任务奖励和传感器延迟后测闭环；250 Hz物理步频与每秒Gym步数是不同指标。

从代码看，FieldSim主要在 `field_sim/env.py`、`terrain.py` 中用CPU NumPy解析射线和准静态车身支撑，省去刚体求解，但地面射线、遮挡与检测仍消耗CPU。Gazebo将叶片合为三个视觉网格，GPU生成雷达/深度，DART/Bullet处理履带与地形接触。因此本机GPU能够抵消一部分物理与消息成本；换到无独立显卡的机器或同时运行真实视觉模型，结果可能变化。

额外运行验证：`probe_08` 在实时限速下行驶约2.43 m，IMU、磁力计、实际云台角和原始传感器持续输出。对20张实际深度帧反投影，114175个有效点中110274个（96.58%）与原始四波地面高度相差不超过8 cm，残差中位数6.38 mm。该检查验证了当前局部视野下的相机Z深度、内参、云台姿态和地形方向，未遍历全场边缘。复核工具为 `simulation/validate_field_capture.py`，结果在 `probe_08/ground_alignment.json`。

迁移时适配了两端地形偏移：碰撞使用collision pose，Ogre2地形使用heightmap position；灰度图采用实际采样极值映射0–255。对应实现依据可查 [Gazebo Physics高度图加载](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/physics/Physics.cc) 和 [Ogre2高度图实现](https://github.com/gazebosim/gz-rendering/blob/gz-rendering8/ogre2/src/Ogre2Heightmap.cc)。早期`probe_06`车辆自由落体，已明确作废；`probe_07`深度地面对齐未通过，均不用于性能结论。

GUI加载检查在 `gui_uwb_smoke/gazebo.log` 中记录了窗口及场景插件正常启动。该旧次运行的UWB未通过，另行修正后，9站网格的 `uwb_10_acceptance/acceptance.json` 已通过：5个不同基站链路共247条有限测距，原始厘米与换算米均保存在 `uwb.csv`。部分远站因75 m硬距离限制不可见，未声称9站同时可用。上游Double未填写采集时间，记录使用回调时最近一次WorldStatistics仿真时间估计，不能作为实机时间同步证据。

查看9站场景可运行 `scripts/field_gazebo.ps1 -Gui -NoMotion -WithUwb -Seconds 3600`；布局为作业区10%、50%、90%位置组成的3×3网格，基站高出当地地面1.8 m，均为待替换的代理布设。
