# Isaac Lab 迁移状态
当前策略观测合同仍为 sensor-minimal-v6；其196,608步训练、开发配对和GUI证据均是历史，训练使用未修复的P-only drive v4。此后源码先修复Newton显式轮端限幅为v5，再加入当前PI候选 drive v6。P/PI隔离确认已在4个预定case通过：每组9,200个物理子步的实测最大轮端effort为2 Nm，初态在协议容差内，两种制动末0.3秒track命令严格为零且v/w门槛通过；已知PI制动probe仍有反向回摆（约0.0729 m反向速度积分、最低约−0.130 m/s），这只是仿真诊断。生产回归现已完成：CPU244 passed/13 skipped、exit0，静态检查exit0，累计13项native检查全部通过/exit0，证据见 `results/isaac_field_diagnosis/drive_pi_v6_tests_20261020/`。独立fresh_side_pi_20261020已完成196,608步训练与配对开发评估，结果见下方trial记录。当前viewer PID198426/exec1490已加载final checkpoint，driver/observation/action合同匹配；重置后0秒基线序列2，随后运行至仿真10.80秒、输出10.85秒、序列219后暂停。60层数值均有限，50/60节点raw张量变化；8项valid和8项age均0项变化，60张卡片无重叠、viewBox一致，console/backend错误为空。这仅验证final模型部署、UI同步和数值有限，不是策略能力评估。验收见 [browser_acceptance.json](results/isaac_field_viewer/side_pi_20261020/runtime_final/browser_acceptance.json)、[compare_before_after.json](results/isaac_field_viewer/side_pi_20261020/runtime_final/compare_before_after.json)、[网络图](results/isaac_field_viewer/side_pi_20261020/runtime_final/network_final.png)及[车辆图](results/isaac_field_viewer/side_pi_20261020/runtime_final/vehicle_final.png)。正式0步baseline仍是随机初始化、未训练模型。旧协议、模型与评分保留原样，详见下文。
历史GUI验收（PI接入前，2026-10-20）：sensor-minimal-v6 final checkpoint 曾在车辆8769/二维网络8770载入。真实GUI载入196608步模型运行0→8.4秒后暂停，60层均有限、50层raw张量变化、2项age变化、viewBox稳定且dev.logs为空；证据见 [final运行记录](results/isaac_field_viewer/minimal_inputs_20261020/runtime_final/browser_acceptance.json)、[raw比较](results/isaac_field_viewer/minimal_inputs_20261020/runtime_final/raw_comparison.json)、[二维图](results/isaac_field_viewer/minimal_inputs_20261020/runtime_final/network_final.png)及[车辆图](results/isaac_field_viewer/minimal_inputs_20261020/runtime_final/vehicle_final.png)。训练前随机0步预览是独立历史预览。CPU全套231 passed/11 skipped、加强专项171 passed/11 skipped；原生Newton 6 passed及磁场异步valid探针通过；真实TorchScript五路输入3×3比较九项最大误差均0。

独立新初始化trial `results/isaac_field_improvement/fresh_minimal_inputs_20261020/` 已完成196608步与配对开发评估。runner PID155311于2026-10-20 07:12:30北京时间真实exit0；launcher exec86650真实返回exit0。protocol SHA-256 `b5e8e9c3b0561bc73733528aa0c5be3120bb5aad3a54a1fb3418ad84f30a6d65`，final `stage_04_turn90.zip` SHA-256 `2c03cc130977cc005adf56527bc10bc4719a8a9d9489fa82c5a1b41f9d3966e5`，正式0步baseline SHA-256 `443b9fc73676bc5dfe6d42450bf2112089852d27c373d12c4b2443f484e43ca9`。训练结束时40个冻结源与当时源码哈希全部匹配，保留29个checkpoint；完成后修复了显式执行器实际力矩限幅，当前源码仅38/40匹配，差异为env.py与fresh训练入口，原40份快照、协议和ZIP未变，详见[完成后限幅变更记录](results/isaac_field_improvement/fresh_minimal_inputs_20261020/post_completion_effort_clamp_change.json)；CPU模型校验通过，Adam 37组状态全有限、step=7531，其余训练分支有限且更新；检测框MLP有限但未更新（训练协议plants=0）。free seeds 87001–87004成功0/4→0/4、平均回报−0.339→−2.363、末距1.680→2.321m；field seeds 82001–82004成功0/4→0/4、平均回报−0.320→−2.037、末距1.837→2.448m；两组损伤和碰撞均0，均为`no_improvement`。这是开发配对结果，不是能力提升结论。验收见[completion validation](results/isaac_field_improvement/fresh_minimal_inputs_20261020/completion_validation.json)与[CPU模型校验](results/isaac_field_improvement/fresh_minimal_inputs_20261020/completion_model_validation.json)。旧v5权重、协议和成绩保留且不可混算；830xx未评估或观察，本轮不做FPS测量。

当前PI训练trial `results/isaac_field_improvement/fresh_side_pi_20261020/` 从新随机初始化启动，未续用或迁移旧权重。协议SHA-256为 `0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8`，41个冻结源码及快照均与协议匹配；正式0步baseline SHA-256为 `bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129`。训练完成196,608步，保留29个checkpoint；最终 `stage_04_turn90.zip` SHA-256为 `f5be877d728037436bcfcca9b65df462de41dd1846f760473374f287f5a40735`。真实runner PID196317的process_exit记录exit0（2026-10-20 08:43:35北京时间），持有的launcher exec/session 62137也实际返回exit0。4个free开发种子的配对结果为成功0/4→0/4、平均末距1.692859→1.412609 m；4个field开发种子为0/4→0/4、平均末距1.840281→1.520151 m。两组mean damage和collision均为0，comparison_summary均判 `no_improvement`；距离变化不能称为导航、田间或安全能力提升。Ki=0.1仍是未标定仿真假设，隔离制动probe已见反向回摆；plants=0训练不学避苗。830xx未评估或观察。权威完成校验18/18通过，见 [completion_validation_pi_v6_verified.json](results/isaac_field_improvement/fresh_side_pi_20261020/completion_validation_pi_v6_verified.json) 和[验证日志](results/isaac_field_viewer/side_pi_20261020/validation/completion_validation_pi_v6_verified.log)；final模型CPU检查见 [final_196608.json](results/isaac_field_viewer/side_pi_20261020/validation/final_196608.json)。

训练流程：从正式随机初始化的0步baseline开始；4个环境每个环境采样128步，形成512个transition的rollout，再由RecurrentPPO以batch 128、最多5个epoch更新。预定四阶段按固定步数逐步扩大目标转向角，每8,192步保存checkpoint，不按成功率自动升课。每个episode/reset都会清除LSTM状态；阶段交界重建环境、RNG与rollout，但承接策略权重和Adam状态。已按同初态完成baseline与final的free和field开发配对。plants=0训练不包含植物，不能学习避苗；阶段窗口中的成功率只是在线过程数据，不是正式成绩。

本项目当前的仿真、训练和交互查看入口为 Isaac Lab + Newton，完整场景已实际启动并可交互查看。2026-10-17 按用户要求将旧后端、旧模型和实验结果移入 `归档/legacy_before_isaac_20261017/`，保留原相对目录和文件内容；Isaac 原始实验结果仍在 `results/`。归档说明见该目录的 `README.md`，原路径和 SHA-256 见 `archive_manifest.json`。场景可运行不等于物理参数已经完成实车标定，也不等于策略具备完整作业能力。

## 当前实现

- Isaac Lab checkout 固定在提交 `5870804e4f6aee57bdef68d59f67d328b990d1e1`，运行环境为 Ubuntu 24.04 WSL、Python 3.12。安装脚本使用 Isaac Lab 锁定的依赖；额外安装 `sb3-contrib==2.9.0` 时不做依赖解析，避免覆盖 Isaac Lab 固定的 PyTorch/SB3 版本。
- `isaac_field/assets.py` 生成 30 kg 代理底盘 USD，包含左右各 7 个、合计 14 个驱动滚轮关节。七对代理滚轮的首末中心跨度按实车接地长度设置为 595 mm；每侧滚轮代理宽度为 60 mm。它是滚轮代理结构，不是连续履带接触模型。
- 已确认的底盘传动参数为履带接地长度 595 mm、主动驱动链轮直径 60 mm、减速比 4:1。USD 根 prim 写入接地长度、链轮半径/直径、减速比和资产规格 SHA-256。实际 30 mm 链轮半径与 45 mm 支持滚轮代理半径是不同尺寸；支持滚轮保持 45 mm，轮轴高度保持原代理值。
- 车身宽度 300 mm 不含履带。用户确认整车总外宽 470 mm、每侧履带宽 60 mm，因此履带中心距按 `0.47 - 0.06 = 0.41 m` 派生，整车安全外廓宽度为 470 mm。该代理几何用于资产与安全边界/出生检查；尺寸更新不代表仿真已完成动力学标定。
- 默认地图是 100 m 方场，约 400 株作物以局部行列布局生成；行距 0.8 m、株距 0.3 m。布局不是均匀铺满整个场地。地形在一次运行中固定并由各仿真实例共享；每次 reset 会随机生成植物布局、出生位置与朝向、以及任务目标。目标作物类别按运行配置固定。
- 连通性快速检查使用 0.1 m 栅格和 0/90 度两个允许朝向，保守检查车身 footprint 和可达性；它只用于场景有效性检查，不向策略提供地图，也不执行路径规划。完整离线校验另外检查干叶目标的视线样本。
- 当前源码观测输入为 `sensor-minimal-v6`：LiDAR `[B,1,400]`、深度 `[B,1,160,88]`、检测 `[B,16,8]`、状态 `[B,33]`、反馈 `[B,37]`。LiDAR策略通道只含按8 m量程归一化的距离；最终valid为假的射线置0，400束与有效距离不降采样，frame内valid仍用于点云和诊断。深度只输入相机Z/量程、无效值0。检测保留cx/cy/w/h、class、confidence、range和range_valid，删除presence与保留常数列；有效框由w>0且h>0识别。state[0]为mode数值（0=search、1=goto），目标局部XY位于1:3、估计绝对XY位于3:5、heading sin/cos位于5:7、上次四维动作位于7:11、时间比例位于11；LiDAR/深度/相机的时序与位姿补偿依次占12:19、19:26、26:33。state不含真值车体线速度、偏航角速度或真值横滚/俯仰；heading、目标相对量和运动补偿共用云台IMU姿态加实测云台q反算出的车体姿态估计，UWB模拟位置还含每个采样时刻独立白噪声。重复读取同一时刻观测不会推进滤波器或随机数。渲染用物理真值位姿与策略估计位姿分离；估计位姿配置随观测元数据记录。PI-v6当前源码与验证状态见文首更新。
- 反馈向量 `gimbal-imu-drive-feedback-v2` 共37维：前21维按 `accel, gyro, mag, attitude, gimbal_yaw, gimbal_pitch, motor_left, motor_right` 排列，具体为云台IMU加速度3、角速度3、磁场原始有符号计数3、四元数xyzw 4、云台偏航q/qd 2、云台俯仰q/qd 2、左侧Iq/RPM 2、右侧Iq/RPM 2；接着是8组valid，再接8组age。Euler与温度不再进入策略向量；磁场保留，raw温度样本仍供诊断。仿真姿态来自VQF。单位尺度在 `feedback_config()` 中声明，不裁剪有效值：加速度除9.80665、角速度除π、磁场counts除32768、四元数不缩放、云台q/qd除π、Iq除100 A、机械RPM除10000、age除2 s；valid为0/1。加速度为传感器proper acceleration（含重力），不是线性加速度；磁力计没有µT标定证据。任一必需字段缺失或超时会将对应组数值置零并清valid；age有界，保持新鲜度信息。左右电机各自为一组，任一Iq或RPM无效则该侧整组无效。气压/海拔帧ID5丢弃。
- v5到v6的总输入元素数由15,120降为14,678，少442（2.92%）；深度保持14,080个元素不变，非深度部分由1,040降为598（减少42.5%）。该计数只描述输入元素，不代表参数量、FLOPs或实测速度的同比变化。当前动作schema仍为 `direct-track-speed-v1`：四维连续输出依次是左履带目标速度、右履带目标速度、云台偏航角速度指令、云台俯仰角速度指令，范围均为 `[-1,1]`；左右速度独立控制，每侧目标范围约 `[-4.1667, 4.1667] m/s`，速度变化率上限1.5 m/s²。
- v5时期（PI接入前）的驱动合同与限幅修复记录为 `independent_left_right_track_speed_explicit_v5`。每个履带轮的 `actuator_effort_limit_nm=2` 通过Newton `NewtonMaxEffortClampingAPI` 对合并后的P+D+feedforward力矩限幅；保留 `joint_effort_limit_nm=2`。当前反馈Iq由限幅后的Newton `applied_effort` 按每侧7个代理轮关节、`Kt=0.035 Nm/A`及效率1假设换算，RPM按实际代理轮角速度、主动链轮30 mm半径、45 mm代理轮半径和4:1齿比换算；仍是模型等效值，不是实测电流、实测电机RPM或接触力。云台使用隐式Newton/MJWarp驱动，`joint_effort_limit=1 Nm`通过solver的`actfrcrange`限幅；`actuator_effort_limit`对隐式驱动只用于模型侧telemetry，不作为此solver限幅。速度伺服增益0.04、履带每轮2 Nm及云台1 Nm均为数值参数，不是实车标定；代理仍无连续履带接触。2026-10-20 native smoke前，v4四种同号/反号FF模式共80个活动物理子步最高实测 `applied_effort=4.000027 Nm`，确认v4的2 Nm关节字段没有约束Newton explicit FF；v5修复后相同80个活动子步全部不超过2 Nm，运行时Newton `ClampingMaxEffort.max_effort` 的14个轮关节值均为2.0，两个云台joint solver limit均为1.0。两次均为隔离物理诊断，不是训练或FPS测试，原始数据在 `results/isaac_field_diagnosis/minimal_inputs_20261020/effort_limit_smoke/` 和 `effort_limit_smoke_v5/`。旧v4 trial与其成绩属于修复前，不能据v4 metadata推断feedforward受到2 Nm限幅，也不与v5物理合同混算。当时8769/8770进程仍是修复前加载的v4环境；旧GUI进程PID166936后来停止。PI-v6已另用near 8,192步checkpoint部署至新viewer PID198426/exec1490，当前旧trial历史不因此改写。
- 当前源码候选驱动合同为 independent_left_right_track_speed_explicit_pi_v6：保留Newton原生每轮P=0.04与每轮2 Nm实际限幅，增加默认Ki=0.1的左右两状态积分前馈；每policy step按实际 self.step_dt 更新每侧7个轮速目标与实测轮速的均值误差，并把该侧积分值广播到7个代理轮。Ki=0允许关闭积分作P-only对照；积分限幅为每代理轮±2 Nm，条件抗饱和使用侧均值P估计，Newton原生限幅仍是合并P+FF后的最终限制。该Ki、代理轮聚合和仿真电机参数均未标定，不是实车控制器参数。
- 实车反馈接入现有硬件数据路径和仿真模型仅部分对应：用户确认10轴 ATK-IMU901随云台偏航、俯仰转动，且有云台两轴实测角度/速率和左右驱动器电流/机械转速回读。仿真采用 `SIM_SENSOR_MODEL` 中记录的噪声与偏置假设、每回合独立随机数，并以20 Hz调用VQF 2.1.2；这不等同于厂家固件融合频率或实车噪声标定。`isaac_field/telemetry.py` 提供协议解码/组装接口，不打开串口/CAN设备；项目硬件实现位于 `电力/FOC驱动模块/软件/速度反馈接口.c`、`.h`、`速度反馈接口测试.c`、`运行主机测试.ps1`，协议和黄金向量同目录。C主机测试是96项host checks、0 fail，不是实机联机或硬件台架测试；Python telemetry/feedback专项分别为10项和8项通过。PDF镜像文件名为 `docs/hardware/atk_imu901/ATK-IMU901_User_Guide_v1.2.pdf`，但手册封面标为V1.0；它是下载镜像，未验证与实物硬件版本或最新V1.2固件一致。磁场物理单位、各通道噪声、安装外参及电流标定仍需实机核实。
- 损伤评分默认使用 `stem-contact-v2`：仅履带接触茎基或车身接触主茎时计损伤，叶片擦碰及旧 `Scene.radius` 根保护圈不计损伤。履带按 595×60 mm 接地带作离散扫掠，并检查与茎基的垂直重叠；当前履带中心距为410 mm、整车外宽470 mm；车身按当前 root 高度及俯仰/横滚扩展的垂直包围范围检查主茎。车身接触仍是简化 2.5D、离散扫掠近似，不是完整三维接触求解。`root-protection-v1` 保留用于旧结果复现；旧 checkpoint 配置未记录规则时回落到 v1。episode、checkpoint 与评测元数据记录规则、惩罚和几何配置。
- 新权重记录动作、驱动、观测、编码器与损伤合同；v6 checkpoint须匹配 `sensor-minimal-v6`、`depth64-lidar-single1d-minimal-v3` 及 `direct-track-speed-v1`，缺字段或配置不符会在SB3加载前拒绝。sensor-v5及更早权重与sensor-minimal-v6 shape/encoder不兼容，必须重新训练；不得通过补metadata让旧权重冒充新合同。旧的sensor-minimal-v6/P-only drive-v4 checkpoint仍可在当前PI-v6物理环境中重评，PolicyRunner保留drive mismatch提示而不以drive差异拒绝加载；报告须并列训练drive与实际env drive，不能改写checkpoint metadata或挪用旧分数。旧权重、训练成绩和协议仍保留为历史数据。PI-v6当前源码与验证状态见文首更新。深度 `Box` 仍声明为 `[-1,1]` 以便Isaac Lab官方 `Sb3VecEnvWrapper` 按已归一化浮点图原样传递；实际深度数组范围仍为 `[0,1]`。
- 当前源码策略编码器 schema 为 `depth64-lidar-single1d-minimal-v3`。五路支路输出仍为 LiDAR32、深度64、检测32、状态32、反馈32，拼接为192维；RecurrentPPO LSTM仍为128维。LiDAR环形 `Conv1d` 改为1→16→32（核5、步长2、填充2），深度 `Conv2d`（1→8→16）及 `[8,6]` 自适应池化保持不变；检测候选框由8字段汇聚、presence由宽高推导。网络结构仍是预设结构，后续训练只学习参数，不做自动架构搜索。
- 深度观测的实际数组仍由 `np.where(valid, Z, 0) / depth_range` 生成，范围为 `[0, 1]`。环境 `Box` 声明为 `[-1, 1]`，是为了让 Isaac Lab 官方 `Sb3VecEnvWrapper` 将浮点深度图按已归一化图像处理并精确保留数值；这不是改变传感器数据范围。精确值专项通过 `1 passed, 37 warnings`。
- 旧驱动的九个逐步诊断、参数扫描和物理探针已原样归档到 `results/project_history/drive_v1_scripts/`；本批两个高增益 v2 诊断脚本也原样归档到 `results/project_history/drive_tuning_20261017/`，其结果 JSON 保留在 `results/isaac_field_diagnosis/drive_v2/`。这些脚本记录历史实验，依赖原目录或固定输出路径，不应作为当前生产入口重跑。原生运动与制动回归入口为 `tests/test_env_integration.py`（需显式设置 `RUN_ISAAC_PHYSICS=1`）；`scripts/diagnose_isaac_ground_contact.py` 保留作可选 A/B/C/D 接触诊断，不测训练吞吐量或决策帧率。
- 本批固定参数植物姿态/传感器诊断脚本归档至 `results/project_history/plant_diagnostics_20261017/`；它们没有生产代码引用，原始 JSON 仍留在 `results/isaac_field_diagnosis/plant_pose_consistency/`，近株射线的独立原生回归保留在 `tests/test_env_integration.py`。
- v5历史浏览器验收：完整展开60个实际层、71条实际边，无整体视图模式；缩窄左导航抽屉、连续曲线、选中层及其真实直接父/子边高亮。宽屏1280 px视口顶栏约130 px；窄屏745×871 px顶栏111 px、画布高760 px；抽屉可展开，点击动作区后自动收起，无页面横向溢出。该记录只对应旧v5模型和源码。另有sensor-minimal-v6 final在PI接入前的浏览器验收，见文首证据；它不代表当前PI-v6源码部署。历史运行时间/raw张量及hardware-input边界见 [v5 UI验收记录](results/isaac_field_viewer/imu_drive_20261020/README.md)。车辆三维页8769和独立二维页8770截图均为历史证据；PI-v6当前源码与验证状态见文首更新。
- 历史viewer/reset验收（2026-10-17）：旧exec94767属于250 mm中心距；exec18556曾在410 mm/470 mm几何加载旧73,728步权重，随后报Warp CUDA999并退出，浏览器已断开。旧viewer session30380句柄失效；恢复child旁启动的viewer PID11788曾因原生GUI reset race（`_last_results=None`）退出。修复版中文viewer PID17740曾由Edge browser3/tab1967487223验收：截至2026-10-17 23:34:59，训练PID8424和viewer PID17740均存活；98304步GUI模型下切策略/Run产生真实运动，native Reset显示“官方仿真已重置；已刷新当前状态”，viewer未退出，drive/observation/damage均匹配。21项专项viewer reset测试通过。一次73728步、seed82001的20秒GUI试跑超时，最终距离1.00 m、0 damage、0 collision；这是开发观察，不是正式评估或成功证据。前端可切换搜索观察模式，但foundation训练协议只有goto，不能据此声称模型学会搜索。浏览器截图与验收记录见`results/isaac_field_viewer/chinese_reset_fixed_49152_20261017/chinese_preview_98304.png`和`browser_verification.json`。GPU健康检查三项通过、wrapper/probe exit0，但没有运行Isaac仿真，不能据此认定CUDA999根因已解决。历史初态与截图留在`results/isaac_field_viewer/width470_seed82001_parent73728/`；交互说明见[使用说明.md](使用说明.md)。
- `adversary` 已完成一次真实 NaiveMinimax 自检：教师16步、学生256步，两个策略权重均更新，教师布局经真实物理rollout评分；不代表策略能力已提升，也不是PAIRED。证据：`results/isaac_field/adversarial_smoke/round_summary.json`。

## 已验证结果

2026-10-18此前模型解剖改为独立二维页面：车辆三维8769、模型二维8770，实时显示真实逐层激活、独立价值诊断、LSTM状态/经核验的门值重建、全量张量和权重。网络三维展示按用户要求删除，伪彩改用Matplotlib标准Turbo全光谱并显示同源色条。实际浏览器验证二维单独运行、重置清帧及前后两帧激活变化/视野不变；CPU151 passed/8 skipped，compileall与HTML Node语法检查通过。操作和证据见 [本轮说明](results/isaac_field_viewer/network_2d_20261018/README.md) 与 [使用说明](使用说明.md)。Netron TorchScript与原策略3组×3步数值一致的旧记录保留；被拒的Netron后台启动未重试。该次仅UI改动时，训练、策略和物理源码未改变；后续输入简化改变了策略输入，详见下条新记录。

2026-10-18历史输入简化记录采用 `sensor-depth-only-v3`：单通道深度、43维状态、无额外目标增益，旧训练权重与实验记录保留。当时8769/8770预览使用0步随机初始化模型，该段也记录了当时尚未启动新训练；这不代表后续状态。CPU155 passed/8 skipped，静态与专项检查和当时的新结构验收见 [输入简化记录](results/isaac_field_viewer/input_simplification_20261018/README.md)。历史foundation/GAE/探索对照脚本的100倍增益及baseline引用仅供旧源码快照复现，不可作为当前direct-track训练入口。

历史动作合同提示：2026-10-19及更早的viewer验收、拓扑记录、checkpoint和训练对照沿用前进/差速动作。它们只保留为历史证据；当前训练、评估和Viewer会拒绝缺少或不匹配 `direct-track-speed-v1` 的旧权重。

实车方面，用户确认履带车可在硬地和农田土面原地向左、向右转向。这是现场实车能力信息。旧纯差速原生探针使用增益20，轮速目标或实际关节转速不能替代底盘偏航测量。2026-10-17将代理更新为470 mm总外宽/410 mm履带中心距后，8项原生集成测试于exec69112真实exit0全部通过；默认gain0.5、平地无作物、前进命令0、差速命令±0.2各运行2 s，车身展开yaw分别为−0.42649和+0.40545 rad，最大中心漂移分别为0.0832和0.1052 m。8株场景还报告body_count=25、joint_dof_count=22、joint_coord_count=23、shape_count=89；植物未增加关节自由度。原始日志为 `results/isaac_field_diagnosis/stem_contact_v2_20261017/native_width470_integration.log`。这验证更新几何下的代理接触/传感器回归，不代表土面实车角速度标定、导航能力或性能提升。

旧 v2 功能探针 `results/isaac_field_diagnosis/drive_v2/drive_v2_functional_probe.json` 使用伺服增益 20，源码现归档于 `results/project_history/drive_tuning_20261017/diagnose_isaac_drive_v2.py`：目标速度、加速度约束和左右转向符号通过，满差速 3 s 的实际偏航约 +0.86/−0.77 rad；其中异常混合响应和制动漂移是高增益历史结果，不代表当前默认配置。配套状态/接触数据 `root_state_consistency_probe.json` 及原始两份结果均未改写。

2026-10-17 的 A/B/C/D 原生接触对照及尾段姿态统计记录于 `results/isaac_field_diagnosis/ground_contact/ground_contact_tail_pose_summary_20261017.json`。增益 20 的 A/B/D 对照中，车体速度峰值约 3.14–3.4 m/s，而代理轮面速度约 0.5 m/s，停车后出现约 1.3–1.7 m 的反向净漂移；D 在原生 MJ contacts 设置下仍复现。C 使用原三角网格地面与增益 0.5，前进 2 s 的速度峰值约 0.621 m/s（轮面约 0.521 m/s），停车阶段 2 s 位移约 0.070 m、最终速度约 7.5×10⁻⁶ m/s；混合阶段停车后位移约 0.072 m、最终速度约 6.8×10⁻⁶ m/s。该结果支持 0.5 作为当前数值稳定化控制参数；它没有证明 Newton 底层存在特定求解器缺陷，也没有完成实车驱动标定或闭环制动验收。C 原始记录为 `results/isaac_field_diagnosis/ground_contact/ground_contact_ab_20261017_191201.json`。该段对应的浏览器会话是旧驱动和 smoke 权重历史记录；exec18556旧checkpoint预览现已退出，不作为当前可用预览。历史验证快照曾有 38 项纯测试及 4 项原生物理集成测试通过；这两个数字仅描述当时的覆盖范围，不代表当前测试快照。

截至 2026-10-17，Isaac Lab runtime 的官方 Cartpole GPU probe 已通过：4 个环境、32 步，设备 `cuda:0`，PyTorch `2.11.0+cu130`，记录在 `results/isaaclab_migration/runtime_probe.json`。这只验证 runtime 与 GPU stepping，不代表本项目机器人场景的结果。

旧44维/双通道契约时，本项目 full-field Isaac smoke 已执行 1 个环境、32 步，Newton 后端报告 14 个滚轮关节；LiDAR、深度、检测和状态观测分别为 `[1,2,400]`、`[1,2,160,88]`、`[1,16,10]`、`[1,44]`。此 smoke 证明短程 step、传感器观测和接口可运行，不是性能测试，也未证明任务成功。

已保存的训练 smoke 共训练 512 个时间步，产生 8 个 episode；8 个均以 time limit 结束，成功数为 0。它只证明训练流程能执行，不证明策略已经学会任务。当前没有本轮速度/吞吐量测试，也不据此给出性能结论。详细原始数据保留在 `results/isaac_field/`。

2026-10-17 完成一次短程策略前后对照，入口为 `scripts/improve_isaac_policy.py`。从 `results/isaac_field_migration/field_policy_16384_latest.zip` 继续训练 4096 步，使用 2 个并行环境、16 株作物、行距 0.8 m、株距 0.3 m、平地、goto 目标距离 0.6–1.2 m、角度 ±30°、单 episode 上限 8 s；训练距离代价为 1， held-out 评测距离代价为 0，使用相同的 4 个 seeds（51001–51004）。原始逐回合数据和新权重保留在 `results/isaac_field_improvement/short_goto_20261017_175932/`。

前后成功数都是 2/4（50%），平均 raw return 为 2.4398 → 2.4441，平均最终距离为 0.8826 m → 0.8765 m，平均损伤和碰撞均为 0；两个阶段各有 2/4 episode 到达 8 s time limit。一个此前失败的 seed 变为成功，同时一个此前成功的 seed 变为失败；总体没有可辨识的能力提升。四个 seed 是小样本接线与行为检查，不足以证明统计显著性、泛化能力或完整场地能力。训练完成和保存权重也不代表策略已学会任务。

2026-10-17 又完成 32,768 步 staged-goto 训练，使用训练种子 46001/46002、16 株平地场景、1–3 m 目标距离和 ±60° 目标角度；held-out 使用固定种子 76001–76012。CPU 核验的最终权重为 `results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip`，记录 53,248 总训练步。held-out 到达数为 0/12 → 1/12，平均最终距离为 1.89916 m → 1.42249 m，平均损伤保持 0.5。唯一到达目标的 seed 76008 造成 4 次损伤且无碰撞，因此零损伤、无碰撞的安全到达仍为 0/12 → 0/12。该结果只显示目标到达计数小幅增加，不能证明可靠导航、避障能力或完整作业能力。协议、前后逐回合数据和评估 JSON 均保留在该结果目录。

策略改进脚本新增显式 `safe-goto` reward profile；它不改变 legacy 默认值。该 profile 使用 20 s episode、`damage_penalty=20`、`goal_progress_budget=5`、训练与 held-out 评估的 `distance_cost=0`，16 株平地，目标距离 1–3 m、角度 ±60°。普通 `staged-goto` 仍使用 legacy：`damage_penalty=4`、`goal_progress_budget=None`，训练 `distance_cost=1`、held-out `distance_cost=0`。safe-goto parent 续训因 reset 崩溃以 exit 1 结束，最后保存点为总计 110,592 步的 `results/isaac_field_improvement/safe_goto_20261017_reward_v1/stage_02_medium/checkpoints/field_policy_medium_110592_steps.zip`；该 parent 没有 after 评估。之后的 resume1 已以 exit 0 完成，最终权重为 `results/isaac_field_improvement/safe_goto_20261017_reward_v1_resume1/field_policy_after_resume1.zip`（118,784 步）。严格配对的 24 个地图、起点和有效目标一致：53,248 步权重为 5 次成功、5 次无损成功、4 个受损回合、平均损伤 0.25、0 碰撞；118,784 步权重为 9 次成功、8 次无损成功、5 个受损回合、平均损伤 0.3333、0 碰撞。评估标记为 `success_safety_tradeoff`；按优先不压苗的标准不通过，也不能据此宣称可靠导航。完整报告保留在 `results/isaac_field_improvement/safe_goto_20261017_reward_v1_resume1/`。

对 53,248 步权重用开发回归 seeds 76001–76004 做了 4 个 20 s goto 实际轨迹评测，详细原始数据和派生统计分别在 `results/isaac_field_diagnosis/policy_53248_trajectory/evaluation.json` 与 `analysis.json`。四例都超时，0 成功、0 损伤、0 碰撞；最小目标距离为 0.744–2.361 m，均未进入 0.4 m 到达范围，hold 始终为 0，因此这四例不是“进目标区后没停住/hold 不足”。每例差速动作都与初始负 bearing 同号，但名义无滑移偏航需求均值约 −0.92 至 −1.10 rad/s，实测 root 偏航角速度均值只有 −0.0057 至 −0.0121 rad/s，累计 yaw 与实测角速度积分相符。这是四个开发种子的场景内响应证据，提示实际转向响应弱；不能单独证明唯一物理原因，也不能外推总体 held-out 能力。该轨迹评测不替代平地无作物纯差速原生回归，也不属于性能测试。

对恢复前后轨迹的正 bearing 子组检查见 `results/isaac_field_diagnosis/recovered_baseline_trajectory_analysis.json` 和 `recovered_after_trajectory_analysis.json`：旧权重的 12 个正 bearing 样本差速动作均值均为负，新权重同组没有正向差速动作。新权重中两个正 bearing 候选成功不足以证明策略已学会双向转向。之后完成的 gSDE/Gaussian 同源新优化器对照使用独立评测 seeds 82001–82024，不与 810xx 混算；结果显示 Gaussian 为 2 次无损成功、4 个受损回合、平均损伤 0.5833，gSDE 为 1 次无损成功、5 个受损回合、平均损伤 0.6667，baseline 为 1 次无损成功、3 个受损回合、平均损伤 0.3333，三者均为 0 碰撞。Gaussian 判为 `success_safety_tradeoff`，gSDE 相对 baseline/control 判为 `no_improvement`。三个 directionality 报告显示 baseline 整体差速动作偏负、两个变体整体偏正，但都没有学会按目标方向双向条件转向；这组结果不构成整体导航提升或仿真标定完成。完整对照见 `results/isaac_field_improvement/exploration_gsde_20261017/comparison_summary.json` 和同目录 `directionality_baseline.json`、`directionality_gaussian.json`、`directionality_gsde.json`。GAE lambda .95→.99 单变量对照已 exit0、24/24 配对：baseline/Gaussian/GAE .99 的 clean successes 为 1/2/3、受损回合 3/4/5、平均损伤 0.3333/0.5833/0.5417，均 0 碰撞；两项比较均为 success_safety_tradeoff，按不压苗优先不通过。82001–82024 是已观察开发集，不是盲测；详见 results/isaac_field_improvement/gae_credit_20261017/comparison_summary.json。只读审计 results/isaac_field_diagnosis/plant_body_cost_audit_20261017.md 未发现固定植物root增加6个q/qd坐标的证据；root仍占body/mocap与shape/BVH条目，准确成本未测。继续使用轻量代理，不基于成本猜测改为bodyless植物。

补充传感器核查：准确复现查看器的 goto/seed80003 配置后，植物布局、Newton 实时位姿与 SceneDataProvider 在 0/1/2/10 步保持一致；最近株距车辆31.5404m且在后侧，因此零测距命中及裸地预览合理。Newton BVH包含VISIBLE和COLLIDE_SHAPES，视觉茎叶可供射线查询，不需要增加植物刚性碰撞或修改传感器实现。近株原生回归 `test_live_crop_stem_returns_native_range_without_a_leaf_collider` 以seed67031把车放在茎中心前1.5m，单点测距与水平LiDAR第200束均返回1.488780m；该株8片叶均VISIBLE且无COLLIDE_SHAPES。证据见 `results/isaac_field_diagnosis/plant_raycast/native_crop_stem_ray_20261017.json`。平地纯差速回归 `test_native_pure_differential_turn_has_measurable_yaw_without_forward_command` 另行验证了 ±0.2 命令下的实际车身转向。两个回归分别单独通过；此前5项原生测试未在本次重复运行。这些验证不产生新的策略能力结论。

## 安装与运行

Windows 可双击项目根目录的 `安装IsaacLab.cmd` 安装，或双击 `启动IsaacLab.cmd` 启动默认查看器。WSL 中的命令见 [使用说明.md](使用说明.md)。安装脚本会复用固定提交的 `external/IsaacLab` checkout，并将 runtime 环境放在项目外的共享 WSL 路径；不要把该共享环境当作项目清理目标。

测试命令：

```bash
bash scripts/isaaclab_python.sh python -m pytest -q \
  "$PWD/tests"
```

不设置 `RUN_ISAAC_PHYSICS` 时，`tests/test_env_integration.py` 的原生物理案例会跳过。单独运行原生物理回归：

```bash
RUN_ISAAC_PHYSICS=1 bash scripts/isaaclab_python.sh -m pytest -q tests/test_env_integration.py
```

当前训练状态见上文。

### 历史 v5 IMU反馈合同与新初始化训练（2026-10-20）

该trial使用 `sensor-imu-drive-v5`、39维状态与41维feedback，编码器为 `depth64-lidar-circular1d-feedback-v2`、192维融合，动作是 `direct-track-speed-v1`。这些合同和成绩均为历史记录；当前源码已切换到v6，v5 checkpoint不兼容且不能载入或用新metadata重新标记。此前direct-track trial使用v4/43状态/160融合，同样保留为历史实验。

该v5独立trial当时按冻结源码训练完成：`results/isaac_field_improvement/fresh_imu_drive_20261020/`，40个冻结源文件，协议SHA-256 `7f42fb76117bd9620a47a4c8e031694f3d04c20a17e952e071cb794e0a6a7a39`，新初始化0步基线SHA-256 `afb53b326b8fadc7928363a34a63ea5585ff07b8d746c3909ea0784f23751129`。launcher记录PID 31873 / exec 98623。该trial与此前direct-track-v4实验相互独立；没有加载旧checkpoint，也没有改旧协议或成绩。

该v5 trial最终完成状态：2026-10-20 02:45:20（北京时间）训练达到196,608步，runner PID 31873实际exit 0、exec 98623真实exit 0；共保存29个checkpoint，最终 `stage_04_turn90.zip` SHA-256为 `0a9b310865e854108221cb1983a8804ddbde75c0d77a535e246502d7e83fefd6`。只读完结检查通过：40个协议快照与当时源码哈希全匹配；final权重/动作与观测metadata一致、有限且已更新，Adam状态有限并有正步数。before/after配对开发评估的free场景无损成功0/4→2/4、损伤与碰撞均0，通过预先设定的安全门禁；16株field场景无损成功仍为0/4→0/4，评估结论 `no_improvement`，不是田间能力提升。证据：[completion_validation.json](results/isaac_field_improvement/fresh_imu_drive_20261020/completion_validation.json)、[completion_model_validation.json](results/isaac_field_improvement/fresh_imu_drive_20261020/completion_model_validation.json)、[process_exit.json](results/isaac_field_improvement/fresh_imu_drive_20261020/process_exit.json) 和 [comparison_summary.json](results/isaac_field_improvement/fresh_imu_drive_20261020/comparison_summary.json)。

02:21–02:26的near阶段及8,192步暂停预览仅是当时记录的历史进度快照，不覆盖上述完结状态。旧v5 UI预览曾暂停在8,192步near checkpoint并运行0→1.70 s；该条历史记录没有加载旧v5 trial final。之后sensor-minimal-v6 final在PI接入前另完成了浏览器载入检查（见文首），不代表当前PI-v6源码部署。0步演示权重 `results/isaac_field_viewer/imu_drive_20261020/untrained_imu_drive_preview.zip` 是独立v5 UI预览，不能代替trial baseline。当前PI-v6当前源码与验证状态见文首更新。

该v5 trial用SB3-Contrib RecurrentPPO新随机初始化训练。五路输入编码LiDAR32、深度64、检测32、状态32、feedback32并拼成192维，随后128单元LSTM和四维连续动作头；4环境各采128步形成512 transition rollout，`batch_size=128`、5 epochs、学习率3e-4、γ=0.995、GAE λ=0.95、target_kl=0.03、`log_std_init=-2`、不使用gSDE，每8,192步存checkpoint。四阶段逐步扩大goto目标距离/角度：near 32,768（0.6–1.2 m、±15°）、steer30 32,768（1–2 m、±30°）、steer60 65,536（1–3 m、±60°）、turn90 65,536（1–3 m、±90°），合计196,608步。它是平地基础到点训练课程，不是植物避障或搜寻课程。

训练场景为4个环境、20 s episode、plants=0、无地形/岩石的平地goto，训练seed 69001–69004。奖励含按初始目标距离归一的进度预算5、到点+5、每株新增受损植物−20、岩石碰撞−0.1、时间−0.01/秒及边界惩罚；`distance_cost=0`、`goal_gain=1`。到点为距离≤0.4 m、车速绝对值<0.2 m/s、偏航角速度绝对值<0.3 rad/s并连续保持0.3 s，不要求终态朝向。训练只在无植物场景中接触到基础运动任务；这不能教会策略识别、绕行或保护作物。

配对开发评估已完成：free场景0株、seeds 87001–87004，无损成功0/4→2/4且损伤/碰撞均为0，按预设安全门禁通过；field场景16株、seeds 82001–82004，无损成功0/4→0/4，结论为 `no_improvement`。这些field seeds已观察，不是盲测；830xx仍未评估或观察。当前平地、plants=0训练和16株开发评估尚未提供可靠田间学习/泛化证据；不能据此推断模型绝不可能迁移，也不能宣称已学会避苗或整田导航。实车IMU/电机反馈尚未连入硬件闭环；仿真feedback来源和代理误差见 [本轮UI及硬件验收记录](results/isaac_field_viewer/imu_drive_20261020/README.md)。

训练收集action时从策略高斯分布采样（`use_sde=false`，初始log-std为−2，后续可学习）；评估和交互推理调用 `predict(... deterministic=True)`，使用分布中心而不是随机采样。两者执行前都受归一化动作范围[-1,1]约束。RecurrentPPO在每个episode起点通过episode_start标志重置相应LSTM状态，episode内部跨步保留记忆；独立评估每个seed从新episode和初始记忆开始。网络UI的critic/value支路则是另一个仅供观察的额外forward，使用独立诊断LSTM记忆，不控制车辆，也不替代训练时critic。

### 历史 direct-track 训练（旧v4观测合同）

`results/isaac_field_improvement/fresh_direct_tracks_20261020/` 已完成196,608步，真实训练进程exit 0（2026-10-19 17:11:13 UTC，即2026-10-20 01:11:13北京时间）。最终 `stage_04_turn90.zip` SHA-256为 `342237b1b2cacfb08135535891f9aaf1d4eb5ef957a68d85168bfe2f43e21329`；协议SHA为 `85b9b4dd44c9b9292078d77eb1fb3f56d06725007af2679011dd97efecce4ab0`，保留28个源文件冻结快照。它按near 32,768、steer30 32,768、steer60 65,536、turn90 65,536步完成，4个环境、阶段种子69001–69004；正式0步基线SHA为 `896c7776ee9ee719941cdb8ee85c665e94ffbe2ac2d373e43ccba88066ed5335`。

配对开发评估各4个种子：free场景无损成功0/4→1/4，前后损伤和碰撞均为0；16株field场景无损成功0/4→0/4，平均末距1.797 m→2.811 m。结果只显示free场景一个无损到达、field场景未改善，不作整体可靠性或能力提升结论；830xx未观察。这是旧v4观测合同模型，当前checkpoint加载器会拒绝它。模型、协议、成绩与失败记录均原样保留。

旧direct-track trial部署后的二维网络验收及其CPU/静态记录见 `results/isaac_field_viewer/expanded_layout_20261020/`，只代表旧模型结构。本轮v5固定源码的全CPU日志 `results/isaac_field_viewer/imu_drive_20261020/final_cpu_verified.log` 为222 passed、11 skipped、1项NumPy float32极端值溢出warning、pytest exit0；较早的shell变量转义失败记录 `final_cpu.log` 保留。fresh runner专项16 passed与compileall exit0分别记录在 `fresh_runner_validation.log`。v5 trial已完成，协议快照、checkpoint、模型权重及配对评估的历史完结核对见 `results/isaac_field_improvement/fresh_imu_drive_20261020/completion_validation.json`；其UI验收和final GUI未完成加载的边界见 [v5反馈UI与训练记录](results/isaac_field_viewer/imu_drive_20261020/README.md)。

运行时 probe：

```bash
bash scripts/isaaclab_python.sh python "$PWD/scripts/isaaclab_runtime_probe.py"
```

短程策略前后对照（默认使用 4096 训练步和 4 个固定 held-out seeds）：

```bash
bash scripts/isaaclab_python.sh python "$PWD/scripts/improve_isaac_policy.py" \
  --device cuda:0 --timesteps 4096 --eval-episodes 4
```

选择安全奖励 profile 时使用 `--protocol safe-goto`；不传 `--timesteps` 时为 65,536 步，且会自动保存紧凑轨迹。该选项不改变默认 `smoke` 或 legacy `staged-goto` 行为：

```bash
bash scripts/isaaclab_python.sh python "$PWD/scripts/improve_isaac_policy.py" \
  --protocol safe-goto \
  --checkpoint results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip \
  --device cuda:0
```

历史训练记录（2026-10-17，前进/差速动作合同）：当时的四阶段 fresh foundation parent使用`train_isaac_foundation.py`，阶段顺序为near 32,768步、steer30 32,768步、steer60 65,536步、turn90 65,536步。它使用4个训练环境、seeds 69001–69004，safe配置damage penalty20、goal progress budget5、distance cost0。parent协议已冻结；原始exec69866因Warp CUDA999错误terminal exit1，parent文件与记录保留，最后8192步checkpoint作为恢复输入。独立child `results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192`已从该checkpoint准备并启动。Root于2026-10-17 23:34:59确认Windows PID8424与viewer PID17740均存活；child仍为`running_steer60`，独立正式评估尚未开始。中文viewer当时已完成一次Edge浏览器验收；98304步是预览模型metadata，不是训练或独立评估成绩。中文化/reset控制补丁只改变可视化代码，不是策略、环境或学习代码变更，也不是模型能力证据。恢复child冻结时`viewer.py` SHA为`d5c9f217…2a4e`，当前SHA为`da205084…0566`，20个冻结源文件中其余19个仍匹配。此后续差异记于`resume_8192/ui_followup_reset_fix.json`；原`protocol.json`、`recovery_manifest.json`和冻结快照未改动。运行核验见`results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192/runtime_verification_20261017_232933.json/.log`。GPU健康检查的PyTorch CUDA拷贝、cuDNN前反向及Warp kernel均通过，wrapper和probe exit0；检查没有运行Isaac仿真，CUDA999根因仍未确认。下面第一组是原始fresh prepare/launch命令，作为parent协议入口记录：

```bash
bash scripts/isaaclab_python.sh python "$PWD/scripts/train_isaac_foundation.py" \
  --prepare --output results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive
bash scripts/isaaclab_python.sh python "$PWD/scripts/train_isaac_foundation.py" \
  --launch --output results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive \
  --device cuda:0
```

恢复runner仅接受terminal failed/interrupted父状态，核验原协议和源码快照；仅`viewer.py`可记录旧新SHA，其余runtime源码必须匹配。它也会核对指定near 8,192步checkpoint的SHA、ZIP完整性和策略/Adam条目。本次child已按以下命令准备并启动；child仍在运行，不要对同一输出目录重复启动：

```bash
bash scripts/isaaclab_python.sh python "$PWD/scripts/train_isaac_foundation.py" \
  --prepare \
  --resume-parent results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive \
  --resume-checkpoint results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/stage_1_near/checkpoints/field_policy_near_8192_steps.zip
bash scripts/isaaclab_python.sh python "$PWD/scripts/train_isaac_foundation.py" \
  --launch --output results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192 \
  --device cuda:0
```

该续训从checkpoint中的策略权重与Adam优化器状态继续，保留全局timestep并训练near剩余24,576步、steer30 32,768步、steer60 65,536步、turn90 65,536步，计划总计196,608步。各阶段使用新环境并重设规范seed；checkpoint不含中断时的环境RNG、LSTM隐状态或未完成rollout，因此不是逐位或中途状态恢复。Root于2026-10-17 23:34:59确认child仍为`running_steer60`；独立正式评估尚未开始。82001–82024是已观察开发集，不是盲测。中文viewer更新只改变可视化UI，不能作为模型能力或策略性能证据；当前浏览器预览已通过本次UI验收，但不构成策略评估。

最终评估计划为free-space seeds 87001–87024及16株场景 seeds 82001–82024；820xx是已观察开发集，不是盲测。训练和最终评估当前均未完成，不能据此声称策略改善。

## 尚未验证或完成

- 完整400株场景曾在官方Viser打开，模型控制下车辆、云台和雷达帧实际更新；这只是历史预览。2026-10-17 23:34:59 Edge浏览器中的中文3D预览已验收：策略切换/Run产生真实运动，native Reset提示“官方仿真已重置；已刷新当前状态”且viewer进程未退出。当前预览显示98304步模型，drive/observation/damage一致。另一次73728步seed82001 GUI 20秒运行超时，最终距离1.00 m、0 damage、0 collision；该单次开发观察不是正式评估，也不代表成功。前端可切换搜索观察，但foundation训练协议只有goto，不能宣称模型学会搜索。独立正式评估仍未开始。截图及验收记录见`results/isaac_field_viewer/chinese_reset_fixed_49152_20261017/chinese_preview_98304.png`和`browser_verification.json`；21项专项viewer reset测试通过。最新CPU suite及静态检查见`results/project_history/cleanup_and_validation_20261017_batch8.md/.log`；8项新几何原生回归由root此前单独运行，见`results/isaac_field_diagnosis/stem_contact_v2_20261017/native_width470_integration.log`。
- 历史512步训练 smoke 的8个 episode均未成功，8个都以time limit结束；历史full-field smoke运行32步。它们只验证运行和训练接线，不能代表当前策略能力。相关原始记录仍保留在 `results/isaac_field/`。
- 历史 staged-goto 固定12个 held-out seeds 对照为0/12→1/12到达；唯一到达的seed压坏4株，干净成功仍为0/12。最新 safe-goto 24 组严格配对评估提升了到达数但损伤均值上升，按不压苗优先标准不通过。两项结果都不支持可靠导航、避障或完整作业能力。
- 早期4-seed短程对照未显示总体提升，不应称为能力改善或泛化验证。16步教师/256步学生对抗实验也只是双向更新和真实评分的接线自检，不证明教师布局设计或学生策略能力提升，且不是PAIRED。
- 当前未做速度或吞吐量测试。
- 旧`fresh_foundation_20261017_stem_v2`为250 mm中心距，因用户确认470 mm总外宽而有意中断（exec15841 exit2/SIGINT，最后检查点73,728步）；协议、快照、权重和中断记录保留。原始parent trial `fresh_foundation_20261017_width470_progressive`的exec69866因Warp CUDA999错误exit1，8192步checkpoint已保存；该parent后续阶段及free-space 87001–87024、16株82001–82024最终评估未运行。独立恢复child当前状态及最新检查点见上文核验记录。GPU correctness probe三项通过，但原仿真错误根因仍未确认。820xx为开发集而非盲测；不作训练能力结论。
- 历史69项批量删除曾被自动审批以 `blocked by policy` 拒绝，未执行。用户本次另行要求移动归档并明确包含旧模型和结果；已完成58个文件/目录根、2754个文件的移动，全部 SHA-256 核验一致，40个当前 Isaac 源码、脚本、测试和配置文件哈希未变。旧 `ISAAC_CLEANUP_CANDIDATES.json` 原样保存在 `归档/legacy_before_isaac_20261017/` 供历史追溯，当前迁移事实以该目录的 `archive_manifest.json` 为准；此次未删除数据，也未改写历史实验记录中的旧路径。

## 对抗闭环实测

`results/isaac_field/adversarial_smoke/round_summary.json` 记录16步地图教师PPO和256步探索学生RecurrentPPO，两方权重均发生变化，教师布局由真实物理rollout评分，学生混合使用教师布局与随机布局。它是朴素minimax接线自检；没有证明布局设计能力或探索能力已经提升。不是PAIRED。

## 浏览器显卡与核显负载

2026-10-17通过Windows DXGI适配器LUID和GPU引擎计数器确认：内嵌浏览器ChatGPT.exe与dwm.exe使用Intel核显。关闭内嵌三维tab后，ChatGPT.exe的3D读数约46%→3%；NVIDIA CUDA仍有仿真进程。官方Pause Rendering冻结服务端状态，不停止浏览器WebGL循环。

按用户选择已打开独立Edge的 http://127.0.0.1:8768/，设置Edge高性能图形首选项（GpuPreference=2），原值备份于results/isaac_field/edge_gpu_preference.json。Edge原本已运行，设置不等于已验证旧GPU进程切换；若需重启，应保留其他网页并先获得用户答复。不要声称该设置已证明独显渲染。使用说明无需内嵌浏览器。

已按用户明确授权重启Edge并恢复原有网页。2026-10-17 18:18，Windows NVIDIA-SMI列出Edge GPU进程29788；同进程在NVIDIA LUID 1978F有300847104字节独立显存，证明当时的新Edge进程使用独显。证据：results/isaac_field/edge_gpu_verified.json；界面截图：results/isaac_field/edge_dedicated_preview.png。当前观测到的Edge PID为36908，与已验证PID不同，尚无该当前进程的适配器证据，因此不能确认当前是否由独显渲染。保留内嵌三维页关闭。

验证记录：Isaac Lab WSL早期CPU测试38 passed；第七批完整CPU suite为101 passed、8 skipped，compileall及两个bash wrapper语法检查均exit0，详见`results/project_history/cleanup_and_validation_20261017_batch7.md/.log`。更新至470 mm外宽/410 mm中心距后的8项原生物理回归另由root运行并通过；没有开展训练速度或FPS测试。


2026-10-19第十批拓扑纠错与整理完成：Gaussian参数log_std独立连接到标准差；LSTM按输入、门、c、h的计算依赖展示，跨步反馈折叠为输入；执行器列随动作输出排列，滚轮目标依赖履带指令。完整CPU156 passed/8 skipped，compileall、wrapper/bash与HTML内嵌JS语法检查通过。见 [第十批整理](results/project_history/cleanup_and_validation_20261019_batch10.md)。此为卷积结构调整前的记录。

历史记录（2026-10-19，前进/差速动作合同）：编码器拓扑更新已在 [本轮记录](results/isaac_field_viewer/encoder_topology_20261019/README.md) 中验收：完整CPU回归172 passed/9 skipped，官方Sb3VecEnvWrapper深度精确值专项1 passed/37 warnings。早期0步预览在浏览器从仿真0秒运行到20秒、记录序列401；深度/LSTM/动作图像变化，LiDAR保持全量程且未变化。该 ZIP 保存的深度 Box 为[0,1]，仅保留历史浏览器与Netron证据，不用于当前[-1,1]环境。前一代 v2 0步基线 `untrained_baseline.zip` SHA-256为`b2ecd57b73ebb8fdc9fa95622b166af82c1b0ab964a2daaf226be37ffffc954a`。车辆页随后真实加载v2的8192步checkpoint并运行到5.15秒：LiDAR输入、两层Conv1d、深度池化、LSTM h与动作图像均变化，errors为空；所选Conv1d值从-1.08233726变为-0.602363467。`checkpoint_8192_validation.json` 验证通过：卷积/LSTM/动作权重更新且有限，Adam有35项状态、最大step300，源码哈希及旧模型哈希匹配。v2训练在15:46:15 UTC曾记录24,576步；当时near阶段32,768步已完成并进入steer30。协议SHA-256为`f32707b8fdfa8098613d766b6ce38d0edbd74bf396c2c6abea432f698b9d9e84`，快照27个源码文件。v1 wrapper边界失败及0步记录保留。旧196608步checkpoint `resume_8192/stage_04_turn90.zip` SHA-256仍为`11f6c287e8bfdbb348fb1c9d9205c2a95692f1629ef8cee4626aef3947ae506f`；该模型与当前观测和动作合同均不兼容，原权重和实验记录保留。

2026-10-20第十一批五项整理完成。第五项为二维网络页全局自适应色标：所有层有限激活值及绑定模型参数共同决定完整值域的对称范围±M，M取最大绝对值，不使用百分位；零点位于中间，Turbo色图不变。新帧出现更大值立即扩展，回落仅在新帧时发生且每帧最多5%；重复HTTP读取或同一帧token不缩小。reset/bind清除范围历史，页面无手动范围控件，后端拒绝非auto数值scale参数。真实viewer使用196,608步final checkpoint通过浏览器检查：初始和参数视图±2.146289825439453，14.75秒运行时±17.186372756958008，19.90秒暂停帧±21.128650665283203；重复读取帧399范围稳定，reset回到±2.146289825439453。跨state/fusion相同0值RGB为[164,252,60]，errors为空；12张卡片、无重叠、sliderCount=0，最后停在融合160维。证据位于 results/isaac_field_viewer/adaptive_global_color_20261020/。

功能CPU验收日志为176 passed、9 skipped、1 warning；staging日志为12 passed、1 warning。唯一warning为极端float32测试触发numpy标准差平方溢出，范围输出保持安全。整理静态检查和源码冻结核对见 results/isaac_field_viewer/adaptive_global_color_20261020/static_validation.log。

历史训练记录（2026-10-19，前进/差速动作合同）：Fresh v2训练已完成196,608步并真实exit 0，协议SHA-256 f32707b8fdfa8098613d766b6ce38d0edbd74bf396c2c6abea432f698b9d9e84；stage_04_turn90.zip SHA-256 fbb2e39f741bfa812b192c9b9320f711bb82187e46dc64fc5461b17a6296cc77。有效配对开发比较free与field各4个seed，两者均为no_improvement；不是盲测或整田导航结果，830xx未观察。该模型与当前直接左右履带动作合同不兼容，评测和Viewer会拒绝加载；SHA和评估数字只作历史记录。部署manifest记录训练exit 0后才更新viewer的network_2d.py、network_2d.html；27个训练源码快照SHA全部匹配，当前只有这两个viewer源码与快照不同，其余25项一致。部署前后权重和协议未改。
