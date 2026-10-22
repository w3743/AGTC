# Isaac Lab 项目状态

2026-10-22 RSL-RL全链迁移：独立worktree `E:/worktree/rsl-rl-migration/先进通用履带底盘工程`，分支`feat/rsl-rl-completion`，基于首轮迁移合入后的main `bb54e169`。新训练/续训调用固定Isaac Lab官方`run_train_cli`，默认任务`Isaac-Field-FastStop-Rsl-Direct`；复用官方wrapper、OnPolicyRunner、PPO更新/存储、RNN、Gaussian及日志。新模型活动入口包含训练/续训、完整评估与显式smoke、LASSC消费/回放、全图点选/跟随预览、真实网络检查和JIT/ONNX导出，没有新增训练器、生命周期管理或Tracking服务。

输入为`sensor-absolute-goal-v1`/encoder v4，五路形状为LiDAR[1,400]、depth[1,160,88]、detections[16,8]、state[33]、feedback[37]，内容与轴不变；仅RSL任务边界将policy下五路提升为根级tensor以适配官方存储。复用原192维编码器，actor/critic各512×1 LSTM、64/Tanh头，共享encoder只在actor注册，PPO优化器和裁剪不重复计算该参数。四维动作顺序及单位继续由action_contract定义。fast_stop_v1/0.4m/真实XY<0.05m/s/完整角速度模长<0.1rad/s/连续0.5s保留；全场随机起终点，flat、plants0、rocks0、原联动导向轮及零稳定奖励。60s有限horizon失败不bootstrap，关闭随机episode_length_buf偏移。未把未完成的v2奖励混入迁移。

RSL检查点为带field_policy_contract的原生.pt，记录完整task_config、导向轮/稳定性配置、输入/编码器/动作/观测、drive/damage及actor/critic配置；真实采样步数在官方process_env_step后累计，续训恢复Adam和计数。五路编码器抽为纯Torch SensorEncoderCore，SB3变为optional legacy extra；旧isaac_field.policy.SensorEncoder类路径和state_dict键保持。RSL网络检查采集实际模块及LSTM门值，critic独立诊断状态，暂停预览不推进runner或车辆。生产打包器从同一任务合同复原绝对目标和归一化；非零heading、异坐标目标数值断言已覆盖。旧ZIP/历史课程保持显式legacy，不转换；新旧原始记录不改。

CPU全套pytest真实exit0：585 passed、23 skipped、55 warnings、31.52s（CUDA_VISIBLE_DEVICES为空，RUN_ISAAC_PHYSICS=0），见results/isaac_field_diagnosis/rsl_completion_smoke_20261022_attempt2/cpu_final.log；其后仅HTML首次层选择修复的前端专项19 passed、1条既有NumPy warning，真实exit0。compileall、两个bash语法检查、TOML解析、宿主git diff --check通过；WSL Git因Windows worktree指针不可读而失败，宿主复查通过。屏蔽SB3导入的子进程仍可导入新活动路径并构造/预测RSL模型。训练时56份源码快照aggregate为aa6f997dcf444362027fefba04363cc2a9a3c7ec90f3dfce6f75df2dcc9c49a6；保留其原字节，之后仅模型卡文案和网络HTML补丁另存UI源及SHA。attempt1保留579 passed/23 skipped/1 failed的原log与exit1；唯一旧ZIP测试补显式legacy选项后通过，生产门槛未放宽。首轮迁移PR18的530 passed记录属于上一批。

原生验收已按预算完成：官方RSL在cuda:0上4env×128步×1迭代，日志Total steps=512、正常打印Training time后Python真实exit0，未追加训练。检查点logs/rsl_rl/isaac_field_rsl_completion_smoke/2026-10-22_completion_smoke_attempt2/model_0.pt，36,853,539字节，SHA256为4565d308855a0eea0f33f35e9f63c32e7fa737b2008d665b183381fe301be83a。CPU安全重载确认实际512步，37项Adam状态均step20，actor/critic/optimizer有限，非零有限moments；没有初始baseline，未声称逐权重前后比较。同checkpoint原生评估seed87001、1场景最多2步、0.1s后由评估步数上限截断，Python真实exit0；不是60s任务超时，assessment=not_assessed，不报告成功率或能力改善。

同checkpoint的LASSC CPU回放18项检查通过、真实exit0；绝对目标按完整task配置组包。JIT保存/重载的动作与h/c核对，ONNX导出/checker通过，未做ORT推理。最终只读审计cpu_checkpoint_audit_attempt4报告与外层直接捕获exit0一致；早期audit路径/导入错误和attempt3外层引号错误保留，attempt3 marker的0不视为独立实测child exit，不以报告内容代替退出证据。

真实RSL预览在8783/8784，results/isaac_field_viewer/rsl_completion_smoke_20261022_attempt3记录loaded/paused/time0。浏览器已核全图四角、点选目标、跟随视角、reset与读取帧：未点击运行策略，最终frame2/input0/仿真0、running=false、errors为空，网络展开61个真实RSL节点，截图及ui_verification_final.json保留。早期attempt2预览主动SIGINT停下补模型卡文案，exit2/KeyboardInterrupt留档；未重训。网络页首次selected层原硬编码SB3名称导致HTTP400，现从真实节点采纳默认层，未知层后端仍严格拒绝；仅热读HTML重开网络页，无第二份仿真。当前卡明确显示“迁移smoke（≤512步），未完成控制能力验收”。

默认4×128×384=196608仍只是正常配置预算，本轮没有完整训练。RSL与SB3的初始化、梯度裁剪和KL处理非数值等价，fixed学习率下desired_kl不等于SB3 target_kl早停；不以迁移跑通宣称到点停稳改善。硬件只有CAD/原理图，没有实车验收。旧runtime/失败ablation现场及两项PAUSED automation保持。

2026-10-22 00:01最新状态：fast_stop512 mastery课程已于2026-10-21 17:15:19北京时间以196608步自然完成，真实exit 0；最终near为6/16，未晋级且未建立能力提升，正式global best仍为untrained_baseline.zip。新的绝对目标输入合同`sensor-absolute-goal-v1`及编码器`depth64-lidar-single1d-absolute-goal-v4`已16文件冻结并审核通过：仅将33维state的`state[1:3]`改为`task.goal/size*2-1`，其余31槽不变；旧`sensor-minimal-v6`相对语义保留，不能混用。最终CPU回归511 passed、23 skipped、43 warnings、23.04s、exit 0；首次502 passed、23 skipped、4 error、exit 1因缺固定6源的llm-viz依赖，补齐后通过且保留失败记录。临时prepare/preflight已验证47源、新schema、512×1与固定Isaac Lab依赖并清理临时目录；未启动正式trial（`training_started=false`）。98304步v6模型CPU加载与forward有限值通过，仅为CPU检查，不是物理/GPU验证；无新训练/GPU启动，两项自动化仍PAUSED。第十八批当前4/5：此前3项加本次绝对目标功能。此worktree为`E:/worktree/absolute-goal-schema/先进通用履带底盘工程`；原目录固定在`runtime/legacy-v6-preview`分支`f24d7c43`，原47 live/snapshot与模型及PID415预览仅属原目录，不代表此worktree状态。

2026-10-21 12:50:55训练历史快照：恢复控制组与rough_a/b配对开发评估已完成；8个paired seed的clean成功与进入0.4 m半径均为0/8→0/8，未见损伤或碰撞，但不构成学习提升。fast_stop512 mastery课程仍在运行；run_status最近记录为`mastery_probe`/49152步（12:50:55北京时间），WSL进程PID458的实际argv已核实。47个训练源码及protocol保持冻结。CPU-only Viser 1.0.30浏览器交互验收已通过；这不包含原生策略物理测试，仍须等待课程自然完成并核实exit 0后才启动实际策略预览。

后续跟进状态：原周期训练automation与专用heartbeat“训练结束后打开点选目标预览”当前均PAUSED，不承诺自动打开；当前手工核验不改变训练进程。恢复与评估明细见TRAIN-01。

恢复前曾只读核实Ubuntu-24.04为Stopped，昨日SIGSTOP的原PID和内存现场已不能确认；没有对旧PID发SIGCONT，也不是逐位原位续跑。新子环境、RNG、LSTM/rollout重新初始化，训练加载阶段逐tensor核实权重与Adam等于父57344检查点。下述2026-10-20启动/暂停等记录属于历史时点；当前运行状态以上述控制组/配对评估完成证据及已启动的课程状态为准。
用户已确认将本轮目标改为最短时间到点停稳。新入口 `random_rough_fast_stop` 与独立目录 `results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020/` 已于18:01:53北京时间正式启动；协议SHA `3ca58fcc05b9d7ac34798f836704c91b5ec608d62f7424c3897ed291451a620c`，45源冻结。LSTM512、新随机权重、gamma1、成功+10/失败−10/每秒−0.1；距离潜势仅辅助学习并在终态抵消，途中稳定奖励权重为0，姿态统计/硬失败保留。停车为距目标≤0.4m、真实水平速度<0.05m/s、完整车体角速度<0.1rad/s持续0.5s，不要求朝向。物理、全场随机goal、4环境、196608预算和60s上限保持。CPU377 passed/20 skipped/8 warnings、编译/脚本/源码稳定性实际0，原生Newton20 passed/61 warnings真实0；第十六批已整理。初始化已核实actor/critic各512×1、3065801独立参数，训练诊断已有实际PPO更新、reward分项对账一致，尚无能力提升结论。细节见本轮整理与真实运行记录。

此前旧奖励512实验在17:42:43经核实argv后SIGTERM停止，实际Python exit143；末完整40960步模型SHA `67bcb7b73dd5f22c1b01e33a1a0e86cf8f8ed82c25587088ceffa907c1df33d5` CPU验收通过，模型/协议/检查点保留。旧run_status的running是停止后陈旧值，不代表仍运行。

旧奖励512实验的原始启动、源码、测试和首检查点记录保存在 `results/project_history/lstm512_experiment_20261020/`，真实停止及末检查点CPU记录在 `results/project_history/fast_stop_objective_20261020/`。当前新实验不迁移旧权重，两种奖励/停车规则的成绩不得混算。自动定时任务继续暂停。

## 当前车辆与控制链

项目使用本地 Isaac Lab、Newton MJWarp 和 SB3-Contrib RecurrentPPO。车体、轮子和云台使用轻量代理，原14个负重轮执行器各保持2 Nm限幅；没有车体速度伺服或手写规划器替代策略。物理步长1 ms，每50个子步执行一次策略，摩擦系数0.8，P=0.04、侧积分Ki=0.1。以上参数仍需实车标定。

用户要求的抬高前导向轮已经接入：左右各一只，半径45 mm、宽60 mm，轮心在车体中心前方420 mm，比支撑轮轴高50 mm。两只轮的质量从车体扣除，整车仍为30 kg。启用 `--front-guides` 时，导向轮通过Newton公开mimic/equality接口跟随本侧轮0；同侧其余负重轮也跟随轮0。

| 项目 | 关系与数量 |
|---|---|
| 每侧负重轮 | 轮0带6个从轮，左右合计12条关系 |
| 前导向轮 | 左右各加1条关系，每环境总计14条 |
| 转速关系 | ω导向轮=(R负重轮/R导向轮)×ω本侧轮0；当前半径相同，1∶1 |
| 新增执行器/PID | 0；导向轮通过约束传递已有驱动力矩 |
| 场景隔离 | 仅连接同一world内、本侧的关节 |
| 不启用前导向轮 | 保留原14轮资产、12条关系及旧合同 |

`FrontGuideSpec` 是规格来源；资产生成、质量分配、环境、Newton约束、drive metadata和fresh实验合同都使用这份规格。Isaac配置层用普通dict传递，进入环境后恢复只读规格，避免configclass修改冻结dataclass。自由转动模式只留作诊断对照，正常前导向轮默认联动。

这是带抬高轮的离散接触代理，还没有连续履带斜段、张紧机构、土壤变形或完整实车单电机传动模型。不能把“导向轮会转、能够碰到台阶”直接等同于已经具备真实履带的翻越能力。

## 当前任务与网络

当前新任务是：零植物、零独立岩石、崎岖连续地形、全场随机起点/朝向/目标，训练安全到点和车身稳定。`random_rough` 配置100×100 m、地形幅值预算0.35 m、波长4–10 m、60秒回合、4环境、seed69009、新初始化196608步。每次运行固定一份物理地形，每回合重新采样起点、朝向和目标；开发配对分别使用80003/80004两份地形和87001–4/82001–4开发种子。830xx仍未观察。

`--body-stability` 增加超过35°试验软阈值的总倾角惩罚和车体X/Y轴角速度平方积分惩罚。仿真真值只用于奖励与统计，不增加策略输入；云台IMU不冒充车体角速度。原1.2 rad硬倾角终止保持，边界/roll/pitch原因分别记录。35°不是实车安全角标定；最大角度和RMS是策略步采样统计，不是每个物理子步的极值。

网络仍为 `sensor-minimal-v6` / `depth64-lidar-single1d-minimal-v3`：LiDAR1×400、深度1×160×88、检测16×8、state33、feedback37，五路融合192→LSTM→连续4维动作（左右履带速度、云台偏航/俯仰速率）。通用默认与历史模型为128维，本轮独立实验指定512维。没有新增导向轮动作或观测通道。

当前前导向轮与稳定性native专项3项、原有环境集成14项均真实通过；整理后的CPU回归349 passed/18 skipped/8 warnings、compileall及项目bash语法检查通过。同源码、同初态60毫米台阶固定低速动作对照中，自由轮/联动轮都记录到接触，但都未越过，行进距离近似相同，不能称翻越改善。详细实际退出、失败尝试和接触对照见本轮整理记录：

- [联动实现与原生验证](results/isaac_field_diagnosis/front_guide_coupled_20261020/README.md)
- [本轮整理和归档记录](results/project_history/front_guide_coupled_cleanup_20261020/README.md)

## 训练与界面边界

原128维新几何准备目录 `results/isaac_field_improvement/random_global_rough_coupled_guides_20261020/` 保留prepared/0步历史证据，协议SHA为5de27ea7b7e1714b8822ec8fe11f0b53e6f08d194577b253d376913a00405980；512入口改源后不能再据旧preflight称其可直接启动。恢复控制组和rough_a/b配对开发评估均已完成；两个split分别no_improvement，合并8个paired seed后clean成功与进入0.4m半径均为0/8→0/8。mastery fast_stop512课程的2026-10-21 12:50:55北京时间历史快照为`mastery_probe`/49152步；当次WSL核实PID458仍以`train_isaac_mastery.py --launch`运行，未见完成或晋级证据。手动检查使用8781车辆页，不加载旧checkpoint；此前实际运行至5.45秒并暂停归零/reset回0秒、console无error（1条上游弃用warn）；旧8769/8770模型回放属于原14轮几何，不作为新增导向轮或512策略验收。运行命令见 [使用说明](使用说明.md)。

界面固定深色，车辆三维和完整二维网络保留；不恢复网络三维或整体视图。网络共60个监控卡片，并非60个神经层；连续曲线不随选层点亮。主图所有采集激活和参数共用整帧完整值域自适应Turbo色标；详情单图可独立自动缩放。无checkpoint的手动检查不产生训练策略实时激活。

另有独立 `train_isaac_mastery.py` 训练方法入口，提供原奖励诊断、能力门槛课程、旧任务新轨迹采样、安全选模与单变量GAE对照。它与当前全场rough方案分开；此前三份method_v2协议是在旧源码下准备，当前源码已变，不能直接启动或改写原协议。相关方法和限制见使用说明及原始 `results/project_history/training_method_solutions_20261020/`。

## 历史模型和限制

| 实验 | 真实结果 |
|---|---|
| fresh_coupled_tracks_20261020 | 原14轮几何，20秒回合，196608步完成；free、field各4种子无损成功0→1，均零损伤/碰撞，仍各3条超时 |
| continue_coupled_tracks_20261020 | 同原几何，改60秒，从父权重及Adam续至393216步，真实exit0；同60秒重评free成功1→1、field1→0，均零损伤/碰撞，未通过提升门槛 |

旧成绩仅属于对应物理、配置和初态；不与新几何或新地形混算。四种子属于开发小样本，不能称盲测。模型、协议、源码快照和原始成绩均保留。旧续训final尚不能据历史viewer标签宣称已部署。

上述历史实验采用距离≤0.4 m、低速稳定0.3秒，不要求终态朝向；当前fast_stop_v1采用本文开头的严格停车判据，两者不得混算。优先不压苗；只有同配置/同初态下无损成功增加，且损伤回合数、平均损伤、碰撞不恶化，才能称导航提升。当前plants0不代表已经学会避苗。粗糙地形可靠导航、整田搜索、对抗训练、真实IMU/FOC串口/CAN/HAL联调与车辆标定均未完成。

用户已关闭定时任务，原生automation保持PAUSED。更早运行日志、逐批整理和旧命令仍保存在results及归档；本轮整理前的三份主文档原字节及SHA清单在 [docs_before](results/project_history/front_guide_coupled_cleanup_20261020/docs_before/manifest.json)，历史“正在运行/当前”等描述仅属于当时时点。

## TRAIN-01：感知契约升级与训练验收方案

状态：前期离线方案已落实为生产消费桥、真实检查点CPU回放及独立QA门槛，通过后按新授权恢复原仿真控制组。用户确认位置范围保留0.4 m，先严格停稳：真实水平XY速度<0.05 m/s、完整车体角速度模长<0.1 rad/s，连续0.5 s，最终朝向不限。LSTM保持512维。下文保留旧基线与后续新观测实验的差异，当前实现与命令见本节末“已落地与运行证据”。整车指标由项目根README维护，组织状态由项目根项目协作.md维护。

### 历史暂停现场与恢复前可恢复证据

2026-10-21主代理复核WSL清单同样为Stopped，没有进入Ubuntu运行ps。昨日暂停证据仍保留在 [pause_record.json](results/project_history/fast_stop_objective_20261020/pause_20261020/pause_record.json) 和 [pause_audit.json](results/project_history/fast_stop_objective_20261020/pause_20261020/pause_audit.json)，它们描述昨日现场，不覆盖今天的WSL状态。

| 核查项 | 本轮结果与边界 |
|---|---|
| 磁盘模型 | `random_global_rough_fast_stop_lstm512_20261020/stage_1_global_random/checkpoints/field_policy_global_random_57344_steps.zip`，37,400,451 B，ZIP可读且含`policy.optimizer.pth` |
| 模型实算SHA-256 | `db473f5e78038c066226f61cb6a5ee02ea71615823f28e519366cb0809ecbb8a`，与manifest一致 |
| 协议SHA-256 | `3ca58fcc05b9d7ac34798f836704c91b5ec608d62f7424c3897ed291451a620c`；45/45 live及snapshot源码一致 |
| 最后完整诊断 | JSONL记录到65,024步，比磁盘模型多7,680步；日志不含这些步的完整可恢复训练状态 |
| 完成与恢复 | 未达到196,608步预算；没有final正常退出证据。检查点含Adam不等于保存了全部仿真/RNG/rollout现场，磁盘恢复不能称逐位连续续跑 |

上述是恢复授权前的只读核查结果；后续没有覆盖原暂停记录，也没有改父45份冻结源、协议、权重或实验成绩。旧PID不得直接复用发信号。恢复原仿真控制组与实施新感知训练仍是两项独立工作，不能把接口通过当作训练观测已经改变。

### 旧基线与新契约的差异

唯一数值契约为[深度接口契约](../视觉算法/深度接口契约.md)，schema为`depth-z-u16le-v1`；板端接口已部署，消费桥与真实模型离线集成已验收，实际数据接入就绪另行判定。本节只承接，不另定义字段真源。当前线上帧缺少已验证时钟映射和外参，正确报告not-ready；不能将合成正例的校准信息填到线上。板端运行与部署依据[现行视觉实现](../视觉算法/软件/数值深度接口候选/README.md)、[已校验历史快照](../视觉算法/备份/RK3588_当前运行算法_20261021_011952/backup-info.json)及[带时点的板端证据](../视觉算法/软件/LightStereo源码/evidence/rk3588-hardware-live.json)。

| 项目 | 冻结仿真基线 | 板端已查证行为／新方案要求 |
|---|---|---|
| 策略输入 | 单通道`(1,160,88)`，有效Z/5，无效为0 | 传输候选为小端uint16毫米轴向Z、0无效；兼容适配保留旧形状与归一化，新增mask通道另立实验 |
| 数值与量程 | camera_z，0.3–5 m | 备份代码同为轴向Z、0.3–5 m，内部float32及valid；显示JPEG不是数值矩阵。数十米是新目标，不能用更大编码范围证明测距准确 |
| 几何 | H160×W88，有K | 默认fast代码推导H240×W131，显式LightStereo分支640×352；真实运行模式由现场配置核实。每帧K须对应旋转、裁剪、尺度及光学参考系；按射线映射，深度与valid同索引，禁止宽高独立拉伸替代标定 |
| 有效性 | 明确valid后才归一化 | 从原始值先计算valid再裁剪并显式传入；0、非有限、越界均是未知。300/5000 mm旧边界有效、299 mm无效、1000 mm策略值0.2。valid不等于校准置信度，不把缺测填成可通行 |
| 时序 | 周期0.4 s、模拟延迟0.3 s；策略配置20 Hz | 板端用PTS调度，不能套仿真频率。payload/seq/K/源PTS/会话epoch/发布时间须同帧；采集单调时间及外参未确认时显式unready，不以发布时间替代曝光时间 |
| 帧龄与同步 | 旧固定延迟模型；已有age_ratio、newframe和相对捕获位姿输入 | 按契约用left_pts映射到producer monotonic的source_mono计算帧龄，包含排队；跨host还需时钟映射与误差界。PTS参考不等于已验证的曝光时刻。左右软件PTS可容许25 ms不代表曝光硬同步。采集30 FPS、历史深度吞吐均不能代替当前整链时延 |
| 高分辨率／ROI | 不在旧输入合同内 | 全幅近处低延迟＋远处原始像素密度ROI为候选。ROI携带自身K、位置、时间、有效域；缺失/过期/乱序/跨epoch需有可验证处理，不用旧帧冒充新帧 |

对齐条件为同一整流左光学坐标系且外参不变时，现有标定近似K目标`fx=fy=98.324796,cx=44,cy=80`，fast源`fx=fy=147.487194,cx≈65.766881,cy≈121.086791`；候选映射约为`u源=1.5u目标−0.233119`、`v源=1.5v目标+1.086791`。须由契约fixture固定像素中心约定并复算，不能把这个近似写死适用于所有模式。相机→云台→车体外参及源时间对应的两轴实测角度也要核实；云台IMU不能直接充当车体IMU。

### 最小对照顺序与资源

复用本地Isaac Lab＋Newton MJWarp、SB3-Contrib RecurrentPPO和现有诊断/安全选模/课程入口，沿用[使用说明](使用说明.md)的运行与留证方式。旧method_v2协议已与live源码不符，只参考方法，未来另建目录prepare。先做CPU接口fixture，不先扩LSTM或重写训练基础设施。

| 阶段 | 对照与要回答的问题 | 将来执行条件 |
|---|---|---|
| C0：离线接口 | 解码字节序、长度、边界、valid先于裁剪；K射线、ROI索引；同帧原子性；重复/乱序/跨epoch/缺帧；源时间1.00、发布1.10、消费1.25 s时帧龄应为0.25 s | 契约及候选fixture落地；只用独立测试输入，不改冻结源 |
| C1：旧输入兼容 | 同一数值深度经旧适配后形状/尺度/valid逐项一致；复用已有age_ratio/newframe/相对捕获位姿。未就绪数据可报告解析成功，但legacy_ready=false，不送入闭环 | C0通过，校准版本和旧合同明确；CPU即可验证，不称车辆轨迹验收 |
| E0/E1：时序单因素 | 同512架构、同训练预算/初始权重与初态，旧0.4 s＋0.3 s对照经实测支持的新帧龄/丢帧模型；输入分辨率与奖励不变 | 明确恢复/新训练授权，实测时间分布可用；新目录/协议/种子/源码与资产冻结 |
| E2：分辨率 | 固定E1时序及量程，仅改空间采样；评估细茎/障碍可观测性、无损成功和板端资源 | 对应K与测距误差数据可用。输入形状或encoder改变需新模型合同；不直接加载不兼容旧状态 |
| E3：全幅＋ROI | 在E2基础上比较远处ROI的增益及ROI缺失代价；双通道valid若需要，另作单因素分支 | ROI选择/时间/几何/输入维度冻结；无手写路径规划器替代策略 |
| D1：真实电驱 | 在选定观测下，独立比较原代理与资格化的电机/母线/制动包络，之后才组合验证 | 电驱负责人提供可追溯边界；保持摩擦0.8，候选值不能冒充标定 |

每个学习对照建议先用现有开发集做小规模可运行性检查，再在固定的配对开发初态上比较至少3个训练种子；实际回合预算在协议中预先固定，不能看结果后挑种子。各对照优先复用相同参数的初始化；新增输入结构须明确初始化/迁移规则及其混杂影响。新量程、时序、几何或驱动不得覆盖旧实验，也不得混用历史成绩。

以下为纯payload及FP32权重算术，MiB=1,048,576 B；不含header、mask、拷贝、梯度、Adam、激活和感知模型，不是RK3588实测性能或已选规格。

| 候选输入 | 单帧B | 候选频率 | 纯payload MiB/s |
|---|---:|---:|---:|
| H160×W88 float32 | 56,320 | 2.5／20 Hz | 0.13428／1.07422 |
| H240×W131 uint16 | 62,880 | 30 Hz | 1.79901 |
| H640×W352 uint16 | 450,560 | 10／30 Hz | 4.29688／12.89063 |
| H512×W512 ROI uint16 | 524,288 | 10 Hz | 5.00000 |

现512模型actor部署口径为含共享encoder和动作head的1,587,016参数，FP32权重约6.054 MiB；训练模型3,065,801独立参数约11.695 MiB。部署不携带critic/Adam仍有激活和运行时开销。640×352的像素数是160×88的16倍，不能只按LSTM权重容量判断余量。将来须在相机采集/识别/深度/策略同时运行条件下记录峰值RAM、CPU/NPU占用、温度、功耗、策略截止时间及源帧龄p95/p99/最大值；逻辑20 Hz不是实测。训练GPU与native检查串行安排，本轮均不运行，不做训练FPS基准。

15 km/h=4.167 m/s；仅按旧0.4 s等待＋0.3 s延迟＋0.05 s策略等待的理想最坏相位估算，0.75 s已前进3.125 m，尚未包含执行器和制动。规划所需观测距离应由整车统筹按`v×总反应时延＋v²/(2a制动)＋余量`评审；后项只是恒减速度近似，实测要覆盖坡度、附着、载荷、温度和SOC。现1.5 m/s²是指令斜坡，不是已知制动能力。

以当前代理主茎直径24 mm和旧fx=98.324796作理想针孔计算，5/20/30 m投影宽度分别约0.472/0.118/0.079 px。这只说明旧采样对细茎的限制，不给出检测概率或真实量程；高分辨率/ROI仍须用实物独立距离基准验证。

### 场景与验收门槛

| 场景或失效 | 必须留证与比较 |
|---|---|
| 风吹叶片、细茎、遮挡 | 优先复用实录双目及独立距离/遮挡标签进行回放，分别统计主茎漏检、背景误深度、无效占比和随风闪烁；仿真运动幅度来自数据，不先自建复杂植物力学 |
| 逆光、弱纹理、反光、明暗变化 | 按距离、照度和目标分层报告有效率及轴向Z误差/离群率；valid=true不能当作正确深度。5/10/20/30 m是候选测试档位，不是已通过量程 |
| 车辆转弯、颠簸、云台移动 | 记录左右源PTS差、曝光/运动模糊、源帧龄、外参链与两轴实测角度；区分静态误差和运动失配，不能仅用发布频率评判 |
| 缺测、过期、重连、ROI失踪 | 冻结后在回放中注入固定事件，同初态比较；不得把未知当自由空间。复用已有newframe/age_ratio，核实其新时间来源；如需新增缺测/ROI质量输入则另立观测版本。超龄失效响应经系统安全接口验收 |
| rough、植物及岩石 | 保留相同地形USD路径/SHA、资产和初态；分别测试零植物基础控制与有植物/岩石任务。当前plants0/rocks0没有避苗或岩石感知证据 |
| 驱动与回馈受限 | 提供两台电机侧扭矩/速度/电流/母线电压/热持续时间、四象限驱动制动、回馈可吸收能量及故障/超龄响应；覆盖SOC/温度/坡度，不能假定无限回馈 |

电驱依据见[FOC审计](results/hardware_limits/foc_20261020/README.md)：14轮各2 Nm、Kt=.035、效率1均是未标定代理；推算每侧电机Iq约66.67 A不是真实反馈。1.8 kW·1 s设计候选、卖家80 A及软件参考限值也不等于连续资格。后续应承接电路与机械负责人的现行边界，不以历史候选直接改冻结动力学。

开发门槛先检查同配置/同初态的无损成功增加，且损伤回合数、平均损伤、碰撞不恶化；再比较双方共同无损成功初态上的停车用时，并同时报告超时/失败，避免只挑成功样本。严格停车保持本节开头合同；报告完整车体角速度、真实XY速度、连续hold时长及终止原因。每项保留原始轨迹、奖励对账、版本/源SHA、资产、种子及真实退出。不同成功规则的旧报告不进入同一成绩表。

810xx/820xx/870xx继续作为已观察开发数据；830xx保持未观察，仅在开发安全门槛通过、方案预注册后作独立验证。小样本通过只作为继续实验依据，不能称整田可靠导航；风吹/光照回放通过也不能替代闭环实车验收。

### 下一依赖与启动条件

接口软件C0/C1已完成下述真实验收。后续新观测及实车闭环仍依赖整车统筹给出可追溯的源帧龄、有效量程、外参/同步和制动边界。外参或采集时钟仍unready时可继续接口fixture，不能放行高速实车闭环。

新观测训练启动前仍须区分原仿真恢复与新合同新训练；所选阶段所需的时间/几何/电驱输入已资格化或明确限定为候选仿真范围，prepare/preflight基于新的冻结源码和独立目录通过。原仿真恢复的用户条件授权与接口门槛已满足并已执行。现有自动任务保持PAUSED。

### 已落地与运行证据（2026-10-21）

| 交付 | 实际结果 |
|---|---|
| [LASSC消费桥](scripts/lassc_consumer.py) | 直接复用RK生产DepthAdapter及FieldTask捕获校验/观测打包；valid在裁剪前生成并显式传入；保留已有age/new_frame/相对位姿，按源时刻的有界PoseHistory查历史；坏帧克隆后校验，只有ready且fresh才提交有效状态 |
| [真实模型回放](scripts/lassc_replay.py) | 复用可信checkpoint保存的spaces并逐键校验；生产wire→consumer→真实57344 PolicyRunner.predict→连续LSTM/reset→4维动作→RPM/CAN CRC与云台rad/s，独立CPU回放17项通过；无CAN/串口传输、无伪造物理轨迹 |
| [独立QA门槛](results/lassc_interface_20261021/qa_gate.json) | status=passed；原样线上62880B帧7项拒绝/不推进状态检查通过；契约最终69项及consumer专项4项通过。正例明确是合成几何/时钟/遥测，deployment_ready=false |
| [续训入口](scripts/resume_fast_stop.py) | 新child冻结45训练源+续训入口共46源，另核QA源/证据和严格白名单的3个外部依赖SHA；父模型与Adam复制不变、全局57344→196608；真实profile/launch守卫回归13 passed |
| [恢复子实验](results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2/protocol.json) | protocol SHA `5b1b1c91fc1647a3c84b9cecfa9ad6d37b9fb193c4fc621898020c1807dec2b7`；prepare/preflight真实exit0；从57344权重/Adam恢复并追加139264步至196608；自然完成且真实进程exit 0 |
| [最终检查点与审计](results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2/resume_summary.json) | summary指定 `stage_01_global_random.zip`，SHA `c5ab17ceb785e1abfed0a1bfc292b1ad31912801ea91b6e77f672463a508585b`；独立CPU审计24/24通过，见 `final_checkpoint_cpu.json` / `.log`；attempt1谓词失败JSON/log保留 |
| [首新增模型审计](results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2/first_periodic_checkpoint.json) | 全局65536，ZIP SHA `6879e3f405e71ab910c26d5f80d5d1a103e6cfdd61e96fd003635f8fce06da27`；策略与Adam均相对父57344更新。这是训练中间检查点，不代表final，也不作为能力提升结论 |
| [配对过程量评估runner](scripts/evaluate_fast_stop_resume.py) | 6项测试在Windows与WSL Isaac Lab Python均通过；真实WSL完成gate exit 0，核验child46/parent45源。attempt2评估真实exit 0/completed，rough_a/b各4个paired seed有效、共16行；两split分别no_improvement，合并8个paired seed后clean成功0/8→0/8、entered-radius 0/8→0/8、max_hold 0，无损伤/碰撞，boundary事件3→0。`evaluation_cpu_audit.json` 24项检查通过；attempt1/2审计问题记录保留，原报告未改 |

首新增模型另经[独立CPU审计](results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2/first_checkpoint_cpu.json)实际exit0、19/19检查通过：actor/critic均512×1，独立参数3065801；权重和Adam有限，37/37项Adam step由2173增至2473；父45与子46源码、协议、模型SHA稳定。首次审计读错首模型记录字段而exit1的JSON/log均保留，修正字段映射后只读核验通过，不改训练记录或门槛。

最短CPU回放命令（不启动仿真/GPU）：

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/isaaclab_python.sh scripts/lassc_replay.py \
  --checkpoint results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020/stage_1_global_random/checkpoints/field_policy_global_random_57344_steps.zip \
  --output results/lassc_interface_20261021/manual_replay.json
```

恢复控制组和配对评估均已完成，勿重复执行launch或覆盖评估输出。评估实际argv为 `python scripts/evaluate_fast_stop_resume.py --output results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2_evaluation --device cuda:0`；报告显示rough_a/b各4个paired seed有效，两个split均no_improvement，安全/过程指标见上表。runner严格选择`resume_summary.checkpoint_final`并按SHA核验manifest，拒绝同为196608步的periodic checkpoint。审阅/未来复用恢复命令是 `scripts/resume_fast_stop.py --prepare --gate-report results/lassc_interface_20261021/qa_gate.json`、`--preflight`、`--launch --device cuda:0`；脚本对已存在目录/启动锁拒绝复用。初次prepare因tuple/list的JSON序列化比较失败，失败目录及[真实失败记录](results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021/prepare_failure.json)保留；修复采用已有严格JSON comparator，未降门槛。首次后台dispatch后WSL停止、PID不存在，尚未建启动锁；[记录](results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021_attempt2/dispatch_attempt1.json)保留，后改用持久exec会话。模型首回放空间边界错误保留在replay_attempt1.json，最终直接复用checkpoint spaces。

学习改进方向独立于恢复：既有[mastery入口](scripts/train_isaac_mastery.py)增加兼容旧默认的`--profile fast_stop_512`，课程先零植物/零岩石平地near→turn30→turn60→turn90→range，512×1/gamma1/GAE .95与严格停车固定；沿用连续两轮16回合clean≥.8、零损伤/碰撞、真实旧阶段回放和安全选模。全场rough_a/b固定为迁移开发探针，不等于已经有flat→rough训练课程。最终候选为 [fast_stop512_mastery_20261021_attempt2](results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2/protocol.json)，协议SHA `182b8862f08dc5d92bd7101d8e2a0d0541ab7f01cdc751e760c60c2373519c76`；[独立CPU准备验收](results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2/cpu_prepare_acceptance.json)acceptance_pass=true，47 live/snapshot冻结。该候选已由下述attempt2启动，不再是prepared/未启动状态；旧候选目录保留superseded.json，修复tuple(None)及global_suite控制器标签后重新prepare，不覆盖旧协议/快照。24项runner/selection回归通过，其中真实经过聚合器、controller与checkpoint比较路径；不以CPU准备声称原生课程已运行。新课程初始化、地形和采样安排不同，不能称与恢复组是单因素因果对照。

另有[四项独立CPU验收](results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2/fast_stop_mastery_acceptance_pytest.json)，WSL Ubuntu-24.04 exit 0、4 passed；记录中的mastery runner与测试源SHA均匹配当前文件，证据限于CPU协议/控制器验收，不代表训练或学习效果。恢复控制组和rough_a/b配对评估已真实完成并核实exit 0；两个split各4个paired seed有效，共16行、分别no_improvement；合并8个paired seed后clean成功与进入0.4m半径均0/8→0/8，max_hold为0，无损伤/碰撞，boundary事件3→0。配对评估CPU审计24项通过，attempt1/2审计问题记录保留，原报告未改。课程`baseline_cpu_audit.json/.log`为真实WSL CPU exit 0、21/21通过：0步baseline为512×1、3065801独立参数、gamma1/GAE .95、Adam state entries 0，47源和原观测匹配。run_status记录03:38:10 UTC进入`mastery_probe`/8192；近5632步诊断已见n_updates=50、reward reconcile matched、action finite，首8192步near checkpoint `steps_0008192_near.zip`已保存。其[first_checkpoint_cpu_audit.json/.log](results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2/first_checkpoint_cpu_audit.json)真实WSL CPU exit 0、21/21通过，ZIP SHA `72629971ce9bbb5f85b1a4222aad3a611e0c08d9b88a3092d5d38e63e5f69340`；65个state张量/3023977元素更新（包含alias，不等于独立参数数），37个Adam step均为320且有限，gamma1/GAE .95、512×1/3065801参数及47源、protocol、baseline稳定。此次审计依据完整ZIP和run_status已离开training进入mastery_probe；checkpoint_manifest尚未生成，待16个near探针结束后再写并核验，不能称checkpoint与manifest已匹配或训练已完成。当前16个near探针仍在进行，尚无near能力成绩或晋级。不能据步数或boundary事件变化宣称能力提升；同为196608步的periodic checkpoint可能早于末次PPO更新，不能替代summary指定的final。`scripts/evaluate_fast_stop_resume.py`已验证逐步真实root COM XY速度与完整XYZ角速度，并从before/after轨迹计算min_distance、是否进入0.4m范围及max_hold；830xx继续不观察。

父训练诊断的终点距离与0成功只能描述回合结束状态，不能推断从未进入0.4 m。配对runner把真实逐步root COM XY速度、全XYZ角速度与trajectory/goal_stop按时刻对齐，报告最小距离、是否进入0.4m范围和最长连续hold，再区分未靠近和经过但未停稳；合并两split的8个paired seed后无clean成功或进入目标半径，max_hold为0，故总体评估为no_improvement。boundary事件减少3→0但不改变无成功门槛结论，不得据步数、检查点或该单项变化认定能力提升，也不得仅凭动作均值或critic EV认定GAE根因。830xx继续不观察。



### 课程全场过程采样

98304步global的8个开发seed中，7次60秒超时、1次31.25秒边界终止；报告没有轨迹，不能补造min_distance、entered_radius、max_hold。初始距离20.11–104.05 m、相对方位−32.1°至+162.3°，与near的0.669–1.183 m及约±15°平地输入同时改变了距离、角度和地形，不能把0/8归因于单个因素。按4.1667 m/s履带目标指令上限估算，8个初距的直线时间约4.83–24.97秒，均小于60秒；它只是忽略转向、加速、崎岖地形和制动的名义参考，既不能证明实际可达，也不支持“仅因初距就必然超时”。该参数不是已标定的车体物理速度硬上限；直接履带动作路径不使用`max_turn_rate`来限制真实转向时间。

后继入口[scripts/evaluate_fast_stop_mastery.py](scripts/evaluate_fast_stop_mastery.py)从冻结`transfer_probe.environment_config`构造rough配置，复用已有`evaluate_fast_stop_resume._evaluate_checkpoint`及其逐步root COM trace adapter，不复制环境/采样基础设施。对选定模型的rough_a/b各4个固定seed单独落盘，记录before/after、真实XY合速度、完整XYZ角速度、最小距离、是否进入0.4m、最长hold及终止原因；不覆盖原trial或回填既有成绩，也不将单模型过程诊断称为学习提升。默认`--check`只读，`--launch`需真实完成门禁、无并行仿真且输出目录不存在。命令统一见[使用说明](使用说明.md)“课程全场过程采样”。

当前只准备工具和CPU合成验证，未执行该后继GPU采样。排期先满足用户亲自地图点选预览；用户结束预览且GPU空闲后再串行诊断，不关闭用户正在用的窗口来抢GPU。两自动项保持PAUSED，不以新任务或轮询替代。

本项CPU验收（历史版本）：`tests/test_mastery_process_evaluation.py`与既有`tests/test_fast_stop_resume_evaluation.py`首次合计22 passed；补齐`python -m scripts.*`及`isaac_field`仿真CLI入口识别后，同组回归扩充为31 passed（真实exit 0），对应`py_compile`真实exit 0；`py_compile`与WSL `--check`均真实exit 0。测试覆盖完整rough配置及固定seeds透传、配置不一致拒绝、初态入圈后离开、终止帧入圈、两回合hold不串接、无效真值/缺快照拒绝、未完成/并行仿真/输出覆盖拒绝。并行检查已修复误识别自身bash父进程的问题，放行自身与LM Studio，识别Python训练/评估/预览；另在真实WSL仅调用该检查，预期拒绝PID458正在运行的课程，exit 0。最终`--check`时课程仍running、`launch_allowed=false`，没有创建评估输出。CPU证据不等同原生模型过程成绩；当时第十八批本项计1/5，属历史进度。

### 严格停车地图点选预览（原目录v6历史UI验收）

原目录`runtime/legacy-v6-preview`中的历史UI验收记录：[地图点选预览](scripts/view_fast_stop_goal.py)使用官方Isaac Lab Viser运行真实策略物理预览。原目录当时viewer为WSL PID 415、持久exec 27961，地址[http://127.0.0.1:8783/](http://127.0.0.1:8783/)；旧输出为`results/isaac_field_viewer/fast_stop512_click_goal_follow_20261021_verified/`，截图`results/isaac_field_viewer/fast_stop512_click_goal_follow_20261021_verified/model_identity_vehicle_follow.png`。责任对话当时核验模型`steps_0098304_near.zip`、98304步、actor/critic 512×1及near-best开发候选身份；全图与车辆跟随视角互斥，重复回调问题已修复。真实trial `--check --selection near-best --port 8785` exit 0；27 passed、1 skipped的CPU专项真实exit 0，唯一skip为旧`GOAL_MAP_REAL_TRIAL`集成项。根`8784/api/frame`当时返回running=false、time=0、done=false、vw=0、steps=98304；这只是该次UI验收快照，不保证用户后续操作仍为0，也不表示本worktree做过物理验证。此前用户回合曾运行60秒，旧输出仍保留于`results/isaac_field_viewer/fast_stop512_click_goal_20261021/`。47份live源与47份snapshot共94项hash、protocol与模型不变仅适用于原目录；专用heartbeat与原周期训练automation仍PAUSED。该历史验收时第十八批为3/5；当前为4/5（此前三项加本次绝对目标功能）。同功能回调修复不另计。

历史版本CPU验收证据（保留原成绩与SHA）：[全套CPU回归](results/project_history/fast_stop_objective_20261020/preview_batch17_wsl_cpu_after_attempt2.log)为428 passed、22 skipped、8 warnings、exit 0（在显示缺陷修复前）；[显示修复专项](results/project_history/fast_stop_objective_20261020/preview_batch17_wsl_ui_fix.log)为18 passed、`py_compile`和`--check`均exit 0。此前CPU验收时`--check`选出的near-best是`checkpoints/steps_0016384_near.zip`（16384步，SHA `9d374b52335464ae2873a1654b69f8a437e04d26ce0e541a40de471c1e686de3`，固定near探针10/16）；这不等同final。protocol SHA为`182b8862f08dc5d92bd7101d8e2a0d0541ab7f01cdc751e760c60c2373519c76`，47/47 live源与snapshot匹配。预览脚本SHA `90f16974401cf34efbb0e206e85ec772d701d327a73347a491b27ebb2eb17ed7`，测试SHA `0d0f8f2bad374785bea66b3d279481bdbbb54222b3074c6adef1694910800aab`；MDX显示文案已改为“低于”，地图文件夹设为`order=-10`并置顶。

截至106496步已落盘探针，near最高成绩候选为`checkpoints/steps_0098304_near.zip`（16/16）；实际启动时重新按完整manifest/SHA和固定合同选择，不能等同正式global best。

训练完成后先用CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/isaaclab_python.sh scripts/view_fast_stop_goal.py \
  --check --trial results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2 --selection near-best --port 8785
```

核实真实训练exit 0并确保所用输出目录不存在后，再启动官方预览：

```bash
bash scripts/isaaclab_python.sh scripts/view_fast_stop_goal.py \
  --launch --trial results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2 \
  --port 8783 --output results/isaac_field_viewer/fast_stop512_click_goal_follow_20261021_verified
```

在`http://127.0.0.1:8783/`展开顶部“地图点选目标”目录，点“俯视整张地图”；地图世界坐标范围0–100 m，原点在左下，+Y朝上。保持“启用地图点选”勾选后点击目标位置。程序先检查当前车辆姿态与目标在完整场景中可达；通过后从当前位置保持yaw开启新回合、清零时间和LSTM并暂停。再按“运行任务”开始策略控制。无效或不连通的点击不会改变当前目标。地图点击是预览中的人工目标测试；每个训练reset仍独立随机采样起点、yaw和目标。mastery的固定16-seed探针用于可重复的候选验收：相同seed会复现同一测试初态以比较checkpoint，不会让训练每回合重复同一个起点或目标。两者的结果不混为训练成功率。
