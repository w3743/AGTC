# 项目协作规则

## 复用与协作

- 完成任务前先查找项目已有实现及现成、开源解决方案，以最小复杂度完成任务；优先复用训练库、仿真器和标准接口，避免另写同类基础设施。
- 查找、机械性修改、计算、测试、验证等低推理需求任务，尽可能交给子代理，使用低额度模型 `gpt-6-luna`、`max` 推理档。新建此类子代理使用 `fork_turns="none"`，提供明确范围和必要上下文。
- 勤于向用户提问，优先确认会影响目标、架构或验收的关键问题；已明确的要求直接推进。

## 每五项改动整理一次

- 每完成 **5项独立的功能改动或问题修复**，必须先对项目进行一次整理，再开始下一项改动。
- 同一功能涉及的实现、测试与文档算一项；不得按文件或代码行重复计数，也不得把无关功能合并为一项来绕过整理。
- 整理包括：清理确认无用或重复的实现，统一受影响的接口、命名与目录组织，同步说明文档和运行命令，并执行必要的测试、静态检查与验证。
- 性能相关结论保持相同测量口径；原始实验记录与需要的模型保留，整理不得覆盖或改写实验成绩。
- 整理围绕已修改范围开展，不扩大成无关重构。
- 主代理统一记录改动计数和最近整理结果；子代理报告独立改动，避免并行时重复记数。更新计数记录本身不另算一项。

## 改动与整理记录

- 当前新批次计数：4 / 5。第七批前四项：GAE lambda 单变量 runner；用户确认的 stem-contact-v2 损伤规则与 legacy v1/元数据；60 mm履带宽代理和整车安全外廓；复用 train_curriculum 的 fresh foundation 课程入口。第五项完成后、开始第六项前整理。最近整理为2026-10-17第六批；备份 `results/project_history/AGENTS_before_20261017_batch6_cleanup.md`，记录及日志 `cleanup_and_validation_20261017_batch6.md/.log`。
- 第六批验证：CPU92 passed、8 skipped，compileall和两个wrapper语法均exit0。原生8项此前在reset修复后通过exit0，日志 results/isaac_field_diagnosis/validated_reset_repair/native_env_integration_retry_20261017.log；后续仅学习/评测/探索改动，未重复原生物理套件。用户要求不做训练速度/FPS基准。
- 现有后端：仅发展Isaac Lab + Newton MJWarp，复用external/IsaacLab与SB3-Contrib RecurrentPPO。此前旧后端69项批量删除被自动审批blocked by policy拒绝，未执行/未绕过重试；共享运行时、upstream及原实验模型保留。
- 实车事实：30kg、车身0.8×0.3m、15km/h、接地595mm、履带宽60mm、主动轮直径60mm、4:1减速，硬地/农田均能原地转向。0.3m仅为车身宽度；履带中心距现用0.25m代理但未获实车确认，当前整车安全外廓按max(0.30,0.25+0.06)=0.31m。转一圈用时暂缺，不反复索取、不称已标定。七对45mm代理滚轮区别于30mm主动轮半径；gain0.5/每轮2Nm为数值代理参数。云台0.4m高、俯仰−25°..60°，雷达随云台俯仰，8m/10Hz/400束/360°；深度88×160，preview与policy同输入。
- 用户要求：模型自己学，连续四维动作，无Nav2/手写规划器；优先不压苗，允许更慢或超时。到点只要求0.4m位置范围并低速稳定hold0.3s，不要求车身/云台终态朝向。
- 用户最新建模方向：担心高精度机械模型消耗性能；继续车身盒体/七对圆柱滚轮/双轴云台的轻量训练代理，无需用户额外重画。现有机械/车.SLDASM、炮台.SLDASM可作尺寸/轴心参考，尚未打开CAD核对其内部零件；优先真实轮心/轮距、质量/质心/惯量和传感器外参，不把外观精细程度当作物理标定。当前训练没有导入这些高精度CAD。
- reset修复已冻结：最终start/yaw/goal先采样再完整校验，推进RNG重试并检查grid anchor、实际车身footprint；不再验证后覆盖端点。源码/CPU/原生证据见validated_reset_repair目录。
- 感知已核查：Newton BVH含VISIBLE和COLLIDE_SHAPES，叶片仅VISIBLE仍可测距；近株1.5m时LiDAR及单点测距均1.488780m。“植物透明/射线只查碰撞”是已纠正误判，不再传播。历史准确seed80003最近株31.54m导致空命中合理，不是植物位姿失效。
- 原safe_goto_20261017_reward_v1为真实exit1，110592保存点、无after；父日志/协议/18文件快照保留。独立resume1从该权重与Adam续8192至118784，真实exit0；未保存环境RNG/中途LSTM rollout，不称bitwise恢复。24题严格配对：原53248 clean5、damaged4、mean damage0.25；候选118784 clean8、damaged5、mean0.3333，均0collision，assessment=success_safety_tradeoff，不通过用户不压苗优先的验收。
- 转向诊断：recovered_baseline_trajectory_analysis.json中12个正bearing样本差速均值全负；recovered_after_trajectory_analysis.json中候选同组没有正差速动作。负bearing的四个旧开发样本不能证明双向条件化正常；不能把错误方向命令只归因于机械死区。
- 新对照：results/isaac_field_improvement/exploration_gsde_20261017，19文件快照/完整Config/协议已冻结，exec97640真实exit0，24组初态配对通过。原53248同源actor/critic/LSTM权重，两组新Adam，各16384新步到69632；native校准seed78001的mean latent norm²=7.260320408，不能使用dummy1.601接口检查值。gSDE每8逻辑步重采样noise matrix，非固定动作8步。训练seed68001、4env；独立评测82001..82024、envseed80003，flat16、20s、goal1..3m ±60°、damage20/budget5/cost0。新24例baseline clean1/damaged3/meanDamage0.3333；Gaussian clean2/damaged4/mean0.5833；gSDE clean1/damaged5/mean0.6667，均0collision。Gaussian为success_safety_tradeoff，gSDE相对baseline/control均no_improvement；不选作能力升级，不可与810xx数据混算。
- 后续诊断方向：三个directionality*.json显示baseline全局负转、Gaussian/gSDE都全局正转，条件化未解决。GAE lambda从.95单变量调至.99的配对实验已完成，但结果仍是安全与到达数权衡，不能据此认定长时信用分配是唯一原因；critic bootstrap仍可传播长期信息。避免继续盲增CAD精度或网络复杂度。
- 新原生/CPU证据：goal_conditioning_20261017中固定原生车姿态与传感器，7个2m目标方位只有state[2:4]变化，编码误差7.49e-10；三权重goal_gain100生效，目标到actor梯度非零，未发现目标通道断开或训练时使用陈旧cache。±90°目标的首步差速均同号，Gaussian全7方位跨度仅0.00390；64次静态重复输入仅为神经反事实，不能当实际运动/记忆失败。reward_motion_audit_20261017.json记录首秒确有前进，后续速度及转向低；没有满足明显原地转向阈值的2秒片段。缺训练总episode日志则不猜总数。
- 第七批第1项 GAE 对照已完成：`results/isaac_field_improvement/gae_credit_20261017`，实际exec93757 exit0，24/24评测初态配对。源53248权重新建Adam，Gaussian同口径续训16,384步，唯一算法变量为model与rollout_buffer的GAE lambda .95→.99；seed68001、4 env、safe Config/hyperparameters不变。已观察开发集82001–82024上，baseline为clean1/damaged3/meanDamage0.3333，Gaussian control为clean2/damaged4/mean0.5833，GAE .99为clean3/damaged5/mean0.5416667，均0 collision。对baseline与Gaussian均为`success_safety_tradeoff`，按不压苗优先不通过，不能称能力提升或长时信用问题已解决；若继续验证需另用830xx未观察种子。原始结果、检查点与协议均保留在该目录。
- 第七批第2项损伤规则：用户确认只有履带压到茎基或车身碰到主茎才损伤；叶片擦碰及旧Scene.radius保护圈不算。新配置默认`stem-contact-v2`：履带接地带按595×60mm扫掠，须与`root_z±stem_radius`垂直相交；车身按当前root Z和pitch/roll扩张的垂直AABB检查主茎高度，水平外廓保持简化2.5D且扫掠为离散近似。`root-protection-v1`保留旧规则，旧配置缺metric时checkpoint解析回落v1。episode info、checkpoint/eval damage metadata记录metric与几何/每株惩罚。
- 第七批第3项几何校正：用户确认每侧履带宽60mm；代理滚轮半径仍45mm。车身宽300mm不含履带；轨距当前0.25m仍是未确认代理，因此`Config.robot_width`新默认整车安全外廓0.31m，旧checkpoint存有的300mm值原样保留。资产规格哈希随60mm滚轮宽度变化，旧生成USD会按规格摘要刷新；旧训练/评估分数不与stem-v2混比。
- 第七批第4项 fresh foundation 课程runner：新协议`results/isaac_field_improvement/fresh_foundation_20261017_stem_v2`已`--prepare` exit0并冻结20个文件；训练exec15841仍运行。near阶段32,768步完成，最近10个训练episode回调在8k为9/10、16k/24k/32k均10/10；turn ±90°阶段使用seed69002，49,152步最近10个为0/10，57,344步为2/10且damage=0。以上是空地转向训练中的滚动窗口，不是held-out评估。后续评估free870xx，并按stem-v2同口径重新评估baseline/after的16株820xx（不复用旧报告）；820xx为已观察开发集，不是盲测，尚无能力结论。
- 第七批统一回归：WSL `pytest tests -q`真实exit0，101 passed、8 skipped；日志`results/isaac_field_diagnosis/stem_contact_v2_20261017/cpu_regression.log`。未额外运行8项native integration套件或速度基准；后续课程训练与viewer已在native cuda:0实际运行。
- 模型成本审计：只读源码审计`results/isaac_field_diagnosis/plant_body_cost_audit_20261017.md`没有证据表明当前固定植物root每株增加6个q/qd坐标；仍有刚体/位姿、mocap、几何与BVH条目，实际内存和solver成本未测。当前训练代理继续使用车身盒体、14个圆柱滚轮和双轴云台，不导入高精度CAD；暂不基于未证实的坐标节省猜测重写bodyless植物路径。
- Viewer：旧exec5626/WSL PID9807已由root确认后SIGINT主动关闭（exit1/KeyboardInterrupt，非训练失败）。新viewer exec94767在8768使用native cuda:0、stem-contact-v2、flat16/20s、envseed80003/episodeseed82001、protect_plants和damage20/budget5。8192与32768步权重均核对field_drive_config、field_observation_config、field_damage_config三项match；当前UI保留32768步near终态：20s timeout、distance2.02m、damage0/collision0，UI四舍五入显示终态x55.49/y3.28/yaw1.42/v0.06/w0.02。Reset复用同一82001初态（start54.81/1.58/yaw1.15，goal53.64/4.09）。详情和截图见`results/isaac_field_improvement/fresh_foundation_20261017_stem_v2/viewer_inspection.json`及同目录`viewer_seed82001_*_terminal.png`；非全轨迹逐位对照或held-out能力评估，UI尚未加载转向stage checkpoint。
- Edge历史PID29788曾验证独显；当前GPU PID36908不同，不能声称当前已独显渲染。内嵌三维页保持关闭，不擅停用户其它进程。
- 完整搜索、粗糙地形/石块、对抗训练及整田可靠导航尚未达成。更多历史与现状见ISAAC_LAB.md、使用说明.md和原始结果；不以局部测试或训练跑通标记总体目标完成。
