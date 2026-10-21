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

- 第八批第3项：复用Windows CIM性能计数器与现有nvidia-smi，提供有限采样、只读JSON采集脚本 results/system_monitor_diagnosis_20261017/collect_windows_perf.ps1；默认3次，GPU保留原始LUID，不硬编码进程或适配器映射。实际运行后JSON解析通过（583条GPU Engine计数）；原始before/after及验证文件保留。已关闭本任务已断开的Viser预览页，其余用户网页不操作；原Edge GPU PID36908退出，after后两组DWM为1%/4%、ChatGPT为1%/3%（单进程引擎值，不作为GPU总占用）。未安装监控服务或新软件，未重启WSL/驱动；CUDA999仍未解决。

- 当前批次计数：3 / 5。批次8第1项为渐进角度课程runner；第2项为安全评估的损伤/几何/惩罚/驱动/观测契约门禁（允许旧metadata权重在同一新环境下重评）。CPU task+learning回归65 passed，日志`results/isaac_field_diagnosis/stem_contact_v2_20261017/batch8_contract_regression.log`。新trial `fresh_foundation_20261017_width470_progressive`已prepare成功（20个源码文件快照），但exec69866已因Warp CUDA999错误terminal exit1；最后8192步checkpoint已保存。4个训练环境seed69001–69004及safe配置（damage penalty20/budget5/distance cost0）和后续free-space 87001–87024、16株82001–82024评估计划保留，最终评估未开始；820xx为开发集而非盲测。viewer exec18556也已退出（exit0），退出前报Warp CUDA999 `wp_memcpy_d2h`并显示Server stopped；浏览器已断开。退出前UI显示drive mismatch、observation match、damage mismatch。root/physics代理正在诊断GPU driver状态，尚未恢复运行；模型metadata由policy代理CPU审计。批次7已整理（2026-10-17），其五项与整理记录见下文历史记录。
- 第六批验证：CPU92 passed、8 skipped，compileall和两个wrapper语法均exit0。此前原生8项在reset修复后通过，日志 `results/isaac_field_diagnosis/validated_reset_repair/native_env_integration_retry_20261017.log`；这份历史日志使用旧250 mm代理中心距，不替代410 mm几何下的本批原生回归。用户要求不做训练速度/FPS基准。
- 现有后端：仅发展Isaac Lab + Newton MJWarp，复用external/IsaacLab与SB3-Contrib RecurrentPPO。此前旧后端69项批量删除被自动审批blocked by policy拒绝，未执行/未绕过重试；共享运行时、upstream及原实验模型保留。
- 实车事实：30kg、车身0.8×0.3m、15km/h、接地595mm、履带宽60mm、主动轮直径60mm、4:1减速，硬地/农田均能原地转向。用户确认整车总外宽470mm；按每侧履带宽60mm，履带中心距为410mm，车身0.3m宽度不含履带。轨距用于当前代理和安全外廓，不等于完成实车物理标定。转一圈用时暂缺，不反复索取、不称已标定。七对45mm代理滚轮区别于30mm主动轮半径；gain0.5/每轮2Nm为数值代理参数。云台0.4m高、俯仰−25°..60°，雷达随云台俯仰，8m/10Hz/400束/360°；深度88×160，preview与policy同输入。
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
- 第七批第3和第5项几何更新：用户确认总外宽470mm与单侧履带宽60mm，因此当前代理履带中心距410mm、整车安全外廓宽470mm；车身盒体仍为300mm宽，接地长度595mm，滚轮代理半径45mm。`Config.robot_width`新默认使用470mm安全外廓；旧序列化配置保存的300mm值保持原样以复现旧场景。资产规格摘要会刷新对应USD；不同几何下的策略/物理结果不得混作同一配置。
- 第七批第4项 fresh foundation 课程runner的旧配置记录：协议`results/isaac_field_improvement/fresh_foundation_20261017_stem_v2`已prepare并冻结快照。旧250mm轨距配置因用户确认470mm总外宽而被有意中断，训练exec15841退出码2、viewer进程94767退出码1（SIGINT/KeyboardInterrupt），不是数值训练崩溃；最后稳定检查点为turn-stage总73,728步。该旧配置的final held-out评估未运行；中断记录、原protocol、快照、权重和run.log均保留。当前新trial及预览状态见本文件开头，不续用旧几何检查点。滚动训练窗口不是held-out结果。
- 第七批统一回归记录于 `results/project_history/cleanup_and_validation_20261017_batch7.md/.log`：CPU 101 passed、8 skipped；compileall与两个wrapper语法检查exit0。root另在470 mm外宽/410 mm中心距下原生运行8项并全部通过（49 upstream warnings），日志 `results/isaac_field_diagnosis/stem_contact_v2_20261017/native_width470_integration.log`。未开展速度/FPS基准。
- 模型成本审计：只读源码审计`results/isaac_field_diagnosis/plant_body_cost_audit_20261017.md`没有证据表明当前固定植物root每株增加6个q/qd坐标；仍有刚体/位姿、mocap、几何与BVH条目，实际内存和solver成本未测。当前训练代理继续使用车身盒体、14个圆柱滚轮和双轴云台，不导入高精度CAD；暂不基于未证实的坐标节省猜测重写bodyless植物路径。
- Viewer历史验收：exec5626/WSL PID9807与旧几何exec94767已由root有意SIGINT关闭；后者是在410 mm轨距代码完成前运行的旧250 mm几何会话。exec18556曾在410 mm中心距/470 mm总外宽下加载旧73,728步权重并暂停于t=0，UI当时显示drive mismatch、observation match、damage mismatch；该viewer现exit0，Warp CUDA999 `wp_memcpy_d2h`后Server stopped，浏览器已断开。记录见`results/isaac_field_viewer/width470_seed82001_parent73728/`，仅作历史状态，不能作为当前可用预览或新几何策略验收。
- Edge历史PID29788曾验证独显；当前GPU PID36908不同，不能声称当前已独显渲染。内嵌三维页保持关闭，不擅停用户其它进程。
- 完整搜索、粗糙地形/石块、对抗训练及整田可靠导航尚未达成。更多历史与现状见ISAAC_LAB.md、使用说明.md和原始结果；不以局部测试或训练跑通标记总体目标完成。
