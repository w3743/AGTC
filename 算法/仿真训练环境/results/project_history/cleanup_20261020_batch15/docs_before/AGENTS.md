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

## 当前批次与运行状态

- 第十五批训练方法方案已收口4项：①原奖励分项/四维动作/PPO诊断，②独立探针连续达标晋级与实际旧任务/植物采样，③固定开发suite逐类安全选模，④独立冻结对照入口与显式GAE参数。当前4/5；另一聊天“归档Isaac项目遗留文件”的random_rough、被动前导向轮、稳定目标/统计3项尚未由其收口，不在此重复计数。其首项收口到5时应先进行第十五批整理，再继续后两项；最近正式整理仍为第十四批，本次专项验证不是下一批整理。
- 本聊天已完成env.py/learning.py写入，保留对方新增body pose、真实body角速度、body_stability_summary及termination_detail可选记录。env.py/newton_drive.py、fresh runner与导向轮/稳定性native交对方接管；本聊天不再写这三个共享Python文件，不再启GPU验证或正式训练。主文档由本聊天先完成追加，对方其后追加其结果；不回退双方改动。
- 本次最终CPU真实exit0：325 passed/16 skipped/8 warnings，compileall0、独立启动器bash-n0，日志/实际退出在 `results/project_history/training_method_solutions_20261020/`。真实Newton专项1 passed/37上游warnings、exit0，仅2个policy步核SB3 fast wrapper与auto-reset终止奖励账目，不是完整14项native或学习验收，发生在跨聊天GPU协调请求前。此前77专项、324全CPU、代理测试各失败/修复记录均保留。
- 新入口 `scripts/train_isaac_mastery.py`/`scripts/run_isaac_mastery.sh` 复用v7/平地/60秒/原奖励/新初始化策略，8阶段依次改变角度、距离、plants0→2→8→16；16seed课程探针连续2次clean≥.8且零损伤/碰撞才晋级，旧任务采新on-policy轨迹；全局32开发seed固定8类各4例，每类不退步且全局clean严格增才选best，latest与best分别记录。合同、seed及实际初态严格配对，83000–83999未用。已prepare `method_control_v2_20261020`、`method_mastery_v2_20261020`、`method_lambda099_v2_20261020`（fixed/mastery同设置，第三组仅λ=.99），各冻结47源，当时preflight真实通过；未启动训练/初始化模型，实际初始权重相等须启动后核policy_tensor_sha256。
- 首次 `method_control_20261020` prepare后因RobotAssetSpec tuple与JSON list比较差异预检exit1，旧协议及失败记录保留；规范化修复/新增回归后另建v2，没有覆盖旧trial。v2准备时协议控制一致，旧父/续训协议及两final SHA仍匹配历史，见prepared_comparisons.json/final_audit.json。后续共享物理源改动会使v2预检拒绝；双方收口后须另建实验目录重新prepare，不能改写旧协议或先称三组现可直接启动。
- 本次没有新学习提升结论，未新增奖励塑形、未放松到点/损伤标准；非对称critic、128→256 rollout、纯延时且保持原时间比例输入的回放仍为后续独立方案，不声称已执行。原生automation继续保持用户要求的PAUSED。
- 用户最新明确要求关闭定时任务。原生automation `automation`（跟进履带联动模型训练与验证）已由automation_update确认PAUSED；不再自动定时跟进，不自行恢复或另建替代任务。仅关闭定时检查，未终止已启动的训练/评估进程；下文ACTIVE及“暂不暂停”均为此前时点记录，不覆盖本条最新状态。
- 2026-10-20 15:18用户问能否到点，已人工核实60秒续训在14:44:06北京时间真实完成393216步/29检查点，process_exit.json及python.exit_code均0、原Python/launcher实际已结束；不是仅看status推断。final SHA仍 `9777af6addd30427026675c0fb9d577c555fd288294bc6129a80b53b1d633a70`。同60秒/同初态开发配对均valid4：free父final1/4→新final1/4，field1/4→0/4；两组损伤episode/平均损伤/碰撞均0，均no_improvement，不能称这轮能力提升。新final free 2超时+1 outside_or_overturned，field2超时+2 outside_or_overturned（组合原因尚未拆分，不武断称全部翻车）。free唯一成功seed87004用10.1s，末距.360113m、hold.3s、v−.159622m/s，满足当前宽松到点定义，不证明精确静止停车。旧20s成绩不混算。
- 同次人工核验已执行新trial `final_cpu_validation/run_final_cpu_validation.sh`，Ubuntu-24.04真实CPU退出0，全12检查true：393216/43源、policy与Adam有限、CPU-only/CUDA0、父baseline及final/manifest哈希稳定、v7合同一致；37项Adam step均14806（父7607），65张量/537107元素较父final更新，12张量未变为plants0检测分支，不称全分支学习。报告/真实退出在该目录，report SHA `28551ec9e0ff784c6b50016bab3725a9bb9c3720af528c70485c62c29644f8f1`。最新仍未部署新final到viewer；8769/8770此前父模型/20s查看器状态须实查，不冒称其已显示393216。定时任务保持PAUSED，未另启训练或评估，计数仍0/5。
- 2026-10-20用户提出超时是否过短，已只读审计新coupled final两组8条开发轨迹：时限20s、dt.05/400步，6条超时都未曾进入.4m、hold0，末5s和末10s目标距离全部增加且仍有运动/履带指令；2条成功分别1.85s/5.05s。因此不支持“只差多给时间”，不外推20秒后必失败。实际v/w有记录，但无逐履带实测反馈，不冒称轮速。证据 `results/isaac_field_diagnosis/timeout_audit_20261020/`，原协议/报告未改，未新增仿真/训练，功能计数仍4/5。注意state含episode_time_ratio=min(1,time/max_time)，直接改40/60会同时改变策略输入；纯延时对照须保留原20s归一化输入，单独放宽终止上限，再做真实同模型/同初态回放；本轮尚未执行延长回放。
- 最新计数：4 / 5（第十五批，本聊天四项训练方法改动；并行物理任务其后收口时继续全项目计数）。第十四批5项已完成并整理：移除连线亮暗、单张独立色标、精简二维介绍、删除辅助侧栏及复审两页文案、新增独立续训入口。整理全CPU255 passed/14 skipped/8 warnings、compileall0、43个bash检查0、Node内联脚本语法0；68项源码/测试哈希仅授权更新的旧侧栏断言改变，其余67一致。首轮旧断言失败及错误捕获0保留；attempt2 pytest实际0，外层因Windows反斜杠哈希路径错误实际1，独立规范路径hash恢复实际0，不混称全外层0。详见 `results/project_history/cleanup_and_validation_20261020_batch14.md`。第十三批物理整理旧状态见 `results/project_history/batch13_before_coupled_tracks_20261020/AGENTS.md` 和 `cleanup_and_validation_20261020_batch13.md`。
- 60秒续训已于2026-10-20 13:49北京时间正式独立启动：`results/isaac_field_improvement/continue_coupled_tracks_20261020/`，保留父final196608权重与Adam，再训练196608至累计393216，四阶段/4env/seed69005–69008，其他物理/输入/策略/奖励不变。协议SHA `fe54ce6233e10705a9bbc393784eeca8cbe4d459ac467af1b2fd3656fb4baa18`，43源码及快照冻结；`parent_final_baseline.zip`与父final同字节SHA `6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5`。prepare/preflight真实0；启动WSL Python328014、launcher328009、exec84372仅作启动记录，后续须重查argv/状态/日志/退出，不据历史PID重启。initialization.json已真实passed：196608步、77权重张量逐项与父相等、Adam111张量相等，37项step均7607且有限；环境/RNG/回合LSTM/rollout重新初始化，不称逐位恢复。每新增8192步保存检查点，首204800；60秒同时改变episode_time_ratio，最终父/子均按同60s配置重新开发配对，不能混旧20s成绩。当前运行中，尚无完成/提升结论。原生automation `automation`已更新为本轮任务并确认ACTIVE，10分钟跟进；仅重要变化通知，完成后暂停。当前旧8769/8770 viewer仍为父模型/20秒配置，后续新final部署须切60秒并验收。
- 用户询问虚拟时间加速，已只读核训练入口无墙钟sleep/限速，env每policy步推进50×1ms=.05s；4env并行，未添加实时viewer，实际仿真倍率未测/不保证。独立viewer循环按cfg.dt补足墙钟属于交互路径，不在训练循环；未为加速改变物理dt或训练配置，未做FPS/倍率基准，也不把未显式设置headless冒称完全无渲染。
- 本轮startup审计真实exit0，43/43当前源及快照、父baseline哈希、60秒/新seeds、live恢复状态与实际Python argv均符合，见新trial的 `startup_audit/`。13:52已实际生成首个累计204800步检查点（新增8192），ZIP data步数及policy/Adam成员存在、读取前后SHA稳定已独立核验，见 `startup_audit/first_checkpoint_metadata.json`；这次仅ZIP元数据审计，不冒称额外CPU模型有限性验证或已完成训练。后续最终有限性与学习效果仍待验收。
- 14:00/14:10/14:20的只读heartbeat审计见新trial的 `heartbeat_checks/`，最新各时点分别245760/286720/327680；14:00首审计曾将早一时点步数硬编码导致自身exit1，保留attempt1，动态读取后实际exit0；这不是训练失败。最终CPU验收脚本已复用父验证器、经root复审准备在新trial `final_cpu_validation/`，仅两个脚本，尚未运行；须先确认训练及自动评估真实结束/exit0，再用Ubuntu-24.04执行 `bash results/isaac_field_improvement/continue_coupled_tracks_20261020/final_cpu_validation/run_final_cpu_validation.sh`。它核393216/43源、父baseline196608且37项Adam7607、final全部Adam步数大于7607、动态manifest final SHA、CPU-only有限性与实际更新；拒绝覆盖既有验证输出，不套旧父固定SHA/步数。此为既有验收准备，不新增独立功能，计数仍0/5。
- 14:40核查：四阶段已达到393216，最终 `stage_04_turn90.zip` SHA `9777af6addd30427026675c0fb9d577c555fd288294bc6129a80b53b1d633a70` 与manifest一致且稳定，43源匹配。真实Python328014仍在 `evaluating_dev_field`，尚无process_exit/python.exit_code，不判整体完成或重复启动；free两报告已写、field baseline已写，最终field/比较仍待完成。审计见 `heartbeat_checks/20261020T1440_0800/`；其首份误选stage子目录ZIP与manifest定位不一致导致自身exit1保留attempt1，根final核对后审计实际0，不是训练失败。root已审阅最终CPU验证器；新60秒viewer启动包装已准备并bash-n0：`results/isaac_field_viewer/continued_coupled_tracks_20261020/run_viewer.sh`，仅结果目录新增，尚未执行。必须完成真实退出/CPU/配对后，核旧viewer实际argv再停止并启动新viewer，然后GUI验证/原始激活对照；暂不暂停automation。
- 第十四批第4项：按用户截图整块移除左侧网络结构/策略流程/01–05/辅助诊断目录及底部支路图例，不移到别处；清除桌面/移动CSS、抽屉/导航事件、REGION_BRANCHES/focusRegion等专属死代码，保留实际SVG分区/层/数据。详情空标题和初始徽标去重复，色标tooltip删实现术语。车辆页删除品牌/渲染介绍，缩短初始/重置后的重复按钮教学，控制状态机不变。Node语法0、54字面DOM引用存在，viewer CPU专项25 passed/py_compile0；第三方说明同步当前剩余复用范围，许可证保留。
- 第4项实际部署验收：旧viewer PID323976核实命令后停止，新viewer WSL PID326346/exec98232，final196608、暂停0秒。二维侧栏数0、截图导航文字不再出现，画布x226→0、宽1055.333→1281.333px（同视口），60卡片71线且所有主图张量图片/全局范围逐项不变；详情深度local±.952539正常。网络console[]；车辆页无错误，1条第三方THREE.Clock弃用warn保留。车辆介绍已实际检查移除、控件/兼容性信息存在。证据/截图 `results/isaac_field_viewer/remove_sidebar_20261020/`。本轮当前源相对42训练快照的预期差异为viewer.py、network_2d.py/html、THIRD_PARTY_NOTICES.md；旧快照/协议/模型/成绩不得改写。
- 第十四批第3项验收：仅精简network_2d.html的副标题、顶部流程介绍、导航说明/副文案、重复预览说明及相应空白；状态徽标改为“60个节点”，移除Graphviz/交叉数等实现介绍。必要控件/aria/title、张量来源/说明/统计、色标、模型和运行状态保留。Node语法0，62处字面DOM引用存在；真实页面60卡片/71线、全部主图张量图片与色标不变、详情单图范围正常、console warn/error[]。1280视口header高度130.333→111.667px，无后端重启，模型仍final196608且暂停0秒。证据及截图 `results/isaac_field_viewer/trim_interface_copy_20261020/`；本次未改模型/协议/快照或运行训练。
- 第十四批验收：HTML删除flow-edge/crossing-bridge的related/muted样式与JS切换，71条真实曲线/箭头及分支颜色保留，切层前后线样式完全相同。详情默认勾选“单图自适应色标”，按当前激活/weight/bias完整有限值±maxabs计算，0同色、全零/非有限fallback±1；可取消勾选使用全局范围。仅detail.image/detail.legend独立，主图nodes/global legend及平滑历史保持。API detail_scale=local|global、默认global兼容，拒绝其他值；切层/张量/模式时防旧响应覆盖。CPU专项17 passed/1既有NumPy warning、compileall与Node语法0；证据 `results/isaac_field_viewer/detail_local_scale_20261020/`，第1项独立证据 `edge_highlight_removed_20261020/`。
- 本轮新部署：仅停止核实的旧本项目viewer PID298144，再启动final196608模型的新viewer，WSL PID323976/exec19150（后续必须重查），8769/8770正常，当前暂停0秒。浏览器实际验证depth local±.952539、weight±1.04657、bias±.0202728；global始终±2.27487，local/global切换主图60张图片完全相同，独立预览图片改变；71条连线无高亮class，console warn/error[]。截图及browser_acceptance.json已存。旧协议/baseline/final与42快照未变，当前训练源仅network_2d.py/html有本轮预期差异，不再声称当前42源全匹配旧训练。
- 用户询问“所有层是否展示”已只读核对：详细页展示全部60监控节点，当前可学习模块未发现漏画；不等于60神经层，含输入/融合/检测公式卡片、LSTM门值重建核验与critic旁路诊断。函数级reshape等不各占卡片，传感器生成不属策略网络图。
- 用户最新授权：自主继续查清转向弱，先修仿真车辆运动和接触，再重新训练验证。用户提出“可能摩擦太小”；不再为已授权的实现、验证或新训练重复确认，不自行复用旧权重。
- 当前已验收正式源码：drive `independent_left_right_track_speed_coupled_pi_v7`，默认每侧轮0带6从轮，12条Newton公开mimic/equality关系/环境；只在构建期注册public world hook，按joint_world隔离，无跨环境连线。1ms物理、decimation/render_interval50、policy周期50ms、elliptic cone/impratio1；原Newton接触源、mu.8/KF1000/几何/P.04/Ki.1/14轮各2Nm限幅不变。`newton_drive.py`缓存canonical实际材料/机制metadata，不能带随num_envs变化的总数进drive合同。
- 输入/策略保持：`sensor-minimal-v6`、`depth64-lidar-single1d-minimal-v3`，LiDAR1×400、depth1×160×88、detections16×8、state33、feedback37，总14678；五路融合192→LSTM128→动作4，`direct-track-speed-v1`为左/右履带目标速度及云台偏航/俯仰速率。无目标100倍增益、无定位噪声常数输入、深度单通道且无效0，磁场保留，温度/Euler等重复输入删除。
- 本轮实现/回归已收口：B env/helper/native，C fresh runner/fixtures（专项55 passed），A全CPU。root已审阅并整理文档/接口；42个当前训练源已生成source_audit，之后新训练冻结期间不得改。B首次落地部分成功、native测试缺reset生命周期、单测CRLF包装的失败记录均保留；最终全14 native使用正确LF runner并全部真实exit0。测试按官方reset flush/compute_observations→env.step检查，无生产reset绕过或修改。
- 诊断证据根目录：`results/isaac_field_diagnosis/steering_contact_repair_20261020/`，README有可靠结论和各原始路径。mu.4/.8/1.2同5ms八动作均真实exit0，增mu转向反而更弱，1.2高差速触2Nm cap；低mu制动回摆稍大，不据此选生产mu或称实车标定。
- 同侧联动1ms最终八动作配对已完成：baseline/mimic各64000物理行、1280policy，真实exit0，预置同初态hash `7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af`，当时41源冻结一致、唯一hook差异；mimic8/8残差及原运动/制动门槛通过，qd差最坏p95 .037625/max .129288rad/s，policy q差max .000519rad；实际effort max1.29885Nm≤2。低pivot末1s yaw +.0704→+.1813/−.0725→−.1641，高pivot−.3795→−.6139/+.3854→+.6422rad/s；制动末max|v|.0007675/|w|.003965，反向回摆积分仍约.0615m。仅固定动作物理响应证据，不是导航提升。详见 `staged_response/same_side_mimic_motion_1ms_v1/paired_comparison.*`。
- 步长按原门槛选择：2ms strong mimic yaw−.171675；1ms−.192453，差.020778>.019245故拒绝2ms、不放宽阈值；.5ms−.184484，1vs.5差.007969≤max(.01,10% finer)=.018448，1ms作为候选。仅该动作yaw收敛，不宣称所有物理量已收敛。v2 solref(.004,1)/solimp(.999,.9999,.001,.5,2)，所有实际12条关系及系数/参数经过Newton/host/MJWarp核查。两world有24条Newton关系，但solver neq12是每world模板拓扑，MJWarp nworld2，不得错判应neq24。
- .5ms物理完整18000子步/180policy，但外层后处理baseline路径错误真实exit1；独立CPU sidecar恢复验收真实exit0，不重跑物理、不混称外层0。summary曾追加/移除acceptance后重序列化，缺事前字节副本，不能称原字节一致；raw/物理数值/失败退出保留。1ms首attempt reset前NameError后新attempt2真实0；其他失败诊断attempt各保留。
- 其他隔离未证明根因：真实轮圆柱轴与铰轴Y一致、22DOF damping/frictionloss/armature0；承载contact有纵向滑移、摩擦饱和，两个切向D相同。只取4快照的末单侧承载不能称持续脱地。KF10000/cone/impratio/解析plane均未消除转向弱，未生产采用；原mesh不支持native MuJoCo contacts的失败保留。EFC vel/force在solve积分前，不能与积分后qvel/pose混配判断力方向。
- 最新完成历史训练：`fresh_side_pi_20261020` 在旧PI-v6/5ms/独立滚轮下完成196608步/29检查点，2026-10-20 08:43:35北京时间真实exit0。protocol SHA `0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8`，baseline SHA `bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129`，final SHA `f5be877d728037436bcfcca9b65df462de41dd1846f760473374f287f5a40735`。free/field成功均0/4→0/4、损伤碰撞0，末距1.692859→1.412609及1.840281→1.520151m，均no_improvement。旧41快照保留；当前源码已改后不再声称live与旧快照一致。
- 新实验 `results/isaac_field_improvement/fresh_coupled_tracks_20261020/` 于2026-10-20北京时间11:44独立启动、12:35:58完成196608步/29检查点及自动开发配对，process_exit.json/python.exit_code和status均确认真实exit0，实际Python进程已结束。新初始化、4env、四阶段，不续旧权重。协议SHA `2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6`；正式0步baseline SHA `beda8fde92236b558433a148245094caeefb88d7c23b9ea73ed94c4cd596f1d5`；final stage_04_turn90.zip SHA `6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5`。42源/快照、依赖和资产清单保留并核对匹配；模型/协议/原始成绩不改写，旧PID297751/297740仅作历史。
- 最终CPU-only验证真实exit0：RecurrentPPO以device cpu加载，CUDA unavailable/0 devices，final196608步、所有权重及Adam有限，37项Adam step均7607。与正式0baseline键/shape一致，65张量/593206元素更新；12个重复extractor的boxes检测分支权重未更新（基础训练plants0），不声称学会检测。报告/脚本/LF runner/真实退出保存在trial `final_cpu_validation/`；这不是额外训练或评估。
- 最终开发配对：free87001–4和field82001–4各4组，合同/初态/目标及轨迹起点逐一匹配，comparison均valid且符合预声明安全判据。free clean成功0/4→1/4、平均末距1.685283142→1.523644775m；field0/4→1/4、1.835561316→1.801870071m；两组受损episode/平均损伤/碰撞均0→0、各3/4仍超时。仅小样本开发改进，非盲测或整田能力、不证明联动单一因果；通用24-seed helper标志false，本trial明确4seed，不能混同两者。最终核查 `heartbeat_checks/20261020T123707_0800/final_pair_audit.json`；12:36 free_pair_check是写入期过渡快照，最终以12:37完整核查及原报告为准。830xx未观察，未做FPS。
- 首个8192步checkpoint SHA `f0f4c58ff912bcf7b6c986b1e402b5cf7f3ce7d9ae767a2fb179814c0b91dffe`，CPU真实exit0验证所有值有限、Adam37项step300、65张量/524193元素较正式0步改变，协议及42源仍匹配。plants0使检测分支未更新，不称所有权重均学习；见trial内first_checkpoint_cpu_validation.json及退出记录。
- 首8192步Viewer验收历史：核对旧PID198426命令后仅停止该viewer，使用新正式baseline重启WSL PID298144/exec88604，8769车辆/8770二维HTTP200；曾GUI实际载入8192步checkpoint。运行前seq2/0秒、运行后seq349/输入17.30秒/输出17.35秒，60卡片0重叠、viewBox固定、console errors[]；原值60/60有限，50改变/10不变，77/77参数相同。证据及截图 `results/isaac_field_viewer/coupled_tracks_20261020/runtime_8192/` 保留；当前已改为以下final模型，不沿用此历史模型标签。
- Final Viewer已GUI实际载入stage_04_turn90.zip，显示196608步、drive/observation/action及损伤合同一致；reset到0秒后实际策略运行并暂停6.50秒。帧seq2/input0→seq132/input6.45/output6.50；车辆(48.25,21.28,yaw−2.32)→(47.18,20.11,−2.43)。前后60卡片0重叠、viewBox精确保持 `-55 -370.655 4723 2642.311`，两页console warn/error[]；最终截图、browser_acceptance、API帧和raw验收见 `results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608/`。这是UI开发回放，不是正式评测。PID及当前模型后续仍须实查；启动argv仍写baseline，不能据此否认GUI已换模型。
- 最终Viewer raw核验CPU真实exit0：60/60采集数组符合形状且有限，50改变/10不变（空检测支路与固定动作std），77/77参数逐元素相同。四份raw导出复制前后SHA稳定且副本与原件一致，原before_frame.json/after_frame.json未覆盖。结果、复制清单、脚本、日志/真实退出及截图均在runtime_final_196608。docs/运行命令已同步final模型；本轮仅完成既有功能的训练/评估/部署验收，无新增独立功能，计数仍0/5。
- 原生heartbeat automation id `automation`、名称“跟进履带联动模型训练与验证”，完成真实退出/权重/开发配对及final viewer验收后，已由原生automation_update确认PAUSED，避免重复跟进。原prompt/间隔/线程/通知偏好保持；不重复创建。后续其他任务须按用户授权另行推进，当前不再有这轮自动训练监控。
- 12:03/12:13/12:23/12:33 heartbeat运行期只读核查仍保留trial的heartbeat_checks子目录；其“运行中”仅属对应时点，最新完成状态以上文真实退出与最终完整评估为准。训练期间未改42冻结源，没有另起重复GPU评估。
- 最新整理前验证：PI-v6全CPU244 passed/13 skipped/8 warnings，13 native均通过、compileall/实际bash语法0；只描述旧快照。本轮当前结果须另记。旧批次详录见归档AGENTS及 `cleanup_and_validation_20261020_batch12.md`。

## 任务边界与验收

- 后端仅发展Isaac Lab + Newton MJWarp，复用本地external和SB3-Contrib；模型自己学，连续四维动作，无Nav2或手写规划器替代策略。
- 不做训练速度/FPS基准。必要的物理正确性、学习效果、PC负载和UI核验仍做；逻辑20Hz配置不冒充实测帧率。
- 优先不压苗，允许更慢或超时；到点为0.4m位置范围、低速稳定hold0.3s，不要求终态朝向。
- 安全验收必须同配置/同初态配对：clean成功增加，且损伤episode数、平均损伤、碰撞不恶化；不能把成功/损伤权衡称能力提升。旧权重可在当前物理重评，但不同规则/几何/驱动/观测下的旧报告不混算，不重贴metadata。
- 810xx/820xx/870xx是已观察开发数据，830xx保留未观察，只有开发安全门槛通过后才做独立验证。滚动训练窗口不是held-out；CPU固定观测反事实不是车辆轨迹。
- 完整植物搜索、粗糙地形/石块、对抗训练及整田可靠导航未达成，不以局部通过或训练跑通标记总体完成。当前基础课程plants0，不称已经学会避苗。
- 每次训练保留协议、源码哈希/快照、配置、种子、检查点、依赖/资产及真实退出。观察超时不是进程结束；句柄失效须核实际进程/端口，不能只看状态文件猜测或重启。
- 旧后端69项批量删除曾被自动审批拒绝，未执行或绕过。用户后续授权移动，已原样移至 `归档/legacy_before_isaac_20261017/`；不能把历史拒绝视为删除授权。共享运行时/upstream/Isaac模型和记录保留；原路径按archive_manifest.json与SHA查阅。

## 实车事实与代理限制

- 30kg；车身0.8×0.3m，300mm不含履带；整车最外宽470mm；每侧履带宽60mm，中心距410mm，接地595mm，最高15km/h。
- 两台BDUAV N5065 270KV，4:1减速，主动轮直径60mm；硬地和农田均能原地转向。转一圈用时暂缺，不反复索取、不称已标定。
- 轻量代理是盒体、七对圆柱轮和双轴云台，不导入高精CAD。45mm代理轮半径不同于30mm主动轮；P.04、Ki.1、2Nm/轮、Kt.035和效率1均未标定。14个轮执行器+同侧约束仍不是两台实车电机的完整模型；不额外施加车体速度/转向伺服。
- 用户已批准IMU/驱动反馈/协议及硬件修改，并授权自主推进；此前“需批准/尚未实施”属历史，不再阻塞。ATK-IMU901为10轴、随云台偏航和俯仰一起动；云台两轴实测角度/速率、左右驱动电流/机械RPM均可回读，驱动器自研文件在项目内。云台IMU不直接等于车体IMU，不用动作指令冒充实测速率。
- 已有Python telemetry解码/组装与FOC C主机96checks/0fail，但未完成实机串口/CAN/HAL联调；不能把仿真等效Iq/RPM称实测或接触力。硬件路径 `电力/FOC驱动模块/软件/`；厂家手册镜像 `docs/hardware/atk_imu901/ATK-IMU901_User_Guide_v1.2.pdf` 封面标V1.0，未核实与实物/最新固件一致。各通道单位、噪声、电流/安装外参仍需实机核实。
- 云台中心离地.4m、俯仰−25°..60°，雷达/相机一起俯仰。X2雷达360°/8m/10Hz/400束；策略与预览同88×160轴向Z深度。气压/海拔不入策略，磁场保留，无重复Euler/温度输入。
- 玉米约1m，行距.8m、株距.3m，允许健康/干叶混合/遮挡；UWB多基站覆盖。识别相机/单点测距与云台同动，须对准目标再测。
- 损伤默认stem-contact-v2：履带压茎基或车身撞主茎；叶片擦碰及旧9–15cm保护圈不算。履带595×60mm带和主茎垂直相交，车身当前Z/pitch/roll扩张AABB；水平扫掠是离散2.5D近似，不称精确接触/弯折。旧配置缺metric用root-protection-v1保持复现。
- 固定植物root未证实每株增加6个q/qd；有位姿/mocap/几何/BVH成本，未测不猜收益，不据此重写bodyless。

## UI与关键证据

- 用户选定bbycroft LLM Visualization风格，external/llm-viz MIT实际复用范围记在 `isaac_field/visualization/THIRD_PARTY_NOTICES.md`。固定深色，无主题切换；只保留完整二维网络展开，无整体视图/网络三维，车辆三维保留，连续曲线连线。
- 色标为整帧全部采集层及绑定参数最大绝对值±全范围、0同色，扩大立即/新frame收缩最多5%，暂停重复GET/选层不重定标；reset/bind清历史。完整值域无百分位截断。不能仅用PNG变色判断raw激活变化。
- reset先确定最终start/yaw/goal再完整校验，推进RNG重试；不验证后覆盖端点。证据 `validated_reset_repair/`。生产reset采用H/V矩形连通验证，0.1m网格可能错过狭窄锚点；Scene.radius导航保护区与stem_radius损伤语义不同，尚未修复，不断言策略无法驶入行间。
- Newton BVH包括VISIBLE与COLLIDE_SHAPES，视觉叶片也能测距；历史空命中因最近株31.54m。近株1.5m LiDAR/单点实测1.488780m，不再传播植物透明/射线只查碰撞的已纠正误判。
- 旧53248/safe118784/Gaussian-gSDE/GAE对照及转向偏置见文档归档和原results，未通过整体不压苗优先门槛。目标通道正确、梯度非零，未发现训练cache陈旧；不称GAE是唯一根因。
- WSL必须显式Ubuntu-24.04。localhost/NAT UTF16提示曾用/bin/true实证actual exit0，不把warning当失败；后台检查真实exit和进程。PowerShell会展开双引号里的$rc，状态采集优先保存bash文件再执行。不要擅自重启WSL/驱动/电脑。
- 详细命令与流程见使用说明.md，当前项目状态见ISAAC_LAB.md；更早历史在 `results/project_history/AGENTS_before_20261017_batch8_cleanup.md` 及本批文档归档。历史PID/源码计数/“当前”只属于相应时点，不覆盖最新记录。
