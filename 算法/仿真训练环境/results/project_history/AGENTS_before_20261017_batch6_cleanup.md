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

- 当前新批次计数：4 / 5。
- 本批第3项：修复连续合法出生点与导航栅格锚点不一致、同种子无效重试及验证后覆盖起终点问题；最终出生姿态/目标先采样再校验，显式yaw检查实际车身占地。CPU完整套件87 passed、8 skipped；原生8项通过exit0，含新增同对起终点回归；日志 results/isaac_field_diagnosis/validated_reset_repair/native_env_integration_retry_20261017.log。
- 本批第4项：恢复实验resume1已exit0，110592保存点续8192至118784；启动前19文件快照。24例地图/起点/目标严格配对通过；旧53248 clean5、damaged4、mean damage0.25，新118784 clean8、damaged5、mean damage0.3333，均0碰撞，assessment=success_safety_tradeoff，未达到用户优先不压苗的安全判据，不认定合格提升。原parent失败结果保留。
- 本批第1项：CLI确定性回放新增独立episode seed、显式保护植物reward profile与紧凑初态审计输出，环境seed仍控制几何；原生同题核对待执行。本批第2项：Viewer目标控件从实际goal初始化/重置同步，Reset不把旧goal套入新map，终态Set goal保持终态并提示先Reset。相关完整CPU suite 60 passed、7 skipped，编译及wrapper语法通过；运行中的旧viewer尚未重启，不得冒称界面已修复生效。
- 最近整理：2026-10-17（第五批）。五项为真实车身原地转向回归、近株雷达/单点测距回归、默认关闭的compact trajectory、可选每目标净进度预算及奖励配置边界、复用两阶段PPO的safe-goto协议。完整旧规则记录在 results/project_history/AGENTS_before_20261017_batch5_cleanup.md；整理记录及日志为 results/project_history/cleanup_and_validation_20261017_batch5.md/.log。
- 整理：两份固定参数植物诊断脚本归档到 results/project_history/plant_diagnostics_20261017；同步ISAAC_LAB.md和使用说明.md。原始结果/模型保留；未触碰共享运行时、upstream和旧后端。此前69项旧后端批量删除被自动审批以blocked by policy拒绝，未重试。
- 验证：WSL完整CPU suite 56 passed、7 skipped；compileall与两个bash wrapper语法通过。本批两项新增原生回归分别通过；整理未重复GPU测试，未做训练速度/决策FPS基准。
- 实车事实：30kg、0.8×0.3m、15km/h、接地595mm、主动轮直径60mm、4:1减速；硬地/农田都能原地转向，转一圈用时未知。七对45mm数值滚轮独立于30mm主动轮半径；gain0.5/每轮2Nm是代理参数，未标定实车转向速度。
- 感知核查：准确goto seed80003中布局/实时位姿/provider在0/1/2/10步一致，最近株距31.5404m，零命中合理；近株1.5m时雷达及单点测距均1.488780m。Newton BVH含VISIBLE与COLLIDE_SHAPES，叶片仅VISIBLE仍可测距。“雷达只查碰撞、玉米透明”是已更正的局部源码误判，不能再传播为故障。
- 当前已完成权重53,248步：results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip；旧12例到达1/12但唯一成功损伤4，干净成功0/12。新24例基线在safe奖励口径下干净成功3/24，损伤回合6/24，平均损伤0.416667，碰撞0。不同评测集/奖励的raw return不可直接比较。
- 当前真实轨迹：results/isaac_field_diagnosis/policy_53248_trajectory，开发seed76001–76004均未进0.4m目标圈，非刹停/hold失败；小差速下真实yaw响应弱，不能由名义无滑移角速度替代。旧arrival trace只属16,384/20,480步。
- 已中断parent实验：results/isaac_field_improvement/safe_goto_20261017_reward_v1，exec26398确切exit1，near完成、medium保存于110592，reset失败后无after。原日志/协议/检查点/18文件快照保留。修复后的独立resume1已完成，如上第4项；不是恢复环境RNG/中途LSTM rollout的bitwise重演。正在准备官方gSDE与Gaussian的同源新优化器对照，未启动；底盘/观测/奖励不变。
- Viewer：独立Edge http://127.0.0.1:8768/，exec5626 / WSL python PID9807；envseed80003/episodeseed81001、protect_plants、flat16、20s、goal1..3m ±60°。初态与后台基线的植物hash/起点/目标均精确匹配，replay_pair_audit.json已保存。目标控件同步、重复Reset和终态Set goal守卫已通过真实浏览器验证。页面经Load checkpoint切到118784候选（drive/observation match），seed81001仍timeout、distance0.98m、damage0collision0；候选截图browser_candidate118784_terminal.png存于初态目录。当前停在Reset后的policy初态，加载不意味着验收通过。Edge当前核显少量占用，不能声称本次独显渲染。完整搜索、粗糙地形、对抗能力和整田可靠导航尚未达成。
- 用户最新验收：到点只要求位置到达并停稳，不要求车身/云台朝向；保持0.4m位置容差、速度阈值和0.3s hold。实际转圈时间暂缺，不再重复索取，不将代理转速称为已校准。
- 新轨迹证据：recovered_baseline_trajectory_analysis.json 中旧53248正bearing12/12例持续负差速、0 clean success；负bearing组5/12 clean success。seed81001正bearing+1.0249rad但差速均值−0.0268、真实yaw增量−0.2256，策略确实朝错误方向发命令，不能只归因于机械转向死区。
