# 第十六批整理与验收

本批五项：同侧前导向轮速度联动；车体稳定奖励与统计；全场随机rough独立入口；可配置512维独立实验；fast_stop_v1到点停稳与最短用时实验。实现、测试和文档按每项功能统一计数，训练/检查点核验不重复计数。本批已完成整理，下一批为第十七批0/5。

本次整理范围是第5项及它接触的接口。复用现有Config、FieldTask、Newton真值、SB3-Contrib、官方wrapper、diagnostic_step_infos、episode_results与训练诊断回调。移除子任务曾添加的重复_terminal_results缓存和SingleFieldAdapter扩展，沿用原评估auto_reset=False路径与官方SB3训练路径；成功/失败奖励常量由task.py单一导出，协议引用同一来源。三个profile共享rough判断，旧profile与新profile分别冻结成功/奖励/终止合同，不改旧报告。

第4项旧奖励512实验按用户新目标于2026-10-20 17:42:43主动停止。SIGINT未结束后台Python后，经实际argv再核对发SIGTERM；父shell记录真实Python exit143。最后完整40960 checkpoint CPU-only 13项验证exit0，protocol、baseline和全部检查点保留。原run_status仍running是陈旧值，真实停止证据见fast_stop_objective_20261020/previous_stop_outcome.json。不得把它称正常完成196608或训练失败；是按新目标中断。

第5项采用新随机512权重，不迁移旧实验。停车条件为0.4m内、真实世界XY COM速度<0.05m/s、完整车体角速度模长<0.1rad/s连续0.5s，朝向不限。gamma1和真终态超时确保距离潜势全回合首尾抵消；零植物/零岩石下主要总分为成功10−.1T、失败−10−.1T。途中平稳惩罚权重0，姿态统计与1.2rad硬终止保留。没有新策略观测、导航规划器、机械驱动或硬件能力包络。

完整CPU最终为377 passed/20 skipped/8 warnings，compileall、项目及本轮Bash语法、源码/测试SHA前后稳定均实际exit0，见fast_stop_objective_20261020/cpu_attempt3.log与cpu_attempt3_actual_exit.json。原生Newton为20 passed/61 upstream warnings、真实Python exit0，包含16环境测试、前导向轮与原诊断测试；见native_attempt1.log与native_attempt1_actual_exit.json。两项新native分别核root COM速度/局部reset隔离，以及官方SB3 wrapper超时done且TimeLimit.truncated=False、自动reset后终态goal_stop原值保留。

失败记录保留：task子任务首轮helper重复关键字导致4 failed/61 passed，修复后70 passed，其首轮输出仅在对话工具记录、无另存原始日志；不补造。root完整CPU attempt1为376 passed/1 failed：新测试把碰撞1→1误当恶化，修为1→2；同次交权后的晚到写入造成源稳定检查exit1，非训练失败。attempt2为376 passed/1 failed：阈值测试改变速度后没同步一致性标志，触发了更早的正确拒绝，修正测试夹具后attempt3通过；未放宽生产验收门槛或覆盖失败日志。

12份历史模型/协议SHA全部未变，旧512全部45份源码快照未变；live仅task.py/env.py/learning.py/fresh runner四个预期源变化。cleanup_audit.json记录全部逐项结果。文档原件在docs_before，八个修改文件原字节在before及runner/task manifest；当前使用说明与ISAAC_LAB/AGENTS同步新目标、旧实验真实停止、新实验独立命令及尚无学习提升结论。训练或评估成绩未覆盖、重写或跨规则混算。

新独立prepare/preflight在整理后执行，均实际exit0；正式目录为results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020，45训练源冻结。运行、首检查点与最终结果以后续真实记录为准，不将本次通过的正确性测试当作导航能力提升。830xx未观察，不做训练速度/FPS基准，原生automation仍PAUSED。
