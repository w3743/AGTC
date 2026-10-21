# 最短时间到点停稳：512维独立实验

最新状态：用户明确要求暂停，已于2026-10-20 18:15:36北京时间核argv后SIGSTOP Python336098，实际ps状态Tl+；进程未退出，现场保留内存、显存仍分配。launcher336095仍等待，不自动恢复、不重新launch。最近完整磁盘checkpoint57344步，SHA `db473f5e78038c066226f61cb6a5ee02ea71615823f28e519366cb0809ecbb8a` 与manifest/实算一致，ZIP完整且含Adam。最新完整rollout数与系统状态见pause_20261020/pause_audit.json；原run_status仍running只是应用不感知SIGSTOP，不据此声称仍训练。下文“训练继续”属于暂停前记录。
用户确认按最短时间到点停稳方向继续。新trial `results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020/` 于2026-10-20 18:01:53北京时间正式启动，Ubuntu-24.04 Python336098/launcher336095/exec37814仅作启动记录，后续必须查实际argv/日志/检查点/真实退出。原生automation仍暂停。

当前状态：已生成8192步checkpoint、完成CPU-only 15项审计实际exit0，训练继续，预算196608 transitions。单次新实验不证明512优于128；首检查点与滚动日志不代表到点能力提升。

## 目标与约束

actor/critic各单层LSTM512、融合192、独立policy参数3065801；新随机初始化，不迁移旧权重。训练仍为4环境、seed69009、100×100m固定崎岖地形、全场随机目标/起点/朝向、60s、植物0/岩石0、原前导向轮联动及2Nm/轮/指令斜坡，未套用硬件审计候选电驱包络。

停车为目标距离≤.4m、真实世界XY COM速度模长<.05m/s、完整车体角速度模长<.1rad/s持续.5s，朝向不限。世界与车体角速度向量经旋转后模长相同，不用云台IMU或动作命令代替真值。策略输入14678字段/五路融合192保持。

成功+10、所有失败−10、每秒−.1。γ=1同步至model与rollout buffer；Φ=5×(初始距−当前距)/max(初始距,.4)，增量Φ下一步−Φ当前，所有终态Φ=0。在本轮无植物/无岩石场景，完整回合距离项相消，主要总分成功10−.1T、失败−10−.1T。超时是真终态、禁止TimeLimit bootstrap。途中稳定奖励系数0，仍记录姿态/摇摆统计并保留越界/roll或pitch>1.2rad硬失败；同一步失败优先于成功。

## 冻结与启动

protocol SHA `3ca58fcc05b9d7ac34798f836704c91b5ec608d62f7424c3897ed291451a620c`，45源冻结，prepare/preflight/外层实际0。初始化实际架构已核实，新0步baseline SHA `3366b22d4b2bb76f2688dcd3dd43b2b3447996cf52f9e23e5713523e7d2190d0`。带锁launcher拒绝重复输出，父shell记录真实python.exit_code/process_exit.json；没有另建定时任务。

首8192 checkpoint SHA `49280e36e068049ba7e811355311c0b228f718878ac330ec8fd8508f856ae933`。CPU-only审计核γ1、严格停车/零稳定权重、架构/编码/动作/驱动合同、45源与模型/协议哈希稳定、权重/Adam有限、实际更新，15项全true；37项Adam step300，65个state张量/3021076元素相对0步改变。state_dict含共享别名；植物0检测分支没有学习证据。见first_checkpoint_cpu.json/log/exit_code。最终CPU复查可复用audit_checkpoint_cpu.py，须先核真实正常退出与196608终步数，使用stage_01_global_random.zip和新的报告路径，不能覆盖首验收。

## 运行诊断与评估

新profile复用训练诊断JSONL，保留每rollout的动作/奖励分项及PPO logger，并标注PPO值可能对应上一更新的对齐语义。startup_diagnostics.json与其完整rollout副本核实已发生PPO更新、动作/奖励有限、逐步reward reconciliation matched、time每步−.005、body_stability为0。完整goal_stop在正式评估行保存，训练compact info不含它，不虚称全部已采集。

训练自然完成后现有runner自动在地形80003/80004、开发seed87001–4/82001–4上做正式0baseline/final同初态配对。核真实USD路径/SHA/字节数、真实terrain seed、起点/目标、严格终态速度/hold与姿态统计。先比较无损成功与安全门槛；只有非空且相同的成功及无损成功seed集合、共同无损成功逐条不变慢且至少一条更快、安全指标不退化时，才标记配对用时提升。均值下降不能替代配对。830xx未观察，不做训练FPS基准，不部署viewer。

## 整理与历史保留

本项是第十六批第5项，已完成整理，当前第十七批0/5。完整CPU最后377 passed/20 skipped/8 warnings、compileall/Bash/80份源码测试SHA稳定实际0；真实Newton20 passed/61 upstream warnings实际0。失败attempt1/2日志保留，最终以attempt3为准，详见上级cleanup_and_validation_20261020_batch16.md。复用原diagnostic_step_infos/episode_results，移除重复终态缓存；成功/失败常量由task.py单一导出。

旧奖励512实验于17:42:43主动停止，真实Python exit143，最后完整40960 checkpoint SHA `67bcb7b73dd5f22c1b01e33a1a0e86cf8f8ed82c25587088ceffa907c1df33d5` CPU审计通过。旧run_status仍running是陈旧值，以previous_stop_outcome.json和父shell实际退出为准。12份历史模型/协议、45旧源码快照全未变。八个改动文件、三份文档原字节分别保存在before/与docs_before/；cleanup_audit.json记录差分与真实验证。不将旧奖励与新严格停车规则的成绩混算。
