# 第六批整理（2026-10-17）

本批五项：确定性Viewer episode回放、目标控件与终态守卫、最终起终点一致的reset采样修复、独立恢复实验、官方gSDE/Gaussian同源对照支持。实现、测试和对应文档按功能计数。

整理将已结束的恢复实验与仍准备中的对照分开记录；统一评测追溯信息到 `learning.evaluate_model` 的可选紧凑轨迹输出，包含实际植物哈希、起点和目标。默认输出不变。旧恢复脚本及其源码快照保留为原始研究记录，不改写已完成成绩。删除的仅有验证代理自己的临时shell脚本和空日志；未重试旧后端批量删除，未触碰共享运行时/upstream。

`safe_goto_20261017_reward_v1_resume1` 真实exit0，最终118784步。24组初态配对通过；清洁成功5→8，但损伤回合4→5、平均损伤0.25→0.3333，安全判据不通过。浏览器已验证episode重放、目标同步、终态守卫及候选metadata；初态精确匹配，未宣称CPU预览与CUDA评估全轨迹逐位相同。补充用户的到点要求：位置到达并停稳，不要求终态朝向。

对照支持复用SB3-Contrib gSDE、Isaac Lab adapter、现有训练/评测/安全判据。两组复制同一源策略均值及价值权重，均新建Adam；gSDE只重建噪声参数并在真实actor latent样本上匹配平均方差，不保持动作8步、不改变逻辑控制周期。实际53248源checkpoint的CPU接口检查也通过；dummy校准值仅为接口自检，正式训练必须重新采集native训练种子样本。实验此时尚未prepare/launch。

验证日志：`cleanup_and_validation_20261017_batch6.log`。完整CPU suite为92 passed、8 skipped，pytest exit0；compileall与两个bash wrapper语法均exit0。未启动GPU或速度/FPS基准。此前独立原生8项exit0日志为 `results/isaac_field_diagnosis/validated_reset_repair/native_env_integration_retry_20261017.log`；其后改动限于学习/评测记录与探索初始化，未改底盘、传感器或reset物理，因此未重复该原生套件。

实现源码冻结，整理完成，计数归零。整理前规则备份：`AGENTS_before_20261017_batch6_cleanup.md`。
