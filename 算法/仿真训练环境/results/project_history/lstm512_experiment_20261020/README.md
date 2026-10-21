# 512维独立实验，2026-10-20

用户授权“先跑512维，可以开始实验”。17:33:58北京时间正式启动，trial为 `results/isaac_field_improvement/random_global_rough_lstm512_20261020/`，WSL Ubuntu-24.04。Python335230/exec50333仅作启动记录，后续必须重新核实际argv、日志和真实退出，不据历史PID重启。automation保持暂停。

配置：shared encoder192，actor/critic各1层LSTM512，head各64，policy独立参数3065801。新随机初始化，不迁移旧权重。现有全场随机rough任务/前导向轮联动/车体稳定奖励/4env/seed69009/196608 transitions/60s/植物0岩石0保持，未混入尚未定义的硬件电驱包络。运行中45源码冻结，protocol SHA `010486bd7ceeb2ed121a1a1b765b6901e7732208496a9c699cc89da53fdad954`。此为单次512实验，不能据baseline/final结果证明比128好。

实现为本批第4项（当前4/5）：prepare专用宽度参数、冻结协议、创建与加载的实际宽度校验、初始化记录。runner/learning/两测试原字节与SHA在before/；文档原件在docs_before/。没有修改网络默认128、奖励或物理。

验证：首次全CPU为2 failed/357 passed/18 skipped/8 warnings，原因是新增测试漏导入sys；原日志与实际exit1保留。补导入后完整CPU359 passed/18 skipped/8 warnings，compileall/bash语法实际0，见cpu_tests_attempt2.log与cpu_attempt2_actual_exit.json。prepare/preflight/外层实际0，分别记录日志及exit_code。Windows定向runner曾9 passed，learning因Windows缺SB3未运行；正式验收以WSL完整结果为准。

17:36:10已产生8192 checkpoint。CPU-only helper真实exit0、13项检查全true，45源live/snapshot匹配，actor/critic512×1/input192，权重与Adam有限，37项Adam step289。65个state张量/3028603元素相对0步改变，state_dict含共享别名，不能混称独立参数数目。初始baseline SHA `9fbad2d40f53aec3ab4763f20890abd49578d6b62c3d9bf01d96b697e3af8675`，8192 SHA `c363f75703adbd740dc2b43a1c81065a889ad3cb8cae601c6e716964e02ae817`。详见first_checkpoint_cpu.json及其log/exit_code。该验证不创建环境、不训练或测FPS。

启动wrapper使用独立mkdir锁、拒绝旧启动证据，父shell等待Python并记录真实python.exit_code/process_exit.json。训练自然完成后现有runner会做rough_a/b两地形上正式0baseline/final同初态开发配对；成功、损伤/碰撞、越界/姿态失败按原门槛判断。830xx未观察。首检查点更新与滚动训练0/5均不是学习效果结论。最终CPU验证可复用audit_checkpoint_cpu.py，但必须用stage_01_global_random.zip和新报告名，检查真实退出/终步数后执行。

用户后续询问奖励函数，已只读核对：每步time -0.01*dt，距离进度5/max(初始距,.4)*(d_old-d_new)，成功停稳+5，稳定性dt*[2*超35度倾角^2+.02*(body_wx^2+body_wy^2)]扣分，损伤-20/新增株，岩石碰撞-0.1/步，越界小项及越界/过倾终止-5；goto超时无额外终局罚，distance_cost0。停稳为距离≤.4m、|前向v|<.2m/s、|yawrate|<.3rad/s持续.3s，无朝向要求。没有在运行中改这些设置。

historical_artifacts_before/after.json保留5份旧协议/模型SHA，未覆盖旧实验。旧128粗糙prepare保留，但本轮源码已变化，不能沿用旧preflight声称可启动。
