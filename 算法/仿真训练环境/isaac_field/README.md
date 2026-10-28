# Isaac Field 仿真代码

本目录是 Isaac Lab 农田履带机器人任务的 Python 包。模块按源码职责列出；训练、运行历史和验收证据见[仿真训练环境上级说明](../README.md)。这里的模块说明不代表所有后端或实验结果均已通过验收。

| 文件 | 用途 |
| --- | --- |
| __init__.py | 声明 Isaac Field 仿真任务包。 |
| __main__.py | 提供按任务选择的训练、评估和查看命令入口。 |
| action_contract.py | 定义左右履带速度与云台角速度动作的共享字段、标签和配置。 |
| adversarial_training.py | 执行一个同场景的朴素极小极大教师/学生训练轮次。 |
| adversary.py | 定义用 RSL-RL PPO 训练的地图设计者环境；其目标是使学生得分降低，不实现 PAIRED。 |
| assets.py | 以 OpenUSD 物理和几何 schema 生成机器人及小型场景 USD 资产。 |
| body_stability.py | 提供训练/评估使用的车体稳定性目标和诊断量；这些不是传感器观测或硬件安全额定值。 |
| curriculum_control.py | 根据门槛结果计算课程采样控制决策，不负责创建环境或训练。 |
| drive_control.py | 为显式代理轮驱动提供批量侧积分控制辅助函数。 |
| env.py | 实现 Isaac Lab 批量环境、机器人驱动和实时 GPU 传感器采集。 |
| feedback.py | 定义传感器反馈字段，并按环境估计 IMU 姿态及驱动反馈。 |
| layouts.py | 生成带植物的程序化农田布局。 |
| learning.py | 串行执行 RSL-RL 课程训练阶段并记录留出评估。 |
| live_state.py | 将实时环境状态原子写出，供明确接入的查看器读取。 |
| navigation.py | 对重置候选点进行双朝向连通性校验；不生成策略地图或路径。 |
| newton_drive.py | 定义 Newton 后端的同侧轮耦合和规范化物理元数据。 |
| physx_drive.py | 生成 PhysX mimic 约束及接触/后端身份元数据。 |
| policy_contract.py | 定义策略输入、观测形状及编码器契约，供多个 RL 适配器共享。 |
| policy.py | 定义 SB3 循环策略适配与检查点契约验证，避免静默重解释权重。 |
| rsl_rl_cfg.py | 声明 fast-stop-v2 绝对目标任务使用的 RSL-RL 环境和训练配置。 |
| rsl_rl_env.py | 将嵌套策略观测适配到 Isaac Lab/RSL-RL 所需的向量环境边界。 |
| rsl_rl_export.py | 将五路传感器输入的 RSL-RL actor 封装为导出推理模块。 |
| rsl_rl_inference.py | 将 RSL-RL 检查点接入已有回合评估器的预测接口。 |
| rsl_rl_policy.py | 将传感器字典接到 RSL-RL 策略，并把策略契约写入检查点。 |
| runtime.py | 管理 Isaac Sim Kit 启动、复用及关闭；仅导入任务契约不会启动 Kit。 |
| scene_assets.py | 使用标准 USD 几何和显式植物后端变体构建任务场景资产。 |
| sensor_encoder.py | 实现 SB3 无关、由多个策略后端共享的 PyTorch 传感器编码器。 |
| task.py | 定义任务配置、农田场景、任务状态和单步参考逻辑。 |
| tasks.py | 注册 Isaac Lab Gym 环境 ID 和对应配置入口。 |
| telemetry.py | 解码外部 IMU/CAN 数据帧并转换为类型化样本；模块本身不进行串口或 CAN I/O。 |
| tensor_sensors.py | 对设备端批量传感器张量执行有效性、裁剪和噪声处理。 |
| tensor_task.py | 批量计算 fast-stop-v2 的奖励、终止和绝对目标观测，不负责模拟场景或传感器。 |
| terrain.py | 生成带岩石的确定性地形；其中 raycast 供离线场景校验使用。 |
| training_diagnostics.py | 记录 SB3-Contrib RecurrentPPO 可取得的附加训练诊断值，不从缺失日志反推数据。 |
| training.py | 使用 RSL-RL 官方 runner 训练，并提供策略推理计时辅助。 |
| viewer.py | 扩展 Isaac Lab Viser 查看器的任务控制和策略监测界面。 |

二维策略网络查看器的独立模块见 [visualization/README.md](visualization/README.md)。它是查看工具，不属于训练路径的必需导入。
