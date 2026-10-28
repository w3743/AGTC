# Isaac Field 仿真代码

本目录是 Isaac Lab 农业履带机器人任务的 Python 包；运行和历史状态见[上级说明](../README.md)。下表概述同目录模块的职责。

| 文件 | 用途 |
| --- | --- |
| `__init__.py`、`__main__.py` | 包初始化与训练、评估、查看命令入口。 |
| `action_contract.py`、`policy_contract.py` | 动作及策略输入、编码器的共享接口约定。 |
| `adversarial_training.py`、`adversary.py` | 对抗式师生训练及地图设计环境。 |
| `assets.py`、`scene_assets.py` | 机器人和任务场景 USD 资产。 |
| `body_stability.py` | 车体稳定性目标及诊断。 |
| `curriculum_control.py`、`learning.py` | 课程控制决策及顺序训练与留出评估。 |
| `drive_control.py`、`newton_drive.py`、`physx_drive.py` | 驱动控制辅助计算以及 Newton、PhysX 驱动与接触配置。 |
| `env.py`、`rsl_rl_env.py` | Isaac Lab 批量任务环境与 RSL-RL 观测适配。 |
| `feedback.py`、`sensor_encoder.py`、`tensor_sensors.py` | 传感反馈、传感编码及批量激光雷达和深度采样辅助函数。 |
| `layouts.py`、`terrain.py` | 植物布局与地形生成。 |
| `live_state.py`、`telemetry.py`、`viewer.py` | 实时状态导出、遥测解码及任务查看器。 |
| `navigation.py` | 农田任务重置的连通性检查。 |
| `policy.py`、`rsl_rl_policy.py` | SB3 和 RSL-RL 策略适配。 |
| `rsl_rl_cfg.py`、`rsl_rl_export.py`、`rsl_rl_inference.py` | RSL-RL 训练配置、策略导出和检查点推理。 |
| `runtime.py` | 管理训练和评估使用的 Isaac Sim 运行时。 |
| `task.py`、`tasks.py`、`tensor_task.py` | 任务状态、Gym 注册和批量任务计算。 |
| `training.py`、`training_diagnostics.py` | RSL-RL 训练及可选的训练诊断记录。 |

子目录包含可视化等配套模块；实验输出保存在上级的结果或验收目录中。
