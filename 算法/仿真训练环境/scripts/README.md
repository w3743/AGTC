# 仿真与训练脚本

脚本可能启动训练、创建运行目录或连接查看器；执行前请阅读对应文件的参数和[上级说明](../README.md)。历史脚本的存在不表示对应实验仍在运行。

| 文件 | 用途 |
| --- | --- |
| `compare_isaac_exploration.py`、`compare_isaac_gae.py` | 准备或运行动作噪声和 GAE 参数的配对比较。 |
| `continue_isaac_foundation.py`、`train_isaac_foundation.py`、`train_isaac_foundation_fresh.py` | 继续或启动基础课程训练。 |
| `train_isaac_mastery.py`、`run_isaac_mastery.sh` | 准备或运行受控 mastery 训练。 |
| `improve_isaac_policy.py` | 运行策略微调的冒烟或分阶段检查。 |
| `evaluate_fast_stop_mastery.py`、`evaluate_fast_stop_resume.py`、`evaluate_rsl_curriculum.py` | 对相应检查点或续训实验执行评估。 |
| `resume_fast_stop.py`、`resume_safe_goto_recovered.py` | 准备并运行指定 fast-stop 或 safe-goto 续训。 |
| `export_rsl_policy.py` | 导出 RSL 策略的 TorchScript。 |
| `diagnose_isaac_ground_contact.py`、`isaaclab_runtime_probe.py` | 地面接触与 Isaac Lab 运行时诊断。 |
| `verify_isaacsim_environment.py`、`verify_isaacsim_training.py`、`run_native_check.py` | 有界环境、训练链路及原生检查。 |
| `install_isaaclab_windows.ps1`、`install_isaaclab.sh` | Windows 和 shell 安装入口。 |
| `isaaclab_python.ps1`、`isaaclab_python.sh` | 使用相应环境运行 Python 的入口。 |
| `lassc_consumer.py`、`lassc_replay.py` | 深度消费者适配和 CPU 回放检查。 |
| `view_fast_stop_goal.py`、`view_policy_netron.py` | 目标选择预览和策略结构导出查看。 |
| `view_training_state.py`、`view_training_state_web.py` | 查看最新训练状态的独立查看器和网页服务。 |
| `watch_training_completion.py` | 监测训练状态并处理完成事件。 |
| `README.md` | 本目录索引。 |
