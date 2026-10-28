# Native 开发工具与测试

本目录放置吞吐测量辅助工具与不依赖第三方测试框架的 Python 检查。测试文件通过名字标记覆盖范围；本 README 仅说明其用途，没有运行这些脚本。

| 文件 | 用途 |
| --- | --- |
| benchmark_training.py | 基于 train_wsl.py 启动短 native WSL 测量，并以只读 auditor 检查运行记录和身份。 |
| test_benchmark_training.py | 检查 benchmark_training 参数处理、运行和只读审计逻辑。 |
| test_task_contract.py | 对 Native V2 task contract canonical identity 与 SHA 行为做 CPU-only 检查。 |
| test_trial_runners.py | 用标准库检查冻结训练/评估 runner 的命令、参数和输出门禁。 |
| test_v2_protocol_gates.py | 检查 V2 protocol/recipe gate；不启动模拟器，也不反序列化 checkpoint。 |

这些文件不包含正式训练入口。正式 run、版本和验收信息见 [train/README.md](../train/README.md) 与[迁移与验收说明](../MIGRATION_ACCEPTANCE.md)。
