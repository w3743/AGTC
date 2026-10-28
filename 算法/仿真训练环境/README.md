# 仿真训练环境

本目录汇集履带底盘的仿真任务、训练程序及运行证据。不同文档记录不同时间的方案与状态，应结合文档日期和对应证据阅读，不把历史运行状态当作当前实况。

| 同目录文件 | 用途 |
| --- | --- |
| `AGENTS.md` | 本目录的协作和技术约束。 |
| `使用说明.md` | Isaac Lab 环境的安装和使用说明，含历史平台记录。 |
| `ISAAC_LAB.md` | Isaac Lab 项目状态与历史实验记录。 |
| `NATIVE_STACK_PLAN.md` | 原生 ARM64 训练栈迁移的历史计划和边界。 |
| `PERF_PLATFORM.md` | 训练平台性能改造过程和实测记录。 |
| `PERF_SUMMARY.md` | 性能改造的状态入口及证据索引。 |
| `WSL_TRAIN_PLAN.md` | WSL 原生训练迁移决策和准备记录。 |
| `pyproject.toml` | `isaac-field` Python 项目的构建、依赖及包配置。 |
| `安装IsaacLab.cmd` | Windows 下的 Isaac Lab 安装入口脚本。 |
| `启动IsaacLab.cmd` | Windows 下的 Isaac Lab 启动入口脚本。 |

子目录中的源码、工具和项目资料按各自说明使用。`results/`、`归档/`、`native/acceptance/` 等保存实验或验收记录，须保留原始字节；`external/` 中的第三方代码按项目固定版本管理。
