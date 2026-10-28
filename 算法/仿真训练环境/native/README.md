# Native 训练栈

> 2026-10-28：历史训练和评估 run 的原始输出已按用户要求清理，其主要结果见[训练结果汇总](../训练结果汇总.md)；当前工程代码及非训练物理检查资料保留。

本目录存放不启动 Isaac Sim/Kit 的 WSL2 ARM64 原生训练与评估代码。Native V2 的冻结协议、task identity、当前运行状态和验收证据以[迁移与验收说明](MIGRATION_ACCEPTANCE.md)为准；本文件的命令示例只说明入口，不表示某个 run 正在运行或已经通过。

## 目录与说明

- [proto/README.md](proto/README.md)：M0–M3 原型、观测/物理适配器、调试与历史测量工具。
- [train/README.md](train/README.md)：训练/评估入口、协议、身份计算和 runner 校验。
- [viz/README.md](viz/README.md)：Native 回放、实时状态及模型可视化辅助。
- [tools/README.md](tools/README.md)：训练吞吐审计器及标准库测试。
- [remote/README.md](remote/README.md)：LAN worker 和状态拉取工具。
- acceptance/：按 trial 保存的冻结验收输入、运行日志与证据快照；本轮不改其中内容。
- runs/：运行时生成的位置；目录名在 .gitignore 中排除。

## 本目录直接文件

| 文件 | 用途 |
| --- | --- |
| .gitignore | 排除 runs、模型权重、日志、临时文件和本地回放/live 状态输出。 |
| MIGRATION_ACCEPTANCE.md | 记录从 Isaac 到 Native 的迁移决策、协议身份、验收状态和历史证据索引。 |
| requirements-native.txt | 记录旧版 WSL2 ARM64 原生环境观察到的依赖版本；文件注明未验证全新安装。 |
| requirements-native-v2.txt | 记录 V2 匹配环境观察到的依赖版本；文件注明当前隔离 overlay 未验证全新安装重建。 |

## V2 环境

当前冻结路线由 Newton MuJoCo Warp V6、匹配的 MuJoCo 与 mujoco-warp 3.15.0、普通 eager GPU 执行。版本细节和已完成的门禁见迁移与验收说明及 [V2 协议](train/PROTOCOL_NATIVE_V2.md)。V2 requirements 记录已观察到的版本，不是安装成功保证；旧 requirements 和 V1/V1.1 入口属于历史路线。

## 训练和评估入口

正式 V2 runner 为 train/run_native_v2_trial.sh。每次使用唯一 tag；runner 校验协议与固定配方，并从头创建 run。评估入口为 train/run_eval_v2.sh，支持 pilot 与 formal 开发评估。实际解释器、协议 SHA、输入身份、参数、状态和验收边界以迁移与验收说明为准。

~~~bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_native_v2_trial.sh <unique-tag> \
  --live-state-file /mnt/c/pioneer/agtc-native-acceptance-backup/native_v2_live/state.json \
  --live-state-interval 5
~~~

评估入口示例：

~~~bash
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_eval_v2.sh <checkpoint.pt> <new-output-dir> pilot
~~~

## 状态查看和能力边界

Native viewer 使用 train 发布的 live-state 数据；静态几何和传感器来源、域差以及 viewer 能说明的范围见迁移与验收说明。吞吐、平台链路、短测或开发评估不等同于导航能力或实车验收。冻结 trial 快照位于 acceptance/，不应通过此目录 README 对它们的内容或结论作改写。
