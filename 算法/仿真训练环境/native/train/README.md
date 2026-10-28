# Native 训练与评估入口

本目录包含 Native 训练/评估脚本、训练协议、身份计算及验收门禁。V2 是当前冻结路线；正在运行的 trial、实际进程状态和最终验收结论请查[迁移与验收说明](../MIGRATION_ACCEPTANCE.md)，本 README 不复制实时状态。

V2 使用 Ubuntu 24.04 ARM64 WSL2、Newton MuJoCo Warp V6、MuJoCo/mujoco-warp 3.15.0 匹配环境、普通 eager GPU 和 NOT_GUARANTEED。协议 SHA 为 83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e；task identity SHA 为 209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2。完整固定项与限制以 PROTOCOL_NATIVE_V2.md 为准。

## 正式训练

使用新的唯一 tag；V2 runner 校验协议与配方，为新 run 建立独立目录并从头训练。

~~~bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_native_v2_trial.sh <unique-tag> \
  --live-state-file /mnt/c/pioneer/agtc-native-acceptance-backup/native_v2_live/state.json \
  --live-state-interval 5
~~~

固定配方为 64 env × 32 steps × 300 iterations、seed 0、LSTM-512、fast_stop_v2、五路观测、1 ms × 50 和 zero-nonfinite gate。协议明确允许的附加 live-state 参数、版本限制与验收门请以 V2 协议及 runner 校验结果为准。

## 开发评估

~~~bash
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_eval_v2.sh <checkpoint.pt> <new-output-dir> pilot
~~~

pilot 使用 seed 87010–87013；formal 使用 seed 87010–87025。评估 runner 会在创建环境前校验协议与 checkpoint contract。不匹配时应拒绝运行。formal 开发评估仍不等于 held-out 能力验收。

## 文件索引

| 文件 | 用途 |
| --- | --- |
| check_sha_scope.sh | 运行小型 CPU 身份检查，核对任务契约哈希对运行目录元数据的处理及不同输出目录间的身份一致性。 |
| contract_identity.py | 规范化任务/物理契约，并计算训练和评估共用的版本化 SHA256 身份。 |
| eval_native.py | 对 native policy 做多 seed 回合评估，输出成功、终止原因和轨迹指标；脚本明确保留 native 域能力限制。 |
| make_protocol_v1_1.py | 从 V1 文本生成 V1.1 协议的历史工具脚本；不是编辑已冻结协议的入口。 |
| policy_lstm.py | 实现原生训练使用的五路传感编码器、LSTM-512 actor-critic 与序列前向逻辑。 |
| PROTOCOL_NATIVE_DRAFT.md | 保存早期待审阅的 Native 协议草案。 |
| PROTOCOL_NATIVE_V1.md | 保存冻结的 V1 原生栈协议文本。 |
| PROTOCOL_NATIVE_V1.sha256 | 保存 V1 协议的 SHA256 sidecar。 |
| PROTOCOL_NATIVE_V1.1.md | 保存冻结的 V1.1 协议文本；V1.1 取代 V1，原文保留作历史版本。 |
| PROTOCOL_NATIVE_V1.1.sha256 | 保存 V1.1 协议的 SHA256 sidecar。 |
| PROTOCOL_NATIVE_V2.md | 定义冻结的 V2 平台验收范围、物理/任务身份、训练配方、运行门和能力边界。 |
| PROTOCOL_NATIVE_V2.sha256 | 保存 V2 协议的 SHA256 sidecar。 |
| protocol_sha.py | 按协议文件的规范化规则计算、验证或嵌入协议 SHA；默认目标是 V1.1/V1，V2 可显式指定。 |
| run_eval_16seeds.sh | 对给定 V1.1 checkpoint 执行 16-seed formal 开发评估。 |
| run_eval_pilot_4seeds.sh | 对给定 V1.1 checkpoint 执行 4-seed pilot 评估。 |
| run_eval_v2.sh | 校验后运行 V2 pilot 或 16-seed formal 开发评估。 |
| run_native_v1_trial.ps1 | 在 Windows/PowerShell 下执行 V1.1 冻结配方、保存完整日志并核验生成的契约。 |
| run_native_v1_trial.sh | 在 WSL/Linux 下执行 V1.1 冻结配方、保存完整日志并核验生成的契约。 |
| run_native_v2_trial.sh | 在 WSL/Linux 下执行冻结的 V2 正式训练配方；拒绝未冻结或不匹配的协议/参数。 |
| train_wsl.py | WSL 原生训练 harness：运行 PPO、定期保存 checkpoint，并记录任务身份和训练日志。 |
| trial_checks.py | 仅依赖 Python 标准库的冻结协议、固定配方、输入身份和 trial 产物验收门。 |

V1、V1.1、V2 协议文本及 SHA 文件均是版本化记录；冻结版本不要原地重写。旧协议 runner 不代表 V2 runner。当前迁移状态与历史证据索引见上级说明。
