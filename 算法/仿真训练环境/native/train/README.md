# 原生训练与评估入口

本目录使用 `../proto/` 的 Newton XPBD/Warp 环境、仓库五路编码器与 LSTM-512/PPO，无需 Isaac Sim/Kit。状态、证据与本机路径限制见 [迁移验收说明](../MIGRATION_ACCEPTANCE.md)。

正式配方为64 env×32 steps、LSTM-512、fast_stop_v2、五路真实观测、1ms/50。旧CLI默认保留原型兼容性，不能省略传感器/任务参数冒充正式配方。

```bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native
PYTHON=/opt/venv/bin/python bash train/run_native_v1_trial.sh <new-tag>
/opt/venv/bin/python tools/benchmark_training.py --help
PYTHON=/opt/venv/bin/python bash train/run_eval_pilot_4seeds.sh <checkpoint.pt> <new-out-dir>
PYTHON=/opt/venv/bin/python bash train/run_eval_16seeds.sh <checkpoint.pt> <new-out-dir> 87010-87025
```

产物：metrics.jsonl、identity.json、observation_contract.json、checkpoint；成功结束才有summary.json。失败记录不得用旧summary或目录名冒充完成。resume必须核对策略、任务/奖励/终止、物理、观测契约；使用新输出目录保存恢复测试。

身份规范native_task_contract_v2排除profile、并行world数及运行元数据，规范每环境拓扑，支持训练64环境→评估1环境；不同task/物理/传感器仍拒绝。旧无版本checkpoint默认拒绝，只有eval可用显式task+legacy opt-in探索，输出不会标为strict_match。

PROTOCOL_NATIVE_V1.1与V1原文及伴随SHA冻结保留。V1.1旧正文仍有5ms示例/旧wsl_train路径，执行以修正入口为准；5ms/10只属实验配置，不能作正式训练。

五路传感器已有真实解析实现；RTX域差、代理物理保真、16-seed配对能力评估仍未完成。capability_metrics_allowed=false；吞吐与链路冒烟不构成导航能力结论。
