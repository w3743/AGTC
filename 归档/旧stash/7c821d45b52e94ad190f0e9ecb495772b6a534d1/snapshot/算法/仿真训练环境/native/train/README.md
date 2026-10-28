# 原生训练与评估入口

本目录使用 `../proto/` 的 Newton XPBD/Warp 环境、仓库五路编码器及 `policy_lstm.py` 的 LSTM-512/PPO。无需 Isaac Sim/Kit。当前状态、性能证据与本机路径限制见 [迁移验收说明](../MIGRATION_ACCEPTANCE.md)。

正式配方为 64 env × 32 steps、LSTM-512、`fast_stop_v2`、五路真实观测及 1 ms/50。旧 CLI 默认保留原型兼容性，不传参数不等于正式配方。

```bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native

# 本机已验证的 WSL Python；用全新 tag/输出目录
PYTHON=/opt/venv/bin/python bash train/run_native_v1_trial.sh pilot02

# 性能测量入口：预热与测量分开报告，完整启动耗时单列
/opt/venv/bin/python tools/benchmark_training.py --help

# 评估入口（必须使用兼容的正式检查点；结果仅限 native 域）
PYTHON=/opt/venv/bin/python bash train/run_eval_pilot_4seeds.sh <checkpoint.pt> <new-out-dir>
PYTHON=/opt/venv/bin/python bash train/run_eval_16seeds.sh <checkpoint.pt> <new-out-dir> 87010-87025
```

`train_wsl.py` 输出 `metrics.jsonl`、`identity.json`、`observation_contract.json`、checkpoint 和成功结束后的 `summary.json`。失败记录不能用旧 summary 或目录名冒充完成。使用 `--resume` 时保留相同策略、传感器、物理和任务契约；不同试验使用新输出目录。

`PROTOCOL_NATIVE_V1.1.md` 和伴随 SHA 文件是冻结历史字节，V1 归档保留。冻结文档中尚有早期 5 ms 示例与旧 `wsl_train/` 路径；执行以修正后的正式入口和本轮迁移说明为准，禁止按旧示例启动正式训练。5 ms/10 只属实验配置。

depth/detections 已有解析真实实现和独立算法对照；与 RTX 的逐像素域等价、代理物理保真和 16-seed 配对能力验收仍未完成。`capability_metrics_allowed=false`；短评估与检查点冒烟不构成导航能力结论。
