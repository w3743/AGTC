# Native V2 训练与评估入口

当前正式路线为 Ubuntu 24.04 ARM64 WSL2、Newton MuJoCo Warp V6、MuJoCo/mujoco-warp 3.15.0 匹配环境、普通 eager GPU 与 NOT_GUARANTEED。协议 V2 已冻结；V6 1/3/64-world physics gate 与 checkpoint resume / strict-eval smoke 已通过，正式 300 轮 run 仍在运行，尚无最终通过结论。完整状态与历史原始证据见 [迁移与验收说明](../MIGRATION_ACCEPTANCE.md)。

冻结身份：
- protocol SHA256：83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e
- task SHA256：209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2
- source revision：7d8a167f
- source manifest SHA256：6026336cad5b961e2fe95885d6010781ae75e59e792d1715988bda45d35e8938

解释器为 /opt/venv_mujoco315_smoke/bin/python；实际依赖版本与隔离 overlay 限制见上级说明。旧 /opt/venv / MuJoCo 3.17.0 不匹配 V2。

## 正式训练

每次使用新的唯一 tag；runner 验证协议与固定配置、创建新的时间戳 run 目录，从头训练。不要覆盖旧 run 或续训 XPBD checkpoint。

~~~bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_native_v2_trial.sh <unique-tag> \
  --live-state-file /mnt/c/pioneer/agtc-native-acceptance-backup/native_v2_live/state.json \
  --live-state-interval 5
~~~

固定配方：64 env × 32 steps × 300 iterations、LSTM-512/PPO、seed 0、threads 8、fast_stop_v2、Newton MuJoCo Warp、1 ms × 50、五路真实观测、ordinary eager、NOT_GUARANTEED、nonfinite-limit 0。额外参数只允许 live-state 文件、发布间隔与间距。

## 评估

~~~bash
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_eval_v2.sh <checkpoint.pt> <new-output-dir> pilot
~~~

pilot 固定 seed 87010–87013；formal 模式使用 87010–87025，最多 1200 policy steps。runner 先做协议与 checkpoint contract 预检；物理、task 或通道身份不同必须拒绝。resume/eval smoke 已验证 task SHA strict match；checkpoint 不保存 RNG 状态，不能声称随机流 bitwise 接续。

## 当前验收状态

V2 正式 run native_v2_formal_v6_live_20261025_224005 于北京时间 22:40:05 启动，目前尚未完成 300 轮最终门。每 5 policy steps 的 live-state viewer 使用训练 raw state/目标/任务时间、CPU 导出的 19-body static geometry 与实际 flat plane；旧 Isaac USD 缺少两个云台和两个前导轮共 4 个 body 的几何，旧 viewer 方块 fallback 不再用于 Native V2。这不含传感器图像，也不等于 GPU shape-array byte comparison。UI、Git 和 CPU geometry export 负载须包含在正式长测实际耗时中。

capability_metrics_allowed=false。短 benchmark、checkpoint smoke 和 viewer 均不能替代正式 300 轮稳定性或导航能力验收；V1/V1.1 与 XPBD 记录只作历史证据。