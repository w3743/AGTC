# 8192 步运行前后原始激活对照

本对照直接读取复制后的 JSON 与 NPZ 数组，CPU-only；没有用 PNG、色标或截图变化代替 raw tensor 对比。原始导出未移动或覆盖，零步 baseline 帧留在 `network_exports`，未混入这组配对。

## 帧与元数据

- 运行前：`frame_20261020_115035_283818`，seq=2，sim_time=0.0 s，暂停预览（未施加车辆）。
- 运行后：`frame_20261020_115246_663781`，seq=349，sim_time=17.30000000000011 s，output_time=17.35000000000011 s，实际驾驶推理（动作前观测）。
- 两帧动作合同和 checkpoint label 相同：`checkpoint loaded; task competence is not implied`。运行后 seq 增加 347，sim_time 增加 17.300000 s。
- 两帧的 NPZ/JSON 都是从 `results/isaac_field_viewer/network_exports/` 原样复制；逐文件 SHA-256 相同，详见 `raw_comparison.json`。

## Raw 数组结果

- 60/60 activation 数组与 metadata shape 一致且全部有限；其中 50 层 raw 数组改变，10 层逐元素相同。
- 77/77 权重数组有限且运行前后完全相同；本组变化来自运行时输入/状态，而非检查点权重切换。

### 改变的 activation 层

- `input.lidar`
- `features_extractor.laser.0`
- `features_extractor.laser.1`
- `features_extractor.laser.2`
- `features_extractor.laser.3`
- `features_extractor.laser.4`
- `features_extractor.laser.5`
- `features_extractor.laser.6`
- `features_extractor.laser.7`
- `input.depth`
- `features_extractor.depth.0`
- `features_extractor.depth.1`
- `features_extractor.depth.2`
- `features_extractor.depth.3`
- `features_extractor.depth.4`
- `features_extractor.depth.5`
- `features_extractor.depth.6`
- `features_extractor.depth.7`
- `input.state`
- `features_extractor.proprioception.0`
- `features_extractor.proprioception.1`
- `input.feedback`
- `features_extractor.feedback.0`
- `features_extractor.feedback.1`
- `features_extractor`
- `lstm_actor`
- `lstm_actor.h0`
- `lstm_actor.c0`
- `lstm_actor.h`
- `lstm_actor.c`
- `lstm_actor.i`
- `lstm_actor.f`
- `lstm_actor.g`
- `lstm_actor.o`
- `mlp_extractor.policy_net.0`
- `mlp_extractor.policy_net.1`
- `action_net`
- `lstm_critic`
- `lstm_critic.h0`
- `lstm_critic.c0`
- `lstm_critic.h`
- `lstm_critic.c`
- `lstm_critic.i`
- `lstm_critic.f`
- `lstm_critic.g`
- `lstm_critic.o`
- `mlp_extractor.value_net.0`
- `mlp_extractor.value_net.1`
- `value_net`
- `action.applied`

### 保持不变的层与可由 raw 证据支持的原因

| 层 | 观测到的原因 |
|---|---|
| `input.detections` | input layer raw array is identical in both exports；输入数组 min/max=0/0，非零元素 0/128 |
| `features_extractor.boxes.0` | all declared parent activations are unchanged; no upstream raw change is observed |
| `features_extractor.boxes.1` | all declared parent activations are unchanged; no upstream raw change is observed |
| `features_extractor.boxes.2` | all declared parent activations are unchanged; no upstream raw change is observed |
| `features_extractor.boxes.3` | all declared parent activations are unchanged; no upstream raw change is observed |
| `detections.masked` | all declared parent activations are unchanged; no upstream raw change is observed |
| `detections.pooled` | all declared parent activations are unchanged; no upstream raw change is observed |
| `detections.winner` | all declared parent activations are unchanged; no upstream raw change is observed |
| `action.log_std` | actual policy distribution parameter is identical in both frames; all exported checkpoint weight arrays also match |
| `action.std` | all declared parent activations are unchanged; no upstream raw change is observed |

检测支路输入及其后续编码保持相同；raw 导出显示检测输入未变化。对父激活未变的层，仅报告其上游输入未变化，不推断更深原因。策略 `log_std` 参数也保持不变，且所有 77 个权重数组完全相同。

## 验收

- CPU 比较结果：`passed`；所有 metadata、60 层 finite/shape、权重一致、原始文件哈希及 baseline 排除门禁通过。
- 机器可读逐层记录：`raw_comparison.json`。实际 CPU 执行退出码见 `raw_comparison_cpu.exit.json`。
- 这仅证明同一个已加载 8192-step checkpoint 在暂停与实际运行帧之间产生了原始 activation 变化；不代表任务能力已学会或导航效果已提升。
