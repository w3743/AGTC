# 编码器拓扑与预览记录（2026-10-19）

本记录描述当前策略编码器结构及其验收边界。策略实现见 `isaac_field/policy.py`，环境观测接口见 `isaac_field/env.py`。传感器/状态数据契约仍为 `sensor-depth-only-v3`；checkpoint 内的策略编码器 schema 为 `depth64-lidar-circular1d-v1`。两者分别标识输入数据语义和编码网络结构。

## 当前结构

四路输入形状为 LiDAR `(2, 400)`、深度 `(1, 160, 88)`、检测 `(16, 10)`、状态 `(43,)`。每条支路的结构预先固定，RecurrentPPO 在训练中学习其权重，包括卷积核权重；这不是自动架构搜索。

| 支路 | 编码路径 | 输出 |
| --- | --- | ---: |
| LiDAR | `Conv1d 2→16, k=5, s=2, p=2, circular`: 400→200；`Conv1d 16→32, k=5, s=2, p=2, circular`: 200→100；自适应平均池化到50；展平1600；线性层1600→32 | 32 |
| 深度 | `Conv2d 1→8, k=3, s=2, p=1`: 160×88→80×44；`Conv2d 8→16, k=3, s=2, p=1`: 80×44→40×22；自适应平均池化到8×6；展平768；线性层768→64 | 64 |
| 检测框 | 每个候选框10→32→32；乘以输入第8列presence，再沿16个候选按特征取最大值 | 32 |
| 状态 | 43→32 | 32 |

拼接顺序为 LiDAR、深度、检测、状态，得到160维特征，再送入 RecurrentPPO 的循环策略。动作仍是四维连续动作。旧编码器权重即使输入空间维度相同，只要 checkpoint 没有当前 encoder schema 或记录了其他 schema，也会被拒绝加载；不能把旧权重当作新结构权重继续使用。

深度像素仍按 `np.where(valid, Z, 0) / depth_range` 生成，实际值在 `[0, 1]`，无效像素为0且没有额外输入掩码。环境 `Box` 的声明范围为 `[-1, 1]`，用于让 Isaac Lab 官方 `Sb3VecEnvWrapper` 将浮点深度图识别为已归一化图像并原样传递；这没有把深度数据改成负值，也没有改变传感器的量程归一化。官方 wrapper 的逐值验证通过，见 `native_sb3_depth.log`。

## 验收与运行状态

- `cpu_validation_after_depth_space.log`：完整 CPU 回归 **172 passed, 9 skipped**。
- `native_sb3_depth.log`：官方 `Sb3VecEnvWrapper` 深度精确值专项 **1 passed, 37 warnings**；warning 来自依赖弃用/兼容提示。
- 早期0步随机初始化预览曾在二维浏览器从仿真时间0秒运行至20秒，记录到序列401（输入时间19.95秒、反馈时间20.00秒）。`runtime_acceptance.json` 记录无错误、训练步数0；深度激活、LSTM和动作 PNG 改变。该片段 LiDAR 输入保持全量程且未变化，因此第一层 LiDAR 卷积也未变化。该预览包保存的深度 Box 为 `[0, 1]`，而当前环境已改为 `[-1, 1]`；其浏览器/Netron结果保留作历史证据，不能再作为当前viewer checkpoint使用。
- `layout_verified.json` 记录61张卡片、无重叠，数据更新前后 viewBox 相同。
- 兼容当前观测空间的0步随机初始化基线为 `results/isaac_field_improvement/fresh_depth_radar_20261019_v2/untrained_baseline.zip`，SHA-256 为 `b2ecd57b73ebb8fdc9fa95622b166af82c1b0ab964a2daaf226be37ffffc954a`；v2训练协议从新初始化开始，不迁移旧权重。
- `netron_export.log` 记录旧0步预览的 TorchScript 导出及3组×3步非零 actor/critic h/c 数值对照，最大误差为0；这是静态结构/数值导出证据，不代表它可用于当前viewer，也不验证任务能力。
- `results/isaac_field_improvement/fresh_depth_radar_20261019/` 中的 v1 试验在官方 wrapper 的 image-bound 检查处失败，退出码1、训练步数0；该失败输出保留。v2 已在 WSL PID 3165 / exec 20811 启动；15:46:15 UTC 的中间状态为24,576步、`training_near`，协议 SHA-256 为 `f32707b8fdfa8098613d766b6ce38d0edbd74bf396c2c6abea432f698b9d9e84`，冻结27个源码文件。截至本轮文档更新，训练到32,768步，near阶段完成并进入steer30；训练仍在运行，尚无最终训练或任务能力结论。
- 车辆页已真实加载 v2 的8,192步checkpoint `stage_1_near/checkpoints/field_policy_near_8192_steps.zip`，`../../isaac_field_improvement/fresh_depth_radar_20261019_v2/checkpoint_8192_validation.json` 的 `validation_passed` 为 true：卷积/LSTM/动作权重更新且有限，Adam有35项状态、最大step为300，27个源码哈希及旧模型哈希匹配。`trained_8192_runtime_acceptance.json` 核对二维激活：输入时间1.45→5.10秒、输出时间1.50→5.15秒；LiDAR输入、两层Conv1d、深度池化、LSTM h和动作图像均变化，错误列表为空。所选Conv1d数值从 `-1.08233726` 变为 `-0.602363467`，检查后暂停。该记录验证checkpoint完整性、模型加载与激活同步，不代表导航能力或FPS。
- 历史196608步权重仍保留在 `results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192/stage_04_turn90.zip`，SHA-256 为 `11f6c287e8bfdbb348fb1c9d9205c2a95692f1629ef8cee4626aef3947ae506f`，与既有记录一致。它属于旧编码器 schema，不能加载到当前结构；原始权重、旧协议和成绩没有被本次结构更新改写。

本轮没有开展训练速度/FPS基准。完整项目说明见 [ISAAC_LAB.md](../../../ISAAC_LAB.md) 和 [使用说明.md](../../../使用说明.md)。

## 训练参数真实性补充

`fresh_depth_radar_20261019_v2/checkpoint_8192_validation.json` 只读CPU核对通过：8192步雷达/深度卷积、LSTM及动作头权重相对正式0步基线实际更新且均为有限值；Adam含35项参数状态且step>0，27个源码快照/当前源哈希和旧模型SHA一致。查看器当前加载8192步快照，右侧“参数：weight”为该检查点的只读权重；车辆运行只更新激活，不会自动载入训练中产生的新权重。`trained_8192_kernel_weights.png` 记录首层Conv1d的16×2×5权重。
