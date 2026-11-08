# 按键预测 20 神经元

本地运行的 20 神经元数字预测原型。每次按下 0–9，系统先揭晓并计分上一轮封存的预测，再训练并生成下一轮预测。

**迁移说明：**原目录的启动脚本与历史验证记录已迁入本目录。历史结果对应记录中的代码和环境，不代表迁移后的版本已经重新验证。

## 运行

需要安装项目依赖中的 CPU 版 PyTorch。启动脚本默认使用旧机器的 Python 路径；在其他环境中请通过 `-Python` 指定解释器。默认启动：

```powershell
& '.\start.ps1'
```

可指定 Python、端口、运行数据目录和随机种子：

```powershell
& '.\start.ps1' -Python 'C:\pioneer\mnist-local-neuron-env\Scripts\python.exe' -Port 8765 -DataRoot 'C:\pioneer\local-neuron-runs' -Seed 0
```

服务仅监听本机，默认端口为 `8765`。网页由 `dist/` 提供。

## 文件

- `model.py`、`server.py`：预测模型、训练逻辑与本地 HTTP 服务。
- `start.ps1`、`requirements.txt`：启动脚本与 Python 依赖。
- `dist/`：网页文件 `index.html`、`app.js`、`style.css`。
- `test_model.py`、`test_server.py`：模型与服务测试。
- `验证记录.md`：实验协议、结果与证据路径。

## 固定训练机制

数字以 10 维 one-hot 输入网络，网络有 20 个神经元。每次训练重放最近最多 32 个数字，用窗口内全部相邻数字对的平均交叉熵训练；有可训练数字对时，每键最多做 3 次 Adam 更新，学习率为 `0.015`。每次更新后仅对输入权重 `Win` 做强度 `λ=0.02` 的逐元素软阈值收缩，可将权重置为精确零。循环权重不做该收缩。窗口状态从零重放，模型参数和优化器跨按键保留。

## 运行记录与验证

最新流式实验：[单位输入、连续状态与在线RTRL](experiments/STREAMING_RTRL.md)。该实验每次只消费一个新数字，保持活动与连接痕迹，包含在线梯度验证及三类对照；实验代码与网页默认模型独立。

活动保留对照：[神经元活动保留系数](experiments/ACTIVITY_RETENTION.md)，在连续状态RTRL中比较原活动更新、固定0.5保留与逐神经元可学习保留，三组使用相同输入、种子与更新次数。

全输入记忆：[30路输入源记忆与历史实验对照](experiments/INPUT_MEMORY.md)，为外部数字及神经元间信号增加历史摘要，保留当前信号直接通路，覆盖既有九项任务。

2026-10-28 的实验入口：[遗忘参数权重衰减对照](experiments/README.md)、[独立连接痕迹与逐神经元遗忘速度](experiments/PER_NEURON.md)、[同计算预算代表网络对比](experiments/REPRESENTATIVE_BENCHMARK.md)、[0–9 猜数字六任务对比](experiments/DECIMAL_GUESS.md)、[20 神经元独立线性输出层对照](experiments/LINEAR_READOUT.md)、[原二进制三任务全量对比](experiments/BINARY_READOUT.md)。逐神经元版本及独立输出层为实验专用模型，默认模型仍使用共享遗忘参数。

遗忘参数 `decay_logit` 单独设置 `weight_decay=0`，由预测损失学习保留比例，避免正则项将其拉向 `sigmoid(0)=0.5`。其他参数保留 Adam 的 `weight_decay=1e-4`；输入权重的软阈值收缩保持原设置。连接痕迹仍按当前保留比例自然衰减。此修改不表示命中率已经提高，旧对照成绩属于修改前的设置。

优化器现有两个参数组。旧检查点的模型权重名称和形状不变，但旧单组优化器状态不能直接加载到新优化器，需要先迁移参数组与状态；当前服务没有检查点恢复入口。

运行日志、检查点和源码快照写入项目外的数据目录。每次启动都会创建新的 run 和会话，并保留既有记录。最新验证中，模型测试 6 项通过；服务端 40 次按键、117 次更新及输入权重计数、快照和检查点检查通过。seed 2 的 600 键循环与三连循环命中率为 98.33% 和 85%（单 seed）；历史实验协议和完整证据见[验证记录](验证记录.md)。
