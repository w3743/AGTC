# 全输入源记忆 RNN 研究

当前研究版本是20神经元的连续状态 RTRL 模型：10路外部数字和20路上一时刻神经元活动各保留一份历史摘要；输出层可学习，不使用活动保留。每次先预测并计分，再使用揭晓的数字训练，最后推进一次状态。输入保留率可学习，模型参数、活动、连接痕迹和输入摘要持续保留。

## 运行

安装 `requirements.txt` 中的 PyTorch 后，在本目录运行：

```powershell
python .\run.py --digits 000111000111
```

每行输出揭晓前的预测、实际数字、命中情况和预测概率。不传 `--digits` 则进入交互输入，输入 `q` 结束；关闭程序后状态不会自动保存。模型 API 可从 `experiments/input_memory_rtrl.py` 的 `InputMemoryRTRL(scope="all", mode="learned")` 使用。旧网页使用窗口重放模型，已归档，不会由此命令启动。

## 文件

| 文件 | 用途 |
| --- | --- |
| `run.py` | 全输入源记忆模型的命令行运行入口 |
| `experiments/input_memory_rtrl.py` | 全输入源记忆和在线学习实现 |
| `experiments/streaming_rtrl.py`、`experiments/linear_readout.py`、`model.py` | 当前模型依赖的连续状态、输出层和基础网络组件 |
| `experiments/test_*.py` | 输入记忆及其基础组件的机制测试 |

在 `experiments` 目录执行 `python -m unittest test_input_memory_rtrl test_streaming_rtrl`。

长程记忆测试涵盖延迟复述、按键查值和模 4 加减，并与六种标准网络配置比较。任务、评分和复跑说明见[长程记忆评测](experiments/LONG_MEMORY_EVAL.md)。

1024 神经元稀疏原型使用固定 32 入度和完整序列 BPTT。成绩、CPU 资源测量口径与复跑命令见[稀疏原型评测](experiments/SPARSE_MEMORY_EVAL.md)。

六种标准模型的两档参数量、独立验证选参、训练量和长度外推结果见[标准模型公平对照](experiments/STANDARD_MEMORY_EVAL.md)。

修正版评测固定前缀，只交换查询键或加减指令。题型、数据隔离和复跑命令见[查询切换配对评测](experiments/QUERY_PAIR_EVAL.md)。

自适应记忆图让查询根据内容选择实际历史连边。三组对照、换边干预和成绩见[自适应记忆拓扑实验](experiments/ADAPTIVE_GRAPH_EVAL.md)。

六个模型冻结权重，在新题上复测到 1,024 步并加入线索换序和拆散。见[拓扑复测与图形化成绩](experiments/ADAPTIVE_GRAPH_RETEST.md)。

固定容量槽图、八模型公平对照、封存门槛和第一轮诊断见[多维度超越实验](experiments/MEMORY_FRONTIER.md)。

共享地址图用独立留出题检验零步和长延迟查值，并通过复用旧边测试查询换边的贡献。见[共享地址路由诊断](experiments/SHARED_ADDRESS_EVAL.md)。

模 4 组合实验分离取证和读出指令，并用相邻关系表及证据消融定位两操作数成绩的来源。见[指令组合诊断](experiments/CONDITIONAL_READOUT_EVAL.md)。

层状分支和带环图的冻结权重外推，见[结构深度实验](experiments/STRUCTURE_DEPTH.md)。

从普通全连接 ReLU 网络起步，比较学习重连、随机重连和始终全连接，并用隐藏层断边检验双证据分工。见[全连接拓扑演化实验](experiments/DENSE_REWIRE.md)。

512 神经元 RNN 的生存动力学、再生和匹配对照见[循环连接生存实验](experiments/EVOLVING_RNN.md)。多任务中断、恢复和新组合的低维诊断见[IMMW 实验](experiments/IMMW.md)。

持有本地归档的协作者可在仓库根目录的 `归档/神经网络研究_20261104/` 查阅历史对照、旧网页、代码快照和运行记录。全输入源记忆的早期实验记录位于该目录下的 `experiments/INPUT_MEMORY.md`。归档只保存在原工作区，当前运行入口和机制测试使用本目录的源码。

## 本地运行环境与证据目录

- 虚拟环境：`.venv`（Python 3.12.14 + PyTorch 2.14.1 CPU），不纳入版本管理。
- 实验证据：`runs/`（各实验的报告、权重、原始预测与日志，约 3.1GB），不纳入版本管理。实验文档中的复跑命令原先引用仓库外旧目录（`rnn-research-runs`），2026-11-07 已随目录迁入统一改为 `C:\pinoeer\AGTC\算法\rnn研究\runs`；`归档/` 下的历史记录保持原始字节，不作改写。
- 合并说明：外部目录 `rnn-research-runs`、`rnn-research-env` 已分别迁入本目录的 `runs/` 与 `.venv/`；`agtc-brake-review-20261105` 审查工作树已并入仓库并删除，审查侧差异字节保存在 `归档/审查分支差异/`（含 `manifest.json`）。
