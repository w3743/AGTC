# 结构深度实验

模型从打散的符号序列为局部候选边打分。候选由三符号窗口提出，符号身份作为共享地址。查询提供起点和指令序列，两类指令与两类记录符号之间的对应由答案损失学习。各步重复使用同一组路由权重。题目给定了边记录和程序边界。

模型用 1–4 跳和最长 64 步尾部干扰训练。每个种子 625 次更新、批量 16。图模型 4,260 参数，学习率 0.01。共用路由消融参数与训练配置相同。GRU 为 100,124 参数，学习率 0.003，训练题及更新次数相同。每个测试条件固定 128 个题组，联合正确要求原题、改单条指令和改单条边都答对。

宽度 4、深度 8 的层状图含 32 个内部节点和 64 条边。三个种子在 1,024 步末尾干扰时均达到 100% 联合正确率和全路径对齐。冻结权重后，带环图在 8、16、32、64 跳均达到 100%。64 跳题每条路径重复访问节点 37–53 次。对换指令题沿用原指令的路由，联合正确率为 0%。

这里的结构是从输入中的显式记录构造出的每题工作图。学习发生在边的识别和使用上，逐步传播是架构先验。实验尚未测量从隐含线索发明新关系或搜索未知路径。

## 复跑

在本目录执行，使用新的输出目录。

```powershell
python structure_depth.py C:/pinoeer/AGTC/算法/rnn研究/runs/structure_depth_reproduction --workers 9
python structure_cycles.py C:/pinoeer/AGTC/算法/rnn研究/runs/structure_depth_reproduction
python structure_depth_report.py C:/pinoeer/AGTC/算法/rnn研究/runs/structure_depth_reproduction
```

实际配置、权重、原始预测和离线报告位于 `C:/pinoeer/AGTC/算法/rnn研究/runs/structure_depth_v1`。
