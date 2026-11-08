# 指令条件化组合诊断

本阶段的主指标是零步、换序并拆散的两操作数模 4 题。三个固定种子的联合正确率都须达到 85%。每组固定历史，只换最后的加减指令，再分别改每个操作数。四个变体都答对才记为一组正确。另测 384 和 768 步，以及三操作数边界。零步只表示记录结束到查询之间没有尾部干扰。

## 取证和组合

- 指令门控让查询投影逐维调节历史投影。证据门控以历史投影生成门，查询仍以加法参与。原读出使用查询与历史向量的拼接。
- 最终配置以边界符号查询历史，将真实指令留给读出。相邻关系表对整个历史中的有向相邻符号计数，按前一符号归一化后继分布，并学习投影。它保留角色与值对应的线索。模型同时保留稀疏图读取的残差。
- 相邻关系表按符号身份归并，未引用操作数、加减规则、正确答案和正确边。符号相邻关系是一项结构先验。整个原型保留完整历史。
- 对照包括共用关系表的证据门控、共用关系表的原读出，以及原共享地址图。每组 107,877 个参数，训练问题和损失相同。主轮训练只用两操作数。三操作数是未见操作数数量的探针。

## 留出结果

三种关系表读出在两种布局及 0、384、768 步的三种子上均为 100% 联合正确。拆散布局的原共享地址图均值依次是 30.47%、30.47%、31.25%。固定指令门控的权重，移除稀疏图残差后仍为 100%。只保留稀疏图时，拆散题依次为 4.17%、2.86%、2.08%。这些对照将主要收益定位到相邻关系表。

只换读出指令，在两操作数的换指令变体上正确率为 100%。沿用旧读出指令则为 0%。三操作数的联合正确率为 0%，即使稀疏图在零步拆散题上的操作数覆盖达到 99.48%。本轮证明了两操作数的指令组合和长度外推，稀疏拓扑独立保存关系及泛化到三操作数仍待验证。

## 复跑

在本目录执行。输出目录应为新的目录。命令使用原题及对应的指令反事实，每批 16 条序列，共 1,000 次更新。三个种子使用相同口径，只有开发集全部达标才生成留出预测。

```powershell
python conditional_readout_eval.py train C:/pinoeer/AGTC/算法/rnn研究/runs/conditional_readout_reproduction --workers 8 --contrastive --train-arities 2 --neutral-routing --binding --ranking --relation-bank --modes conditioned history_gate shared original
python conditional_readout_eval.py test C:/pinoeer/AGTC/算法/rnn研究/runs/conditional_readout_reproduction --workers 8
python conditional_readout_report.py C:/pinoeer/AGTC/算法/rnn研究/runs/conditional_readout_reproduction
python -m unittest -v test_frontier_memory
```

实际配置、权重、原始预测和报告在 `C:/pinoeer/AGTC/算法/rnn研究/runs/conditional_readout_v7`。同名的 v1 至 v6 目录保存此前只用开发集的对照。
