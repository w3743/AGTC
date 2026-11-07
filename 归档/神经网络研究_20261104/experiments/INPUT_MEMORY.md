# 输入记忆 RTRL 实验与报告协议

## 实验问题

在既有在线二进制和十进制下一符号预测任务上，为每个输入源增加一个共享指数记忆摘要，评估其与现有 RTRL 基线的差异。模型按相同任务、种子和 600 步序列配对；每条序列在首个预测后执行 599 次优化更新。新旧实验的架构、数据任务、训练步数和计算量不完全相同，因此历史对照用于定位结果，不构成严格的等算力因果结论。

## 模型定义

每步输入源为 `s=[x,h_prev]`，共 10 个外部输入通道和 20 个上一时刻隐藏活动。对每个源只维护一个记忆摘要：

```text
m_new = rho * m_old + (1-rho) * s
candidate = 原有候选状态表达式 + V * m_old
```

其中 `V` 的形状为 `20×30`，并初始化为零。记忆是按源共享的摘要，不为 600 个时间步各自保存一份，也不保留无限长度的输入数字序列。既有隐藏状态 `h` 和快速权重 `F` 按原机制持续更新；RTRL 灵敏度增加 `E_m`，用于传播这 30 个记忆维度的影响。

| 模型 | 输入记忆率 | 总参数 | RTRL core 参数 |
|---|---|---:|---:|
| `baseline_rtrl20` | 无输入记忆 | 851 | 641 |
| `fixed_input_memory20` | 固定 `rho=0.8` | 1451 | 1241 |
| `learned_input_memory20` | 30 个可学习 `rho` | 1481 | 1271 |

输入记忆率 `rho` 的权重衰减为 0，其余训练设置沿用现有 RTRL 协议；本实验不增加 activity retention `a`。具体学习率、普通权重衰减、梯度裁剪、近似计算代理及机器环境以新 run 的 `protocol.json` 与 `compute_profile.json` 为准。

## 配对任务与样本量

三种模型分别运行以下 9 项任务、3 个种子（`2`、`20261027`、`20261028`），总计 81 次 run、48,600 个预测步、48,519 次非首步优化更新。

| 任务族 | 任务 ID | 展示名称 |
|---|---|---|
| 二进制 | `binary_alternating` | `01` 交替 |
| 二进制 | `binary_triples` | `000111` 三连 |
| 二进制 | `binary_period16` | 周期 16 |
| 十进制 | `decimal_cycle` | 循环 |
| 十进制 | `decimal_triples` | 三连 |
| 十进制 | `decimal_zigzag` | 锯齿 |
| 十进制 | `decimal_fibonacci_mod10` | Fibonacci mod 10 |
| 十进制 | `decimal_switching` | 切换 |
| 十进制 | `decimal_iid_random` | IID 随机 |

注意：`binary_triples` 与 `decimal_triples` 是不同任务，不能因名字相似而混为一组。历史阶段 `decimal_guess`、`linear_readout` 的任务属于十进制任务族；其他阶段的 `alternating`、`triples`、`period16` 属于二进制任务族。legacy `retention_no_decay` 任务名和 last-50 指标按原始定义保留，不映射成新实验任务或 last-100 指标。

## 结果文件约定

报告脚本读取传入的完成 run 目录，要求至少有：

- `aggregate.json`：27 个模型×任务聚合行，包含准确率、首/末 100 步准确率、平均 NLL 等。
- `runs_summary.json`：81 个种子级 run 条目，包含种子、模型、任务、更新数和结果目录。
- `audit.json`：必须给出明确通过状态；报告脚本会读取并拒绝没有通过标记的输入。
- 每条 run 的 `events.jsonl`：600 个连续预测步，供学习曲线读取。
- `compute_profile.json`：3 个模型配置的参数量、完整 run 代理和稳态代理、资格迹/eligibility 开销等计算量口径。

事件曲线按任务、种子读取 `hit`，计算 50 步滚动准确率，再对三个种子求均值。报告不会把旧阶段的计算预算折算为新实验排名，也不会将结果差值解释成输入记忆的单独因果效应。`aggregate.json` 是种子结果的汇总，不能作为额外训练 run 再计一次。

## 历史 catalog 处理

默认优先读取新 run 目录中的 `history_catalog_normalized.json`，否则读取 `C:\pioneer\rnn-research-runs\input_memory_20261031\history_catalog_normalized.json`；也可用 `--catalog` 指定文件。冻结 catalog 的 463 条原始记录保存在 normalized JSON 的 `original_records` 字段；规范化记录根据每条原 summary/aggregate 重建，修正 retention weight decay、per-neuron 两组的模型名和嵌套 aggregate 展开。报告输出完整规范化历史 CSV 和原 catalog 463 条记录 CSV。463 是 catalog 条目数，不等于 463 次独立训练。

历史聚合表按阶段/模型/任务分组，二进制三项和十进制六项分别呈现，权重衰减模式、shared/per_neuron 模型以及19神经元模型均保留自己的模型名。一个分组若有 seed run，就不把 aggregate-only 行再当作一条训练记录；跨阶段重复仍保留，并标注来源复用或依赖关系，不能当作独立证据。十进制旧阶段强制归为十进制任务族，避免 task 名中的 `triples` 被错误并入二进制。

规范化脚本 `normalize_history_input_memory.py` 逐文件读取历史 run 的 `checksums.json` 与 `analysis_checksums.json`，核对 aggregate、protocol、summary、input sequences 和引用到的 source snapshots。规范化 JSON 的 `checksum_verification` 分开报告 manifest 匹配、未被旧 manifest 列出以及找不到原 run manifest 的文件。legacy `retention_no_decay` 没有旧 run checksum manifest，只能记录采集时 SHA-256，不能表述为历史 manifest 验证。workspace 当前的实验定义脚本不作为历史数据完整性的依据。

十进制历史阶段单列，避免进入二进制排名。历史中没有相同任务定义的记录保留原任务名，作为描述性参考。catalog 提供的 checksum 和 source origin 只用于溯源及发现重复来源，不推断随机种子间完全独立。

## 生成离线报告

在得到完整 run 目录后执行：

```powershell
$py = 'C:\Users\wangj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py 'C:\pioneer\先进通用履带底盘工程\算法\rnn研究\experiments\report_input_memory.py' 'C:\pioneer\rnn-research-runs\<new-run-directory>'
```

若 catalog 不在默认位置，可附加 `--catalog <catalog.json>`。脚本使用 `C:\pioneer\rnn-report-deps` 中的 Matplotlib 依赖和 Windows 微软雅黑字体，生成完全离线的 `report.html`，并在 run 目录旁写入：

- `input_memory_binary_learning_curves.png` 与 `.svg`：二进制 3 任务曲线。
- `input_memory_decimal_learning_curves.png` 与 `.svg`：十进制 6 任务曲线。
- `input_memory_new_results.csv`：新实验模型/任务聚合表。
- `input_memory_history.csv`：按阶段、模型、任务及种子整理的历史对照表。
- `input_memory_history_binary_architectures.csv`、`input_memory_history_decimal_architectures.csv`：历史主要架构按任务族分别聚合。
- `input_memory_history_all_records.csv`：完整规范化历史记录；`input_memory_history_catalog_records.csv` 保留冻结 catalog 的463条原始记录。

## 正式运行结果（2026-10-31）

正式 run 位于 `C:\pioneer\rnn-research-runs\input_memory_20261031\run_20261031T042507_512398Z`。81 条训练均完成：3 个模型 × 9 项任务 × 3 个 seed；共 48,600 次预测、48,519 次更新。以下准确率为 1,800 个预测的跨 seed aggregate；差值为可学习输入记忆相对 baseline 的百分点。

| 任务 | baseline | fixed | learned | learned − baseline | learned last100 | learned mean NLL |
|---|---:|---:|---:|---:|---:|---:|
| binary alternating | 99.56% | 99.56% | 99.56% | +0.00 | 100.00% | 0.033 |
| binary triples | 87.11% | 58.28% | 70.67% | −16.44 | 100.00% | 0.338 |
| binary period16 | 43.94% | 40.56% | 41.72% | −2.22 | 40.00% | 0.814 |
| decimal cycle | 95.06% | 91.00% | 91.50% | −3.56 | 100.00% | 0.291 |
| decimal triples | 40.72% | 17.06% | 34.56% | −6.17 | 61.33% | 1.911 |
| decimal zigzag | 70.11% | 83.33% | 80.56% | +10.44 | 100.00% | 0.560 |
| decimal Fibonacci mod 10 | 15.00% | 17.39% | 14.83% | −0.17 | 15.67% | 2.429 |
| decimal switching | 64.72% | 67.39% | 69.94% | +5.22 | 91.67% | 1.138 |
| decimal IID random | 9.00% | 9.28% | 10.39% | +1.39 | 10.33% | 2.562 |

可学习记忆在 decimal zigzag（+10.44 pp）与 switching（+5.22 pp）上高于 baseline；IID random 高 1.39 pp，但 10.39% 仍接近十类机会水平，不表示可预测真正随机输入。五项低于 baseline，binary alternating 持平。因此本轮结果没有形成全面优势，也不应将跨任务均值或历史对照写成整体胜出结论。decimal triples 和 binary triples 是各自独立的任务，不能跨族合并排名。

历史 MLP/TCN/GRU 架构表的 binary period16 参考值分别为 MLP 97.11%、TCN 94.89%、GRU 88.11%；decimal Fibonacci mod 10 的 MLP 参考值为89.06%，19神经元 `custom_linear20` 为66.39%。历史模型训练步数和更新数与本轮不同，且参数数、全程 proxy 也不等，因此这些数字只作背景。binary_readout 的63/72个旧架构 run 是同种子、同输入下从头重训，并与 representative_compute 的部分事件轨迹逐步完全复现；这是确定性训练复现，不是来源文件复用，也不增加新的种子证据。

| 模型 | 参数 | 稳态 proxy | 全程 proxy | eligibility 元素 | float32 理论字节 | 相对 baseline 全程 proxy |
|---|---:|---:|---:|---:|---:|---:|
| baseline_rtrl20 | 851 | 3,177,051 | 1,906,187,965 | 269,220 | 1,076,880 | 1.000× |
| fixed_input_memory20 | 1,451 | 7,743,362 | 4,645,939,104 | 558,450 | 2,233,800 | 2.437× |
| learned_input_memory20 | 1,481 | 7,930,295 | 4,758,097,251 | 571,950 | 2,287,800 | 2.496× |

Proxy 是协议定义的算术计数，不是硬件 FLOPs。新模型参数与资格迹均大于 baseline；fixed/learned 的全程 proxy 分别约为 baseline 的 2.44× 和 2.50×，所以相同更新数不代表相同计算预算。历史阶段的更新数、架构和计算代理也不一致，报告只按任务族和阶段给出描述性参照。

独立审计通过（`audit.json: passed=true`）：验证 755 个新文件哈希、17 个冻结源码文件、81 个 run、48,600 条预测、567 个检查点、27 个重算 aggregate；另与 5,400 条既有 binary baseline 事件核对，重放 540 条冻结前缀事件，并通过 6 项冻结测试。normalized 历史 checksum 引用验证 569 项，其中 532 项与原始 run manifest 匹配；33 个旧源文件没有可用 run manifest，4 项 legacy 哈希仅为采集时记录。该限制不影响新实验审计通过，但历史溯源强度有差异。

最终离线报告：`C:\pioneer\rnn-research-runs\input_memory_20261031\run_20261031T042507_512398Z\report.html`。报告附二进制三任务与十进制六任务学习曲线 PNG/SVG、27 个新聚合结果、逐任务 baseline 差值、compute profile、分任务族历史架构表、规范化历史 CSV 与冻结 catalog 原始 463 行 CSV。历史模型表明确保留权重衰减两种模式、per-neuron/shared 及 19 神经元模型；decimal_guess/linear_readout 与 binary 历史不混排。

HTML 表格支持本地筛选；PNG 图嵌入 HTML，SVG 与 CSV 作为相邻文件链接提供。报告只有在读取并确认 `audit.json` 显式通过、27 个聚合行、81 个种子记录以及 81×600 个连续事件后才会输出。
