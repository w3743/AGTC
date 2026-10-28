# BMS 跨板审查记录

本目录支持[跨板统一审查](../README.md)中的 BMS 逐针和故障链结论。当前审查入口为 `BMS全原理图逐针与故障链复核.md`；当前源相关文件由同目录官方快照、网表/BOM 和针网覆盖记录绑定。

## 本目录直接文件

| 文件或文件族 | 用途与边界 |
|---|---|
| [BMS全原理图逐针与故障链复核.md](BMS全原理图逐针与故障链复核.md) | 说明当前 BMS 原理图的逐针和故障链复核结论。 |
| [BMS全原理图逐针与故障链复核.json](BMS全原理图逐针与故障链复核.json) | 保存该复核的结构化数据。 |
| `current-official-snapshot.json`、`current-official-bom.csv`、`current-official-netlist.net`、`current-pin-net-coverage.json` | 保存当前官方源快照、BOM、ENET 网表和针网覆盖记录。 |
| `baseline-official-snapshot.json`、`baseline-bom.csv`、`baseline-netlist.net`、`baseline-api-components-library-search.json` | 保存逐针审查使用的基线快照、BOM、网表和库查询基线。 |
| `current-official-snapshot.pre-full-audit.de5e951c.json` | 保存全图审查前的官方源快照。 |
| `AFE元件连接.before-TAP-B9-update.json`、`BOM注释定点更正.json`、`official-api-boundary-text-fix-diff.json`、`official-api-text-fix-diff.json` | 保存 TAP_B9 更新前或文字/注释修正批次的差异记录。 |
| `P1.before-boundary-text-fix.48247383.esch2`、`P1.before-external-boundary.4d274083.esch2`、`P1.before-temperature-interface.esch2`、`P1.before-text-dedupe.a713d448.esch2`、`P1.before-vishay-text-fix.56f8f62e.esch2` | 原理图修改前的局部源快照，按文件名所指修改批次回溯，不作为当前 P1。 |
| `invoke_cli.py` | 将命令行参数传给指定的嘉立创 EDA CLI `invoke` 会话并返回输出。 |
| `probe_before_edit.js` | 从活动 EDA 会话读取工程/器件属性，用于编辑前探查。 |
| `pad-library-search.json`、`rh5009-official-geometry-review.json`、`temperature_raw_window_candidate.json` | 保存封装库查询、RH5009 几何复核及温度 RAW 窗口候选记录。 |

目录内修改前 `P1`/基线/API 快照用于追溯历史批次；如与当前 BMS 模块 README 或当前官方报告冲突，按当前源 SHA 与正式复核报告解释。
