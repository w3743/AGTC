# 9S4P BMS

**当前责任边界与成本约束（2026-10-22）：**原理图、器件选择、接口合同和原理级计算由当前自动化审查维护；原理图之后的全部工作，包括PCB、封装/布局布线、制造、打样、装配、固件、台架和整车验证，均由人类完成。全电力系统以最低总成本取得最好性价比；BMS约200元板内目标尚未证明，列入BOM不代表已采购。

当前根工程：[BMS-9S4P.eprj3](BMS-9S4P.eprj3)，与同目录 `sch/` 配套。P1 SHA256 `bdba7a90f7a4bbbdf0d03ac986a015a66336af80b713635e28921e73b03e890e`；EPRJ3 SHA256 `4de6f6a845074505476aeec2e3dc4d57c0643c259daa0e84f4aa3d01a47b4805`；采购候选BOM SHA256 `665c9d5c1bde1e44a2aa0fb1a162d1281688810316eee891c23ac5af3a1bb7ba`（248行）。全局当前证明 [跨板并料复核.json](../检查/统一审查/跨板并料复核.json) SHA256 `cce09a3c0439f40b7c8cc882c304d68d2e0b85dc68f8558cd4498ebcd610ff09`，当前官方 ENET SHA256 `537caa4389b5fae40fe5508d4ae1c194f69ed53982c99c0fb8fe04959f53c6eb`，原始API输出ZIP SHA256 `1acad083beca8d0b75e966760734da69ab140dd190c4c39f631f3eb76c57c6fb`。

官方保存后关闭会话、根工程重开、ENET/live API读回已完成：226位号、740针，逐针网名0差异；严格DRC为0错误、9组warning（8组 `SHOWDANDLING`、1组 `ATTRMATCHCODE`）。该结果仅绑定当前EDA源，不是PCB或硬件资格结论。完整入口见[当前设计状态](检查/当前设计状态.md)、[当前源快照](../检查/统一审查/BMS/current-official-snapshot.json)、[740针覆盖](../检查/统一审查/BMS/current-pin-net-coverage.json)和[全原理图故障链复核](../检查/统一审查/BMS/BMS全原理图逐针与故障链复核.md)。

基线说明：validated 215/710源已包含之前的3颗电芯NTC共6针与4个裸孔共4针。当前边界增加11对象/30针；13个B9端点由 `CELL_B9` 改接 `TAP_B9`，使源端保险在分流前有效。不要用旧700针直接加30解释740。 Netport `bdb539cd08ac77ae` 位于 `TAP_B9`，Add into BOM=false、Convert to PCB=false，不计采购/PCB实例。

温度候选：板外 `NTC_CELL1/2/3` 为Semitec `103AT-2`；旧Vishay symbol/device/footprint只作 `Convert to PCB=false` 的CAD几何载体。板内 `NTC_BOARD/NTC_MOS` 为时恒 `MF52A103F3950(A1)`，采用厂家 P477-4B R/T表；P163-6为历史。GPC输入已准备，系数未返回，BQ写入/读回与传感器热耦合和温度故障测试未完成。RAW格式为signed 32-bit little-endian counts；250000/4700000开短窗只是未资格化候选值。

当前11颗1uF电容 `C_BAL_CELL1..9/C_REG1/C_CAN_VCC2_2` 统一使用Samsung `CL10B105KA8NNNC` / LCSC `C29936`。偏压有效容量和硬件效果仍待验证。当前成本以[冻结BOM](原理图准备/BOM.csv)及[元件封装库当前商城成本](../元件封装库/索引/当前商城成本.md)为准；[当前主板成本.csv](检查/当前主板成本.csv)是旧本地快照，不是现行成本入口。

| 内容 | 入口 |
|---|---|
| 当前要求与待办 | [项目要求](项目要求.md) · [TODO](TODO.md) |
| 当前状态与证据 | [设计状态](检查/当前设计状态.md) · [检查目录说明](检查/README.md) · [全图逐针/故障链报告](../检查/统一审查/BMS/BMS全原理图逐针与故障链复核.md) · [温度保护配置](检查/温度保护配置闭合.md) |
| 原理图设计输入 | [原理图准备目录说明](原理图准备/README.md) · [当前采购候选BOM](原理图准备/BOM.csv) |
| Layout与接口交接 | [Layout交接](原理图准备/Layout交接.md) |
| 当前计算与模型 | [结果索引](仿真/结果/README.md) · [源码](仿真/源码/) · [输入](仿真/输入/) |
| 历史恢复 | 当前工作区缺少旧记录所列的BMS历史ZIP和根目录整理JSON；现行盘点见[电力目录盘点](../检查/统一审查/inventory.json)。 |

BMS仅经CAN上传温度，独立冷却板自行控制；板上无风扇供电、驱动或测速。5V使用海凌科URB4805YMD-10WR3，输入从保险下游TAP_B9取得。AFE设置、预充硬截止、光耦全温余量、低压保险清除配合、热耦合与整车回馈均未完成硬件资格验证。当前设计保持HOLD。

## 本目录直接文件

| 文件 | 用途 |
|---|---|
| [项目要求.md](项目要求.md) | 汇总BMS用户要求、系统输入、设计边界与验收条件；需求状态和未决项以该文件及当前状态检查为准。 |
| [BMS-9S4P.eprj3](BMS-9S4P.eprj3) | 当前嘉立创EDA原生工程入口，与本目录的`sch/`原理图工程文件配套。 |
| [TODO.md](TODO.md) | 列出后续待办及关闭条件；条目状态需与当前原理图和检查证据核对。 |

## 原生 EDA 文件

| 位置 | 用途与边界 |
|---|---|
| [sch/Schematic1/P1.esch2](sch/Schematic1/P1.esch2) | 当前BMS原理图页面源；源哈希和逐脚回读状态见[当前设计状态](检查/当前设计状态.md)。 |
| [sch/Schematic1/Schematic1.ecfg](sch/Schematic1/Schematic1.ecfg) | 与当前原理图页面同目录的嘉立创EDA页面/设计规则配置文件。 |
| [pcb/PCB1.epcb2](pcb/PCB1.epcb2) | 项目树中保留的PCB格式工程文件。其存在不表示已完成PCB布局、制造交付或实物验收；这些后续工作由人类执行。 |

原生工程内部的其它器件、页面和库对象由EDA文件维护，不在模块说明中逐个展开。
