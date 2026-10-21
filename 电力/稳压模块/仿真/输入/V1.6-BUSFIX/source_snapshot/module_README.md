# 稳压模块

共用器件入口：[全项目元件封装库](../元件封装库/README.md)。选型先查共用库，满足本板要求时复用已有精确型号；现行工程或BOM变更后同步中央索引。

当前候选为 9S4P 电池公共 BUS 与 LICAP SC0350-270-RSS 超容组之间的双向 DC/DC。以下是唯一现行工程与交付入口。

## 当前工程与交付

- 当前原生文件：[sc-regulator.eprj3](eda工程文件/sc-regulator/sc-regulator.eprj3)、[原理图](eda工程文件/sc-regulator/sch/sc-regulator/01%20稳压模块全部电路.esch2)、[工程配置](eda工程文件/sc-regulator/sch/sc-regulator/sc-regulator.ecfg)。
- 当前交付：[完整工程 ZIP](eda工程文件/导出/20261010_BUS输入修复/稳压模块_V1.6-BUSFIX_eprj3完整工程.zip)、[逻辑原理图 PDF](eda工程文件/导出/20261010_BUS输入修复/稳压模块_V1.6-BUSFIX_逻辑原理图.pdf)、[完整候选 BOM](eda工程文件/导出/20261010_BUS输入修复/稳压模块_V1.6-BUSFIX_完整候选BOM.csv)。
- 人读复核：[原理图复核结论](检查/原理图复核结论.md)；结构化证据：[当前原理图证据](检查/当前原理图证据.json)；归档明细与 SHA 记录：[文件整理记录](检查/文件整理记录.json)。

本轮原生源 DRC PASS（fatal=0，errors=0，warnings=0，strict=true，sourceReopened=true）；交付副本 DRC PASS（fatal=0，errors=0，warnings=0，strict=true，sourceReopened=true）。SC1–SC8 已绑定用户提供的候选封装 LICAP_SC0350-270-RSS_USER（2f4e50aef1bc46dd9511974b979b6c70）；用户确认中心脚 1 正、四周同号脚 2 负，但机械适配仍待实物核验。本轮没有重跑仿真；LICAP350F 仿真结果是冻结的 V1.5 历史模型，不代表 BUSFIX 验证或电气、PCB、采购、实物及整车验收。

四个负极圆铜盘已由4 mm加大到5 mm，圆角孔角部计算铜环约0.503 mm；孔位、3×3 mm孔形、0.3 mm圆角及编号未变。参考资料中的原库保留原样；设计使用当前工程内已修正封装。


## 工作材料

- [原理图准备](原理图准备/)：当前选型、[BUSFIX 设计计划](原理图准备/design-plan.json)、预期连接与接口核对。
- [参考资料](参考资料/)：厂商资料与必要设计参考。
- [仿真](仿真/)：源码、冻结的历史模型输入和既有结果。
- [历史与调试归档](归档/历史与调试.zip)：本模块唯一历史 ZIP；原件另按整理记录保留在根目录的可恢复材料区。
- 模块级设计背景：[设计构想](稳压模块设计构想.md)、[元器件方案](最简元器件方案.md)、[计算与保护边界](计算与保护边界.md)、[可编程稳压控制](可编程稳压控制.md)、[超容串并联取舍](超容串并联取舍.md)。
