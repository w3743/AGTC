# 电力工程

每块板仅维护两份现行设计文档：`设计需求.md`说明要达到的功能、指标及边界；`设计细节.md`说明采用的电路、器件、接口、计算、成本、验证状态和待解决问题。后续原位更新，不再另建模块README、TODO、专项设计报告或审查结论。

| 板卡 | 设计需求 | 设计细节 | 现行 Code-as-EDA / KiCad 工程 |
|---|---|---|---|
| BMS | [需求](BMS模块/设计需求.md) | [细节](BMS模块/设计细节.md) | [BmsBoard.kicad_pro](BMS模块/eda工程文件/exports/kicad/BmsBoard.kicad_pro) · [SVG](BMS模块/eda工程文件/exports/schematic.svg) · [BOM](BMS模块/eda工程文件/exports/bom.csv) |
| FOC驱动，两块同版 | [需求](FOC驱动模块/设计需求.md) | [细节](FOC驱动模块/设计细节.md) | [FOCBoard.kicad_pro](FOC驱动模块/eda工程文件/exports/kicad/FOCBoard.kicad_pro) · [SVG](FOC驱动模块/eda工程文件/exports/schematic.svg) · [BOM](FOC驱动模块/eda工程文件/exports/bom.csv) |
| 独立超容稳压 | [需求](稳压模块/设计需求.md) | [细节](稳压模块/设计细节.md) | [RegulatorBoard.kicad_pro](稳压模块/eda工程文件/exports/kicad/RegulatorBoard.kicad_pro) · [SVG](稳压模块/eda工程文件/exports/schematic.svg) · [BOM](稳压模块/eda工程文件/exports/bom.csv) |
| 水泵驱动 | [需求](水泵驱动模块/设计需求.md) | [细节](水泵驱动模块/设计细节.md) | [PumpBoard.kicad_pro](水泵驱动模块/eda工程文件/exports/kicad/PumpBoard.kicad_pro) · [SVG](水泵驱动模块/eda工程文件/exports/schematic.svg) · [BOM](水泵驱动模块/eda工程文件/exports/bom.csv) |

新设计输入为9S5P、保留8颗超容、稳压独立。超容首先承接无机械刹车车辆的电制动回生，启动有余电则辅助，无余电允许受限慢加速。四套主电力板均已替换为冻结施工合同对应的 `Code-as-EDA (tscircuit)` / `KiCad` 原理图成果并完成 100% 逐针校验；每板设计细节区分冻结底稿、施工合同、模型验证与后续人类 PCB/固件/实物实施。

### 交付与证据入口

| 板卡 | 施工合同 / 逐针校验 | 采购 / 仿真 / 检查 |
|---|---|---|
| BMS | [施工合同](BMS模块/检查/闭合候选_9S5P/BMS_9S5P施工合同.json) · [逐针报告](BMS模块/eda工程文件/exports/netlist_verify_report.json) | [闭合复算](BMS模块/检查/闭合候选_9S5P/BMS_9S5P闭合复算.json) |
| FOC | [施工合同](FOC驱动模块/检查/闭合候选_164Rev2/FOC_164Rev2施工合同.json) · [逐针报告](FOC驱动模块/eda工程文件/exports/netlist_verify_report.json) | [闭合复算](FOC驱动模块/检查/闭合候选_164Rev2/FOC_164Rev2闭合复算.json) |
| 稳压 | [施工合同](稳压模块/检查/闭合候选_P5Rev2/稳压P5Rev2施工合同.json) · [逐针报告](稳压模块/eda工程文件/exports/netlist_verify_report.json) | [闭合复算与仿真](稳压模块/检查/闭合候选_P5Rev2/稳压闭合复算与仿真.json) |
| 水泵 | [施工合同](水泵驱动模块/检查/闭合候选_隔离CAN/水泵闭合施工合同.json) · [逐针报告](水泵驱动模块/eda工程文件/exports/netlist_verify_report.json) | [整车立创一键采购 BOM](检查/联合仿真/整车立创一键采购BOM.csv) |

整车联合门禁见[画图前合同完整性校验](检查/联合仿真/画图前合同完整性校验.json)与[6场景故障联合仿真](检查/联合仿真/降本后典型故障联合仿真.json)。历史过程文档、旧版嘉立创 `.eprj3` 及原归档目录已按成果精简策略清理（完整历史快照保留在 Git 提交历史中）。

本机根目录可能存在被 Git 忽略的 SQLite 文件 `0819f05c4eef4c71ace90d822a990e87`。2026-11-08 只读检查显示它的 projects、schematics、documents、components、devices、resources 等表无记录，users 和 db_versions 各一条；库标识也出现在设计导出中，但这不证明此本机文件是工程必需品。该文件不随仓库交付，本次未验证应用重建或替换过程，暂保留；不要把标识相同当作可恢复性证明。

共同要求见[基本设计](基本设计.md)、[规则](规则.md)和[代理约定](AGENTS.md)。共用器件与采购入口为[元件封装库](元件封装库/README.md)。

原理图、BOM、连接合同JSON、CSV、计算脚本、原始检查输出和数据手册是两份文档引用的附件。`检查/`保存机器证据，`仿真/`保留源码、输入和结果，`原理图准备/`保留选型及连接数据。冻结模型`source_snapshot`和第三方原始工具资料保持原字节，不作为额外的现行板级设计文档。

各模块的文档归并映射记录旧路径与新的正文位置；历史机器证据内的原路径保留其溯源含义，不表示旧文档仍是当前入口。全局证据索引见[检查目录](检查/README.md)。原生源和BOM一致性不等于电气、PCB或实物验收；原理图后的PCB、制造、装配、固件和实物工作由人类完成。
