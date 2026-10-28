# CAD接力闭合验证

日期：2026-10-15。范围仅限隐藏 CLI 打开的隔离副本；初始复制的5个文件与正式工程 SHA256 一致。随后仅经官方 API 在副本中持久化修正 U1 左侧文字布局；副本现在与正式源不同，正式 EPRJ3/P1/BOM 未改，未建 PCB。

## 结果

- **基线与网表：PASS（只证明当前副本网表）**。完整副本的 P1 页面非空：872 条 COMPONENT 记录（含网络端口）；网表单独确认物理器件为 213；官方 `SCH_ManufactureData.getNetlistFile` 输出 213 个器件、711 pins（659 已连接、52 未连接）。与保存的官方网表忽略 project UUID 后，213 refs集合及逐器件完整数据完全一致。872=213 个器件 COMPONENT+659 个 netport COMPONENT 记录，不是所有图元总数。证据见 `tmp/接力_netlist_compare.json`。
- **DRC：未通过／无法逐项验收**。正确213器件副本调用详细模式 `SCH_Drc.check(true,false,true)` 返回数组 `[{"type":"warn","count":10}]`，仍只有聚合类别与数量，没有逐项内容；不能判PASS。此前false-only重载曾返回`false`且无日志，现以这次正确页面上的详细模式结果为准。
- **R_PSU_FB_TOP：属性不一致，尚未换绑**。API读回 Name=`RT0603BRD0795K3L`，Manufacturer Part同为该目标，Device/符号源=`RT0603BRD0730K1L`，Supplier Part=`C705764`，Footprint=`R0603`，Value/Tolerance为空，pin 1→`VOUT_5V`、pin 2→`PSU_FB`。BOM注释目标为95.3kΩ±0.1%；器件源/供应商字段指向30.1kΩ carrier。精确目标库器件搜索为空，因此未创建替换。它是载体/采购元数据一致性问题，不能据此断言实装为30.1kΩ或输出必为2.4V。早次元件状态未含属性显示开关；本次属性API确认Designator/Name可见，见补充核验。
- **其余 carrier**：`U_FAN_SW` 当前源名`SGM2593AXN6G/TR`、Manufacturer Part目标`SGM2593XN6G/TR`；`U_LOAD_POWER_VALID` 当前源名`VO617A-4X007T`、Manufacturer Part目标`VO617A-2X007T`。本轮未换绑。
- **L_PSU：API修改路径未闭合**。当前焊盘距11.999976 mm，目标11.25 mm；需两焊盘各向内0.374988 mm至±5.625 mm。正确副本的`getDocumentFootprintSources()`返回0；LIB_Footprint.get仅返回元数据，modify无pad几何字段，updateDocumentSource所需源数据不可得。该 UUID 属于共享/系统库，未写共享库。此前隔离探测记录 copy 报错、create返回null且搜索为空、打开不存在的项目封装返回undefined，详见 `L_PSU封装官方API只读回读_20261015.json`。
- **官方raw PDF：边界裁切FAIL；后处理导航审阅件：内容完整性PASS、排版可读性FAIL**。raw P1 PDF为1页、14400×14400 pt；普通页面文本提取命中104/213 refs，但无限裁切区命中213/213。PDF文本边界（x=15..22597、y=4..19229 pt）和矢量边界（x=−1227..22602、y=−4.625..20112 pt）越出默认MediaBox，确认为页面边界裁切，不是隐藏位号，也不说明原生网表缺器件。旧的A4尺寸参数尝试在59.75s/60s后超时，未重试。另有后处理导航审阅PDF通过213/213文本覆盖及213书签核验；但整页稀疏，U1局部文字重叠，仍不适合作为细节可读版。

## 可执行修订工单

1. 在项目副本中核对 `R_PSU_FB_TOP` 的实际目标采购编码与 `C705764`；明确写入95.3kΩ±0.1% Value/容差并保持目标MPN，或按确认后的真实采购料更正MPN，保存后检查网表/BOM。不要只凭carrier名称推断电路输出。
2. 将 `U_FAN_SW` 的A型号carrier校正为确认后的非A目标；将 `U_LOAD_POWER_VALID` 的-4型号carrier校正为确认后的-2目标。保存后比对所有pin网络。
3. 通过 EasyEDA 项目库工作流复制 `L_PSU` 封装到项目库，将焊盘中心改到±5.625mm并保持编号、层、尺寸、方向及网络；保存关闭重开后复核。
4. 重新运行官方逐项 ERC/DRC，保存详细结果；重新导出并核对213 refs是否均落入PDF页面。

## 补充API、DRC与导出核验（2026-10-15）

- **API文档与本机CLI有签名差异**。技能完整参考 `.agents/skills/easyeda-api-skill/references/classes/SCH_Drc.md` 明确列出两个BETA重载：`check(strict,userInterface,false): Promise<boolean>` 以及 `check(strict,userInterface,true): Promise<Array<any>>`。本机 `doc api --class-name SCH_Drc --method-name check` 输出（证据 `tmp/handoff_sig_SCH_Drc_check.out`）只暴露`includeVerboseError:false`及boolean返回；本轮仍按完整参考对正确P1副本实测true重载，结果为聚合`warn:10`，运行时没有返回逐项条目。官方 `SCH_ManufactureData.getNetlistFile` 则成功用于213器件/711引脚网表，不重试先前超时的 `sch_Netlist.getNetlist`。
- **原生位号可见性与PDF裁切原因**。官方API读取正确P1页：872个COMPONENT记录（含213器件与659个netport）、213个唯一Designator，且官方网表213 refs/711 pins；872不等于213器件。`SCH_Primitive.getPrimitivesBBox`、`SCH_PrimitiveAttribute.getAll`及属性状态getter返回的目标 `Designator` 和 `Name` 均为`valueVisible=true`（包括L_PSU、R_PSU_FB_TOP、U1、U_PSU、U_FAN_SW、U_LOAD_POWER_VALID）；位号没有被隐藏。位置、bbox及完整属性回读见 `tmp/cad_followup_drc_geometry.json`。结合上面MediaBox外的文本/矢量范围，104/213是raw PDF页面剪裁导致。
- **导出/范围API能力边界**。完整 `SCH_ManufactureData.getExportDocumentFile(fileName?,fileType?,typeSpecificParams?,object?,objectSpecificParams?)` 签名仅提供 `size`（Original Size/字符串/width-height-unit）、主题、线宽、属性菜单、对象及sheet `range`/合并方式；没有导出原点、按图元边界fit或页面内缩放参数。`DMT_EditorControl.zoomToAllPrimitives(tabId?)` 与 `zoomTo(x?,y?,scaleRatio?,tabId?)` 只回报/调整画布视口范围，不设置PDF页面。`SCH_Primitive.getPrimitivesBBox(primitiveIds:Array<string|ISCH_Primitive>)` 只读bbox；`SCH_PrimitiveAttribute.getAll(parentPrimitiveId?:string)` 只读属性，配合状态getter能核实位号显示，但不修导出MediaBox。故未找到有文档依据的官方API修正raw PDF裁切路径；后处理审阅件仅修PDF页边界。
- **导航审阅PDF**：`当前原理图导航审阅.pdf` 是对官方raw PDF进行等比例0.5的矢量页面复用与扩边，不改CAD源；1页11940×10080 pt，源/输出213/213位号唯一命中、213书签，JSON完整性`PASS`。SHA256=`bd0845f745c49bfb12a4345b7891b25e5b23bcc3dc06ce824dc13263c5dfd4b3`。完整性不代表可读性：图面原始排布稀疏、全页缩略难读，U1局部引脚/网络文字重叠，故可读性仍未通过；此PDF也不代表ERC、电气或实物验收。细节见 `当前原理图导航审阅.json`。
- **历史跟进阶段无修订**。下述 DRC/PDF 跟进当时没有保存 CAD 变更；后续隔离副本 U1 左侧标签布局修正与验证见下节。当时 CLI 会话关闭证据保留于 `tmp/cad_followup_session_close.json`。


## U1 左侧网络名布局试修（隔离副本，2026-10-15）

- **结果：局部持久化修复 PASS，正式源未应用。** 副本入口 `BMS模块/检查/tmp/接力CAD闭合副本/eda工程文件/BMS-9S4P.eprj3`；本轮仅用 `SCH_PrimitiveAttribute.modify` 改 U1 左侧 23 个可见 wire `Name` 属性（U1 的 40 个 wire 端点连接保持不变），仅调整文字位置与对齐，不改网络名或网络值。各属性 x 移至对应 wire 左端 minX−10（实际 x=21795），y 、值、可见性不变，alignMode=8 (`RIGHT_MIDDLE`)。未改 pin/port 锚点、wire 几何、NET 值或共享符号库。
- **保存重开与电气对照：PASS。** 保存返回 true；新建隐藏 CLI 会话 `e0662ff3-9f4d-46fa-94f0-7f3c9d73dfc9` 重开副本，API 重读 23/23 属性符合目标位置与对齐。重开导网表后 213 refs/711 pins 与基线一致（canonical SHA256 均为 `3A6AD8937A9A4CA724CD0AB30A67EAD833F438F56AF70A844C3420B91E3E9950`）；659/659 wires 的 ID、线几何和 NET 全部相同。
- **局部目视：PASS（限 U1 左侧）。** 主代理核对改后裁图，23 个标签不再遮挡 pin 号和符号左边界。改后局部图 `tmp/cad_u1_left_bulk_trial_local.png`；改前参考 `tmp/当前原理图导航审阅_U1.png`。这只验收改后 U1 局部文字布局，不改变原 PDF 的其他全页排版结论，不等于 ERC 或电气验收。
- **会话关闭与文件边界：PASS。** 重开会话关闭命令退出码 0，返回 `ok=true/status=closed`；随后 `session list` 显示该会话 closed。报告 JSON 保存每个证据路径和本轮正式 EPRJ3/P1/BOM SHA256；仅副本 `P1.esch2` 已变更，与正式源不同。

## 正式文件与会话

正式文件复核 SHA256：EPRJ3 `FE4F325C50613ABDC9C8A4F719E138FB5752FB6A646692D0710554B0DC073DD9`；P1 `CC1A211574E829C44D0557182A67790F24DAAA5C81F594FF5271ED384BE990C2`；BOM `43B7C50E53DA9509B600904B6FD300B3EEFB9C47E40A13C84555A0D0DD6FA00E`。先前会话`c2d2cb1b-99fb-4285-90af-bf6ffdd6d52b`已关闭；本次跟进会话`ecc0a01e-078d-46c2-a48b-21290171ed63`也已关闭，`session list`确认为closed。

总体结论：已取得 U1 隔离副本持久化左侧标签排版修复，重开后 213 refs/711 pins 与 659 条导线网络几何对照不变，U1 局部目视 PASS；正式源仍未修订，carrier/L_PSU/ERC 未闭合。DRC 只返回 10 条聚合 warning，不能逐项 PASS；raw 官方 PDF 页边界裁切，后处理导航审阅 PDF 内容 213/213 完整但整页可读性未通过。正式三文件 SHA256 未变。详情与机器可读证据见同名 JSON。
