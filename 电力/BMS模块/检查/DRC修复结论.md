# BMS 当前DRC修复与Layout证据结论

日期：2026-10-16。当前输入：[正式根路径准入核对](Layout验证/layout-entry-check.json)、[发布清单](Layout验证/当前发布.json)、[同字节副本重开读回](Layout验证/root-after-cleanup-reopen.json)、[几何核验](Layout验证/final-geometry.json)。采购映射细节见[采购身份核查](DRC修复/采购身份核查.md)。

## 当前计数与验证状态

| 项目 | 当前读回 |
| --- | --- |
| 板上元件 / 归一化采购MPN | 206 / 51（原始MPN字符串 52） |
| CAD位号 / 针 | 208 / 700（含2个板外载体） |
| 正式根EPRJ3与P1页官方CLI/API打开 | PASS：源SHA前后不变，rows/ports/wires/texts与当前发布快照一致（208 CAD位号/700针、206板上、285 ports/331 wires/237 texts）；详见layout-entry-check.json。字节一致副本保存/重开PASS另见历史读回记录。 |
| 用户导出DRC（19:55） | 0 fatal / 0 error / 1 warning / 72 info（72个互异位号信息）；19:55:29单条供应商属性警告列出19对ref/instance，具体见layout-entry-check.json的latestUserExportReview。 |
| API DRC（16:51准入核验，历史记录） | 当时API记录1个warning组；保留为历史证据，不与本轮用户导出日志混称。 |
| DRC review / layoutEntry / manufacturingRelease | 供应商属性warning仍在；READY_FOR_HUMAN_LAYOUT（仅可回退分区初摆）/ HOLD |
| 警告明细 | 19:55用户日志的19对ref/instance与旧16:39审计列表顺序及ID完全一致（新增0、缺失0）；此日志不是API运行，也未由此证明属性不匹配已解决。本轮未重审其余18个字段或光耦电气兼容。 |
| 日志源哈希对照 | EPRJ3、P1.esch2、BOM与layout-entry-check记录的SHA256完全一致；ECFG哈希变化，但对照归档原件只见DOCHEAD.client从894da77061c0bf1d变为6ed036e268826c86，归一化该字段后全文相同。 |
| U18局部连线重画 | 本轮按U18完成VC/SRP局部连线重画读回核验：同字节隔离副本保存/重开、逐针语义PASS；16:51准入核验记录正式根及P1页官方CLI/API打开PASS且当时源SHA未变。19:55用户日志仍有1条供应商属性warning。layoutEntry=READY_FOR_HUMAN_LAYOUT；布线冻结与manufacturingRelease保持HOLD。 |
| 原理图连线几何 | PASS：700针、1246条活动原生LINE、331个wire group、285个netport、388个物理连通簇；严格语义20类缺陷计数全0。该检查不替代ERC/DRC、PCB布局或实物验证。 |
| 独立路径 | 旧记录中的112条警告只作历史，不能作为本轮独立路径通过证据 |
| 板外R_PRE/QPRE_SW | R_PRE在正式快照中存在（MPN CJP300J8R2J；BOM标记=False、PCB标记=False表示该板外件不重复计入）；R_PRE.1=QPRE_SW、R_PRE.2=PACK+、Q_PRE2.2=QPRE_SW均逐针PASS，QPRE_SW已闭合。 |

## 当前模型状态

16A净充电参考算法：PASS executable reference algorithm and deterministic fail-safe tests; NOT firmware/hardware/vehicle acceptance；检查 17/17 项通过，故障注入 18/18 项收敛至CCL=0，母线能量参数情景 756 组；发布门禁=BLOCKED pending physical input qualification, firmware implementation/readback, branch-current, local source-response, and sink validation。保护模型：PASS_WITH_OPEN_ITEMS，检查 17/17 项通过，仍有7项开放事项。预充定时模型：PASS for TMR_ACTIVE/logic-window deassertion only; excludes BJT and MOSFET physical release；FF-B供电至DRV低电平上界 2.8577405271378624 s；MOSFET物理关断最大时间有保证=False，整链物理最大时间=未形成。这些是参考算法/纸面模型结果，不构成固件、硬件、台架或整车验收。

## 既往原理图修复记录

- 清理97项重复NET。
- 修正7处off-grid坐标。
- 补齐1处板外R_PRE原理图表达；当前快照已含R_PRE，R_PRE两针与Q_PRE2.2逐针语义PASS，QPRE_SW已闭合。
- 补齐145个空Value字段。
- 纠正光耦错误C号映射；采购MPN `VO617A-2X007T` 身份保持不变，不用不同等级载体号覆盖采购身份。

本轮通过官方文件API从11个wire group清理29条重复 `WIRE ATTR key=NET`，其余P1原生记录未变。此前官方文件API在16:51准入核验中记录正式根EPRJ3与P1页打开PASS、快照字段一致及1个warning组；该API结果保留为时间限定的历史证据。19:55用户导出日志摘要为0致命、0错误、1警告、72信息，单条供应商属性警告含19对ref/instance，与16:39审计列表逐项一致。此次核查只读日志与文件哈希，没有调用API或修改CAD；其余18个属性字段和光耦电气兼容未重审。三项源SHA与准入记录完全一致，ECFG仅DOCHEAD.client字段改变且归一化后全文与归档原件一致。现有layoutEntry=READY_FOR_HUMAN_LAYOUT仅限人类可回退分区初摆；布线冻结与manufacturingRelease保持HOLD。

当前记录仍保留U18 VC/SRP局部连线重画及逐针语义核验结果。16:51官方CLI/API入口核验与19:55用户导出DRC日志是两项不同时间和来源的证据；最新用户日志仍报1条供应商属性警告，未据此宣称属性问题已解决。光耦CTR/接收器兼容、采购与PSU支路故障分断/I²t仍OPEN。既有layoutEntry仅允许人类可回退分区初摆；**布线冻结与投板保持HOLD**。PCB布局布线由人类完成，代理未创建PCB。

文档入口：[模块README](../README.md) · [当前设计状态](当前设计状态.md) · [Layout交接](../原理图准备/Layout交接.md) · [项目要求](../项目要求.md)。
