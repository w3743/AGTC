# B9 到海凌科 5V 电源支路保护闭合建议

日期：2026-10-21  
状态：PSU从TAP_B9取电的原理图/接口合同保持；当前P1 SHA为 `56f8f62e754d1e4af8f068bdc7b19c37dfa4788101cdb2852dc2f2254db41e8f`。本轮官方根EPRJ3/P1打开成功，但同源DRC/ENET调用未返回，布局/制造状态HOLD。辅助正极线真实短路电流、保险在37.8V DC下总清除时间及PCB/焊点I²t配合仍OPEN。

## 唯一建议

复用现有源端 `F_CELL_B9 = Eaton PV-5A10F`（5 A、1000 Vdc、gPV）和 `FH_CELL_B9 = Eaton CHPV1U`，不新增板端熔丝、不增加芯片/比较器。将熔丝安装在 B9 电芯取电点附近、整段细线和两半 DT 维护连接器之前，形成唯一受保护支路：

`CELL_B9 源端 → F_CELL_B9/FH_CELL_B9 → TAP_B9 → 16 AWG 线束 → K_CELL_BATTERY.10 → DT → K_CELL_BMS.10 → J_CELL_SERVICE.10 → 板内 TAP_B9 → 海凌科 VIN+`

`B0 → J_CELL_SERVICE.1 → PSU5_EXT.1/5` 为回路返回，不熔断。现有 100 A 主保险不是此细线支路的保护器件，不能代替 5 A 源端熔丝。

连接合同 `接口元件连接.json` SHA256 `DEFED46A6993E23A143D3A7206D1CC16FB9761C1D1AB4B87D5417456295C28EF` 仍将 `PSU5_EXT.2`、`C_PSU_IN1/2/3.1` 接到 `TAP_B9`，三只旁路电容另一端为 `B0`；`J_CELL_SERVICE.10=TAP_B9`，源端 `F_CELL_B9/FH_CELL_B9` 与线束路径见维护连接合同 SHA256 `CC9703F57095DD33EC03FCC8BFD8F575BF45DE755112ED62A754D1311FB132C7`。正式P1 SHA256 `56f8f62e754d1e4af8f068bdc7b19c37dfa4788101cdb2852dc2f2254db41e8f`；本轮未得到该P1同源官方ENET/DRC输出。前一官方a713源的官方网表及当前P1字节差分支持该支路连线未变，但不代替当前同源API复读、PCB DRC、布线/载流验收或实物检查。

## 线束与接口约束

- B9 正极支路全长统一采用 **16 AWG 铜线（约 1.31 mm²）**。其截面积落在现有 TE 0462-201-16141 / 0460-202-16141 触点的 16–20 AWG、0.5–1.5 mm² 范围内，也符合 CHPV1U 端子 1–25 mm² 范围。线材绝缘额定至少覆盖现有 60 Vdc 设计目标；另须覆盖项目实测的最大瞬态电压。按实际线束捆扎、环境温度和端接降额后，线、触点、PCB 焊点/铜箔和连接器每一段的连续允许电流均须不低于 5 A。
- Eaton CHPV1U 是 DIN 导轨模块式保险座。将其固定在靠近 B9 源端的受保护外壳/保险区，端子遮蔽并做线束应力释放；熔丝前未保护的引线保持最短且不可延伸成长线。不能把 CHPV1U 放到 BMS 板端再让长线先经过连接器。
- TE 触点规格给出 13 A 典型额定值，但整条支路仍按实际温度、线束、压接和接口降额验证；板端 `J_CELL_SERVICE.10` 的现有 PTH（孔 2.0 mm、焊盘 4.0 mm）及焊线方式要另外核对 16 AWG 插入、焊接和 5 A 温升，不能从触点额定值推定 PCB 端通过。
- `J_CELL_SERVICE.10`、板内 TAP_B9 走线、PSU VIN 与旁路电容焊盘均按 37.8 Vdc 最高持续电压设计；绝缘/间距不低于现有 60 Vdc 目标，并按最终板材、污染等级、海拔和实测瞬态复核。B9 维护连接器只允许零电流插拔，不能以热插拔作为日常关断方式；海凌科模块资料明确不支持热插拔。

## 电流估算与边界

当前 BMS 不带风扇。现有 5 V 预算中的 300 mA 基础负载规划加 73 mA 模块最低负载预算为 373 mA，即 1.865 W；基础负载不是实测值。海凌科效率 81% 是 48 V 额定输入、满载条件的下限数据，不是 22.5–37.8 V 下限载效率保证。仅将其作为说明性外推，1.865 W 对应 22.5 V 输入约 0.102 A、37.8 V 输入约 0.061 A；10 W 满载外推则约为 0.549 A 与 0.327 A。低线真实效率、启动浪涌和输入电容偏压后有效值仍待核实。

现有主动均衡设计把 2.9 A 作为电芯端控制目标，但不是验收过的最大 B9 电流。以 2.9 A 目标加上模块满额 10 W、22.5 V、81% 效率外推的 0.549 A，得到约 **3.45 A** 的纸面算术情形，低于 5 A 熔丝标称值，余量约 1.55 A。该余量不证明熔丝不会在温度、启动浪涌或实际均衡波形下误断；5 A 是熔丝额定值，不是电流超过 5 A 即瞬断。须用厂家完整时间—电流与总清除 I²t 曲线核查。PV-5A10F 的 19 A²s 是预弧 I²t，不能代替总清除值。

Eaton 公布 PV-5A10F 熔丝为 5 A、1000 Vdc、50 kAIC；Eaton 720147 保险座资料把 CHPV 系列与 PV-A10F 系列熔丝列为配套，并给出 33 kA breaking capacity。对当前 CHPV1U/PV-5A10F 组合，纸面采用 **33 kA 为公开目录的组合分断边界**，不能沿用熔丝单体的 50 kA 值当作组件额定。最高持续工作电压 37.8 Vdc 低于器件额定值。

电池 DC 最大预期短路电流目前未知。EVE 30P 的 1 kHz AC 内阻参数推得约 933 A 的算术代理，不是 DC 最小内阻，也不是最大短路电流上限；不得据此把 33 kA 分断边界标为已验证。正式闭合前必须拿到可引用的电芯/电池 DC 故障电流上限，或用保守源阻抗证明最大故障电流不超过该组件分断额定，并用熔丝总清除曲线核对 16 AWG 线、TE 触点、板端焊点/铜箔的热与电弧承受。F1 100 A 主保险以及保险前未熔断短引线均不纳入 5 A 支路保护证明。

## Layout 条件与首板验证

Layout 可按上述 TAP_B9 取电拓扑实施，但应把“源端装熔丝、熔丝在整段线束及 DT 连接器之前、未保护引线最短、全程按 16 AWG、输入旁路也在熔丝后”作为冻结条件。电气/实物验收保持 OPEN，直至完成：

1. 首板网络回读确认四个 VIN/B9 引脚均为 TAP_B9，找不到绕过 `F_CELL_B9` 的 CELL_B9→PSU 输入支路；确认 `J_CELL_SERVICE.10` 至 PSU/旁路电容走线、焊点与净距符合 5 A 和 60 Vdc 设计目标。
2. 在 22.5 V、37.8 V 两端点，分别覆盖当前实际 5 V 负载与满载边界，并叠加均衡最大实际 B9 支路电流；记录模块 VIN 最低点、输出跌落、输入电流、熔丝/触点/焊点/线束温升。启动时用限带宽足够的电流测量记录峰值/持续时间，确认模块启动和既有熔丝不误断；全过程不热插拔。
3. 先确定受控试验源的最大 DC 短路电流和熔丝总清除曲线，再在限流/能量受控夹具验证支路对 B0/相邻节点短路时的切断与温升。实际电池包短路电流未经证实前，不做硬短路实包测试，也不签发短路分断 PASS。

## 证据来源

本轮输入以当前源为准：`BMS模块/项目要求.md`（U21：BMS 无风扇）、`BMS模块/原理图准备/接口元件连接.json`（当前 PSU 和三只输入电容正端均为 TAP_B9）、`BMS模块/原理图准备/维护线束元件连接.json`（F_CELL_B9、FH_CELL_B9、J_CELL_SERVICE、TE 端子）、`BMS模块/检查/5V电源预算.md`（基础负载计划）及 `BMS模块/检查/抽头保险性价比候选.md`（2.9 A 目标和短路电流边界）。海凌科本地 V1.7 资料：`稳压模块/参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf`，SHA256 `1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d`。

厂商来源： [Eaton PV-5A10F](https://www.eaton.com/us/en-us/skuPage.PV-5A10F.html)、[Eaton CHPV1U](https://www.eaton.com/us/en-us/skuPage.CHPV1U.html)、[Eaton 10×38 mm fuse-holder data sheet 720147](https://www.eaton.com/content/dam/eaton/products/electrical-circuit-protection/bussmann-iec-low-voltage-fuses/bussmann-iec-modular-fuse-holders/eaton-bussmann-series-10x38mm-modular-fuse-holders-datasheet-720147-en-us.pdf)、[TE 0462-201-16141 socket](https://www.te.com/en/product-0462-201-16141.html)、[TE 0460-202-16141 pin](https://www.te.com/en/product-0460-202-16141.html)、[海凌科 UR(A)B_YMD-10WR3 series](https://www.hlktech.com/Goods-278.html)。


## 当前原理图证据与 Layout 边界

- 当前接口连接输入 SHA256 `DEFED46A6993E23A143D3A7206D1CC16FB9761C1D1AB4B87D5417456295C28EF`；维护线束合同 SHA256 `CC9703F57095DD33EC03FCC8BFD8F575BF45DE755112ED62A754D1311FB132C7`。前者记载四个VIN/旁路输入pin均接TAP_B9，后者记载源端F_CELL_B9/FH_CELL_B9及16AWG维护线束段。
- 先前官方ENET源为P1 SHA `a713d448b79c51c11663470647ab0cefe02809e6c9159e58bf61aeb7d97cb049`；当前P1 SHA `56f8f62e754d1e4af8f068bdc7b19c37dfa4788101cdb2852dc2f2254db41e8f`的`WIRE`和`LINE`记录与前者逐字节一致，R_QPRE_PU器件库替换不触及PSU网。故PSU pin网连续性只作为差分派生，当前同源DRC/ENET调用没有返回。现有PCB `PCB1.epcb2`比当前原理图源早，不能据此宣称铜箔、焊点、净距、温升或载流通过。
- 允许人类以源端保险、整段16AWG线束和TAP_B9输入拓扑做可回退摆放；冻结前需检查最薄弱段：板端`J_CELL_SERVICE.10`的2.0mm PTH/4.0mm焊盘、焊料/焊线界面、板内TAP_B9铜箔/过孔、PSU输入焊盘。现有证据只给线缆/端子输入，没有当前PCB铜厚、线宽、过孔或焊点的5A温升/I²t能力。


## 当前正式源绑定与故障分断状态

- 当前P1 `56f8f62e…4db41e8f`，EPRJ3 `fd5caba1…3f283369a6d`，BOM `6690cb1b…b65ae0f3b`（248项数据行）；当前BOM未改变5A保险、端子或维护线束料号。当前源同源DRC/ENET未返回，历史 warning 不代表当前结果，当前layoutEntry/manufacturingRelease均HOLD。
- Eaton PV-5A10F 的5A、1000Vdc、50kAIC和I²t参数必须分开理解：数据表19A²s是预弧值；50A²s总清除值明确绑定1000Vdc测试条件，不是37.8V下任意预期电流/电感时间常数的清除上界。Eaton时间-电流图依可用直流故障电流和试验条件读取；不能把`50A²s`单独拿来核算线材或PCB热承受。5A只是额定电流，不保证电流小于5A或略高于5A的故障能够按期切断。
- **辅助正极线支路故障/短路切断能力保持OPEN。** 尚缺电池包最小DC源阻抗/最大预期短路电流、实装PV保险的可用DC时间-电流/总清除曲线与试验条件、源端至PCB各串联段的最薄弱件额定，以及焊点/铜箔/过孔的短时I²t边界。必须取得预期短路电流和适用DC曲线，以总清除能量核对16AWG、TE触点、`J_CELL_SERVICE.10`焊线/焊点、TAP_B9走线/过孔、PSU输入焊盘，再在受控低能夹具故障注入；未闭合前不得签发故障分断PASS。来源：[Eaton PV-5A10F官方数据表](https://www.eaton.com/content/dam/eaton/products/electrical-circuit-protection/fuses/data-sheets/bus-ele-ds-10121-pv10m.pdf)。
