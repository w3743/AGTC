# BMS 全原理图逐针与故障链复核

审查日期：2026-10-22。当前正式源为 P1 `bdba7a90f7a4bbbdf0d03ac986a015a66336af80b713635e28921e73b03e890e`，EPRJ3 `4de6f6a845074505476aeec2e3dc4d57c0643c259daa0e84f4aa3d01a47b4805`，采购候选 BOM SHA `665c9d5c1bde1e44a2aa0fb1a162d1281688810316eee891c23ac5af3a1bb7ba`。当前官方网表 `跨板并料_当前.enet` SHA `537caa4389b5fae40fe5508d4ae1c194f69ed53982c99c0fb8fe04959f53c6eb`；保存、关闭、根工程重开、API/ENET复核和最终关闭均已完成。此报告仍不是 PCB 审查或硬件资格结论。

## 当前证据边界

- 官方正式会话保存、关闭并从根 EPRJ3 重开后，当前 P1 直接复读 **226 位号、740 针、246 条 TEXT**；当前 ENET 与 live API 针网差异为 0。严格 DRC 为 **0 error、9 warning groups**：8 组 `SHOWDANDLING`、1 组 `ATTRMATCHCODE`。本结果只说明当前EDA源与检查结果，不构成 PCB、实物或制造资格。 Netport `bdb539cd08ac77ae` 的 Add into BOM 与 Convert to PCB 均为 false，因此不形成额外采购/PCB实例。
- 完整 740 脚映射见 [current-pin-net-coverage.json](current-pin-net-coverage.json)，当前版本 SHA256 `0d47680be75332f48788a98e72e00dca4838ff5b93429cc3bb908f16b9730698`，按当前官方 ENET 的 Designator 刷新，并与 live API 逐脚比对为 0 mismatch。全局proof [跨板并料复核.json](../跨板并料复核.json) SHA256 `cce09a3c0439f40b7c8cc882c304d68d2e0b85dc68f8558cd4498ebcd610ff09`，原始API输出ZIP SHA256 `1acad083beca8d0b75e966760734da69ab140dd190c4c39f631f3eb76c57c6fb`；本报告和网表均由该当前proof绑定。
- 层次正确的基线：P1 `56f8f62e…` 已有 **215/710**，其中已经包含 3 路电芯 NTC（6 针）和 4 个裸孔焊线点（4 针）；`56f8…→4d274…` 仅修改 TEXT `58db56e9107cff1e`。当前边界增加 `F_CELL_B1..9`、`F1`、`J_CELL_SERVICE` 共 11 对象/30 针。当前网表还确认 13 个 B9 端点由 `CELL_B9` 改为 `TAP_B9`，其中包括 `D_BAT.2`、`D_BREG.2`、`F_CELL_B9.2` 与服务端 pin10；最终 **226/740**。较早的 700 针不应直接加 30。
- 当前采购候选 BOM：248 行数据，SHA256 `665c9d5c1bde1e44a2aa0fb1a162d1281688810316eee891c23ac5af3a1bb7ba`；采购状态仍以BOM注明状态为准，不将列入BOM视为实购。11 个 1uF 项目电容 `C_BAL_CELL1..9/C_REG1/C_CAN_VCC2_2` 当前统一为 Samsung `CL10B105KA8NNNC` / C29936；有效偏压容量和硬件表现仍待验证。NTC_CELL1/2/3 列 Semitec 103AT-2 候选料号；9 路外置熔断器/座不增加板上 BOM。当前成本入口为[当前商城成本](../../元件封装库/索引/当前商城成本.md)+冻结BOM；模块内旧“当前主板成本.csv”是旧快照。

## 单体抽头、保险和板外接口

新源把每个抽头保险明确串在电芯源端和保护板输入之间；保险座只保留在外部采购合同，不另画并联的两针“座”符号。

| 源端 | 保险端 1 → 保险端 2 | 板侧抽头 |
|---|---|---|
| `CELL_B1` | `F_CELL_B1.1` → `F_CELL_B1.2` | `TAP_B1` |
| `CELL_B2` | `F_CELL_B2.1` → `F_CELL_B2.2` | `TAP_B2` |
| `CELL_B3` | `F_CELL_B3.1` → `F_CELL_B3.2` | `TAP_B3` |
| `CELL_B4` | `F_CELL_B4.1` → `F_CELL_B4.2` | `TAP_B4` |
| `CELL_B5` | `F_CELL_B5.1` → `F_CELL_B5.2` | `TAP_B5` |
| `CELL_B6` | `F_CELL_B6.1` → `F_CELL_B6.2` | `TAP_B6` |
| `CELL_B7` | `F_CELL_B7.1` → `F_CELL_B7.2` | `TAP_B7` |
| `CELL_B8` | `F_CELL_B8.1` → `F_CELL_B8.2` | `TAP_B8` |
| `CELL_B9` | `F_CELL_B9.1` → `F_CELL_B9.2` | `TAP_B9` |

- `F1`: `CELL_B9` → `BATT+_FUSED`，对应外置 Littelfuse JLLN100.VXP 100A 主保险；它与 9 路抽头保险是两种不同保护边界。
- `J_CELL_SERVICE`: pin 1=`B0`，pins 2–10=`TAP_B1..TAP_B9`。这是无采购、无 PCB 转换的图面边界符号；物理合同为 1×10、5.00 mm pitch、2.00 mm 成品孔、4.00 mm 铜盘。尚无与此几何匹配的布局读回，不得拿普通 2.54 mm 排针替代。人类 PCB 布局、爬电距离、线束固定和装配方向保持 HOLD。
- `F_CELL_B1..B9` 是 5A Eaton PV-5A10F 保险的电气串联表达，Eaton CHPV1U 是机械座采购身份。保险额定电流及高压短路指标不能替代 37.8V 电池的直流时间-电流和总清除 I²t；额定值以下的故障可能不熔断。需用电池预期故障电流、实际 37.8V DC 曲线、线束/座/端子/板上最弱截面热承受能力做配合验证。F1 同理需核 100A 连续/1s、曲线和低压包短路电流。

## AFE、电芯测量与供电

| 电阻 | 源端 → AFE 滤波网 |
|---|---|
| `R_VC0` | `B0` → `VC0_FILT` |
| `R_VC1` | `TAP_B1` → `VC1_FILT` |
| `R_VC2` | `TAP_B2` → `VC2_FILT` |
| `R_VC3` | `TAP_B3` → `VC3_FILT` |
| `R_VC4` | `TAP_B4` → `VC4_FILT` |
| `R_VC5` | `TAP_B5` → `VC5_FILT` |
| `R_VC6` | `TAP_B6` → `VC6_FILT` |
| `R_VC7` | `TAP_B7` → `VC7_FILT` |
| `R_VC8` | `TAP_B8` → `VC8_FILT` |
| `R_VC15` | `TAP_B8` → `VC15_FILT` |
| `R_VC16` | `TAP_B9` → `VC16_FILT` |

- `U1` BQ7695202PFBR 48 针连接逐脚如下。VC9–VC14 未用通道均停在 `VC8_FILT`；9S 读数候选为 VC1–VC8 与 VC16−VC15。实际通道启用、增益/偏置校准、阈值和故障响应尚未由目标器件配置读回。

| Pin | Signal | Net | Pin | Signal | Net |
|---:|---|---|---:|---|---|
| 1 | VC15 | `VC15_FILT` | 25 | ALERT | `BQ_ALERT_IC_N` |
| 2 | VC14 | `VC8_FILT` | 26 | SCL | `BQ_I2C_SCL` |
| 3 | VC13 | `VC8_FILT` | 27 | SDA | `BQ_I2C_SDA` |
| 4 | VC12 | `VC8_FILT` | 28 | HDQ | `NC` |
| 5 | VC11 | `VC8_FILT` | 29 | CFETOFF | `BQ_CFETOFF` |
| 6 | VC10 | `VC8_FILT` | 30 | DFETOFF | `DFETOFF` |
| 7 | VC9 | `VC8_FILT` | 31 | DCHG | `BQ_DCHG` |
| 8 | VC8 | `VC8_FILT` | 32 | DDSG | `NC` |
| 9 | VC7 | `VC7_FILT` | 33 | RST_SHUT | `RST_SHUT` |
| 10 | VC6 | `VC6_FILT` | 34 | REG2 | `NC` |
| 11 | VC5 | `VC5_FILT` | 35 | REG1 | `REG1` |
| 12 | VC4 | `VC4_FILT` | 36 | REGIN | `REGIN` |
| 13 | VC3 | `VC3_FILT` | 37 | BREG | `BREG` |
| 14 | VC2 | `VC2_FILT` | 38 | FUSE | `NC` |
| 15 | VC1 | `VC1_FILT` | 39 | PDSG | `NC` |
| 16 | VC0 | `VC0_FILT` | 40 | PCHG | `NC` |
| 17 | VSS | `B0` | 41 | LD | `BQ_LD` |
| 18 | SRP | `SRP` | 42 | PACK | `BQ_PACK` |
| 19 | NC | `NC` | 43 | DSG | `DSG` |
| 20 | SRN | `SRN` | 44 | NC | `NC` |
| 21 | TS1 | `BQ_TS1` | 45 | CHG | `CHG` |
| 22 | TS2 | `NTC_BOARD_NODE` | 46 | CP1 | `CP1` |
| 23 | TS3 | `BQ_TS3` | 47 | BAT | `BAT` |
| 24 | REG18 | `BQ_REG18` | 48 | VC16 | `VC16_FILT` |

- `D_BAT.2`、`D_BREG.2` 现都接 `TAP_B9`，之后分别经 BAT、BREG 支路供 AFE；保护采样 `VC16/VC15` 保持独立分压/滤波域，没有把 `VC` 电芯串联域改成 BAT/BREG 电源域。`PSU5_EXT` 同样从 `TAP_B9` 取电，`F_CELL_B9` 在源端先串联。
- `RSHUNT=0.5 mΩ`，pin1=`B0`，pin2=`PACK-`；SRP/SRN 各经 100 Ω，`C_SRP` 跨差分端。纸面耗散为 60 A 时 1.8 W，100 A/1 s 时 5 W。分流器连续/脉冲额定、Kelvin 走线、温升、ADC 校准及实际保护阈值动作仍需目标板验证。

## 主动均衡

八个 ETA3000D2I 通道覆盖九个相邻电芯对。每个通道的完整 9 针映射见机器可读覆盖 JSON；核心 pin map 模式如下：`BATC=TAP_Bn`，`BATP=TAP_B(n+1)`，`BATN/EP=TAP_B(n−1)`（第 1 路为 B0），`SW=BAL_ETA_SW_n`，`BIAS/ISET/EN` 各自独立。

| 通道 | BATN / EP | BATC | BATP | SW / BIAS / ISET / EN | STAT |
|---:|---|---|---|---|---|
| 1 | `B0` / `B0` | `TAP_B1` | `TAP_B2` | `BAL_ETA_SW_1` / `BAL_ETA_BIAS_1` / `BAL_ETA_ISET_1` / `BAL_ETA_EN_1` | NC |
| 2 | `TAP_B1` / `TAP_B1` | `TAP_B2` | `TAP_B3` | `BAL_ETA_SW_2` / `BAL_ETA_BIAS_2` / `BAL_ETA_ISET_2` / `BAL_ETA_EN_2` | NC |
| 3 | `TAP_B2` / `TAP_B2` | `TAP_B3` | `TAP_B4` | `BAL_ETA_SW_3` / `BAL_ETA_BIAS_3` / `BAL_ETA_ISET_3` / `BAL_ETA_EN_3` | NC |
| 4 | `TAP_B3` / `TAP_B3` | `TAP_B4` | `TAP_B5` | `BAL_ETA_SW_4` / `BAL_ETA_BIAS_4` / `BAL_ETA_ISET_4` / `BAL_ETA_EN_4` | NC |
| 5 | `TAP_B4` / `TAP_B4` | `TAP_B5` | `TAP_B6` | `BAL_ETA_SW_5` / `BAL_ETA_BIAS_5` / `BAL_ETA_ISET_5` / `BAL_ETA_EN_5` | NC |
| 6 | `TAP_B5` / `TAP_B5` | `TAP_B6` | `TAP_B7` | `BAL_ETA_SW_6` / `BAL_ETA_BIAS_6` / `BAL_ETA_ISET_6` / `BAL_ETA_EN_6` | NC |
| 7 | `TAP_B6` / `TAP_B6` | `TAP_B7` | `TAP_B8` | `BAL_ETA_SW_7` / `BAL_ETA_BIAS_7` / `BAL_ETA_ISET_7` / `BAL_ETA_EN_7` | NC |
| 8 | `TAP_B7` / `TAP_B7` | `TAP_B8` | `TAP_B9` | `BAL_ETA_SW_8` / `BAL_ETA_BIAS_8` / `BAL_ETA_ISET_8` / `BAL_ETA_EN_8` | NC |

`STAT` 未接入状态采集。均衡电流/方向、温升、启动默认关断、单路 EN 卡高、抽头开路/短路时的停机需要台架验证；配置/固件当前没有物理读回证明。

## 主 MOS、硬许可与预充

| 器件 | Source / Drain / Gate | 作用与未闭合处 |
|---|---|---|
| `Q_CHG1..3` | `BATT+_FUSED` / `FET_MID` / `CHG_G1..3` | 充电侧 3 颗并联 N-MOS（HYG022N10NS1TA）；需热均流、SOA、栅极驱动实测 |
| `Q_DSG1..3` | `PACK+` / `FET_MID` / `DSG_G1..3` | 放电侧 3 颗并联 N-MOS（HYG022N10NS1TA）；需热均流、SOA、栅极驱动实测 |
| `D_TVS` | `PACK+`–`PACK-` | SMCJ45A 用作瞬态钳位，不是持续能量卸载件 |
| `Q_PRE1/Q_PRE2` | D=`BATT+_FUSED` / `QPRE_SW`；共同 S=`QPRE_SRC`；共同 G=`QPRE_G` | 背靠背 P-MOS；外置 `R_PRE=8.2Ω/300W` 在 `QPRE_SW` 与 `PACK+` 间 |

- `MAIN_CLR_N` 组合 BQ alert、本地允许和预充逻辑；`PRECHARGE_WINDOW` 组合本地允许、`PRE_CTRL_3V3` 与 `TMR_ACTIVE`。默认下拉能覆盖 MCU 复位时的请求默认态，但不能推出所有电源故障模式都已失效安全。
- `DFETOFFPinConfig=0x42` 是 active-high `BOTHOFF` 候选，可同时关 CHG/PCHG/DSG/PDSG。它改变了“需另加 CFETOFF 才能关 CHG”的旧判断；仍需实际写入/读回配置，并观察 BQ 错误/复位时 FET 输出。`R_DFETOFF_PU` 拉到 `REG1`：若 REG1 单独掉电、而 BAT/REG18 仍存在，硬件是否将 DFETOFF 拉高没有原厂保证证据，保持 OPEN。`NTC_CELL3` 留在 CFETOFF。
- `R_QPRE_PU` 的采购 BOM 已选项目复用 Yageo `RC0603FR-074K7L`/C99782、4.7 kΩ ±1%、0603、100 mW、75 V；电压分压纸面低端约 11.04–11.26 V，高于 10 V RDS(on) 条件。37.8 V 时若 11.4 V 齐纳开路，纸面栅压约 19.1 V，低于 ±20 V绝对最大值但余量不宽。4.7 kΩ相较旧 10 kΩ使关断 RC 约快 2.13 倍。
- **属性复核已闭合：**同根官方 API 直接读回 `R_QPRE_PU`（primitive `826afdaef88d2dc2`）以及当前 P1 属性记录均显示 YAGEO `RC0603FR-074K7L` / C99782、4.7 kΩ、1%、100 mW、75 V，pin1=`QPRE_SRC`、pin2=`QPRE_G`。本轮源 SHA 与起始相同；旧 10 kΩ字段只存在于旧派生针网 JSON，现已更正。
- `TPL5111` 逻辑窗纸面值约 2.858 s；这不是 R/C/器件公差下已证明的最大硬截止，也不证明电流已归零。没有独立 `PRE_OK` 或电流零反馈。需在 9S 上下限、热态、异常负载测 VGS、预充电流和截止时间；固件须在闭合主 MOS 前独立确认 PACK 电压。

## PSU、外部保险与默认许可

- `PSU5_EXT`：pin1 Vin−=`B0`，pin2 Vin+=`TAP_B9`，pin3 `VOUT_5V`，pin5 GND=`B0`，pin6 CTRL=`CTRL_MODULE`。`SW_RUN_STORE` 在预充控制、REG1 与模块 CTRL 间切换。短路电流、转换器温升/降额、CTRL 各状态和 5V 负载预算尚未做样机验收。
- `F_CELL_B1..B9` 与 `F1` 的保险配合必须按低压直流实际清除曲线计算。1000 VDC 额定值、熔断器额定电流或某一点 pre-arc I²t，均不能单独证明整车包故障选择性。
- `Q_LOAD_POWER_VALID_DRV` 的基极/G-S网络在 `FET_RELEASE` 无效时把光耦 LED 关断；BMS 光耦输出是串接在 `LOAD_EN_BIAS_EXT` 和 `LOAD_POWER_VALID` 之间的双线接点，外部 FOC 板提供 bias 和基极负载。按 FOC 双板 KCL 角点结果（SAFE_5V=4.75–5.25 V、A 板独有 1 kΩ bias、A/B 每板 4.7 kΩ base 与 100 kΩ pulldown），总电流范围约 1.046–1.425 mA，最差单支可用基流 0.5061 mA；对约 0.4560 mA sink 需求仅有 50.1 µA、约 11% 的假设余量。该纸面计算不保证 VO617A 的 0–60°C CTR/VCEsat 或目标器件角落；需在 SAFE 电源边界、两板并接和温度角落实测。

## CAN、MCU 和连接器

- `J_CORE_P1` 外接 5 V、B0、本地许可、BQ alert、PRE/MAIN 请求和 CAN RX/TX；`J_CORE_P2` 外接 3.3 V、B0、I²C 与 BAL_ENABLE_CMD。未列信号的位号在当前针网覆盖中 NC。采购器件、线束方向、电压域与板外控制器未做实际接口验证。
- `U_CAN` CA-IS2062A：逻辑侧 TX/RX=`CAN_TX/CAN_RX`，逻辑地 B0；隔离总线侧地 PACK−，CANH/CANL 到 `J_CAN`；隔离电源相关脚为 `CAN_ISO_5V`。板上没有 120 Ω 终端；按实际总线两端终端配置验收。TVS 有器件表示，耐压/浪涌/布线和隔离爬电距离没有 PCB 证据。
- 默认低的 `BMS_LOCAL_OK`、`MAIN_REQ`、`PRE_REQ` 与 ALERT/CAN_TX 上拉只说明所示电阻逻辑，不证明 MCU 固件在 watchdog、棕断、I²C 锁死或重复复位时锁存失效安全。

## 五路温度

| 当前位号 | 热区 | pin1 信号 | pin2 | 当前采购/模型状态 |
|---|---|---|---|---|
| `NTC_CELL1` | A | `BQ_TS1` | `B0` | Semitec 103AT-2 候选，BOM 待询价未购买；TI 18k default byte `0x07` 候选 |
| `NTC_CELL2` | B | `BQ_DCHG` | `B0` | Semitec 103AT-2 候选，BOM 待询价未购买；TI 18k default byte `0x07` 候选 |
| `NTC_CELL3` | C | `BQ_CFETOFF` | `B0` | Semitec 103AT-2 候选，BOM 待询价未购买；TI 18k default byte `0x07` 候选 |
| `NTC_BOARD` | 板内 | `BQ_TS2`/`NTC_BOARD_NODE` | `B0` | 时恒 MF52A103F3950(A1)，R/T 表为时恒 P477-4B；P163-6仅作历史资料。GPC 输入包已准备，系数和目标芯片写入/读回仍未完成 |
| `NTC_MOS` | MOSFET | `BQ_TS3` | `B0` | 时恒 MF52A103F3950(A1)，R/T 表为时恒 P477-4B；P163-6仅作历史资料。GPC 输入包已准备，系数和目标芯片写入/读回仍未完成 |

采购候选 BOM 为 NTC_CELL1/2/3 列出的 Semitec 103AT-2；当前待询价，未视为已采购。原理图旧 Vishay symbol/device/footprint 仅作 CAD 几何载体，`Convert to PCB=false`，两种身份在采购/设计数据中分开标识，不构成采购 MPN 冲突。板外探头、线束、绝缘与热耦合尚未实物验收。A/B/C 外置通道的 TI 18k 默认模型字节 `0x07` 是候选；NTC_BOARD/NTC_MOS 为时恒 MF52A103F3950(A1)，使用厂家 P477-4B R/T曲线；P163-6只作历史资料。A1 P477-4B 的 GPC 输入包已准备，但系数返回未收到，目标 BQ 写入及读回均未完成。raw 格式已确认：DASTATUS6 `0x0076` 中 CFETOFF bytes 12–15、TS1 24–27、TS2 28–31；DASTATUS7 `0x0077` 中 TS3 bytes 0–3、HDQ 4–7、DCHG 8–11，均为 signed 32-bit little-endian counts。候选窗 `250000 < raw < 4700000` 尚未对实际芯片/通道完成阈值资格验证，不能作为已验证固件常量；不要将 0.1 K 温度命令读数混同 raw counts。三外置通道任一开路/短路/读超时/配置读回不符应关闭充放电、预充与均衡并锁定故障；该固件策略未见实现证据。

## 优先未闭合项

1. 当前源已有官方保存/重开 ENET 与严格 DRC：226/740、0 net/live mismatch、0 error/9 warning groups。警告均需保留审查；此项不是硬件放行。
2. BQ76952 的 9S 通道、保护阈值、BOTHOFF `0x42` 与三外置热敏 `0x07` 尚未在实物写入/读回；REG1 单独掉电的 DFETOFF 状态没有证明。
3. 温度配置仍未 commissioned：A/B/C 的 0x07 候选及 TS2/TS3 A1 自定义模型都未在目标器件写入/读回；P477-4B GPC输入包已准备但系数未返回。signed-32 LE raw 格式已确认，250000/4700000 counts 仍是未资格化的候选开短窗；BOM 列出的 Semitec 候选 MPN 与旧 Vishay CAD carrier 已明确分离，不是采购冲突；传感器尚待询价。
4. 每路 5A 抽头保险与 100A 主保险在 37.8V 电池上的切断选择性、低于额定的故障、线束/座/PCB 最弱段 I²t 未闭合。
5. 预充 2.858 s 是逻辑时间窗，不是全温最大电流清零保证；需物理量测、主接触/允许延迟及异常负载 SOA 通过。
6. 双板光耦静态角点纸面 margin 约 50.1 µA/11%（单支最差 0.5061 mA 对约 0.4560 mA sink）；VO617A 0–60°C CTR/VCEsat及两块 4.7 kΩ基极负载组合没有全角保证，必须实测。
7. 5 mm 抽头接口、9 个外保险座、主保险座和大电流主回路未纳入本轮 PCB 审查；不得以符号/采购合同声称布局或装配已经完成。
8. `R_QPRE_PU` 的 BOM、当前原理图元件属性及直接 API 读回已一致；此前派生针网 JSON 中的旧 10 kΩ文本已更正，无需 native 修改。

## 针对性故障验证

| 验证 | 通过条件 |
|---|---|
| Cell-tap open/fuse open/miswire injection | Only intended cell differential is lost; no cross-cell short or false-good status; F_CELL_Bn series path and external insulation withstand verified. |
| VC filter and AFE threshold/config readback | Verify exact 9S enabled channels, unused VC parking, conversion polarity/gain and all safety thresholds from target readback. |
| AFE power-path faults | Open F_CELL_B9, D_BAT, D_BREG and each regulator rail independently; confirm state defaults and no unintended charge/discharge. |
| Active balance startup/stuck-enable/open cell | All eight channels default off, one selected pair operates with correct direction/current, and any invalid cell data forces every EN low. |
| Precharge timeout/stuck MOS/load short | Across pack/temperature corners measure gate, branch current and current-zero time; no MAIN close until independently verified voltage match; any timer/MCU fault opens safely. |
| DFETOFF BOTHOFF config and REG1 failure | Read back 0x42 setup; inject local reset/ALERT, REG1-only loss and BQ fault while BAT/REG18 remain up; prove both charge/discharge gates reach specified safe state. |
| Fuse coordination | Use pack prospective-current and vendor 37.8V DC TCC/total-clear data; verify 5A branches and 100A main fuse clear before every wire, holder, connector and board segment exceeds I²t/thermal limit; test below-rated fault behavior. |
| NTC open/short/swap/thermal ramp | Raw signed-32 LE field map is known; qualify the 250000/4700000-count candidate window on each configured channel, read back TI 0x07/A1 coefficients, inject open/short at each TS1/DCHG/CFETOFF/TS2/TS3, then verify fail-closed action, sensor identity, thermal coupling and hysteresis. |
| MCU reset/watchdog/I2C/CAN faults | Prove all permits default low, AFE alert behavior, no stale permission, bus short/open/termination limits and isolation withstand. |
| Shared opto simultaneous two-board load | At 0/60°C and SAFE_5V min/max, measure LED current, opto VCE, LOAD_POWER_VALID, each 4.7k base and collector clamp; validate all four dual-board load states and SAFE rail loss. |
| PSU load and CTRL switch faults | Verify input range at 9S extremes, inrush, 5V rail/load short, converter temperature, CTRL open/ground/store/run and source fuse operation. |

## 原厂参考

- [TI BQ76952 Technical Reference Manual](https://www.ti.com/lit/ug/sluuby2b/sluuby2b.pdf)
- [Vishay VO617A datasheet](https://www.vishay.com/docs/83430/vo617a.pdf)
- [Eaton PV-5A10F product data](https://www.eaton.com/us/en-us/skuPage.PV-5A10F.html)
- [Eaton CHPV holder family data](https://www.eaton.com/content/dam/eaton/products/electrical-circuit-protection/fuses/data-sheets/bus-ele-ds-10430-chcc-chm-chpv.pdf)
- [Littelfuse JLLN data](https://www.littelfuse.com/assetdocs/jlln-datasheet?assetguid=3a7bc9bf-d932-4401-bdc7-b39f302195cf)

`R_QPRE_PU` 属性与当前正式P1及BOM一致，且由当前官方proof绑定。本次只刷新派生审查入口/JSON/文档，没有改native或BOM、没有改PCB，也没有执行中央库同步；下游PCB及全部实物工作由人类完成。

机器可读完整针网：[current-pin-net-coverage.json](current-pin-net-coverage.json)。报告结构：[BMS全原理图逐针与故障链复核.json](BMS全原理图逐针与故障链复核.json)。
