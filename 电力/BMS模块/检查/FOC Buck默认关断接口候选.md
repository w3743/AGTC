# FOC Buck 默认关断许可接口候选

日期：2026-10-14。仅为接口候选，不是已实施源或已验证设计；不修改 FOC、BMS 连接 JSON 或 BOM。

## 当前源读回

- BMS 现有 `LOAD_POWER_VALID` 保持正逻辑和原引脚定义：`U_LOAD_POWER_VALID` pin 4 接外部 EN 偏置、pin 3 输出 `LOAD_POWER_VALID`；`J_LOAD_POWER_VALID` pin 1/2 分别为偏置输入/正逻辑许可输出。LED 只有 `VOUT_5V` 已由 `SW_RUN` 启用且硬件 `FET_RELEASE` 有效时才亮，MCU/CAN 不参与。此接口语义不变。
- FOC 的 `U1` pin 47 是 `BUCK_RT_SD`，由 `U1_BUCK_RT=280 kΩ` 接至 `BUS`。`J_PWR_SENSE` 当前没有 Buck 许可输入，`DRV_ENABLE` 只用于门极驱动许可。现有 `SAFE_5V` 独立从 `BUS` 生成，因此 Buck 可在 SAFE_5V 建立前启动；候选接口要在 BUS 上升时就把 RT/SD 拉低。
- Buck 默认关断不会切掉独立 `SAFE_5V`。既有两块 FOC 的 SAFE_5V 稳压与其负载仍须计入预充期间的母线负载/电流计算，不能把新增 Buck 许可当作消除了这部分预充负载。
- 当前 BMS 合并 BOM 为 487 个实物件、464 个采购位号，分类记录维护线束 56 件/33 位号；144 种归一化型号不含 CORE_BOARD，含 CORE_BOARD 为145种。计数证据 JSON 的分类和 `separateItems.maintenanceHarness` 均为56件/33位号，与当前 BOM 分类及 TODO 一致。

## 最少连接候选

增加一条独立、低有效的 `BUCK_ALLOW` 输出；它与现有正逻辑 `LOAD_POWER_VALID` 使用相同的硬件许可条件，但使用独立光耦输出端，不复用或改写现有输出。

| 位置 | 候选器件/端点 | 连接 |
|---|---|---|
| BMS | `R_BUCK_ALLOW_LED`，510 Ω；`U_BUCK_ALLOW`，VO617A-2X007T | `VOUT_5V → R_BUCK_ALLOW_LED → U_BUCK_ALLOW pin 1`；pin 2 接现有 `LOAD_VALID_LED_K`，由现有 `Q_LOAD_POWER_VALID_DRV` 在 `FET_RELEASE` 有效时下拉。此支路与现有 LED 并联、各自串联电阻。 |
| BMS | `J_BUCK_ALLOW`，2针 | pin 1=`BUCK_ALLOW`，接 `U_BUCK_ALLOW` pin 4（光敏三极管C）；pin 2=`FOC_GND_SYS_RETURN`，接 pin 3（光敏三极管E）。不得接 B0。 |
| 每块 FOC | `Q_BUCK_CLAMP`，NPN；`R_BUCK_CLAMP_B`，100 kΩ | `Q` 发射极接 `GND_SYS`，集电极接 `U1` pin 47/`BUCK_RT_SD`，基极接 `BUCK_ALLOW`；100 kΩ从基极接 `BUS`。新增单针信号连接器接 `BUCK_ALLOW`，回路地复用现有 `J_PWR_SENSE` pin 3=`GND_SYS`。 |

无许可或信号线开路时，光耦输出截止；BUS 经每块板上的100 kΩ给 NPN 基极偏置，NPN 把 RT/SD 拉低，Buck 默认关闭。许可成立时，BMS 光耦把 `BUCK_ALLOW` 拉到 FOC 地，令两块板的钳位 NPN 关断；现有280 kΩ网络再按 LM5008A 正常方式设置 RT/SD，Buck 才能启动。BUS 本身提供钳位偏置，不依赖尚未启动的 SAFE_5V、Buck 5 V或 MCU。

按两块 FOC、BUS=22.5–37.8 V估算，每个100 kΩ支路许可时约0.22–0.37 mA，光耦总集电极电流约0.44–0.74 mA。若每个输入LED用510 Ω，最坏按 LED VF=1.65 V、现有 NPN VCE=0.4 V，要保证 IF≥5 mA，`VOUT_5V(min)`须不低于4.60 V。现有 `Q_LOAD_POWER_VALID_DRV` 将同时下拉两只光耦 LED，LED总流约10–14 mA；当前10 kΩ基极电阻的全温饱和余量未闭合。若原驱动不能保证该电流，需把现有驱动基极电流重新定值或另加一个 NPN 与基极电阻（再增加2件）。

最少器件数为9件：BMS新增光耦、LED电阻、2针接口各1件；每块FOC新增钳位NPN、100 kΩ电阻和1针信号接口各1件。若需独立LED下拉驱动则为11件。线束分支/端子不计入器件数。此数是候选增量，不应与当前BOM计数混记。

## 必须满足的边界

- TI LM5008A pin 6 RT/SD 的绝对最大额定值属于“All other inputs to GND”：−0.3 V至+7 V；远程关断上升阈值全温为0.4–1.05 V（典型0.7 V），TI要求拉到地以关断。为保证全温关断，钳位 NPN 的实际 RT/SD 电压须在全温/电流角点低于0.4 V；候选NPN的数据表必须覆盖其实际低集电极电流，不能只用典型饱和压降推断。关断后内部LDO和Buck开关均关闭。
- VO617A-2的CTR下限63%及VCE(sat)最大0.4 V条件分别为IF=5 mA、VCE=5 V和IF=5 mA、IC=1 mA，表中注明Tamb=25°C；并无这里所需的全温最小CTR保证。两板所需输出电流低于0.74 mA，仍需按实际温度、LED电流与光耦老化余量核算。
- VO617A输出端在许可撤销时可能承受由FOC BUS 经100 kΩ产生的电压；系统BUS最高37.8 V低于该型号80 V集电极—发射极击穿额定值，但母线瞬态峰值仍须低于80 V。
- `BUCK_ALLOW` 断线/拔除为安全关闭；信号短到 `GND_SYS` 会误判为许可并允许Buck启动。此单通道低有效候选不能证明线路短路失效安全，须由系统故障边界接受该风险，否则需另加独立许可通道或短路诊断。
- FOC pin/连接器、BMS输出连接器、光耦及钳位晶体管均尚未进入源/BOM。新增信号不可接入 `LOAD_POWER_VALID` 正逻辑输出以替代原来的中央模块许可。

## 原厂依据

- [TI LM5008A 数据表，Rev. H](https://www.ti.com/lit/ds/symlink/lm5008a.pdf)：§6.1绝对最大额定、§6.5远程关断阈值、§7.3.5及§7.4.1 RT/SD关断。
- [Vishay VO617A 数据表](https://www.vishay.com/docs/83430/vo617a.pdf)：VO617A-2 CTR、输出饱和压降及80 V输出耐压。
- 当前源：[BMS接口连接](../原理图准备/接口元件连接.json)、[FOC原理图连接](../../FOC驱动模块/原理图准备/当前原理图连接.json)。
