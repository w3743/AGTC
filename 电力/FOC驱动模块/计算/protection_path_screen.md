# FOC保护路径纸面筛查

日期：2026-10-11。此报告对应[可复算脚本](protection_path_screen.py)与其JSON输出；计算结果仍是纸面角点筛查，不是固件、PCB、台架或整车验收。当前原理图已保存并回读：158器件、474引脚（461连接、13NC）、97网络、7颗IC；最终交付ZIP重开PASS，3份原生文件SHA与源一致。重开API DRC为0致命、0错误、5警告且未返回明细，旧12:01 GUI报告早于本轮修订，严格零警告未通过。配套生成JSON保留筛查时的计划差分字段，不能作为当前原理图状态记录。范围只含P0-1失电/默认关断和P0-2阈值/延迟；不含散热设计与热校核。

## 结论

**P0-1 未闭合。**当前原理图已将SAFE_ADC连至PB0/P2-9（COMP4_INP/ADC1_IN15），TEMP_MOS_ADC移至PA2/P2-18（ADC1_IN3）；COMP4欠压到TIM1 Break仍需固件配置和寄存器读回验证。它仅在MCU_3V3、VDDA和定时器路径有效时，才能独立于CPU指令流请求关断。SAFE_5V本身失效时，TLV1704负相比较器和UCC27511泄放驱动同时失电，泄放不可用；PWM关断后电机仍可经MOSFET体二极管向BUS回馈，吸能/隔离路径仍需由电力系统闭合。

**P0-2 未闭合。**纸面阈值角点有界，但全故障链没有数据手册可保证的最大响应时间；在 120 A 运行筛查点与最小静态过流门限之间只有约 10.6–11.5 A。瞬态余量还要扣除门限后的比较器、Break、门驱与栅极放电期间电流增长，需按实测最大 `|dI/dt|` 和同步波形验收。

电流环/回馈/5V联合模型共197项筛查；其中20kHz平均电流环给出：±100A指令、5000A/s斜率、20µH条件下，峰值包络最坏112.880A，低于120A筛查线7.120A；110A在20µH网格会越线，只有实测有效电感在饱和、温度及转子角度影响后仍≥44µH，且实测纹波/波形不高于包络时，才保留为条件候选。该模型不闭合全路径最大关断时间，也不把软件限流变成硬件保护。

## COMP4 欠压接法与纸面门限筛查

下表为当前原理图源及回读引脚映射。报告JSON仍保留计算运行时的旧记录和计划差分，因此不作为此处的连通性证据：

| 项目 | 旧筛查记录 | 当前原理图回读 |
|---|---|---|
| R_SAFEADC_TOP | 47 kΩ、1% | 27.4 kΩ、1%，SAFE_5V→SAFE_ADC；Yageo RC0603FR-0727K4L/C165753 |
| R_SAFEADC_BOT | 47 kΩ、1% | 10 kΩ、1%，SAFE_ADC→GND_SYS；Yageo RC0603FR-0710KL/C98220 |
| C_SAFEADC | 10 nF | 1 nF、50 V、C0G，SAFE_ADC→GND_SYS；TDK C1608C0G1H102J080AA/C183116 |
| 核心板映射 | PA2/P2-18=SAFE_ADC；PB0/P2-9=TEMP_MOS_ADC | PB0/P2-9=SAFE_ADC；PA2/P2-18=TEMP_MOS_ADC |
| 硬件路由 | PA2 ADC轮询 SAFE_ADC | PB0同时接ADC1_IN15与COMP4_INP；COMP4→TIM1 Break须由固件寄存器配置并读回验证；PA2继续作温度ADC |

此方案不增加 IC、不增加六路 PWM 输入下拉。PB0 既有 ADC1_IN15 又是 COMP4_INP；PA2 为 ADC1_IN3。阈值名义值为 `1.212 × (27.4k+10k)/10k = 4.53288 V`。使用 STM32G431 数据手册全温 VREFINT `1.182…1.232 V`、COMP scaler offset `±10 mV`、COMP offset `−9…+3 mV`、PB0 TT 输入漏电 `±150 nA` 及两只电阻各 ±1% 独立角点，得 SAFE_5V 静态翻转范围 **4.2824…4.7294 V**。TT 输入漏电经分压上臂约再贡献 ±4.15 mV 的 SAFE_5V 等效误差。`HYST=000` 只有 0 mV typical、没有 min/max，故没有可保证的回差值；SAFE_5V 边界附近可能抖动，需实测并用系统恢复策略避免自动重启。

5.0 V SAFE_5V 时 SAFE_ADC 名义值 1.337 V，比 VREFINT typical 高约 125 mV。以现有 TPS7A4001 反馈分压和器件规格做的输出下限筛查为 4.769 V，较含PB0漏电的COMP4最高翻转角点约高 **39.6 mV**；该计算把 ±2.5% 输出精度套用到外置反馈分压，尚需复核其对该电路、温度和反馈条件是否为保证值，因此不宣布 SAFE_5V 健康裕量已硬件保证。

### 固件与寄存器约束

- COMP4 选择 `INPSEL=0`（PB0）、`INMSEL=011`（VREFINT）、`SCALEN=1`、`BRGEN=0`；COMP4 `POLARITY=1` 反相，使 SAFE 健康时输出低、欠压时输出高。
- COMP scaler 启动最大 200 µs、COMP 启动最大 5 µs；两者使能后至少等待 205 µs，再验证 SAFE_ADC/COMP 状态并允许 ARM。该等待不是电源故障响应时间。
- 在 `TIM1_AF1` 置 `BKCMP4E=1`，并置 `BDTR.BKE=1`。COMP4 输出为高有效故障时 `BKCMP4P=0`；外部 `BREAK_N` 为低有效时，候选 `BKP=1、BKINP=1`。保留 `AOE=0`，复位/故障后 `MOE=0`，显式 ARM 才恢复。STM32G431 的 `COMP_CSR` 没有 `OUTSEL` 字段，不能用其他系列寄存器名替代内部 `TIM1_AF1.BKCMP4E` 路由。
- 启用并读回 BOR level 4 option bytes；数据手册给出的 BOR4 上升阈值为 2.85/2.90/2.95 V、下降阈值为 2.76/2.81/2.86 V（min/typ/max）。实物需核对 option-byte 读回、复位和供电斜坡。
- 在抬高 DRV_ENABLE 前，将六路 TIM1 输出配置为 LOW，并确认 `MOE=0、AOE=0`、Break 源/极性及安全 `OIS/OISN、OSSI/OSSR、deadtime` 配置均读回正确。复位时 `DRV_ENABLE` 由已有 10 kΩ 外部下拉保持低。DRV8353 数据表列出 INHx/INLx 100 kΩ 内部下拉，但该项没有 min/max；本版不把它当作六个输入有保证的板级低电平，也不增加六颗外拉。六个 U1 输入和六路 VGS 的复位/掉电波形必须在台架确认。禁止自动重试或复位后自动 ARM。

## 失电路径与边界

| 条件 | 本地能做什么 | 仍未保证/必须验证 |
|---|---|---|
| SAFE_5V 下降，MCU_3V3/VDDA 有效 | COMP4 可在配置完成后直接请求 TIM1 Break；与 CPU 是否执行无关 | U3 TLV1704 与 U4 UCC27511 已失电，主动泄放不可用；TPS7A4001 输出下降斜率和 SAFE 输出电容/负载决定实际检测时间；须量 SAFE_ADC、BREAK_N、六 PWM、六 VGS、BUS 电压/电流与相电流 |
| MCU_3V3/主 5 V 下降，SAFE_5V 尚在 | DRV_ENABLE 已有 10 kΩ 下拉；BOR4 可使 MCU 复位 | COMP4/TIM1 功能不应再被依赖。六路 PWM 的 100 kΩ 内拉无保证上下限；测六个输入和六路 VGS，验证主板/核心板/USB/Hall 是否造成部分供电 |
| 两路控制电源均失电 | DRV_ENABLE 外部下拉和门驱 UVLO/sleep 可降低驱动 | 泄放不可用。关栅后体二极管仍可把转动电机能量送回 BUS；源端必须能吸收，或由独立制动/隔离/残能路径处置 |
| MCU 复位/看门狗/挂死 | 若 COMP4/TIM1 已完成配置且 MCU/VDDA 保持有效，硬件比较路径可继续触发 Break | 复位期间依赖 ENABLE 外部下拉；CPU 初始化顺序、六 PWM LOW、配置读回、Break 失效注入与显式 ARM 需实测 |
| Q7 漏源短路/泄放电阻开路 | 固件停止回馈；温控开关可切断门驱输入 | Gate-off/温控不能清除漏源短路；开路没有吸能。须由系统级实体断电、源端吸能或独立热/能量处置闭合 |

三路 `CSA_SO_RAW→U3` 的 1 kΩ 串阻只在规定信号/电源条件下限制各 TLV 输入支路电流。以 SO 绝对最大 `VREF+0.3 V`、`VREF≤3.6 V`、SAFE_5V=0、TLV 输入绝对额定轨外 `0.5 V` 和 1 kΩ 最小 990 Ω 条件筛查，每路钳位电流上界约 **3.43 mA**，低于 TI 的 10 mA 输入绝对最大值。它不证明失电 TLV1704 输出状态、没有 SAFE_5V 反向供电，或两路轨之间无部分上电。此结论是有条件的电流限制，不是 fail-safe。

STM32G431 的 PB0/PA2 属 TT 类 IO。ST 表15说明在规定绝对最大输入范围内正向注入电流为 0/不发生；这不能被改写成“存在一个能把异常电流钳到 VDD 的二极管”或据此计算高回灌电流。数据手册还将绝对最大输入限制与 VDD+0.3 V 的正常工作条件分开；VDD/VDDA 失电时 ADC/COMP 功能没有保证。`SO_ADC` 节点由现有 47 Ω/220 pF 支路连接 MCU，掉电行为仍需同时量 MCU_3V3、VREF_CORE、CSA_SO_RAW、SO_ADC 和 SAFE_ADC，报告是否有轨抬升及各引脚电压；不凭空增加隔离器，也不声称部分供电已通过。

## 电流门限与动态筛查

沿用 [当前电流窗口角点计算](../原理图准备/电流窗口误差核算.md)：0.5 mΩ 分流器、GAIN10、正向 DAC3_CH1/CH2 码 2979、负向 VREF_CORE/26.7 kΩ/10 kΩ。计算采用电阻/分流器容差、VREF_CORE ±2.5%、CSA gain/VOS、DAC TUE 和 COMP/TLV 输入失调的角点组合；它不是完整硬件误差保证。

| 方向 | 静态最小门限 | 静态最大门限 | 相对 120 A 筛查最小余量 |
|---|---:|---:|---:|
| 正相（加COMP输入漏电界限） | +131.513 A | +169.767 A | 11.513 A |
| 负相（加TLV输入偏置界限） | −130.588 A（幅值） | −171.158 A（幅值） | 10.588 A |

来源角点文件未计入比较器输入电流。TI TLV1704 每输入全温最大偏置 20 nA；按1 kΩ+1%信号串阻与26.7 kΩ/10 kΩ、各±1%参考分压的最大等效阻抗合并，差分电压误差界限约 **167.2 µV**，按最小 `0.492525 mΩ×9.7=4.7775 mV/A` 换算为负相附加电流误差 **0.0350 A**。正相G431 COMP若接在47 Ω滤波节点，TT输入漏电±150 nA再贡献约7.05 µV，即 **0.00148 A**；此项取决于COMP实际抽头和现行网表，需保存后核对。表中门限和余量已把两项保守地加到来源角点。额外输入电流误差虽小，仍不能替代 CSA 漂移、地偏差、噪声和动态误差。DRV8353 CSA 偏置漂移 `10 µV/°C`、建立时间、带宽与压摆率是 typical 项，没有可用的最大响应界限。

| 电流上升斜率假设 | 正相到最小门限余量时间 | 负相到最小门限余量时间 |
|---:|---:|---:|
| 1 A/µs | 11.513 µs | 10.588 µs |
| 5 A/µs | 2.303 µs | 2.118 µs |
| 10 A/µs | 1.151 µs | 1.059 µs |
| 25 A/µs | 0.461 µs | 0.424 µs |

公式为 `(最小静态门限幅值−120 A)/|dI/dt|`；这只是从 120 A 到门限的时间，**不含门限后任何响应时间和过冲**。作为对照，TI DRV8353 GAIN10 CSA 的 500 ns 是 0.5 V SO 阶跃的 typical 建立时间；TI TLV1704 的 460/560 ns 也是 typical（100 mV 过驱、15 pF、5.1 kΩ 上拉）。在 25 A/µs 筛查斜率下，留给达到静态门限的时间已小于 500 ns，但阶跃/过驱测试条件不同，不能仅凭比较判定通过或失败，必须测波形。

正相COMP路径的47Ω/220pF ADC节点被动时间常数为**10.34ns**，理想RC的10–90%为**22.7ns**；这只描述外部RC。COMP具体抽头、CSA源阻抗和ADC采样窗仍须按当前网表与实物确认。负相U3支路的1kΩ没有并联滤波电容，不以未知输入电容伪造延迟上界。SAFE_5V欠压支路按27.4kΩ∥10kΩ和1nF的名义RC时间常数为**7.326µs**；理想SAFE_5V源从5V瞬间降至0V时，分压节点到COMP4静态角点门限约需**0.385–1.197µs**。这不含LDO输出电容的真实放电、BUS下降、负载变化或接线电感；真实rail decay没有这里可用的最大值。

ST G431 COMP 最大 31 ns（VDDA≥2.7 V）/35 ns（VDDA<2.7 V）只适用于 200 mV 输入阶跃、100 mV overdrive、50 pF 输出负载的指定条件。DRV8353 Table 7.5 的输入到 GHx/GLx 延迟为单列 200 ns，没有 min/max；它不是 VGS 关断或栅极放电上界。TIM1 从 Break 到输出引脚的端到端最大延迟、Break期间 OIS/OISN 与 deadtime 转换、实际栅极放电、负载实际电流斜率均没有由上述数据拼成的总最大值。**不得把典型/单列时间相加后标成 worst-case/PASS。**

台架应同步捕获：三路 SO_RAW、SO_ADC、COMP1–4状态/或 BREAK_N、U1 六路 INHx/INLx、六路 VGS、相电流、BUS 电流/电压、SAFE_5V、MCU_3V3、VREF_CORE、DRV_ENABLE 和 nFAULT；分别注入正/负过流、SAFE_5V下降、主 5 V/MCU 掉电、BUS 过压及复位。用最大实测 `|dI/dt|` 计算门限后电流增长并核对 MOS SOA、BUS 能量与源端吸能。故障锁存、TIM1 输出闲置状态、显式重 ARM 和残压/物理隔离需另项验证。

## 原厂资料

- ST [STM32G431x6/x8/xB 数据手册 DS12589 Rev.6](https://www.st.com/resource/en/datasheet/stm32g431r8.pdf)：Table 12 pp.52–53（PB0 COMP4_INP/ADC1_IN15、PA2 ADC1_IN3）；Tables 14–17 pp.68–71（TT 输入、正注入说明、绝对最大值与工作范围）；Table 19 p.72（BOR level 4）；Table 20 p.74（VREFINT 全温范围）；Table 73 p.143（COMP scaler/offset、启用/传播时间）。
- ST [RM0440 Rev.9](https://www.st.com/resource/en/reference_manual/dm00355726.pdf)：§24.3.2、Tables 199–200（COMP 引脚/内部输入选择，印刷 pp.768–769）；Table 269（TIM Break interconnect，p.1149）；TIM1_AF1/BKIN polarity 寄存器说明（p.1192）；Table 275（Break状态，p.1197）；Table 282（互补输出、MOE/OSSI/OSSR 与 idle 状态，p.1261）。
- ST [STM32G431 CMSIS 设备头文件](https://raw.githubusercontent.com/STMicroelectronics/cmsis-device-g4/master/Include/stm32g431xx.h)：`COMP_CSR` 字段和 `TIM1_AF1_BKCMP4E/BKCMP4P`；COMP_CSR 无 `OUTSEL` 字段。
- ST [STM32G4 LL COMP 头文件](https://raw.githubusercontent.com/STMicroelectronics/stm32g4xx-hal-driver/master/Inc/stm32g4xx_ll_comp.h)：`LL_COMP_INPUT_MINUS_VREFINT` 对应 INMSEL 与 SCALEN，full-VREFINT 不需 BRGEN；启动延迟宏值 200 µs scaler/5 µs comparator。
- TI [DRV835x 数据手册 Rev.A](https://www.ti.com/lit/ds/symlink/drv8353.pdf)：Table 7.5 p.12（INHx/INLx 列示 100 kΩ internal pull-down、tPD 200 ns，均无 min/max）；§7.5 pp.14–15（CSA 增益/VOS/漂移及典型建立时间）；§8.4.1.1 p.53（ENABLE sleep）。
- TI [TLV1704 数据手册 Rev.D](https://www.ti.com/lit/ds/symlink/tlv1704.pdf)：Table 7.1 p.6（输入绝对额定/钳位脚注）；Tables 7.6–7.7 p.7（20 nA 最大偏置、传播延迟为 typical 及测试条件）。
- TI [TPS7A4001 数据手册 Rev.B](https://www.ti.com/lit/ds/symlink/tps7a4001.pdf)：Table 6.5 p.5（参考值和反馈参数测试条件）；可调输出方程与应用建议 pp.9–10。此处 LDO 输出下限仅作电路筛查假设，不宣称全温 min 保证。

## 状态

| 项目 | 判定 | 当前证据 |
|---|---|---|
| SAFE_ADC引脚与分压连通性 | 原理图回读确认 | PB0/P2-9接SAFE_ADC，PA2/P2-18接TEMP_MOS_ADC；COMP4/TIM1固件配置仍须读回 |
| COMP4静态门限计算 | 纸面筛查 | ST数据手册、RM0440与CMSIS/LL头文件；不是实测阈值或全路径保护验收 |
| ±150 A 对 120 A 静态余量 | 角点筛查通过 | 已有 current-window JSON；不是动态限流验收 |
| 失电时 PWM/ENABLE 安全态 | 未闭合 | 10 kΩ ENABLE 下拉已存在；六 PWM 仅依靠无上下限保证的芯片内列值，需实测 |
| SAFE_5V失电时能量去路 | 未闭合 | 主动泄放电源也失效，体二极管仍可回馈；须系统级吸能/隔离方案 |
| 全路径最大关断时间/允许过冲 | 未闭合 | CSA/TLV 延迟只有 typical；TIM1、门驱/栅极及真实电流斜率缺最大边界和同步台架数据 |
| 散热与热验收 | 本报告不覆盖 | 由用户负责 |
