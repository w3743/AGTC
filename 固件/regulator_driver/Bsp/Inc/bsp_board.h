#ifndef __BSP_BOARD_H__
#define __BSP_BOARD_H__

#include "type.h"
#include "stm32g4xx_hal.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 稳压模块（sc-regulator，2026-10-30 接口重排版）—— 板级引脚总表
 *
 * 这是本工程**唯一**定义引脚和 ADC 通道的地方。各 bsp_*.c 只引用、不重复定义。
 * 改板、换脚、改通道，只改这一个文件，不用去翻六个 .c。
 *
 * ── 依据（2026-10-30 版，旧针网已作废）────────────────────────────────────
 *   逐针合同：电力/稳压模块/原理图准备/最终逐针连接.csv
 *             （515 针 / 166 实例 / 21 NC / 84 网，四页 DRC 为空后的只读读回）
 *   GPIO 表 ：电力/稳压模块/设计细节.md §「GPIO及实现交接」
 *   Netlist ：电力/稳压模块/检查/当前官方网表.enet（= 原理图准备/目标连接网表.enet）
 *   MCU 复用：DS12589 Table 12（UFQFPN48 列）
 *
 *   ⚠️ 2026-10-30 之前的 pin 表**整体作废**。那次改版把
 *      “8 颗 BW6101 被动均衡 + 8 路直采抽头 + PMOS_CTRL/TEST”换成了
 *      “3 颗 TMUX1511 选通 + 2 路共用 ADC + 共享故障线 + 开漏许可”，
 *      引脚几乎全动。看到旧文档里的 AD_T1…AD_T8 / PB6 / PB10 一律按本文件为准。
 *
 *   排针 J_CORE_P1 = 核心板 P1（WeAct Normal QFN48），J_CORE_P2 = 核心板 P2，
 *   各 2×12、2.54 mm。下文 P1.n / P2.n 就是这两排的针号。
 *
 * ── 这块板是什么 ────────────────────────────────────────────────────────────
 *   一颗**双向半桥 DC/DC**，在 9S 电池直连的 BUS（22.5–37.8 V）与 8S1P 超容之间
 *   搬能量；外加一路独立软件控制的泄放（dump）电阻、8 节超容抽头经模拟开关选通采样、
 *   3 路 NTC、SC 总压硬件比较（TLV6700）、SC 侧背靠背隔离（LTC7001 + 2×HYG022）
 *   以及一条**共享故障线** DRV_FAULT_N（DRV8161 nFAULT/nDRVOFF、LTC7001 INP、
 *   TLV6700 OUTB、MCU 的 G07 开漏、两个比较器输出全挂在这条线上）。
 *
 *   板上的 MCU 就是核心板上的 STM32G431CBU6 —— **本板自己没有主控**，
 *   只提供 J_CORE_P1/P2 两排母座插核心板。同一颗核心板换插到 FOC 板上，
 *   就是另一套固件（本仓库的 foc_driver 工程）。
 *
 * ── 引脚总表 ────────────────────────────────────────────────────────────────
 *   信号           排针      MCU    复用/通道              方向    说明
 *   ----------------------------------------------------------------------------------
 *   PWM_H         P1.10     PA8    TIM1_CH1    (AF6)     输出    半桥上管（→ DRV8161 INH）
 *   PWM_L         P1.6      PB13   TIM1_CH1N   (AF6)     输出    半桥下管（→ DRV8161 INL）
 *   DUMP_PWM      P1.22     PB7    TIM4_CH2    (AF2)     输出    泄放 PWM（→ U_DUMP_DRV IN+）
 *   PERMIT        P1.11     PA9    GPIO                  输出    经 U_GATE_OD(G07) 开漏拉 DRV_FAULT_N
 *   DRV_FAULT_N   P1.5      PB12   TIM1_BKIN   (AF6)     输入    共享故障线，低 = 故障，硬件关桥
 *   FAULT_FB      P1.12     PA10   GPIO                  输入    同一条故障线的真线回读
 *   FDCAN1_RX     P1.13     PA11   FDCAN1_RX   (AF9)     输入    ⚠️ 与核心板 USB D− 硬连
 *   FDCAN1_TX     P1.14     PA12   FDCAN1_TX   (AF9)     输出    ⚠️ 与核心板 USB D+ 硬连
 *   MUX_SEL1      P1.9      PC6    GPIO                  输出    U_CELL_MUX1.SEL1 → 抽头 T1
 *   MUX_SEL2      P1.15     PA15   GPIO                  输出    U_CELL_MUX1.SEL2 → 抽头 T2
 *   MUX_SEL3      P1.16     PC10   GPIO                  输出    U_CELL_MUX1.SEL3 → 抽头 T3
 *   MUX_SEL4      P1.17     PC11   GPIO                  输出    U_CELL_MUX1.SEL4 → 抽头 T4
 *   MUX_SEL5      P1.18     PB3    GPIO                  输出    U_CELL_MUX2.SEL1 → 抽头 T5
 *   MUX_SEL6      P1.19     PB4    GPIO                  输出    U_CELL_MUX2.SEL2 → 抽头 T6
 *   MUX_SEL7      P1.20     PB5    GPIO                  输出    U_CELL_MUX2.SEL3 → 抽头 T7
 *   MUX_SEL8      P1.24     PB9    GPIO                  输出    U_CELL_MUX2.SEL4 → 抽头 T8
 *   SC_ADC_A      P1.7      PB14   ADC1_IN5              模拟    MUX1 的 D1–D4 公共端（T1–T4）
 *   SC_ADC_B      P1.8      PB15   ADC2_IN15             模拟    MUX2 的 D1–D4 公共端（T5–T8）
 *   BUS_ADC       P2.9      PB0    ADC1_IN15             模拟    母线电压（150k/10k 分压）
 *   IMON          P2.10     PB1    ADC1_IN12             模拟    DRV8161 CSA 输出（兼 COMP1_INP）
 *   (IMON_COMP2)  P2.11     PA7      —                  模拟    同一路 SO 的第二分支，只接 COMP2_INP
 *   (IMON_COMP3)  P2.20     PA0      —                  模拟    同一路 SO 的第三分支，只接 COMP3_INP
 *   NTC_MAIN      P2.6      PB11   ADC2_IN14             模拟    主功率级温度
 *   NTC_SC        P2.12     PC4    ADC2_IN5              模拟    超容温度
 *   NTC_DUMP      P2.15     PA3    ADC1_IN4              模拟    泄放电阻温度
 *   (COMP1_OUT)   P2.14     PA6      —                  输出    COMP1 开漏输出，接 DRV_FAULT_N（未启用）
 *   (COMP2_OUT)   P2.18     PA2      —                  输出    COMP2 开漏输出，接 DRV_FAULT_N（未启用）
 *   VREF_CORE     P2.8      VREF+    —                  电源    ADC 满量程 + CSA 基准
 *
 *   未连接（NC，不要驱动、不要上拉）：P1.21=PB6、P1.23=PB8-BOOT0、P2.5=PB10、
 *   P2.7=PB2、P2.13=PA5、P2.16=PA4、P2.17=PA1、P2.21=PC14、P2.22=PC15、P2.24=PC13。
 *
 * ── 六个必须记住的陷阱（改动前先读）──────────────────────────────────────────
 *
 *  1) PA11/PA12 = FDCAN1_RX/TX + USB D−/D+
 *     核心板把这两个脚同时引到 Type-C。**接 CAN 时不得再插 USB 数据线**，
 *     否则 USB 收发器会成为 CAN 总线的额外负载。
 *
 *  2) PERMIT 是**低有效许可、高禁止**，和 2026-10-30 之前完全相反
 *     老版本是 PB5 推挽直接驱动故障线（高=放行）；现在是 PA9 → SN74LVC1G07 开漏缓冲器 A 端，
 *     Y 端并到 DRV_FAULT_N。G07 是非反相缓冲：A 高 → Y 拉低 → 故障线低 → 禁止。
 *     所以 **PA9 = 高 → 禁止，PA9 = 低 → 放行**。
 *     故障线上另有 R_PB5_PERMIT 1k 上拉到 3V3_CORE（保留原名）与 R_DRV_FAULT_PD 10k 下拉，
 *     静态电平约 3.0 V（未故障）。D 端不能推挽，只能开漏——推挽高电平会顶住整条故障线。
 *
 *  3) PB12 = DRV_FAULT_N，**这次真的接了 BKIN**
 *     老版本 PB12 是 NTC 模拟输入，Break 必须关掉才不会被 NTC 分压误触发。
 *     现在 PB12 就是共享故障线，BreakState 必须 **ENABLE**、极性 LOW，
 *     AutomaticOutput **DISABLE**（刹车后不自恢复，必须软件重新 start）。
 *
 *  4) MUX_SEL 必须**一次只开一路**
 *     两颗 TMUX1511 各自的 D1–D4 是连在一起的（都接同一个 ADC 节点），
 *     同时开两路等于把两节超容通过开关直接短在一起。bsp_mux 里做了 one-hot 保护。
 *
 *  5) 抽头分压比变成 14.6，不是 7.8 / 9.25
 *     选通后 AD_Tx 的 10k 与 ADC 节点的 10k 并联成 5k，于是
 *     K = 1 + 68k/5k = 14.6。**8 路全部一样**，T8 也不再是 7.8。
 *     旧的 47k→TEST 支路随 TEST 网络一起删除，不再有 −c×Vtest 修正项。
 *
 *  6) P2.7 = PB2 是核心板自己的 VBUS 检测（经 SB4 + 板载 100k/10k/10nF）
 *     不要复用、不要改板。同理 P1.19 = PB4 常被误当成 VBUS 检测，它不是。
 *
 * ── 封装硬限制（UFQFPN48，改板时别把线画到这些脚上）──────────────────────────
 *   PC0 / PC1 / PC2 / PC3 / PC5 / PC12 在 UFQFPN48 上**不存在**。
 *   原理图核对时若发现这些脚名，说明看错了封装列（DS12589 Table 12 有 5 种封装，
 *   只有 UFQFPN48 那一列才算数）。
 ********************************************************************************************/

/************************************双向半桥（TIM1）******************************************/
/* 为什么只有两路：DRV8161 工作在 2-pin PWM 模式（DT/MODE 脚 3.3 kΩ 上电锁存），
 * 上下管各吃一根输入线，互补由 MCU 的 CH1 / CH1N 直接给出。
 * 手册真值表：INH=1,INL=0 → GH 高、GL 低；INH=0,INL=1 → GH 低、GL 高；
 *            两者同为高或同为低 → 两栅极都低、SH 进 Hi-Z（这就是它自带的防直通）。 */
#define BSP_BOARD_BRIDGE_H_PORT                                         (GPIOA)
#define BSP_BOARD_BRIDGE_H_PIN                                          (GPIO_PIN_8)    /* P1.10 */
#define BSP_BOARD_BRIDGE_L_PORT                                         (GPIOB)
#define BSP_BOARD_BRIDGE_L_PIN                                          (GPIO_PIN_13)   /* P1.6  */
#define BSP_BOARD_BRIDGE_AF                                             (GPIO_AF6_TIM1)

/* ---- 时基 ---- */
/* ⚠️ 这个频率**尚未定型**。原理图准备文档明确写着「PWM 频率、采样同步、限流策略和
 *    交越停机时序待确定」，所以下面这个 20 kHz 只是**首轮上电用的保守值**，不是结论。
 *
 * 为什么先给 20 kHz：开关损耗最低，第一次通电最容易活下来。
 *
 * 但它带来的电感纹波很大，必须知道这个数：
 *     ΔI_pp = V_bus × D × (1 − D) / (L × f)
 *   本板 L_MAIN_A + L_MAIN_B 串联 = 20 µH，母线取额定 32.4 V：
 *     f = 20  kHz, D = 0.5  → ΔI_pp ≈ 20.3 A
 *     f = 50  kHz, D = 0.5  → ΔI_pp ≈  8.1 A
 *     f = 100 kHz, D = 0.5  → ΔI_pp ≈  4.1 A
 *
 * 中心对齐：f_pwm = f_tim / (2 × ARR)。170 MHz / (2 × 4250) = 20 kHz。 */
#define BSP_BOARD_BRIDGE_PWM_ARR                                        (4250U)
#define BSP_BOARD_BRIDGE_PWM_FREQ_HZ                                    (20000U)

/* 死区（ns）。
 *
 * ⚠️ 这里刻意设成 0，不是漏了。
 *   DRV8161 在 2-pin 模式下自己插死区，典型 370 ns（数据手册范围 316–422 ns）。
 *   两边都加死区不会更安全，只会把有效占空比压缩掉 2×370 ns。
 *
 * TIM1 时钟 170 MHz → tDTS = 5.882 ns，DTG 寄存器 1 个步进约 5.9 ns。 */
#define BSP_BOARD_BRIDGE_DEADTIME_NS                                    (0U)

/* ---- 刹车（BKIN = DRV_FAULT_N，PB12）----
 * 2026-10-30 改版后这条线是真·共享故障线，打开 Break 不再有"被 NTC 误触发"的风险。
 * 极性 LOW：线上任何一方（DRV8161 nFAULT / TLV6700 OUTB / G07 / 比较器）拉低即触发。
 * 滤波 0：故障线已经是开漏+1k 上拉，边沿干净；加滤波只会延长关断延迟。 */
#define BSP_BOARD_BKIN_PORT                                             (GPIOB)
#define BSP_BOARD_BKIN_PIN                                              (GPIO_PIN_12)   /* P1.5  */
#define BSP_BOARD_BKIN_AF                                               (GPIO_AF6_TIM1)

/************************************泄放（TIM4_CH2）******************************************/
/* 泄放电阻不在板上（外置，接 J_DUMP_BUS），由 Q_DUMP 低边开关 + U_DUMP_DRV(UCC27511A) 驱动。
 * 2026-10-30 版新增：除 MCU 的 DUMP_PWM 之外，U_SC_OV 的 OUTA 与 PB7 驱动的 MMBT5551
 * 也能各自拉高 Q_DUMP 栅极（DUMP_OFF 节点），即**硬件与软件是两条并行的开通路径**。
 * MCU 复位期间 Q_DUMP 栅极由板上电阻保持关断。 */
#define BSP_BOARD_DUMP_PWM_PORT                                         (GPIOB)
#define BSP_BOARD_DUMP_PWM_PIN                                          (GPIO_PIN_7)    /* P1.22 */
#define BSP_BOARD_DUMP_PWM_AF                                           (GPIO_AF2_TIM4)

/* 泄放 PWM 频率。
 * 20 kHz 顺手和半桥同频，避免两个开关动作在采样上互相拍频。
 * TIM4 在 APB1（本工程 170 MHz），边沿对齐：ARR = 170 MHz / 20 kHz − 1 = 8499。 */
#define BSP_BOARD_DUMP_PWM_ARR                                          (8499U)
#define BSP_BOARD_DUMP_PWM_FREQ_HZ                                      (20000U)

/************************************功率许可（PA9 → G07 开漏）*******************************/
/* 这是本板最硬的一道软件闸：PA9 高 → G07 把 DRV_FAULT_N 拉低 → DRV8161 停输出、
 * LTC7001 INP 变低 → SC 背靠背隔离撤销。与 TIM1 在做什么完全无关。
 *
 * ⚠️ 极性与老版本相反（老版本 PB5 高=放行）。这里显式写成宏，别在函数体里写魔法电平：
 *      BSP_BOARD_PERMIT_LEVEL_ALLOW   → 放行
 *      BSP_BOARD_PERMIT_LEVEL_INHIBIT → 禁止
 *
 * ⚠️ 上电默认禁止：G07 的 A 端有 100k 下拉（A=0 → Y 高阻 → 故障线由 1k/10k 分压到 3.0 V，
 *    即"未故障"），但半桥此时 INH/INL 都是 47k 下拉、MOE=0，功率级本身是关的。
 *    软件必须在自检通过后才显式放行，见 bsp_gpio.c。 */
#define BSP_BOARD_PERMIT_PORT                                           (GPIOA)
#define BSP_BOARD_PERMIT_PIN                                            (GPIO_PIN_9)    /* P1.11 */
#define BSP_BOARD_PERMIT_LEVEL_ALLOW                                    (GPIO_PIN_RESET)
#define BSP_BOARD_PERMIT_LEVEL_INHIBIT                                  (GPIO_PIN_SET)

/* 故障线真线回读（PA10）。只读，用来区分"故障还在"与"故障已消失"，
 * 以及判断刚才的 Break 是谁拉低的（结合 DRV8161 的 Fault Status 寄存器）。 */
#define BSP_BOARD_FAULT_FB_PORT                                         (GPIOA)
#define BSP_BOARD_FAULT_FB_PIN                                          (GPIO_PIN_10)   /* P1.12 */
#define BSP_BOARD_FAULT_FB_LEVEL_OK                                     (GPIO_PIN_SET)
#define BSP_BOARD_FAULT_FB_LEVEL_FAULT                                  (GPIO_PIN_RESET)

/* 两个比较器开漏输出（PA6/PA2）也并在这条故障线上，但**本版固件不启用 COMP**：
 * 阈值、迟滞、极性、DAC 上电顺序与传播延迟都还没核，开着比不开危险。
 * 引脚保持模拟输入（COMP 关闭时是高阻，不影响故障线）。 */
#define BSP_BOARD_COMP1_OUT_PORT                                        (GPIOA)
#define BSP_BOARD_COMP1_OUT_PIN                                         (GPIO_PIN_6)    /* P2.14 */
#define BSP_BOARD_COMP2_OUT_PORT                                        (GPIOA)
#define BSP_BOARD_COMP2_OUT_PIN                                         (GPIO_PIN_2)    /* P2.18 */

/************************************抽头选通（TMUX1511 ×2）**********************************/
/* 8 节抽头不再直连 ADC：每 4 节进一颗 TMUX1511，四路开关的 D 端并到同一个 ADC 节点。
 *   U_CELL_MUX1：S1–S4 = AD_T1–AD_T4 → D1–D4 = SC_ADC_A（PB14 / ADC1_IN5）
 *   U_CELL_MUX2：S1–S4 = AD_T5–AD_T8 → D1–D4 = SC_ADC_B（PB15 / ADC2_IN15）
 * 两颗芯片的 SEL 各自独立，共 8 根，**必须 one-hot**（理由见文件头陷阱 4）。
 *
 * ⚠️ 上电默认全低：8 个开关全断，抽头只挂自身 68k/10k 分压，ADC 节点由 10k 下拉到 0 V。
 *    这是安全态，也是"没在测"的明确状态。 */
#define BSP_BOARD_MUX_SEL1_PORT                                         (GPIOC)
#define BSP_BOARD_MUX_SEL1_PIN                                          (GPIO_PIN_6)    /* P1.9  */
#define BSP_BOARD_MUX_SEL2_PORT                                         (GPIOA)
#define BSP_BOARD_MUX_SEL2_PIN                                          (GPIO_PIN_15)   /* P1.15 */
#define BSP_BOARD_MUX_SEL3_PORT                                         (GPIOC)
#define BSP_BOARD_MUX_SEL3_PIN                                          (GPIO_PIN_10)   /* P1.16 */
#define BSP_BOARD_MUX_SEL4_PORT                                         (GPIOC)
#define BSP_BOARD_MUX_SEL4_PIN                                          (GPIO_PIN_11)   /* P1.17 */
#define BSP_BOARD_MUX_SEL5_PORT                                         (GPIOB)
#define BSP_BOARD_MUX_SEL5_PIN                                          (GPIO_PIN_3)    /* P1.18 */
#define BSP_BOARD_MUX_SEL6_PORT                                         (GPIOB)
#define BSP_BOARD_MUX_SEL6_PIN                                          (GPIO_PIN_4)    /* P1.19 */
#define BSP_BOARD_MUX_SEL7_PORT                                         (GPIOB)
#define BSP_BOARD_MUX_SEL7_PIN                                          (GPIO_PIN_5)    /* P1.20 */
#define BSP_BOARD_MUX_SEL8_PORT                                         (GPIOB)
#define BSP_BOARD_MUX_SEL8_PIN                                          (GPIO_PIN_9)    /* P1.24 */

/************************************ADC*******************************************************/
/* ⚠️ 通道顺序 = 转换顺序 = 结果寄存器下标，三者必须一致。
 *    改这里必须同步改 bsp_adc.c 里的 Rank 赋值、缓冲区长度和 bsp_adc.h 里的下标宏。
 *
 * 分两个 ADC 不是设计选择，是引脚决定的：PB14 只有 ADC1_IN5，PB15 只有 ADC2_IN15。
 * 所以 8 节抽头被拆成两半，各自配一路 ADC —— 这也是当初选 2 颗 TMUX 而不是 1 颗的原因。 */

/* ---- ADC1 · 规则组（4 通道，连续转换 + DMA 循环搬运）---- */
#define BSP_BOARD_ADC_BUS_PORT                                          (GPIOB)
#define BSP_BOARD_ADC_BUS_PIN                                           (GPIO_PIN_0)    /* P2.9  */
#define BSP_BOARD_ADC_BUS_CHANNEL                                       (ADC_CHANNEL_15)
#define BSP_BOARD_ADC_IMON_PORT                                         (GPIOB)
#define BSP_BOARD_ADC_IMON_PIN                                          (GPIO_PIN_1)    /* P2.10 */
#define BSP_BOARD_ADC_IMON_CHANNEL                                      (ADC_CHANNEL_12)
#define BSP_BOARD_ADC_NTC_DUMP_PORT                                     (GPIOA)
#define BSP_BOARD_ADC_NTC_DUMP_PIN                                      (GPIO_PIN_3)    /* P2.15 */
#define BSP_BOARD_ADC_NTC_DUMP_CHANNEL                                  (ADC_CHANNEL_4)
#define BSP_BOARD_ADC_SC_A_PORT                                         (GPIOB)
#define BSP_BOARD_ADC_SC_A_PIN                                          (GPIO_PIN_14)   /* P1.7  */
#define BSP_BOARD_ADC_SC_A_CHANNEL                                      (ADC_CHANNEL_5)
#define BSP_BOARD_ADC1_MASK                                             \
    (BSP_BOARD_ADC_BUS_PIN | BSP_BOARD_ADC_IMON_PIN |                   \
     BSP_BOARD_ADC_NTC_DUMP_PIN | BSP_BOARD_ADC_SC_A_PIN)

/* ---- ADC2 · 规则组（3 通道）---- */
#define BSP_BOARD_ADC_NTC_MAIN_PORT                                     (GPIOB)
#define BSP_BOARD_ADC_NTC_MAIN_PIN                                      (GPIO_PIN_11)   /* P2.6  */
#define BSP_BOARD_ADC_NTC_MAIN_CHANNEL                                  (ADC_CHANNEL_14)
#define BSP_BOARD_ADC_NTC_SC_PORT                                       (GPIOC)
#define BSP_BOARD_ADC_NTC_SC_PIN                                        (GPIO_PIN_4)    /* P2.12 */
#define BSP_BOARD_ADC_NTC_SC_CHANNEL                                    (ADC_CHANNEL_5)
#define BSP_BOARD_ADC_SC_B_PORT                                         (GPIOB)
#define BSP_BOARD_ADC_SC_B_PIN                                          (GPIO_PIN_15)   /* P1.8  */
#define BSP_BOARD_ADC_SC_B_CHANNEL                                      (ADC_CHANNEL_15)
#define BSP_BOARD_ADC2_MASK                                             \
    (BSP_BOARD_ADC_NTC_MAIN_PIN | BSP_BOARD_ADC_NTC_SC_PIN |            \
     BSP_BOARD_ADC_SC_B_PIN)

/* 只接比较器、不进 ADC 的两路：PA7 = COMP2_INP，PA0 = COMP3_INP。
 * 它们与 PB1 是同一路 SO 的三个分支，本版不启用 COMP，但仍要把脚配成模拟输入，
 * 避免浮空脚在比较器输入端拾噪（将来一开 COMP 就是误触发的源头）。 */
#define BSP_BOARD_ADC_IMON_C2_PORT                                      (GPIOA)
#define BSP_BOARD_ADC_IMON_C2_PIN                                       (GPIO_PIN_7)    /* P2.11 */
#define BSP_BOARD_ADC_IMON_C3_PORT                                      (GPIOA)
#define BSP_BOARD_ADC_IMON_C3_PIN                                       (GPIO_PIN_0)    /* P2.20 */

/* ADC 满量程（mV）。VREF_CORE（P2.8）同时是 DRV8161 的 CSAREF 参考（板载 3.3 V）
 * 和 MCU 的 VREF+，两者同源，所以电流换算与电压换算共用同一个基准。
 * ⚠️ 上电第一件事：万用表量 P2.8 对 GND，确认 3.3 V 再谈任何标定。 */
#define BSP_BOARD_ADC_VREF_MV                                           (3300U)

/************************************FDCAN1****************************************************/
/* TCAN332DR 没有 STB/SHDN 引脚，所以本板没有「收发器待机」这根线，
 * 也就没有 foc_driver 里那种 CAN_STB 控制。想静默总线只能停 FDCAN 外设本身。
 * ⚠️ PA11/PA12 与核心板 Type-C 的 D−/D+ 硬连，接 CAN 时不能同时插 USB 数据线。 */
#define BSP_BOARD_CAN_RX_PORT                                           (GPIOA)
#define BSP_BOARD_CAN_RX_PIN                                            (GPIO_PIN_11)   /* P1.13 */
#define BSP_BOARD_CAN_TX_PORT                                           (GPIOA)
#define BSP_BOARD_CAN_TX_PIN                                            (GPIO_PIN_12)   /* P1.14 */
#define BSP_BOARD_CAN_AF                                                (GPIO_AF9_FDCAN1)

#ifdef _cplusplus
}
#endif

#endif /* __BSP_BOARD_H__ */
