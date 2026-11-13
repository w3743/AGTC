#ifndef __BSP_GPIO_H__
#define __BSP_GPIO_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 功率许可与故障回读 —— 稳压板的两个开关量
 *
 * 2026-10-30 改版后本层只剩两根线（实体引脚定义在 bsp_board.h）：
 *
 *   PERMIT    PA9   P1.11  → SN74LVC1G07 开漏缓冲器 A 端，Y 端并到 DRV_FAULT_N
 *   FAULT_FB  PA10  P1.12  → 同一条 DRV_FAULT_N 的真线回读（纯输入）
 *
 * ⚠️ 和改版前完全不同的两点：
 *
 *  1) **极性翻了。** 老版本是 PB5 推挽直接驱动故障线，高 = 放行。
 *     现在中间隔了一只非反相开漏缓冲器 G07：
 *         PA9 = 低 → G07 的 Y 高阻 → 故障线由 R_PB5_PERMIT 1k 上拉到 3V3_CORE（≈3.0 V）→ 放行
 *         PA9 = 高 → G07 把 Y 拉低     → 故障线被拉低 → DRV8161 停输出、LTC7001 撤隔离 → 禁止
 *     所以 **低 = 放行，高 = 禁止**。宏在 bsp_board.h（BSP_BOARD_PERMIT_LEVEL_*），
 *     函数体里不写魔法电平。
 *
 *  2) **现在能读回来了。** 老版本这根线是单向的，DRV8161 报故障软件看不见。
 *     现在 PA10 直连故障线，可以区分“故障还在”和“故障已消失”，
 *     配合 TIM1 Break 的锁存状态，能判断刚才那次关桥是谁拉低的。
 *
 * ⚠️ 为什么必须开漏（G07）而不是直连：这条线上同时挂着 DRV8161 的 nFAULT（开漏）、
 *    TLV6700 的 OUTB（开漏）和两个比较器输出（开漏）。谁都可以拉低，谁都不能推高。
 *    MCU 直接推挽驱动会在别人报故障时形成对拉 —— 3.3 V 灌进开漏器件的输出级。
 *
 * ⚠️ P-MOS 控制已不在 MCU 手上：改版后 Q_PMOS_CTRL 由 U_SC_OV 的 OUTA 驱动，
 *    纯硬件路径。旧的 bsp_gpio_pmos_set() 一并删除，别再找。
 *
 * ⚠️ 复位期间：PA9 是输入（高阻），外部 100k 下拉让 G07 的 A=0 → Y 高阻 → 故障线≈3.0 V；
 *    但此时 INH/INL 各有 47k 下拉、TIM1 的 MOE=0，功率级本身是关的。
 *    “上电安全”靠的是这三件事叠加，不是靠故障线电平。
 ********************************************************************************************/

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_gpio_init
* Descriptions  :   Configure PERMIT as a push-pull output parked in the INHIBIT state and
*                   FAULT_FB as a plain input. Nothing else is touched.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_gpio_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_gpio_permit_set
* Descriptions  :   Drive the PERMIT buffer (G07) input.
*                   TRUE  = release: PA9 low  -> G07 high-Z -> fault line pulled up -> driver live
*                   FALSE = inhibit: PA9 high -> G07 pulls the fault line low -> driver off,
*                                    LTC7001 INP low -> SC back-to-back isolation removed
*                   This is the "hardest" software gate on the board: with it inhibited the
*                   gate driver stops switching regardless of what TIM1 is doing.
* Input         :
*                   enable: TRUE = release, FALSE = inhibit
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_gpio_permit_set
(
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   bsp_gpio_read_fault
* Descriptions  :   Read the real level of DRV_FAULT_N.
*                   The line is pulled up by 1k to 3V3_CORE and down by 10k to GND, and can be
*                   pulled low by DRV8161 nFAULT, TLV6700 OUTB, the G07 buffer, or the two
*                   comparator outputs. Low therefore only means "someone is asserting a
*                   fault" - it cannot tell you who.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE  = line low, fault asserted (or permit inhibited by us)
*                   FALSE = line high, no fault
*********************************************************************************************/
extern BOOL bsp_gpio_read_fault
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_GPIO_H__ */
