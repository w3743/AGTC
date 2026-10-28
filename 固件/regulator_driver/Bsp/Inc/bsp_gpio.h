#ifndef __BSP_GPIO_H__
#define __BSP_GPIO_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 功率路径与诊断 GPIO —— 稳压板的三个开关量
 *
 * 三根线各管一件事，都在 P1/P2 排针上，实体引脚定义在 bsp_board.h：
 *   PMOS_CTRL  PB6   P1.21   → Q_PMOS_CTRL(MMBT5551 NPN) → Q_BUS_P 栅极
 *   PERMIT     PB5   P1.20   → 经 R_PB5_PERMIT 1k 接 **active-low** 的 PERMIT/DRV_FAULT_N 线
 *   TEST       PB10  P2.5    → 诊断用 TEST 网络
 *
 * ⚠️ 三根线的**有效电平不一样**，照抄会导致上电就通电或永远不通电，见下面每个函数的说明。
 *
 * ⚠️ 复位期间板子靠硬件停在安全态，本层只是把电平"钉死"：
 *     R_QP_BE    100k  Q_PMOS_CTRL 基极下拉 → NPN 截止 → R_QP_GS 47k 把 Q_BUS_P 栅源短路
 *                                            → P-MOS 默认关断
 *     R_DRV_FAULT_PD 10k  把 PERMIT/DRV_FAULT_N 拉低 → 驱动处于"故障"态 → 禁止功率级
 ********************************************************************************************/

/* PERMIT 线的有效电平。
 *
 * 依据：BOM 里 R_PB5_PERMIT 的描述原文是
 *   "1kohm PB5 series connection to **active-low** permit/DRV_FAULT_N line"，
 * 加上 R_DRV_FAULT_PD = 10 kΩ 下拉到 GND_SYS。
 *
 * 推一遍电平：PB5 高 → 3.3 V 经 1 kΩ 与 10 kΩ 分压 → 该线约 3.0 V，高于 VIH，故障解除；
 *            PB5 低 → 直接把该线拉到约 0 V，故障成立；
 *            PB5 高阻（复位态）→ 10 kΩ 下拉到 0 V，故障成立。
 * 所以：**高 = 放行，低 = 禁止**。三态里两种都落在"禁止"上，是安全的失效方向。
 *
 * ⚠️ 这条线同时挂着 DRV8161 自己的 nFAULT（开漏）。也就是说 PB5 拉高只是"不主动
 *    制造故障"，芯片真报故障时它照样把线拉低 —— 1 kΩ 串阻保证芯片能赢过 MCU 的
 *    3.3 mA 上拉电流。因此**不能**把 PB5 当成"屏蔽故障"的手段。 */
#define BSP_GPIO_PERMIT_LEVEL_ENABLE                                    (GPIO_PIN_SET)
#define BSP_GPIO_PERMIT_LEVEL_DISABLE                                   (GPIO_PIN_RESET)

/* PMOS_CTRL 的有效电平。
 *
 * 依据：BOM 里 R_QP_GS 的描述是 "47kohm Q_BUS_P **gate-to-source default-off** resistor"，
 * 驱动链是 PB6 → MMBT5551(NPN) 基极，集电极经 R_QP_COL 10 kΩ 接 Q_BUS_P 栅极。
 * 推一遍：PB6 高 → NPN 导通 → 集电极拉低 → 栅极低于源极 → P-MOS 导通；
 *         PB6 低 → NPN 截止 → 集电极悬空 → R_QP_GS 把栅源短路 → Vgs=0 → P-MOS 关断。
 * 所以：**高 = 导通，低 = 关断**。 */
#define BSP_GPIO_PMOS_LEVEL_ON                                          (GPIO_PIN_SET)
#define BSP_GPIO_PMOS_LEVEL_OFF                                         (GPIO_PIN_RESET)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_gpio_init
* Descriptions  :   Configure PMOS_CTRL / PERMIT / TEST as push-pull outputs and park them
*                   in the safe state (all LOW) before anything else runs.
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
* Descriptions  :   Drive the PERMIT / DRV_FAULT_N line.
*                   TRUE releases the driver, FALSE asserts a fault and inhibits the bridge.
*                   This is the "hardest" software gate on the board: with it asserted the
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
* Function name :   bsp_gpio_pmos_set
* Descriptions  :   Drive the BUS series P-MOS gate.
*                   TRUE closes the switch, FALSE opens it.
*                   WARNING: never open it while the bridge is still switching - the inductor
*                   current has nowhere to go and will punch through Q_BUS_P. Stop PWM first
*                   and let the current decay.
* Input         :
*                   on: TRUE = conduct, FALSE = open
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_gpio_pmos_set
(
    IN  BOOL    on
);

/********************************************************************************************
* Function name :   bsp_gpio_test_set
* Descriptions  :   Drive the shared TEST net.
*                   Must stay LOW in normal operation: T1-T7 each have a 47k resistor from the
*                   tap side to TEST, so the TEST level directly shifts those divider ratios.
*                   Only raise it for a short channel diagnosis, then restore LOW.
* Input         :
*                   high: TRUE = drive high, FALSE = drive low (normal)
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_gpio_test_set
(
    IN  BOOL    high
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_GPIO_H__ */
