#ifndef __BSP_HALL_H__
#define __BSP_HALL_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 霍尔位置反馈 —— 3 路开关量
 *
 * 引脚（依据 当前原理图连接.json + DS12589 Rev6 Table 12）：
 *   HALL_A  PA15  J_MCU_L.15  TIM2_CH1（AF1）/ 普通输入
 *   HALL_B  PB3   J_MCU_L.18  TIM2_CH2（AF1）/ 普通输入
 *   HALL_C  PB10  J_MCU_R.5   TIM2_CH3（AF1）/ 普通输入
 *
 * 三个脚正好是 TIM2 的 CH1/CH2/CH3 —— 这是有意的：TIM2 支持"霍尔接口模式"
 * （SMS=100，TI1/TI2/TI3 三路异或后计数），可以在不占 CPU 的情况下直接得到转速。
 * 本层先用最笨的"当三个普通输入读"，理由：
 *   ① 上电就能用，不需要先确认霍尔相位和 TIM2 配置对不对；
 *   ② 换相表本来就要求逐台电机实测，先读原始码值最直观；
 *   ③ 出问题时能立刻分清"是霍尔坏了"还是"是定时器配错了"。
 * 需要测速时再让 DL 层把这三个脚切到 AF1 + TIM2 霍尔模式，本层不用改。
 *
 * 电气侧（依据原理图）：
 *   霍尔信号经 R_Hx 串联进来，再用跳线 JP_Hx 二选一：
 *     JP 接 1-2（DIRECT）  → 只经 4.7k 上拉到 3V3，适合 3.3V 霍尔
 *     JP 接 2-3（5V_DIV）  → 经分压再进 MCU，适合 5V 开集霍尔
 *   所以**本层绝对不能开内部上拉**：DIRECT 时多一路上拉影响不大，但 5V_DIV 时
 *   内部上拉会把分压点顶高，轻则读数错乱，重则把 PA15/PB3 顶到 3.3V 以上。
 *   板上已有 C_Hx 做滤波，软件不需要再去抖。
 *
 * ⚠️ JTAG 说明：PA15/PB3 复位后默认是 JTDI/JTDO。本层把它们配成普通输入，
 *    JTAG 功能随之让位（SWD 的 PA13/PA14 不受影响，仿真照常）。
 *    如果哪天要用 JTAG 四线调试，这三个脚就得还回去，霍尔另找引脚。
 ********************************************************************************************/

/* 霍尔码位定义：code = A | B<<1 | C<<2 */
#define BSP_HALL_BIT_A                                                  (0x01U)
#define BSP_HALL_BIT_B                                                  (0x02U)
#define BSP_HALL_BIT_C                                                  (0x04U)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_hall_init
* Descriptions  :   Configure the three hall inputs as plain GPIO inputs with no bias,
*                   because the board already provides pull-up / divider.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_hall_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_hall_read
* Descriptions  :   Read the three hall inputs as one 3-bit code
*                   Bit0 = A, Bit1 = B, Bit2 = C. The code is NOT translated into a
*                   sector here: with 120-degree hall placement the valid sequence is
*                   1-3-2-6-4-5 (or its reverse), and which one applies depends on the
*                   motor wiring, so it has to be measured once per motor.
* Input         :
*                   none
* Output        :
*                   p_code: 0 ~ 7
* Return        :
*                   TRUE = code is one of the six valid states, FALSE = 000 or 111 seen
*                   (either no hall power, a broken wire, or the rotor is between sectors)
*********************************************************************************************/
extern BOOL bsp_hall_read
(
    OUT U8      *p_code
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_HALL_H__ */
