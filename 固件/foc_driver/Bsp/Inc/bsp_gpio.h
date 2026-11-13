#ifndef __BSP_GPIO_H__
#define __BSP_GPIO_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 杂项 GPIO 模块
 *
 * ⚠️ 完整引脚总表已集中到 **bsp_board.h**，本文件不再重复维护一份 ——
 *    以前每个子模块各写一张表，改板时必然出现"改了 .c 忘了 .h"的对不上问题。
 *
 * 本模块只负责这三个脚（数值见 bsp_board.h 的 DRV_EN / CAN_STB / INV_BKIN）：
 *
 *   DRV_ENABLE    PB11  GPIO 输出  DRV8353 第 33 脚 ENABLE，高=放行栅极驱动
 *                                  板上 R_DRV_EN_PD 10k 下拉 → 复位期间天然禁用
 *   CAN_STB       PB5   GPIO 输出  TCAN1042 STB：低=正常通信，高=待机
 *                                  板上 R_CAN_STB_PU 10k 上拉 → 复位期间天然待机
 *   BREAK_N       PB12  GPIO 输入  只读回电平做诊断；它的模式由 bsp_inverter 配成 AF（TIM1_BKIN）
 *
 * ⚠️ 2026-10-30 起本地制动/泄放电路整套删除，原第四个脚 DUMP_THERM_OK（PB9、J_MCU_L.24）
 *    在网表里已标 NC，对应的读取接口一并删除。别再按旧文档找它。
 *
 * 其余外设的引脚归属见 bsp_board.h：
 *   六路 PWM + 刹车 → bsp_inverter    霍尔 → bsp_hall    SPI1 → bsp_spi
 *   采样通道        → bsp_adc         CAN  → bsp_can
 ********************************************************************************************/

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_gpio_init
* Descriptions  :   Init GPIO owned by this module and park the board in a safe state
*                   DRV8353 ENABLE = low (gate driver disabled)
*                   TCAN1042 STB   = high (transceiver standby)
*                   All three pins are additionally biased by resistors on the board,
*                   so the safe state also holds before this function runs.
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
* Function name :   bsp_gpio_drv_set_enable
* Descriptions  :   Drive DRV8353 ENABLE pin
*                   ENABLE is the "last gate": it forces all six gates off regardless of
*                   what the PWM pins are doing, so it is the preferred way to cut power.
* Input         :
*                   level: TRUE = enable gate driver, FALSE = disable (gates forced off)
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_gpio_drv_set_enable
(
    IN  BOOL    level
);

/********************************************************************************************
* Function name :   bsp_gpio_can_set_stb
* Descriptions  :   Drive TCAN1042 STB pin
* Input         :
*                   level: TRUE = standby, FALSE = normal communication
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_gpio_can_set_stb
(
    IN  BOOL    level
);

/********************************************************************************************
* Function name :   bsp_gpio_read_break_n
* Descriptions  :   Read the raw level of the shared BREAK_N net
*                   BREAK_N is a wired-OR of DRV8353 nFAULT (pin 28, open drain) and the
*                   U3 comparator outputs, pulled up by R_BREAK_PU 2.2k to 3V3.
*                   It is also wired to TIM1_BKIN, so the hardware cuts the PWM by itself;
*                   this function is only for diagnostics after the fact.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE  : net high, no fault
*                   FALSE : net low, fault asserted
*********************************************************************************************/
extern BOOL bsp_gpio_read_break_n
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_GPIO_H__ */
