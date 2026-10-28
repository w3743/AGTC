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
 * 本模块只负责这四个脚（数值见 bsp_board.h 的 DRV_EN / CAN_STB / DUMP_THERM_OK / INV_BKIN）：
 *
 *   DRV_ENABLE    PB11  GPIO 输出  DRV8353 第 33 脚 ENABLE，高=放行栅极驱动
 *                                  板上 R_DRV_EN_PD 10k 下拉 → 复位期间天然禁用
 *   CAN_STB       PB5   GPIO 输出  TCAN1042 STB：低=正常通信，高=待机
 *                                  板上 R_CAN_STB_PU 10k 上拉 → 复位期间天然待机
 *   DUMP_THERM_OK PB9   GPIO 输入  泄放电阻热许可（U4 比较器输出经 R_THERM_SER 100k 串入）
 *                                  板上无上/下拉，靠比较器推挽驱动
 *   BREAK_N       PB12  GPIO 输入  只读回电平做诊断；它的模式由 bsp_inverter 配成 AF（TIM1_BKIN）
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
* Function name :   bsp_gpio_get_dump_therm_ok
* Descriptions  :   Read the dump-resistor thermal permission signal
*                   Comes from comparator U4 through a 100k series resistor, so the pin is
*                   a plain input with no bias of its own.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE  : thermal condition allows dumping (comparator released)
*                   FALSE : dumping not allowed
*********************************************************************************************/
extern BOOL bsp_gpio_get_dump_therm_ok
(
    VOID
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
