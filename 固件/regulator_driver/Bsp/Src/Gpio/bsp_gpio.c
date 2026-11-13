/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_gpio.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 引脚实体定义已上移到 bsp_board.h（全工程唯一来源）。这里只保留本模块的短别名。 */
#define BSP_GPIO_PERMIT_PORT                                            (BSP_BOARD_PERMIT_PORT)
#define BSP_GPIO_PERMIT_PIN                                             (BSP_BOARD_PERMIT_PIN)

#define BSP_GPIO_FAULT_FB_PORT                                          (BSP_BOARD_FAULT_FB_PORT)
#define BSP_GPIO_FAULT_FB_PIN                                           (BSP_BOARD_FAULT_FB_PIN)


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   bsp_gpio_init
* 功能描述 :   配 PERMIT 为推挽输出并钉在“禁止”，FAULT_FB 为普通输入
*
*               安全态 = PERMIT 高（G07 拉低故障线，驱动停输出）。
*               为什么先写 ODR 再切输出模式：HAL_GPIO_WritePin 写的是 ODR，
*               引脚还是输入时也会记住这个值，一旦切成输出立刻呈现该电平；
*               反过来做会在切换瞬间出现一个不确定电平的窄脉冲。
*
*               FAULT_FB 不加内部上拉：线上已有 1k 上拉 + 10k 下拉，
*               内部 40k 并上去只会让分压变成 3.3×40/(40+10)≈2.6 V，还是高，
*               但没必要去动它 —— 保持高阻输入看真实电平。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_init
(
    VOID
)
{
    GPIO_InitTypeDef gpio_init = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* 先锁存“禁止”电平 */
    HAL_GPIO_WritePin(BSP_GPIO_PERMIT_PORT, BSP_GPIO_PERMIT_PIN,
                      BSP_BOARD_PERMIT_LEVEL_INHIBIT);

    /* PERMIT：推挽输出，低速。它驱动的是 G07 的 CMOS 输入，不需要边沿速度。 */
    gpio_init.Pin   = BSP_GPIO_PERMIT_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_GPIO_PERMIT_PORT, &gpio_init);

    /* FAULT_FB：普通输入，高阻 */
    gpio_init.Pin  = BSP_GPIO_FAULT_FB_PIN;
    gpio_init.Mode = GPIO_MODE_INPUT;
    gpio_init.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(BSP_GPIO_FAULT_FB_PORT, &gpio_init);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_permit_set
* 功能描述 :   控制功率许可（PA9 → G07 开漏 → DRV_FAULT_N）
*
*               这是本板**最硬**的一道软件闸：它直接作用在栅极驱动的故障线上，
*               拉低时 DRV8161 立刻停止输出，并且同时撤掉 LTC7001 对超容隔离管的许可，
*               与 TIM1 在做什么完全无关。
*
*               ⚠️ 与 FOC 板的区别：那块板有独立的 DRV_ENABLE（高有效）；
*                  本板没有独立 ENABLE，只有这条共用 nFAULT 的许可线，
*                  所以"清故障"的概念在这里不存在 —— 放行只是不再主动制造故障。
*
*               ⚠️ 放行不能绕过自检：DRV8161 真报故障时它自己会把线拉低（开漏对开漏），
*                  软件能做的只有"不主动拉低"。真正的故障清除必须靠硬件条件恢复。
*
* 输入参数 :   enable: TRUE = 放行，FALSE = 禁止
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_permit_set
(
    IN  BOOL    enable
)
{
    HAL_GPIO_WritePin(BSP_GPIO_PERMIT_PORT, BSP_GPIO_PERMIT_PIN,
                      (enable == TRUE) ? BSP_BOARD_PERMIT_LEVEL_ALLOW
                                       : BSP_BOARD_PERMIT_LEVEL_INHIBIT);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_read_fault
* 功能描述 :   读 DRV_FAULT_N 的真实电平
*
*               这条线上挂了：
*                 DRV8161 pin4   nFAULT（开漏）+ nDRVOFF（低有效禁用），共脚
*                 TLV6700 OUTB   超容总压窗口比较输出（开漏）
*                 U_GATE_OD Y    MCU 的许可缓冲器（开漏）
*                 COMP1/COMP2    两个比较器输出（开漏，本版未启用）
*               线与关系：任何一方拉低，整条线就是低。
*
*               ⚠️ 读到低**不能区分**是芯片报故障、过压比较动作，还是我们自己把许可关了。
*                  要区分必须结合 bsp_bridge 的 Break 锁存标志和 DRV8161 的
*                  VDS/OCP 状态（本板没有 SPI，读不到）—— 所以这个信号只能当"有事发生"，
*                  不能当故障分类依据。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 低电平，故障/禁止；FALSE = 高电平，无故障
*********************************************************************************************/
BOOL bsp_gpio_read_fault
(
    VOID
)
{
    return (HAL_GPIO_ReadPin(BSP_GPIO_FAULT_FB_PORT, BSP_GPIO_FAULT_FB_PIN)
            == BSP_BOARD_FAULT_FB_LEVEL_FAULT) ? TRUE : FALSE;
}

#ifdef _cplusplus
}
#endif
