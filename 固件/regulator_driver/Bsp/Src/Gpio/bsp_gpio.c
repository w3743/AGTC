/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_gpio.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 引脚实体定义已上移到 bsp_board.h（全工程唯一来源）。这里只保留本模块的短别名，
 * 好处是函数体不用改、读起来短，而改板时仍然只需要动 bsp_board.h 一个文件。 */
#define BSP_GPIO_PMOS_PORT                                              (BSP_BOARD_PMOS_CTRL_PORT)
#define BSP_GPIO_PMOS_PIN                                               (BSP_BOARD_PMOS_CTRL_PIN)

#define BSP_GPIO_PERMIT_PORT                                            (BSP_BOARD_PERMIT_PORT)
#define BSP_GPIO_PERMIT_PIN                                             (BSP_BOARD_PERMIT_PIN)

#define BSP_GPIO_TEST_PORT                                              (BSP_BOARD_TEST_PORT)
#define BSP_GPIO_TEST_PIN                                               (BSP_BOARD_TEST_PIN)


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   bsp_gpio_init
* 功能描述 :   初始化三个开关量输出，并把板子停在安全态
*
*               安全态（三根线全部为低）：
*                 PMOS_CTRL = 低 → Q_BUS_P 关断，BUS 与半桥断开；
*                 PERMIT    = 低 → DRV_FAULT_N 被拉低 → 驱动处于故障态，栅极全关；
*                 TEST      = 低 → T1–T7 的分压比保持设计值。
*
*               三根线低电平都是安全方向，这不是巧合：板上配套的下拉电阻
*               （R_QP_BE 100k、R_DRV_FAULT_PD 10k、R_T*_TEST 47k）就是按这个假设设计的，
*               所以 MCU 复位期间 GPIO 高阻时，板子天然落在同一个状态。
*
*               顺序：先写 ODR 再切输出模式。HAL_GPIO_WritePin 写的是 ODR，引脚还是输入时
*               也会记住这个值，一旦切成输出立刻呈现该电平。反过来做会在切换瞬间
*               出现一个不确定电平的窄脉冲。
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

    /* 三个脚都在 GPIOB 上：PB5 / PB6 / PB10 */
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* 先锁存安全电平 */
    HAL_GPIO_WritePin(BSP_GPIO_PMOS_PORT,  BSP_GPIO_PMOS_PIN,  BSP_GPIO_PMOS_LEVEL_OFF);
    HAL_GPIO_WritePin(BSP_GPIO_PERMIT_PORT, BSP_GPIO_PERMIT_PIN, BSP_GPIO_PERMIT_LEVEL_DISABLE);
    HAL_GPIO_WritePin(BSP_GPIO_TEST_PORT,  BSP_GPIO_TEST_PIN,  GPIO_PIN_RESET);

    /* PMOS_CTRL：推挽输出，低速。
     * 板上有 R_QP_BE 100k 下拉，这里不再并内部下拉 —— 内部下拉只有 ~40k，
     * 并上去会把基极分压改掉，而且没必要。 */
    gpio_init.Pin   = BSP_GPIO_PMOS_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_GPIO_PMOS_PORT, &gpio_init);

    /* PERMIT：推挽输出，低速。
     * ⚠️ 这个脚通过 1 kΩ 直接接到 active-low 的 DRV_FAULT_N 线上，
     *    所以它**必须配成推挽**：开漏的话拉不高（板上只有 10k 下拉，没有上拉），
     *    永远放行不了功率级。 */
    gpio_init.Pin   = BSP_GPIO_PERMIT_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_GPIO_PERMIT_PORT, &gpio_init);

    /* TEST：推挽输出，低速。翻转只在诊断时发生，不需要速度 */
    gpio_init.Pin   = BSP_GPIO_TEST_PIN;
    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_GPIO_TEST_PORT, &gpio_init);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_permit_set
* 功能描述 :   控制功率许可（PERMIT / DRV_FAULT_N 线）
*
*               这是本板**最硬**的一道软件闸：它直接作用在栅极驱动的故障线上，
*               拉低时 DRV8161 立刻停止输出，与 TIM1 在做什么完全无关。
*               急停优先用它，而不是把占空比写 0 —— 占空比 0 在 2-pin PWM 模式下
*               意味着 INH=0/INL=1（下管常通）或双低（Hi-Z），取决于互补极性，
*               不是可靠的"断电"手段。
*
*               ⚠️ 与 FOC 板的区别：那块板的 DRV_ENABLE 是高有效、独立一根线；
*                  本板没有独立的 ENABLE，只有这条共用 nFAULT 的许可线，
*                  所以"清故障"的概念在这里不存在 —— 拉高只是不再主动制造故障。
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
                      (enable == TRUE) ? BSP_GPIO_PERMIT_LEVEL_ENABLE : BSP_GPIO_PERMIT_LEVEL_DISABLE);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_pmos_set
* 功能描述 :   控制 BUS 串联 P-MOS（Q_BUS_P）
*
*               ⚠️ 使用约束（原理图准备文档原文）：
*                  「P-MOS 由 PB6 独立控制，不能和 PWM 关断同时硬切电感回路。」
*                  电感电流不能突变，PWM 还在跑的时候断开 P-MOS，L_MAIN 上的电流
*                  只能在 Q_BUS_P 的体二极管和寄生电容上找出路，Vds 会冲到几十上百伏。
*                  正确顺序：先把半桥占空比降到 0 并停 MOE，等电流衰减，再关 P-MOS。
*
*               ⚠️ 使用前必须确认核心板 SB3（PB6↔PA9）、SB6（PB6↔CC1）实物开路，
*                  否则本函数会顺带驱动 PA9 和 Type-C 的 CC1。
*
* 输入参数 :   on: TRUE = 导通，FALSE = 关断
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_pmos_set
(
    IN  BOOL    on
)
{
    HAL_GPIO_WritePin(BSP_GPIO_PMOS_PORT, BSP_GPIO_PMOS_PIN,
                      (on == TRUE) ? BSP_GPIO_PMOS_LEVEL_ON : BSP_GPIO_PMOS_LEVEL_OFF);
}

/********************************************************************************************
* 函数名称 :   bsp_gpio_test_set
* 功能描述 :   驱动共用的 TEST 网络
*
*               为什么平时必须是低：T1–T7 每一路都有 R_Tj_TEST = 47 kΩ 从抽头侧接到 TEST，
*               名义式是 Vtap = K×Vadc − c×Vtest（K=9.2468、c=1.4468）。
*               TEST 悬空或为高，这个式子里的 Vtest 就不是 0，所有抽头读数都会偏。
*
*               诊断用法（原理图准备文档给出）：
*                 1) 平时保持 LOW，记一组基准读数；
*                 2) 短时拉 HIGH，等 RC 稳定，读 TEST_FB 确认 TEST 真的是高；
*                 3) 看各 tap 读数是否按 c×ΔVtest 变化 —— 不变的那一路就是断线；
*                 4) 恢复 LOW，复核 IDR 与模拟回读。
*               ⚠️ 回读不能代替开短路诊断，也不能覆盖电阻漂移。
*
* 输入参数 :   high: TRUE = 拉高（仅诊断），FALSE = 拉低（正常运行）
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_gpio_test_set
(
    IN  BOOL    high
)
{
    HAL_GPIO_WritePin(BSP_GPIO_TEST_PORT, BSP_GPIO_TEST_PIN,
                      (high == TRUE) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

#ifdef _cplusplus
}
#endif
