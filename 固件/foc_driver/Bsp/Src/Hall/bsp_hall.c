/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_hall.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 引脚实体定义已上移到 bsp_board.h（全工程唯一来源），这里只是短别名。 */
#define BSP_HALL_A_PORT                                                 (BSP_BOARD_HALL_A_PORT)
#define BSP_HALL_A_PIN                                                  (BSP_BOARD_HALL_A_PIN)

#define BSP_HALL_B_PORT                                                 (BSP_BOARD_HALL_B_PORT)
#define BSP_HALL_B_PIN                                                  (BSP_BOARD_HALL_B_PIN)

#define BSP_HALL_C_PORT                                                 (BSP_BOARD_HALL_C_PORT)
#define BSP_HALL_C_PIN                                                  (BSP_BOARD_HALL_C_PIN)


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   bsp_hall_init
* 功能描述 :   把 PA15 / PB3 / PB10 配成普通输入，读霍尔
*
*               为什么不加内部上拉（这条最容易配错）：
*                 板上的霍尔前端有两个分支，由跳线 JP_Hx 选：
*                   DIRECT：信号经 4.7k 上拉到 3V3，MCU 直连；
*                   5V_DIV：信号先经串联电阻 + 对地下拉分压，把 5V 霍尔降到 3.3V 以内。
*                 如果这里再开内部上拉（约 40k），5V_DIV 分支的分压点会被抬起来，
*                 分压比被破坏，重则超过 3.3V。所以 Pull 一律 NOPULL，
*                 偏置完全交给板上的跳线电路。
*
*               速度用 GPIO_SPEED_FREQ_LOW：霍尔是机械换相信号，最快也就几 kHz，
*               低速档反而减少引脚上的开关噪声。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_hall_init
(
    VOID
)
{
    GPIO_InitTypeDef gpio_init = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* HALL_A：PA15。复位后它是 JTDI，配成输入即自动让出 JTAG。 */
    gpio_init.Pin   = BSP_HALL_A_PIN;
    gpio_init.Mode  = GPIO_MODE_INPUT;
    gpio_init.Pull  = GPIO_NOPULL;          /* 板上 4.7k 上拉 / 分压，见函数头说明 */
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_HALL_A_PORT, &gpio_init);

    /* HALL_B：PB3（复位后是 JTDO），HALL_C：PB10，一起配 */
    gpio_init.Pin   = BSP_HALL_B_PIN | BSP_HALL_C_PIN;
    gpio_init.Mode  = GPIO_MODE_INPUT;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BSP_HALL_B_PORT, &gpio_init);
}

/********************************************************************************************
* 函数名称 :   bsp_hall_read
* 功能描述 :   读三路霍尔，拼成 3 位码
*
*               返回 FALSE 的两种情形：
*                 码值 000 —— 霍尔没供电（J_HALL 没插 / 5V_BUCK 没起来），或者转子正好
*                              停在两个扇区之间；
*                 码值 111 —— 同上，或者某一路对电源短路。
*               正常旋转时只可能出现 001/010/011/100/101/110 这六个值里的三个互补对，
*               具体是哪三个取决于霍尔安装角度和相序，**必须实测**，本层不做假设。
*
*               典型用法（换相表实测）：
*                 手动把转子一格一格转过去，每停一次调用本函数，记下 6 个码值和
*                 对应的相序，做成表放进 DL 层。
*
* 输入参数 :   无
* 输出参数 :   p_code: 0 ~ 7，bit0=A，bit1=B，bit2=C
* 返回值   :   TRUE = 六种有效状态之一；FALSE = 000 或 111
*********************************************************************************************/
BOOL bsp_hall_read
(
    OUT U8      *p_code
)
{
    U8 code = 0U;

    if (p_code == NULL)
    {
        return FALSE;
    }

    if (HAL_GPIO_ReadPin(BSP_HALL_A_PORT, BSP_HALL_A_PIN) == GPIO_PIN_SET)
    {
        code |= BSP_HALL_BIT_A;
    }
    if (HAL_GPIO_ReadPin(BSP_HALL_B_PORT, BSP_HALL_B_PIN) == GPIO_PIN_SET)
    {
        code |= BSP_HALL_BIT_B;
    }
    if (HAL_GPIO_ReadPin(BSP_HALL_C_PORT, BSP_HALL_C_PIN) == GPIO_PIN_SET)
    {
        code |= BSP_HALL_BIT_C;
    }

    *p_code = code;

    /* 000 和 111 这两个全同态在正常霍尔里不会出现 */
    return ((code == 0U) || (code == 7U)) ? FALSE : TRUE;
}

#ifdef _cplusplus
}
#endif
