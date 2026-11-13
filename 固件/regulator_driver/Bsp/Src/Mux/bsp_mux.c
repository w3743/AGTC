/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_mux.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BSP_MUX_PIN_T_
{
    GPIO_TypeDef   *port;
    U16             pin;
}BSP_MUX_PIN_T;

typedef struct _BSP_MUX_MODULE_T_
{
    U8              selected;                                           /* 0 = 全断 */
}BSP_MUX_MODULE_T;


/************************************静态变量定义************************************************/
/* 顺序就是通道号：下标 0 未用，1..8 对应 T1..T8。
 * ⚠️ 这张表必须和 bsp_board.h 的 MUX_SEL1..8 顺序一致，改板时两处一起改。 */
static const BSP_MUX_PIN_T s_bsp_mux_pin[BSP_MUX_CH_NUM + 1U] =
{
    { NULL,            0U                    },                         /* 0：占位，全断 */
    { BSP_BOARD_MUX_SEL1_PORT, BSP_BOARD_MUX_SEL1_PIN },                /* 1 → T1 */
    { BSP_BOARD_MUX_SEL2_PORT, BSP_BOARD_MUX_SEL2_PIN },                /* 2 → T2 */
    { BSP_BOARD_MUX_SEL3_PORT, BSP_BOARD_MUX_SEL3_PIN },                /* 3 → T3 */
    { BSP_BOARD_MUX_SEL4_PORT, BSP_BOARD_MUX_SEL4_PIN },                /* 4 → T4 */
    { BSP_BOARD_MUX_SEL5_PORT, BSP_BOARD_MUX_SEL5_PIN },                /* 5 → T5 */
    { BSP_BOARD_MUX_SEL6_PORT, BSP_BOARD_MUX_SEL6_PIN },                /* 6 → T6 */
    { BSP_BOARD_MUX_SEL7_PORT, BSP_BOARD_MUX_SEL7_PIN },                /* 7 → T7 */
    { BSP_BOARD_MUX_SEL8_PORT, BSP_BOARD_MUX_SEL8_PIN },                /* 8 → T8 */
};

static BSP_MUX_MODULE_T s_bsp_mux_module;


/************************************静态函数声明************************************************/
static BSP_MUX_MODULE_T * _bsp_mux_get_module
(
    VOID
);

static VOID _bsp_mux_write
(
    IN  U8      channel,
    IN  GPIO_PinState level
);


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_mux_get_module
* 功能描述 :   取模块实例
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BSP_MUX_MODULE_T * _bsp_mux_get_module
(
    VOID
)
{
    return &s_bsp_mux_module;
}

/********************************************************************************************
* 函数名称 :   _bsp_mux_write
* 功能描述 :   给某一路 SEL 写电平（channel 必须 1..8）
* 输入参数 :   channel: 1..8；level: SET / RESET
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bsp_mux_write
(
    IN  U8              channel,
    IN  GPIO_PinState   level
)
{
    if ((channel < 1U) || (channel > BSP_MUX_CH_NUM))
    {
        return;
    }

    HAL_GPIO_WritePin(s_bsp_mux_pin[channel].port,
                      s_bsp_mux_pin[channel].pin,
                      level);
}

/********************************************************************************************
* 函数名称 :   bsp_mux_init
* 功能描述 :   8 个 SEL 配成推挽输出，初始全低（全断）
*
*               为什么推挽：SEL 是 TMUX1511 的数字输入，高阻时电平不确定，
*               开关可能半开 —— 那比全开还糟（Ron 变大、读数偏、还发热）。
*               为什么低速：这些脚一辈子只翻几次，不需要边沿速度。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_mux_init
(
    VOID
)
{
    GPIO_InitTypeDef    gpio_init = {0};
    BSP_MUX_MODULE_T    *p_mod    = _bsp_mux_get_module();
    U8                  i;

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* 先把 ODR 清干净，再切成输出，避免切换瞬间出现窄脉冲把两路同时打开 */
    for (i = 1U; i <= BSP_MUX_CH_NUM; i++)
    {
        HAL_GPIO_WritePin(s_bsp_mux_pin[i].port, s_bsp_mux_pin[i].pin, GPIO_PIN_RESET);
    }

    gpio_init.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init.Pull  = GPIO_NOPULL;
    gpio_init.Speed = GPIO_SPEED_FREQ_LOW;

    for (i = 1U; i <= BSP_MUX_CH_NUM; i++)
    {
        gpio_init.Pin = s_bsp_mux_pin[i].pin;
        HAL_GPIO_Init(s_bsp_mux_pin[i].port, &gpio_init);
    }

    p_mod->selected = BSP_MUX_CH_NONE;
}

/********************************************************************************************
* 函数名称 :   bsp_mux_select
* 功能描述 :   选通一路抽头（0 = 全断）
*
*               顺序是“先关旧的、再开新的”，不是反过来 —— 见 bsp_mux.h 的 one-hot 说明。
*               中间会有几百纳秒全断，此时 ADC 节点靠 10k 下拉维持在 0 V，
*               对下一路没有影响（每一路都是独立分压后再进开关）。
*
*               ⚠️ 选通后模拟侧需要建立时间：抽头分压 68k/5k 对 1 nF 滤波电容，
*                  τ ≈ (68k ∥ 5k) × 1 nF ≈ 4.7 µs，取 5τ ≈ 24 µs 才稳。
*                 本函数不管建立时间，调用者必须自己等（见 dl_cell.c 的扫描循环）。
*
* 输入参数 :   channel: 0 = 全断，1..8 = T1..T8
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_mux_select
(
    IN  U8      channel
)
{
    BSP_MUX_MODULE_T *p_mod = _bsp_mux_get_module();

    if ((channel > BSP_MUX_CH_NUM) || (channel == p_mod->selected))
    {
        return;
    }

    /* 先全断：只有 8 路，一次循环的代价可以忽略，换来的是“绝不可能两路同开” */
    if (p_mod->selected != BSP_MUX_CH_NONE)
    {
        _bsp_mux_write(p_mod->selected, GPIO_PIN_RESET);
        p_mod->selected = BSP_MUX_CH_NONE;
    }

    if (channel != BSP_MUX_CH_NONE)
    {
        _bsp_mux_write(channel, GPIO_PIN_SET);
        p_mod->selected = channel;
    }
}

/********************************************************************************************
* 函数名称 :   bsp_mux_get_selected
* 功能描述 :   当前选通的通道号
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   0 = 全断，1..8
*********************************************************************************************/
U8 bsp_mux_get_selected
(
    VOID
)
{
    return _bsp_mux_get_module()->selected;
}

#ifdef _cplusplus
}
#endif
