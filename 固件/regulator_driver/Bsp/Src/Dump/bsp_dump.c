/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_dump.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BSP_DUMP_MODULE_T_
{
    TIM_HandleTypeDef   handle;                                         /* TIM4 句柄 */
    F32                 duty;                                           /* 最近一次写入的占空比 */
}BSP_DUMP_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_DUMP_MODULE_T * _bsp_dump_get_module
(
    VOID
);


/************************************静态变量定义************************************************/
static BSP_DUMP_MODULE_T s_bsp_dump_module =
{
    {0},
    0.0F
};


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_dump_get_module
* 功能描述 :   获取泄放模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BSP_DUMP_MODULE_T * _bsp_dump_get_module
(
    VOID
)
{
    return &s_bsp_dump_module;
}

/********************************************************************************************
* 函数名称 :   bsp_dump_init
* 功能描述 :   把 TIM4_CH2 配成普通边沿对齐 PWM，输出到 PB7，占空比一开始就是 0
*
*               为什么是边沿对齐而不是中心对齐：
*                 泄放电阻上的电流由 R 和外置电阻决定，纹波不影响任何测量，
*                 边沿对齐的 ARR 算式更直观（ARR = clk/f − 1），也不用考虑双沿。
*
*               为什么脉冲数（Pulse）初值给 0：
*                 CCR = 0 在 PWM1 模式下等于 CH2 恒低 → UCC27511A IN+ 低
*                 → 栅极被 10 kΩ 下拉（R_DUMP_GATE_PD）钉住 → Q_DUMP 关断。
*                 这是唯一安全的上电状态：泄放电阻不发热。
*
*               ⚠️ 本函数不碰 PB7 的复用后面那些 GPIO 设置 —— 那是 bsp_adc /
*                  bsp_gpio 的活。这里只保证 PB7 配成 AF2 推挽。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_dump_init
(
    VOID
)
{
    GPIO_InitTypeDef    gpio_init = {0};
    TIM_OC_InitTypeDef  oc_init   = {0};
    BSP_DUMP_MODULE_T   *p_mod    = _bsp_dump_get_module();

    /* ── GPIO：PB7 → TIM4_CH2，AF2 推挽 ─────────────────────────────────────── */
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_TIM4_CLK_ENABLE();

    gpio_init.Pin       = BSP_BOARD_DUMP_PWM_PIN;
    gpio_init.Mode      = GPIO_MODE_AF_PP;
    gpio_init.Pull      = GPIO_NOPULL;                                  /* 板上已有 10k 下拉 */
    gpio_init.Speed     = GPIO_SPEED_FREQ_LOW;                          /* 20 kHz 方波，低速档足够 */
    gpio_init.Alternate = BSP_BOARD_DUMP_PWM_AF;
    HAL_GPIO_Init(BSP_BOARD_DUMP_PWM_PORT, &gpio_init);

    /* ── 时基：TIM4 在 APB1 = 170 MHz，边沿对齐 ─────────────────────────────── */
    p_mod->handle.Instance               = TIM4;
    p_mod->handle.Init.Prescaler         = 0;
    p_mod->handle.Init.CounterMode       = TIM_COUNTERMODE_UP;
    p_mod->handle.Init.Period            = BSP_DUMP_ARR;
    p_mod->handle.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
    p_mod->handle.Init.RepetitionCounter = 0;
    p_mod->handle.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
    if (HAL_TIM_PWM_Init(&p_mod->handle) != HAL_OK)
    {
        return;
    }

    /* ── 通道 2 ────────────────────────────────────────────────────────────── */
    oc_init.OCMode     = TIM_OCMODE_PWM1;
    oc_init.Pulse      = 0;                                             /* 0 = 完全不泄放 */
    oc_init.OCPolarity = TIM_OCPOLARITY_HIGH;                           /* 高电平 = 导通泄放 */
    oc_init.OCFastMode = TIM_OCFAST_DISABLE;
    if (HAL_TIM_PWM_ConfigChannel(&p_mod->handle, &oc_init, TIM_CHANNEL_2) != HAL_OK)
    {
        return;
    }

    /* 起计数器 + 使能 CH2 输出。
     * TIM4 是通用定时器，没有 MOE，所以这里必须真的 Start 通道，
     * 不像 TIM1 那样可以"起计数器、等 MOE"。 */
    if (HAL_TIM_PWM_Start(&p_mod->handle, TIM_CHANNEL_2) != HAL_OK)
    {
        return;
    }

    p_mod->duty = 0.0F;
}

/********************************************************************************************
* 函数名称 :   bsp_dump_set_duty
* 功能描述 :   设置泄放占空比
*
*               duty = 0.0 → CCR = 0      → CH2 恒低 → Q_DUMP 关断，电阻不发热
*               duty = 1.0 → CCR = ARR    → CH2 恒高 → Q_DUMP 常通，Pavg ≈ V_bus²/R
*
*               ⚠️ 长期 100% 会把外置电阻烧掉。本层只负责写寄存器，
*                  限幅、限时、按热模型降额全部由 BL 层决定。
*               ⚠️ 写 CCR 立即生效（没有开 CCR 预装载），
*                  所以 duty 的跳变会在当前 PWM 周期内就反映出来，
*                  不会出现"要等下一个周期"的额外延迟。
*
* 输入参数 :   duty: 0.0 ~ 1.0，越界钳位
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_dump_set_duty
(
    IN  F32     duty
)
{
    BSP_DUMP_MODULE_T *p_mod = _bsp_dump_get_module();
    F32               clamped;
    F32               compare;

    if (p_mod->handle.Instance != TIM4)
    {
        return;
    }

    clamped = duty;

    if (clamped < 0.0F)
    {
        clamped = 0.0F;
    }
    if (clamped > 1.0F)
    {
        clamped = 1.0F;
    }

    p_mod->duty = clamped;

    compare = (clamped * (F32)BSP_DUMP_ARR) + 0.5F;
    __HAL_TIM_SET_COMPARE(&p_mod->handle, TIM_CHANNEL_2, (U32)compare);
}

/********************************************************************************************
* 函数名称 :   bsp_dump_get_duty
* 功能描述 :   读回最近一次写入的泄放占空比
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   0.0 ~ 1.0
*********************************************************************************************/
F32 bsp_dump_get_duty
(
    VOID
)
{
    return _bsp_dump_get_module()->duty;
}

#ifdef _cplusplus
}
#endif
