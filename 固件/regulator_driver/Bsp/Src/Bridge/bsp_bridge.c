/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_bridge.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 定时器时钟。SystemClock_Config() 把 APB2 分频设为 1，所以 TIM1 内核时钟 = 170 MHz。
 * ⚠️ 改时钟树必须同步复核这里的值，否则 bsp_board.h 里按 170 MHz 算出来的 ARR
 *    和实际 PWM 频率就对不上了。 */
#define BSP_BRIDGE_TIM_CLK_HZ                                           (170000000U)

/* tDTS = 1 / 170 MHz = 5.882 ns */
#define BSP_BRIDGE_TDTS_NS                                              (5.882F)


/************************************数据类型定义************************************************/
typedef struct _BSP_BRIDGE_MODULE_T_
{
    TIM_HandleTypeDef   handle;                                         /* TIM1 句柄 */
    F32                 duty;                                           /* 最近一次写入的带符号占空比 */
    BOOL                started;                                        /* MOE 是否已置位 */
}BSP_BRIDGE_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_BRIDGE_MODULE_T * _bsp_bridge_get_module
(
    VOID
);

static U32 _bsp_bridge_deadtime_to_dtg
(
    IN  U32     deadtime_ns
);

static U32 _bsp_bridge_duty_to_compare
(
    IN  F32     duty
);


/************************************静态变量定义************************************************/
static BSP_BRIDGE_MODULE_T s_bsp_bridge_module =
{
    {0},
    0.0F,
    FALSE
};


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_bridge_get_module
* 功能描述 :   获取半桥模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BSP_BRIDGE_MODULE_T * _bsp_bridge_get_module
(
    VOID
)
{
    return &s_bsp_bridge_module;
}

/********************************************************************************************
* 函数名称 :   _bsp_bridge_deadtime_to_dtg
* 功能描述 :   把纳秒换算成 BDTR.DTG 的位域编码（四段折线，全量程）
*
*               RM0440 "Dead-time generator"，tDTS = 1/170MHz = 5.882 ns：
*                 0xx  → DT = DTG[7:0]        × tDTS      0 ~ 747 ns
*                 10x  → DT = (64 + DTG[5:0]) × 2 × tDTS  753 ~ 1494 ns
*                 110  → DT = (32 + DTG[4:0]) × 8 × tDTS  1506 ~ 2965 ns
*                 111  → DT = (32 + DTG[4:0]) × 16 × tDTS 3012 ~ 5930 ns
*               四段之间有不连续台阶，必须按段判断，不能线性换算。
*               超出 5930 ns 饱和到最大值；0 返回 0（不加死区）。
*
*               ⚠️ 默认配置是 0（DTG = 0x00），即"不加死区"，由 DRV8161 自己插入
*                  （典型 370 ns）。只有实测确认需要 MCU 侧兜底时才改
*                  BSP_BOARD_BRIDGE_DEADTIME_NS。
*
* 输入参数 :   deadtime_ns: 需要的死区，单位 ns
* 输出参数 :   无
* 返回值   :   DTG 位域值
*********************************************************************************************/
static U32 _bsp_bridge_deadtime_to_dtg
(
    IN  U32     deadtime_ns
)
{
    U32 steps;

    if (deadtime_ns == 0U)
    {
        return 0U;
    }

    /* 换算成 tDTS 个数，向上取整（死区宁大勿小） */
    steps = (U32)(((F32)deadtime_ns / BSP_BRIDGE_TDTS_NS) + 0.999F);

    if (steps == 0U)
    {
        steps = 1U;
    }

    if (steps <= 127U)                                                  /* 第 1 段：1:1 */
    {
        return steps;
    }

    steps = (steps + 1U) / 2U;                                          /* 第 2 段：1:2 */
    if (steps <= 127U)
    {
        if (steps < 64U)
        {
            steps = 64U;                                                /* 该段下限，防止掉进第 1 段的编码空间 */
        }
        return 0x80U | (steps - 64U);                                   /* DTG[7:5] = 10x */
    }

    steps = (steps + 3U) / 4U;                                          /* 第 3 段：1:8（叠加在 /2 之上） */
    if (steps <= 63U)
    {
        if (steps < 32U)
        {
            steps = 32U;
        }
        return 0xC0U | (steps - 32U);                                   /* DTG[7:5] = 110 */
    }

    steps = (steps + 1U) / 2U;                                          /* 第 4 段：1:16 */
    if (steps > 63U)
    {
        steps = 63U;                                                    /* 饱和到最大死区 ≈5.93 µs */
    }
    if (steps < 32U)
    {
        steps = 32U;
    }
    return 0xE0U | (steps - 32U);                                       /* DTG[7:5] = 111 */
}

/********************************************************************************************
* 函数名称 :   _bsp_bridge_duty_to_compare
* 功能描述 :   带符号占空比 → CCR 比较值
*
*               映射：high_side_duty = (duty + 1) / 2
*                 duty = -1 → 0.00 → CCR = 0     上管恒关、下管恒通
*                 duty =  0 → 0.50 → CCR = ARR/2 上下各半（中点）
*                 duty = +1 → 1.00 → CCR = ARR   上管恒通、下管恒关
*
*               为什么用带符号而不是 0~1：这块板是双向的，中点 0.5 才是"零请求"的参照。
*               用带符号以后，上层做 PI 调节时输出直接就是"往哪边搬多少能量"，
*               不需要自己记住 0.5 这个偏移量。
*
*               ⚠️ 但中点 50% **不等于零电流**。开关节点平均电压是 D × V_BUS，
*                  电感电流只有在 D = V_SC / V_BUS 时才停止爬升。9S 母线 22.5–37.8 V、
*                  8S 超容 0–20.4 V，平衡点随 SOC 漂移。50% 只是"最温和的起点"，
*                  真正的平衡占空比由 BL 层按两端电压算。这里不替上层做这个假设。
*
*               越界钳位到 ±BSP_BRIDGE_DUTY_LIMIT，防止上层算飞了直接把管子钉死。
*
* 输入参数 :   duty: -1.0 ~ +1.0
* 输出参数 :   无
* 返回值   :   0 ~ ARR
*********************************************************************************************/
static U32 _bsp_bridge_duty_to_compare
(
    IN  F32     duty
)
{
    F32 clamped;
    F32 ratio;
    F32 compare;

    clamped = duty;

    if (clamped > BSP_BRIDGE_DUTY_LIMIT)
    {
        clamped = BSP_BRIDGE_DUTY_LIMIT;
    }
    if (clamped < -BSP_BRIDGE_DUTY_LIMIT)
    {
        clamped = -BSP_BRIDGE_DUTY_LIMIT;
    }

    ratio   = (clamped + 1.0F) * 0.5F;
    compare = (ratio * (F32)BSP_BRIDGE_ARR) + 0.5F;                      /* 四舍五入 */

    /* F32 比较在这里是安全的：compare 已被 ratio 夹在 [0, ARR] 内，
     * 加 0.5 后最大 ARR+0.5，转 U32 不会溢出，也不会出现负值。 */
    return (U32)compare;
}

/********************************************************************************************
* 函数名称 :   bsp_bridge_init
* 功能描述 :   配置 TIM1 通道 1 + 互补通道 1N，中心对齐，死区按 bsp_board.h
*
*               几个关键取舍：
*
*               1) 中心对齐（CENTERALIGNED1）而不是边沿对齐：
*                  上下管开通点关于周期中点对称，电流纹波更小；
*                  而且计数器顶点是天然的"下管全通"时刻，将来要做同步采样直接等 TRGO 就行
*                  （现在没开注入组，因为本板的采样全是慢信号，见 bsp_adc.c）。
*
*               2) PWM1 模式 + 互补不反相：
*                  CH1 高电平期间为"有效"，CH1N = NOT(CH1)，正好对上 DRV8161 的 INH/INL。
*                  如果哪天发现管子动作反了，先怀疑板上的 INH/INL 接反，
*                  改 OCNPolarity 只会让波形反相，不会修正真正的接线错误。
*
*               3) IdleState 全设 RESET：
*                  MOE=0（刹车、或本层主动 stop）时两路都输出低，
*                  驱动器看到 INH=0/INL=0 → 两栅极关断、SH 进 Hi-Z。这是安全态。
*                  若设成 SET，一停 MOE 反而会让一根管子常通 —— 千万别改。
*
*               4) 刹车的处理：
*                  BreakState = DISABLE，理由见 bsp_bridge.h 第 4 条。
*                  这个定时器的 BKIN 引脚在 PB12 / PA6 上，而这两个脚在本板是
*                  NTC_DUMP 和 AD_T7 的模拟输入，打开刹车等于给 NTC 分压装了个触发源。
*
*               5) 只起计数器、不起 MOE：
*                  这样其他依赖 TIM1 的东西（将来要做同步采样的话）从 init 之后就有节拍，
*                  而栅极一路都是关着的，直到上层显式 bsp_bridge_start()。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_bridge_init
(
    VOID
)
{
    GPIO_InitTypeDef                gpio_init = {0};
    TIM_OC_InitTypeDef              oc_init   = {0};
    TIM_BreakDeadTimeConfigTypeDef  bdt_init  = {0};
    BSP_BRIDGE_MODULE_T             *p_mod    = _bsp_bridge_get_module();

    /* ── GPIO：PA8 → TIM1_CH1，PB13 → TIM1_CH1N，都是 AF6 推挽 ─────────────── */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_TIM1_CLK_ENABLE();

    gpio_init.Mode      = GPIO_MODE_AF_PP;
    gpio_init.Pull      = GPIO_NOPULL;                                  /* 板上每路 47k 下拉 */
    gpio_init.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;                    /* 栅极信号边沿要快 */
    gpio_init.Alternate = BSP_BOARD_BRIDGE_AF;

    gpio_init.Pin = BSP_BOARD_BRIDGE_H_PIN;
    HAL_GPIO_Init(BSP_BOARD_BRIDGE_H_PORT, &gpio_init);

    gpio_init.Pin = BSP_BOARD_BRIDGE_L_PIN;
    HAL_GPIO_Init(BSP_BOARD_BRIDGE_L_PORT, &gpio_init);

    /* ── 时基 ──────────────────────────────────────────────────────────────── */
    p_mod->handle.Instance               = TIM1;
    p_mod->handle.Init.Prescaler         = 0;
    p_mod->handle.Init.CounterMode       = TIM_COUNTERMODE_CENTERALIGNED1;
    p_mod->handle.Init.Period            = BSP_BRIDGE_ARR;
    p_mod->handle.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
    p_mod->handle.Init.RepetitionCounter = 0;
    p_mod->handle.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
    if (HAL_TIM_PWM_Init(&p_mod->handle) != HAL_OK)
    {
        return;
    }

    /* ── 通道 1 + 互补通道 1N ──────────────────────────────────────────────── */
    oc_init.OCMode       = TIM_OCMODE_PWM1;
    oc_init.Pulse        = 0;                                           /* 初始 CCR=0，随时可改 */
    oc_init.OCPolarity   = TIM_OCPOLARITY_HIGH;                         /* CH1  不反相 */
    oc_init.OCNPolarity  = TIM_OCNPOLARITY_HIGH;                        /* CH1N 不反相 → 就是"非 CH1" */
    oc_init.OCFastMode   = TIM_OCFAST_DISABLE;
    oc_init.OCIdleState  = TIM_OCIDLESTATE_RESET;                       /* MOE=0 时 CH1  = 低 */
    oc_init.OCNIdleState = TIM_OCNIDLESTATE_RESET;                      /* MOE=0 时 CH1N = 低 */

    if (HAL_TIM_PWM_ConfigChannel(&p_mod->handle, &oc_init, TIM_CHANNEL_1) != HAL_OK)
    {
        return;
    }

    /* ── 死区 + 刹车 ───────────────────────────────────────────────────────── */
    bdt_init.OffStateRunMode  = TIM_OSSR_ENABLE;                        /* MOE=0 时引脚被驱动到 OISx，而不是高阻 */
    bdt_init.OffStateIDLEMode = TIM_OSSI_ENABLE;
    bdt_init.LockLevel        = TIM_LOCKLEVEL_OFF;                      /* 允许后续改 DTG/OSSI，上板调试方便 */
    bdt_init.DeadTime         = _bsp_bridge_deadtime_to_dtg(BSP_BOARD_BRIDGE_DEADTIME_NS);
    bdt_init.BreakState       = TIM_BREAK_DISABLE;                      /* ⚠️ 见 bsp_bridge.h 第 4 条，别打开 */
    bdt_init.BreakPolarity    = TIM_BREAKPOLARITY_LOW;
    bdt_init.BreakFilter      = 0;
    bdt_init.BreakAFMode      = TIM_BREAK_AFMODE_INPUT;
    bdt_init.Break2State      = TIM_BREAK2_DISABLE;
    bdt_init.Break2Polarity   = TIM_BREAK2POLARITY_LOW;
    bdt_init.Break2Filter     = 0;
    bdt_init.Break2AFMode     = TIM_BREAK_AFMODE_INPUT;
    /* 自动输出关掉：刹车（将来真开了以后）不会自己恢复，必须软件显式重启。
     * 这是防止故障反复自启动，和 foc_driver 的处理一致。 */
    bdt_init.AutomaticOutput  = TIM_AUTOMATICOUTPUT_DISABLE;

    if (HAL_TIMEx_ConfigBreakDeadTime(&p_mod->handle, &bdt_init) != HAL_OK)
    {
        return;
    }

    /* 先把占空比摆到中点，再让计数器起跑。
     * 这样即使有人在 init 之后、start 之前手滑置了 MOE，也是从中点这个最温和的工作点
     * 开始，不会一上来就满占空。注意中点不等于零电流，理由见 _bsp_bridge_duty_to_compare()。 */
    p_mod->duty = 0.0F;
    __HAL_TIM_SET_COMPARE(&p_mod->handle, TIM_CHANNEL_1, _bsp_bridge_duty_to_compare(0.0F));

    /* 只起计数器、不起 MOE */
    __HAL_TIM_ENABLE(&p_mod->handle);

    p_mod->started = FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_bridge_start
* 功能描述 :   使能两路输出并置位 MOE，让 CCR 真正作用到引脚
*
*               ⚠️ MOE 一置位，当前 CCR 立刻生效。所以调用顺序必须是：
*                    bsp_bridge_set_duty(想要的占空比);
*                    bsp_bridge_start();
*                 反过来的话，会先输出上一次残留的占空比（上位机调过之后尤其容易出事）。
*
*               这里不用 HAL_TIM_PWM_Start：它一次只能启一个通道，还会顺带改 CCER；
*               直接置 CC1E | CC1NE | MOE 更清楚，也不会重复开关。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_bridge_start
(
    VOID
)
{
    BSP_BRIDGE_MODULE_T *p_mod = _bsp_bridge_get_module();

    if (p_mod->handle.Instance != TIM1)
    {
        return;
    }

    /* 主输出 + 互补输出同时使能。CH1N 对应 CC1NE 位。 */
    SET_BIT(p_mod->handle.Instance->CCER, TIM_CCER_CC1E | TIM_CCER_CC1NE);
    __HAL_TIM_MOE_ENABLE(&p_mod->handle);

    p_mod->started = TRUE;
}

/********************************************************************************************
* 函数名称 :   bsp_bridge_stop
* 功能描述 :   清 MOE，两路输出回到 OISx 定义的低电平
*
*               驱动器看到 INH=0 / INL=0 → 两栅极全关、SH 进 Hi-Z，电感电流走体二极管续流。
*               计数器不关：省得将来加了同步采样又要重新对齐相位，
*               而且关计数器会让 CCER 之外的状态也一起复位，没必要。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_bridge_stop
(
    VOID
)
{
    BSP_BRIDGE_MODULE_T *p_mod = _bsp_bridge_get_module();

    if (p_mod->handle.Instance != TIM1)
    {
        return;
    }

    __HAL_TIM_MOE_DISABLE(&p_mod->handle);
    p_mod->started = FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_bridge_set_duty
* 功能描述 :   写带符号占空比
*
*               duty = +1.0 → CCR = ARR → CH1 恒高 / CH1N 恒低 → 上管常通
*               duty =  0.0 → CCR = ARR/2 → 常规 50% 互补
*               duty = -1.0 → CCR = 0   → CH1 恒低 / CH1N 恒高 → 下管常通
*
*               ⚠️ 本函数**不检查** MOE 状态：MOE=0 时写进去只是改寄存器，不会输出。
*                  这是故意的 —— 上层可以先把占空比摆好，再调 bsp_bridge_start()。
*               ⚠️ 也**不检查** PERMIT：功率许可由 bsp.c 的总入口和 BL 层负责，
*                  BSP 这一层只做"把值写到硬件"，多层都做同一个判断反而容易互相打架。
*
* 输入参数 :   duty: -1.0 ~ +1.0，越界钳位
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_bridge_set_duty
(
    IN  F32     duty
)
{
    BSP_BRIDGE_MODULE_T *p_mod = _bsp_bridge_get_module();

    if (p_mod->handle.Instance != TIM1)
    {
        return;
    }

    p_mod->duty = duty;

    /* 写 PMOS_CTRL 那一路是另一回事，这里只动 CH1 的比较值。
     * CH1N 是硬件互补，CCR 只有一份，改 CH1 就等于同时改了两路。 */
    __HAL_TIM_SET_COMPARE(&p_mod->handle, TIM_CHANNEL_1, _bsp_bridge_duty_to_compare(duty));
}

/********************************************************************************************
* 函数名称 :   bsp_bridge_get_duty
* 功能描述 :   读回最近一次写入的带符号占空比
*               返回的是"请求值"（钳位前的原始值），不是硬件里那个已经钳过的 CCR。
*               要看硬件实际值请直接读 CCR1。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   -1.0 ~ +1.0（可能超出，如果上层传了越界值）
*********************************************************************************************/
F32 bsp_bridge_get_duty
(
    VOID
)
{
    return _bsp_bridge_get_module()->duty;
}

/********************************************************************************************
* 函数名称 :   bsp_bridge_is_running
* 功能描述 :   查询 MOE 是否置位（两路输出是否真的在动）
*               注意这是"本层认为自己在跑"，不是"引脚上真的有波形" ——
*               如果 PERMIT 被拉低，驱动器那边已经停了，本函数照样返回 TRUE。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = MOE 已置位
*********************************************************************************************/
BOOL bsp_bridge_is_running
(
    VOID
)
{
    return _bsp_bridge_get_module()->started;
}

#ifdef _cplusplus
}
#endif
