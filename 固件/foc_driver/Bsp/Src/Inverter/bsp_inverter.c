/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_inverter.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BSP_INVERTER_MODULE_T_
{
    TIM_HandleTypeDef   handle;                                         /* TIM1 句柄 */
    BOOL                started;                                        /* MOE 是否已置位 */
}BSP_INVERTER_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_INVERTER_MODULE_T * _bsp_inverter_get_module
(
    VOID
);

static VOID _bsp_inverter_gpio_init
(
    VOID
);

static U32 _bsp_inverter_compare_from_duty
(
    IN  F32     duty
);


/************************************静态变量定义************************************************/
static BSP_INVERTER_MODULE_T s_bsp_inverter_module =
{
    {0},                                                                /* 句柄在 init 中填充 */
    FALSE                                                               /* 初始不输出 */
};


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_inverter_get_module
* 功能描述 :   获取逆变器模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BSP_INVERTER_MODULE_T * _bsp_inverter_get_module
(
    VOID
)
{
    return &s_bsp_inverter_module;
}

/********************************************************************************************
* 函数名称 :   _bsp_inverter_gpio_init
* 功能描述 :   配置 TIM1 六路输出与 BREAK_N 的复用引脚
*
*               六路 PWM：
*                 PA8/PA9/PA10   → TIM1_CH1/CH2/CH3     （上桥，AF6）
*                 PB13/PB14/PB15 → TIM1_CH1N/CH2N/CH3N  （下桥，AF6）
*               全部推挽复用、高速档；不加内部上下拉 —— 板上栅极侧有 10k 下拉（R_GS_x），
*               而 DRV8353 的 INHx/INLx 本身也有内部下拉，MCU 这里再并一个没有意义。
*
*               BREAK_N（PB12）：
*                 它是 TIM1_BKIN 的复用输入（AF6）。注意这一脚**同时**是 ADC1_IN11，
*                 本板用作刹车输入，就不再拿它做模拟通道。
*
*               关于 JTAG：本板没有用到 PA13/PA14/PA15/PB3/PB4 的 JTAG 功能，
*               但 PB3/PA15 被 bsp_hall 拿去当霍尔输入，届时 JTAG 自动让位、
*               SWD 仍可用。本函数不碰这几个脚。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bsp_inverter_gpio_init
(
    VOID
)
{
    GPIO_InitTypeDef gpio_init = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* 上桥三路：PA8 / PA9 / PA10 —— 同一个端口，一次配完 */
    gpio_init.Pin       = BSP_BOARD_INV_HIGH_MASK;
    gpio_init.Mode      = GPIO_MODE_AF_PP;
    gpio_init.Pull      = GPIO_NOPULL;
    gpio_init.Speed     = GPIO_SPEED_FREQ_HIGH;     /* 20 kHz 方波，边沿要快，减少开关损耗 */
    gpio_init.Alternate = BSP_BOARD_INV_AF;
    HAL_GPIO_Init(BSP_BOARD_INV_UH_PORT, &gpio_init);

    /* 下桥三路 + 刹车输入：PB13 / PB14 / PB15 / PB12 */
    gpio_init.Pin       = BSP_BOARD_INV_LOW_MASK | BSP_BOARD_INV_BKIN_PIN;
    gpio_init.Mode      = GPIO_MODE_AF_PP;          /* BKIN 在本函数里配成 AF_PP；注意
                                                     * HAL_TIM_PWM_Init() 内部的
                                                     * HAL_TIM_Base_MspInit 随后会把它改成
                                                     * AF_OD + LOW 速（CubeMX 生成），对输入
                                                     * 无功能影响，以最终配置为准 */
    gpio_init.Pull      = GPIO_NOPULL;              /* BREAK_N 板上已有 2.2k 上拉 */
    gpio_init.Speed     = GPIO_SPEED_FREQ_HIGH;
    gpio_init.Alternate = BSP_BOARD_INV_AF;
    HAL_GPIO_Init(BSP_BOARD_INV_UL_PORT, &gpio_init);
}

/********************************************************************************************
* 函数名称 :   _bsp_inverter_compare_from_duty
* 功能描述 :   把 0.0~1.0 的占空比换算成中心对齐模式的比较值
*
*               中心对齐向上/向下计数时，PWM1 模式的高电平宽度约等于 2*CCR 个计数，
*               而一个完整周期是 2*ARR 个计数，所以 CCR = duty * ARR 即可，
*               不需要像边沿对齐那样再乘 (ARR+1)。这就是这里用 ARR 而不是 ARR+1 的原因。
*
*               越界钳位放在这里，调用方不必自己判断。
*
* 输入参数 :   duty: 0.0 ~ 1.0，按绝对值钳位
* 输出参数 :   无
* 返回值   :   0 ~ BSP_BOARD_INV_PWM_ARR 的比较值
*********************************************************************************************/
static U32 _bsp_inverter_compare_from_duty
(
    IN  F32     duty
)
{
    F32 compare;

    if (duty < 0.0f)
    {
        duty = 0.0f;
    }
    if (duty > 1.0f)
    {
        duty = 1.0f;
    }

    /* 四舍五入到整数，避免浮点截断带来固定偏低 */
    compare = (duty * (F32)BSP_BOARD_INV_PWM_ARR) + 0.5f;

    return (U32)compare;
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_get_dtg
* 功能描述 :   把"死区纳秒数"编码成 TIM1 BDTR 寄存器的 DTG[7:0] 字段
*
*               死区发生器有 4 段折线（RM0440 "Dead-time generator"），tDTS = 1/170MHz = 5.882ns：
*                 DTG[7:5] = 0xx → DT = DTG[7:0]            × tDTS          (0 ~ 747ns)
*                 DTG[7:5] = 10x → DT = (64 + DTG[5:0])     × 2 × tDTS      (753 ~ 1494ns)
*                 DTG[7:5] = 110 → DT = (32 + DTG[4:0])     × 8 × tDTS      (1506 ~ 2965ns)
*                 DTG[7:5] = 111 → DT = (32 + DTG[4:0])     × 16 × tDTS     (3012 ~ 5930ns)
*               四段之间有不连续的台阶，所以必须按段判断，不能简单线性换算。
*
* 输入参数 :   ns: 期望死区（纳秒），超出 5930ns 时按最大值处理
* 输出参数 :   无
* 返回值   :   DTG[7:0] 编码值
*********************************************************************************************/
U32 bsp_inverter_get_dtg
(
    IN  U32     ns
)
{
    const U32 tdts_ps = 5882U;                          /* 1/170MHz ≈ 5882 ps，用 ps 保留精度 */
    U32       ticks;

    /* 先换算成 tDTS 个数（向上取整，死区宁大勿小） */
    ticks = (ns * 1000U + tdts_ps - 1U) / tdts_ps;
    if (ticks == 0U)
    {
        return 0U;                                      /* DT = 0，即不插入死区 */
    }

    if (ticks <= 127U)                                  /* 第 1 段：1:1 */
    {
        return ticks;
    }

    ticks = (ticks + 1U) / 2U;                          /* 第 2 段：1:2，先把 tDTS 数除以 2 */
    if (ticks <= 127U)
    {
        if (ticks < 64U)
        {
            ticks = 64U;                                /* 该段下限是 64，否则会掉进第 1 段的编码空间 */
        }
        return 0x80U | (ticks - 64U);                   /* DTG[7:5] = 10x */
    }

    ticks = (ticks + 3U) / 4U;                          /* 第 3 段：1:8 */
    if (ticks <= 63U)
    {
        if (ticks < 32U)
        {
            ticks = 32U;
        }
        return 0xC0U | (ticks - 32U);                   /* DTG[7:5] = 110 */
    }

    ticks = (ticks + 1U) / 2U;                          /* 第 4 段：1:16 */
    if (ticks > 63U)
    {
        ticks = 63U;                                    /* 饱和到最大死区 */
    }
    if (ticks < 32U)
    {
        ticks = 32U;
    }
    return 0xE0U | (ticks - 32U);                       /* DTG[7:5] = 111 */
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_init
* 功能描述 :   把 TIM1 配成 20 kHz 中心对齐三相互补 PWM，并把 BREAK_N 接到 BKIN
*
*               为什么是中心对齐（CMS=CENTERALIGNED1）：
*                 计数器从 0 数到 ARR 再数回 0。PWM1 模式下 CHx 在 CNT<CCR 时有效，
*                 于是"上桥导通"这段被居中在 CNT=0 附近，"下桥导通"这段被居中在 CNT=ARR 附近。
*                 三相下桥同时导通的那一刻正好是 CNT 越过 ARR 的瞬间 —— 这就是低边采样
*                 电阻能安全采样的窗口，bsp_adc 的注入组就靠这个时刻触发。
*
*               为什么 MOE 先不开（OSSI/OSSR 都置 1、OISx 全 0）：
*                 MOE=0 时六路输出被强制成"低"，也就是 INH=0、INL=0；
*                 按 DRV8353 Table 1，INL=INH=0 → GL=GH=L、SH=Hi-Z，六路栅极全关。
*                 这一步是在芯片层面把功率级钉死，比"把占空比写 0"可靠得多。
*
*               为什么 AOE（AutomaticOutput）关掉：
*                 刹车事件之后 MOE 只被硬件清零、不会被自动恢复。故障消失后必须由软件
*                 显式重新 start。宁可让上层代码多写两行，也不要让驱动器自己反复重启。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_inverter_init
(
    VOID
)
{
    TIM_OC_InitTypeDef              oc_init       = {0};
    TIM_BreakDeadTimeConfigTypeDef  bdt_init      = {0};
    TIM_MasterConfigTypeDef         master_init   = {0};
    BSP_INVERTER_MODULE_T           *p_mod        = _bsp_inverter_get_module();
    U32                             dtg;

    _bsp_inverter_gpio_init();

    /* TIM1 挂在 APB2，时钟已经由 SystemClock_Config 配到 170 MHz（APB2 不分频） */
    __HAL_RCC_TIM1_CLK_ENABLE();

    /* ---- 时基：170 MHz / (2 * 4250) = 20 kHz，中心对齐 ---- */
    p_mod->handle.Instance                   = TIM1;
    p_mod->handle.Init.Prescaler             = 0;
    p_mod->handle.Init.CounterMode           = TIM_COUNTERMODE_CENTERALIGNED1;
    p_mod->handle.Init.Period                = BSP_BOARD_INV_PWM_ARR;
    p_mod->handle.Init.ClockDivision         = TIM_CLOCKDIVISION_DIV1;
    /* RCR 留 0。这里不用更新事件去触发 ADC，所以 RCR 只影响中断频率，不影响采样点。
     * （用更新事件触发会有个坑：中心对齐下溢出和下溢都会产生更新事件，
     *   RCR=1 时到底停在溢出还是下溢要看计数初相，示波器不量一遍不敢下结论。
     *   所以触发改用 CH4 的 OC4REF，见下面的通道 4 配置。） */
    p_mod->handle.Init.RepetitionCounter     = 0;
    p_mod->handle.Init.AutoReloadPreload     = TIM_AUTORELOAD_PRELOAD_ENABLE;
    if (HAL_TIM_PWM_Init(&p_mod->handle) != HAL_OK)
    {
        return;
    }

    /* ---- 通道 1/2/3：PWM1 模式，初始比较值 0 ---- */
    oc_init.OCMode       = TIM_OCMODE_PWM1;
    oc_init.Pulse        = 0;
    oc_init.OCPolarity   = TIM_OCPOLARITY_HIGH;
    oc_init.OCNPolarity  = TIM_OCNPOLARITY_HIGH;        /* 互补通道不反相：CHxN 就是"非 CHx" */
    oc_init.OCFastMode   = TIM_OCFAST_DISABLE;
    oc_init.OCIdleState  = TIM_OCIDLESTATE_RESET;       /* 刹车/空闲时 CHx  = 低 */
    oc_init.OCNIdleState = TIM_OCNIDLESTATE_RESET;      /* 刹车/空闲时 CHxN = 低 */

    if (HAL_TIM_PWM_ConfigChannel(&p_mod->handle, &oc_init, TIM_CHANNEL_1) != HAL_OK)
    {
        return;
    }
    if (HAL_TIM_PWM_ConfigChannel(&p_mod->handle, &oc_init, TIM_CHANNEL_2) != HAL_OK)
    {
        return;
    }
    if (HAL_TIM_PWM_ConfigChannel(&p_mod->handle, &oc_init, TIM_CHANNEL_3) != HAL_OK)
    {
        return;
    }

    /* ---- 通道 4：不驱动任何引脚，只用来给 ADC 打触发拍 ----
     *
     * 为什么非要绕这一圈：
     *   本板是低边采样，三相相电流只有在"三相下桥同时导通"时才是真的。中心对齐 PWM 里
     *   这段时间被居中在 CNT=ARR（计数器顶点）。UPDATE 事件在顶点和谷点各来一次，用它
     *   触发就会采到一半有效、一半全 0 的数据。CH4 配成 PWM2 模式后，OC4REF 只在
     *   CNT ≥ CCR4 时为高；把 CCR4 设在 ARR 附近，OC4REF 一个周期就只在顶点翻高一次，
     *   拿它当 TRGO 就是唯一、确定的采样点。
     *
     * 为什么留 8 个计数（≈47 ns）的宽度：
     *   ADC 的触发信号是按时钟采样的，脉冲太窄会被漏掉。ADC 时钟 42.5 MHz（周期 23.5 ns），
     *   8 个计数正好两倍于它，稳。
     *
     * 引脚说明（这一条容易想当然，已按 DS12589 Rev6 Table 12 复核）：
     *   TIM1_CH4 的映射是 **PA11**（AF6），不是 PA3 —— PA3 是 TIM2_CH4，两回事。
     *   而本板的 PA11 分给了 FDCAN1_RX（AF9），所以 CH4 的波形根本到不了引脚。
     *
     *   这恰恰是我们想要的：
     *     · HAL_TIM_PWM_ConfigChannel(..., TIM_CHANNEL_4) 只写定时器寄存器
     *       （CCMR2 / CCER 的 CC4E），不动 GPIO；
     *     · PA11 的 AFR 被 bsp_can 设成 AF9，定时器的 AF6 输出到不了它；
     *     · 但 TRGO = OC4REF 用的是**片内信号**，与引脚无关，照样每个周期打一拍。
     *   结果：既拿到了 ADC 触发源，又没多占一个引脚。
     *
     *   ⚠️ 顺带一个坑：PA11 同时还是 TIM1_BKIN2（AF6）。本层把 Break2State 关成
     *      TIM_BREAK2_DISABLE，所以 PA11 上的 CAN 数据不会被当成刹车输入。
     *      谁要是哪天打开了 Break2，CAN 总线上的每一帧都可能触发刹车。 */
    oc_init.OCMode  = TIM_OCMODE_PWM2;
    oc_init.Pulse   = (BSP_BOARD_INV_PWM_ARR > BSP_BOARD_INV_TRGO_OFFSET)
                      ? (BSP_BOARD_INV_PWM_ARR - BSP_BOARD_INV_TRGO_OFFSET) : 0U;
    if (HAL_TIM_PWM_ConfigChannel(&p_mod->handle, &oc_init, TIM_CHANNEL_4) != HAL_OK)
    {
        return;
    }

    /* ---- 把 CH4 的比较事件送到 TRGO，供 ADC 注入组使用 ---- */
    master_init.MasterOutputTrigger  = TIM_TRGO_OC4REF;                 /* TRGO 源 = OC4REF */
    master_init.MasterOutputTrigger2 = TIM_TRGO2_RESET;                 /* TRGO2 本板不用 */
    master_init.MasterSlaveMode      = TIM_MASTERSLAVEMODE_ENABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&p_mod->handle, &master_init) != HAL_OK)
    {
        return;
    }

    /* ---- 死区 + 刹车 ---- */
    dtg = bsp_inverter_get_dtg(BSP_BOARD_INV_DEADTIME_NS);

    bdt_init.OffStateRunMode  = TIM_OSSR_ENABLE;        /* MOE=0 时引脚被驱动到 OISx，而不是高阻 */
    bdt_init.OffStateIDLEMode = TIM_OSSI_ENABLE;        /* 同上，空闲态也钉死 */
    bdt_init.LockLevel        = TIM_LOCKLEVEL_OFF;      /* 不锁 BDTR，允许运行时改死区 */
    bdt_init.DeadTime         = dtg;
    bdt_init.BreakState       = TIM_BREAK_ENABLE;
    /* BREAK_N 是线与低有效：任何一路故障拉低 → 触发刹车。
     * 极性写成 LOW 表示"低电平有效"。 */
    bdt_init.BreakPolarity    = TIM_BREAKPOLARITY_LOW;
    /* 数字滤波 BKF=5：采样频率 fDTS / 2^5，需连续 8 个样本 → 延迟 ≈ 8 × 32 × 5.882 ns ≈ 1.5 µs。
     * ⚠️ 不是"47 ns"（那是 BKF=0 的量级，只剩输入同步本身）。
     *    1.5 µs 换来的是对 2.2k 上拉 + 长走线耦合噪声的容忍度；若实测过流保护太慢，
     *    可下调：BKF=2 ≈ 188 ns、BKF=1 ≈ 94 ns、BKF=0 无滤波（≈47 ns），
     *    每改一档都要重新评估误刹车率，改完用示波器确认 BREAK_N 的响应。 */
    bdt_init.BreakFilter      = 5;
    /* ⚠️ 第二刹车输入必须保持 DISABLE：PA11 是 FDCAN1_RX，
     * 打开 Break2 就等于让 CAN 总线每收一帧都可能触发刹车。 */
    bdt_init.Break2State      = TIM_BREAK2_DISABLE;
    bdt_init.Break2Polarity   = TIM_BREAK2POLARITY_LOW;
    bdt_init.Break2Filter     = 0;
    bdt_init.AutomaticOutput  = TIM_AUTOMATICOUTPUT_DISABLE;

    if (HAL_TIMEx_ConfigBreakDeadTime(&p_mod->handle, &bdt_init) != HAL_OK)
    {
        return;
    }

    /* 只起计数器、不起 MOE：这样 ADC 的触发源从 init 之后就一直在跑，
     * 上层可以先采电流、再决定什么时候放行栅极。 */
    __HAL_TIM_ENABLE(&p_mod->handle);

    p_mod->started = FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_start
* 功能描述 :   置位 MOE，让六路输出真正跟随占空比
*
*               注意执行顺序：MOE 一置位，当前 CCR 立刻生效。如果此刻 CCR=0（默认），
*               三相下桥会马上全通 = 电制动，而不是"没输出"。所以正确用法是：
*                   bsp_inverter_set_duty(u, v, w);   // 先把想要的占空比写好
*                   bsp_gpio_drv_set_enable(TRUE);
*                   bsp_inverter_start();             // 再放行
*
*               这里不用 HAL_TIM_PWM_Start，因为它一次只能启一个通道，而且会顺手改 CCER；
*               直接置 CCxE/CCxNE + MOE 更清楚，也避免重复调用 HAL 造成多次开关。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_inverter_start
(
    VOID
)
{
    BSP_INVERTER_MODULE_T *p_mod = _bsp_inverter_get_module();

    if (p_mod->handle.Instance == TIM1)
    {
        /* 三路主输出 + 三路互补输出全部使能。
         *
         * ⚠️ 这里故意**不开** CCR 预装载（OCxPE 保持复位默认 0）：
         *    本工程用的这一版 G4 HAL 里 TIM_OC_InitTypeDef 压根没有 OCPreload 字段，
         *    HAL_TIM_PWM_ConfigChannel() 也就不会去置 CCMRx.OCxPE。
         *    结果是 bsp_inverter_set_duty() 写 CCR 立即生效 —— 三路之间最多差两条指令
         *    （170 MHz 下约 12 ns），相对 50 µs 的 PWM 周期是 0.02%，可以忽略。
         *    如果哪天要做更精细的调制（比如要求三相严格同一拍切换），
         *    就在 HAL_TIM_PWM_Init() 之后手动置 CCMR1/CCMR2 的 OCxPE 位，
         *    但那样每次写占空比都要等一个更新事件才生效，FOC 的环路延迟会多一个周期。 */
        SET_BIT(p_mod->handle.Instance->CCER,
                TIM_CCER_CC1E | TIM_CCER_CC1NE |
                TIM_CCER_CC2E | TIM_CCER_CC2NE |
                TIM_CCER_CC3E | TIM_CCER_CC3NE);

        __HAL_TIM_MOE_ENABLE(&p_mod->handle);
        p_mod->started = TRUE;
    }
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_stop
* 功能描述 :   清 MOE，六路输出回到 OISx 定义的"低"，DRV8353 据此进 Hi-Z（自由滑行）
*
*               计数器不关：ADC 注入组还在等 TRGO，关了以后相电流就再也不更新了。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_inverter_stop
(
    VOID
)
{
    BSP_INVERTER_MODULE_T *p_mod = _bsp_inverter_get_module();

    if (p_mod->handle.Instance == TIM1)
    {
        __HAL_TIM_MOE_DISABLE(&p_mod->handle);
        p_mod->started = FALSE;
    }
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_set_duty
* 功能描述 :   一次性写三相占空比
*
*               duty=1.0 → CCR=ARR → CHx 恒高、CHxN 恒低 → INH=1、INL=0 → 上桥常通
*               duty=0.0 → CCR=0   → CHx 恒低、CHxN 恒高 → INH=0、INL=1 → 下桥常通（电制动）
*               中间值就是常规 PWM。
*
*               ⚠️ 这是"逐相独立"的写法，本层不管调制策略。上层如果要用 SVPWM，
*                  也是先算出三相占空比再调这里。
*
*               ⚠️ 三路 CCR 不是同时生效的（逐条写寄存器），会有一个到几个周期的相位差。
*                  20 kHz 下影响可以忽略；如果以后上了更高的开关频率，改成写 CCR 预装载
*                  再靠更新事件统一加载。
*
* 输入参数 :   duty_u / duty_v / duty_w: 0.0 ~ 1.0，越界钳位
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_inverter_set_duty
(
    IN  F32     duty_u,
    IN  F32     duty_v,
    IN  F32     duty_w
)
{
    BSP_INVERTER_MODULE_T *p_mod = _bsp_inverter_get_module();

    if (p_mod->handle.Instance != TIM1)
    {
        return;
    }

    __HAL_TIM_SET_COMPARE(&p_mod->handle, TIM_CHANNEL_1, _bsp_inverter_compare_from_duty(duty_u));
    __HAL_TIM_SET_COMPARE(&p_mod->handle, TIM_CHANNEL_2, _bsp_inverter_compare_from_duty(duty_v));
    __HAL_TIM_SET_COMPARE(&p_mod->handle, TIM_CHANNEL_3, _bsp_inverter_compare_from_duty(duty_w));
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_set_output_enable
* 功能描述 :   只切换 MOE，不改已写的占空比
*               适合做"暂停一下再继续"：停的时候六路全低（滑行），恢复时接着原来的占空比跑，
*               不用重新算三相占空比。
* 输入参数 :   enable: TRUE = 放行输出，FALSE = 回到 off-state
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_inverter_set_output_enable
(
    IN  BOOL    enable
)
{
    if (enable == TRUE)
    {
        bsp_inverter_start();
    }
    else
    {
        bsp_inverter_stop();
    }
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_break_occurred
* 功能描述 :   查询是否发生过刹车事件
*
*               TIM_FLAG_BREAK 由硬件置位并保持，即使故障只持续几百纳秒也抓得到。
*               这是判断"功率级是不是刚刚被硬件切掉"的唯一可靠途径 ——
*               别去读 BREAK_N 的当前电平，那时候故障可能已经消失了。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 发生过，FALSE = 没发生
*********************************************************************************************/
BOOL bsp_inverter_break_occurred
(
    VOID
)
{
    BSP_INVERTER_MODULE_T *p_mod = _bsp_inverter_get_module();

    if (p_mod->handle.Instance != TIM1)
    {
        return FALSE;
    }

    return (__HAL_TIM_GET_FLAG(&p_mod->handle, TIM_FLAG_BREAK) != RESET) ? TRUE : FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_inverter_clear_break
* 功能描述 :   清刹车标志（不动 MOE）
*
*               清完标志以后 MOE 仍然是 0，输出仍然是断的。要恢复必须再调 bsp_inverter_start()。
*               这个"两步走"是刻意的：中间那一步正好留给上层去查 DRV8353 的故障寄存器、
*               判断能不能重启。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_inverter_clear_break
(
    VOID
)
{
    BSP_INVERTER_MODULE_T *p_mod = _bsp_inverter_get_module();

    if (p_mod->handle.Instance == TIM1)
    {
        __HAL_TIM_CLEAR_FLAG(&p_mod->handle, TIM_FLAG_BREAK);
    }
}

#ifdef _cplusplus
}
#endif
