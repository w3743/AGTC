/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_adc.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BSP_ADC_MODULE_T_
{
    ADC_HandleTypeDef       handle1;                                    // ADC1：注入组(相电流) + 规则组(温度/安全)
    ADC_HandleTypeDef       handle2;                                    // ADC2：规则组(母线)
    DMA_HandleTypeDef       dma1;                                       // ADC1 → DMA1_Channel1
    DMA_HandleTypeDef       dma2;                                       // ADC2 → DMA1_Channel2
    U16                     reg1[BSP_ADC1_REG_NUM];                    // 规则组 DMA 目标缓冲
    U16                     reg2[BSP_ADC2_REG_NUM];
    U16                     phase_u;                                    // 最近一次同步相电流
    U16                     phase_v;
    U16                     phase_w;
    BOOL                    phase_valid;                                // 是否已经采到过
}BSP_ADC_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_ADC_MODULE_T * _bsp_adc_get_module
(
    VOID
);

static VOID _bsp_adc_gpio_init
(
    VOID
);

static BOOL _bsp_adc_dma_init
(
    IN  ADC_HandleTypeDef   *p_adc,
    IN  DMA_HandleTypeDef   *p_dma,
    IN  DMA_Channel_TypeDef *dma_instance,
    IN  U32                 dma_request
);


/************************************静态变量定义************************************************/
static BSP_ADC_MODULE_T s_bsp_adc_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_adc_get_module
* 功能描述 :   获取 ADC 模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BSP_ADC_MODULE_T * _bsp_adc_get_module
(
    VOID
)
{
    return &s_bsp_adc_module;
}

/********************************************************************************************
* 函数名称 :   _bsp_adc_gpio_init
* 功能描述 :   把七个采样脚配成模拟输入
*
*               PA1 SO_U / PA3 SO_V / PA0 SO_W / PA2 TEMP_MOS / PB1 TEMP_MOTOR /
*               PB0 SAFE / PC4 BUS
*
*               为什么必须是 GPIO_MODE_ANALOG：
*                 模拟模式下施密特触发器输入被断开、上下拉被切断，输入漏电流降到 nA 级。
*                 如果留成浮空输入，引脚上的数字输入缓冲会吃掉一部分电荷，高阻分压
*                 （比如 180k/10k 的母线分压）会明显偏低。
*
*               ⚠️ 这里和 bsp_inverter 的 TIM1_CH4 没有交集，别搞混（DS12589 Rev6 Table 12）：
*                  PA3  = ADC1_IN4 + **TIM2_CH4**（不是 TIM1_CH4）
*                  PA11 = FDCAN1_RX + **TIM1_CH4** + TIM1_BKIN2
*                  本板的 TIM1_CH4 落在 PA11 上，而 PA11 被 bsp_can 设成 AF9 给 CAN 用了，
*                  所以 CH4 的输出到不了引脚、也不会和任何 ADC 通道抢位置；
*                  ADC 触发只用到片内的 OC4REF→TRGO，详见 bsp_inverter.c 通道 4 那一段。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bsp_adc_gpio_init
(
    VOID
)
{
    GPIO_InitTypeDef gpio_init = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* PA0 / PA1 / PA3：三相电流采样（SO_W / SO_U / SO_V） */
    gpio_init.Pin  = BSP_BOARD_ADC_INJ_MASK;
    gpio_init.Mode = GPIO_MODE_ANALOG;
    gpio_init.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(BSP_BOARD_ADC_SO_W_PORT, &gpio_init);

    /* PA2：MOS 温度 */
    gpio_init.Pin  = BSP_BOARD_ADC_TEMP_MOS_PIN;
    gpio_init.Mode = GPIO_MODE_ANALOG;
    gpio_init.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(BSP_BOARD_ADC_TEMP_MOS_PORT, &gpio_init);

    /* PB1：电机温度；PB0：安全电压 */
    gpio_init.Pin  = BSP_BOARD_ADC_TEMP_MOTOR_PIN | BSP_BOARD_ADC_SAFE_PIN;
    gpio_init.Mode = GPIO_MODE_ANALOG;
    gpio_init.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(BSP_BOARD_ADC_TEMP_MOTOR_PORT, &gpio_init);

    /* PC4：母线电压（只有 ADC2_IN5 这一个映射） */
    gpio_init.Pin  = BSP_BOARD_ADC_BUS_PIN;
    gpio_init.Mode = GPIO_MODE_ANALOG;
    gpio_init.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(BSP_BOARD_ADC_BUS_PORT, &gpio_init);
}

/********************************************************************************************
* 函数名称 :   _bsp_adc_dma_init
* 功能描述 :   给一个 ADC 挂一个循环 DMA 通道
*
*               G4 的 DMA 请求不是固定的通道映射，必须经 DMAMUX 选源：
*                 ADC1 → DMA_REQUEST_ADC1（HAL 内部值是 5）
*                 ADC2 → DMA_REQUEST_ADC2（HAL 内部值是 6）
*               这里用 DMA1_Channel1 和 DMA1_Channel2，两个通道互不干扰。
*
*               数据宽度用半字（16 位）：ADC 的 DR 是 16 位右对齐，正好对上。
*               循环模式 + DMAContinuousRequests 让规则组一轮接一轮自动搬，不用软件重装。
*
* 输入参数 :   p_adc:        ADC 句柄
*               p_dma:        DMA 句柄（由调用者提供存储）
*               dma_instance: DMA1_Channel1 / DMA1_Channel2
*               dma_request:  DMA_REQUEST_ADC1 / DMA_REQUEST_ADC2
* 输出参数 :   无
* 返回值   :   TRUE 成功，FALSE 失败
*********************************************************************************************/
static BOOL _bsp_adc_dma_init
(
    IN  ADC_HandleTypeDef   *p_adc,
    IN  DMA_HandleTypeDef   *p_dma,
    IN  DMA_Channel_TypeDef *dma_instance,
    IN  U32                 dma_request
)
{
    p_dma->Instance                 = dma_instance;
    p_dma->Init.Request             = dma_request;
    p_dma->Init.Direction           = DMA_PERIPH_TO_MEMORY;
    p_dma->Init.PeriphInc           = DMA_PINC_DISABLE;                 // 外设地址固定（DR）
    p_dma->Init.MemInc              = DMA_MINC_ENABLE;                  // 内存地址递增
    p_dma->Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    p_dma->Init.MemDataAlignment    = DMA_MDATAALIGN_HALFWORD;
    p_dma->Init.Mode                = DMA_CIRCULAR;                     // 循环，永不停止
    p_dma->Init.Priority            = DMA_PRIORITY_LOW;                 // 慢信号，低优先级即可

    if (HAL_DMA_Init(p_dma) != HAL_OK)
    {
        return FALSE;
    }

    __HAL_LINKDMA(p_adc, DMA_Handle, *p_dma);

    return TRUE;
}

/********************************************************************************************
* 函数名称 :   bsp_adc_init
* 功能描述 :   配置 ADC1（注入组 + 规则组）与 ADC2（规则组）以及各自的 DMA
*
*               几个关键取舍：
*
*               1) 时钟用 ADC_CLOCK_SYNC_PCLK_DIV4 = 170/4 = 42.5 MHz。
*                  G4 的 ADC 最高 60 MHz，42.5 MHz 留了余量；同步时钟模式不需要额外
*                  配置 RCC 的 ADC 专用时钟源，最省事也最不容易配错。
*
*               2) 注入组和规则组采样时间分开设：
*                  相电流 12.5 周期（≈294 ns）—— 信号来自 DRV8353 的运放输出，
*                  源阻抗低，快采没问题，而且要在死区窗口内采完；
*                  慢通道 247.5 周期（≈5.8 µs）—— 母线分压是 180k/10k（源阻抗 10k），
*                  温度通道还有 10k 上拉，必须给足建立时间，否则读数偏低。
*
*               3) 注入组用 TIM1_TRGO 触发、边沿上升。触发点由 bsp_inverter 用 CH4(OC4REF)
*                  在计数器顶点产生，一个 PWM 周期正好一次。
*
*               4) Overrun 选"覆盖"：慢通道读慢了就丢旧值，不要让 OVR 把 ADC 卡死。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_adc_init
(
    VOID
)
{
    ADC_InjectionConfTypeDef inj = {0};
    ADC_ChannelConfTypeDef   ch  = {0};
    BSP_ADC_MODULE_T         *p_mod = _bsp_adc_get_module();

    _bsp_adc_gpio_init();

    __HAL_RCC_ADC12_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* ⚠️ DMAMUX1 的时钟必须单独开，这一行漏了会出很典型的"静默故障"：
     *    G4 的 DMA 请求不直接连到通道，而是先经 DMAMUX 选源（配 DMAMUX_CxCR 的
     *    DMAREQ_ID）。HAL_DMA_Init() 会去写这个寄存器，但 HAL **不会**帮你开
     *    DMAMUX 的时钟 —— 手册原文就写着"Prior to HAL_DMA_Init the peripheral clock
     *    shall be enabled for both DMA & DMAMUX"（stm32g4xx_hal_dma.c 第 38 行）。
     *    时钟没开时写寄存器不报错也不生效，现象是 DMA 一次都不搬：ADC 在转、
     *    DMA 标志永远不动、缓冲区永远全是 0，而代码看起来完全正常。 */
    __HAL_RCC_DMAMUX1_CLK_ENABLE();

    /* ================= ADC1 主体 ================= */
    p_mod->handle1.Instance                      = ADC1;
    p_mod->handle1.Init.ClockPrescaler           = ADC_CLOCK_SYNC_PCLK_DIV4;
    p_mod->handle1.Init.Resolution               = ADC_RESOLUTION_12B;
    p_mod->handle1.Init.DataAlign                = ADC_DATAALIGN_RIGHT;
    p_mod->handle1.Init.GainCompensation         = 0;
    p_mod->handle1.Init.ScanConvMode             = ADC_SCAN_ENABLE;
    p_mod->handle1.Init.EOCSelection             = ADC_EOC_SINGLE_CONV;
    p_mod->handle1.Init.LowPowerAutoWait         = DISABLE;
    p_mod->handle1.Init.ContinuousConvMode       = ENABLE;               // 规则组一直转
    p_mod->handle1.Init.NbrOfConversion          = BSP_ADC1_REG_NUM;
    p_mod->handle1.Init.DiscontinuousConvMode    = DISABLE;
    p_mod->handle1.Init.ExternalTrigConv         = ADC_SOFTWARE_START;   // 规则组不需要外部触发
    p_mod->handle1.Init.ExternalTrigConvEdge     = ADC_EXTERNALTRIGCONVEDGE_NONE;
    p_mod->handle1.Init.DMAContinuousRequests    = ENABLE;
    p_mod->handle1.Init.Overrun                  = ADC_OVR_DATA_OVERWRITTEN;
    p_mod->handle1.Init.OversamplingMode         = DISABLE;
    if (HAL_ADC_Init(&p_mod->handle1) != HAL_OK)
    {
        return;
    }

    /* 规则组：顺序 = 缓冲区下标 */
    ch.SamplingTime = ADC_SAMPLETIME_247CYCLES_5;

    ch.Channel = BSP_BOARD_ADC_TEMP_MOS_CHANNEL;                         /* PA2  TEMP_MOS   */
    ch.Rank    = ADC_REGULAR_RANK_1;
    ch.SingleDiff = ADC_SINGLE_ENDED;
    ch.OffsetNumber = ADC_OFFSET_NONE;
    ch.Offset  = 0;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_TEMP_MOTOR_CHANNEL;                       /* PB1  TEMP_MOTOR */
    ch.Rank    = ADC_REGULAR_RANK_2;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_SAFE_CHANNEL;                             /* PB0  SAFE       */
    ch.Rank    = ADC_REGULAR_RANK_3;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    /* 注入组：3 个通道，TIM1_TRGO 上升沿触发
     *
     * ⚠️ 下面必须「每个 rank 调一次」，一共三次，**不能**先单独调一次来"只设长度/触发"：
     *    HAL 用句柄里的 InjectionConfig.ChannelCount 递减来攒 JSQR，攒到 0 才写寄存器。
     *    多出来的那次空调用（InjectedChannel/InjectedRank 还是 0）会把计数提前置 3 并
     *    占掉一次递减 —— 结果第 3 次调用就把只含 rank1/rank2 的 JSQR 落盘，rank3 的通道
     *    永远写不进去（症状：JDR3 读到的是 CH0，第三相电流恒错）。
     *    长度/触发的赋值会随第一次 rank 调用一起生效，不需要也不能单独调用。
     *    2026-10-28 审查发现并修复，详见 D:/Hal_code/审查报告_引脚一致性与BSP_2026-10-28.md */
    inj.InjectedNbrOfConversion         = 3;
    inj.InjectedSamplingTime            = ADC_SAMPLETIME_12CYCLES_5;
    inj.InjectedSingleDiff              = ADC_SINGLE_ENDED;
    inj.InjectedOffsetNumber            = ADC_OFFSET_NONE;
    inj.InjectedOffset                  = 0;
    inj.ExternalTrigInjecConv           = ADC_EXTERNALTRIGINJEC_T1_TRGO;
    inj.ExternalTrigInjecConvEdge       = ADC_EXTERNALTRIGINJECCONV_EDGE_RISING;
    inj.InjecOversamplingMode           = DISABLE;

    /* 逐个通道设 rank：顺序必须和 bsp_adc_get_phase_raw 里的 JDR1/2/3 对应 */
    inj.InjectedChannel      = BSP_BOARD_ADC_SO_U_CHANNEL;               /* PA1  SO_U → JDR1 */
    inj.InjectedRank         = ADC_INJECTED_RANK_1;
    if (HAL_ADCEx_InjectedConfigChannel(&p_mod->handle1, &inj) != HAL_OK) { return; }

    inj.InjectedChannel      = BSP_BOARD_ADC_SO_V_CHANNEL;               /* PA3  SO_V → JDR2 */
    inj.InjectedRank         = ADC_INJECTED_RANK_2;
    if (HAL_ADCEx_InjectedConfigChannel(&p_mod->handle1, &inj) != HAL_OK) { return; }

    inj.InjectedChannel      = BSP_BOARD_ADC_SO_W_CHANNEL;               /* PA0  SO_W → JDR3 */
    inj.InjectedRank         = ADC_INJECTED_RANK_3;
    if (HAL_ADCEx_InjectedConfigChannel(&p_mod->handle1, &inj) != HAL_OK) { return; }

    /* ================= ADC2 主体（只有母线） ================= */
    p_mod->handle2.Instance                      = ADC2;
    p_mod->handle2.Init.ClockPrescaler           = ADC_CLOCK_SYNC_PCLK_DIV4;
    p_mod->handle2.Init.Resolution               = ADC_RESOLUTION_12B;
    p_mod->handle2.Init.DataAlign                = ADC_DATAALIGN_RIGHT;
    p_mod->handle2.Init.GainCompensation         = 0;
    p_mod->handle2.Init.ScanConvMode             = ADC_SCAN_DISABLE;     // 只有一个通道
    p_mod->handle2.Init.EOCSelection             = ADC_EOC_SINGLE_CONV;
    p_mod->handle2.Init.LowPowerAutoWait         = DISABLE;
    p_mod->handle2.Init.ContinuousConvMode       = ENABLE;
    p_mod->handle2.Init.NbrOfConversion          = BSP_ADC2_REG_NUM;
    p_mod->handle2.Init.DiscontinuousConvMode    = DISABLE;
    p_mod->handle2.Init.ExternalTrigConv         = ADC_SOFTWARE_START;
    p_mod->handle2.Init.ExternalTrigConvEdge     = ADC_EXTERNALTRIGCONVEDGE_NONE;
    p_mod->handle2.Init.DMAContinuousRequests    = ENABLE;
    p_mod->handle2.Init.Overrun                  = ADC_OVR_DATA_OVERWRITTEN;
    p_mod->handle2.Init.OversamplingMode         = DISABLE;
    if (HAL_ADC_Init(&p_mod->handle2) != HAL_OK)
    {
        return;
    }

    ch.SamplingTime = ADC_SAMPLETIME_247CYCLES_5;
    ch.Channel      = BSP_BOARD_ADC_BUS_CHANNEL;                         /* PC4  BUS_ADC */
    ch.Rank         = ADC_REGULAR_RANK_1;
    ch.SingleDiff   = ADC_SINGLE_ENDED;
    ch.OffsetNumber = ADC_OFFSET_NONE;
    ch.Offset       = 0;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    /* ================= DMA ================= */
    (VOID)_bsp_adc_dma_init(&p_mod->handle1, &p_mod->dma1, DMA1_Channel1, DMA_REQUEST_ADC1);
    (VOID)_bsp_adc_dma_init(&p_mod->handle2, &p_mod->dma2, DMA1_Channel2, DMA_REQUEST_ADC2);

    p_mod->phase_valid = FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_adc_start
* 功能描述 :   校准 → 启动两条规则组 DMA → 武装注入组
*
*               顺序不能乱：
*                 1) 校准必须在 ADC 使能之前做，否则结果不准；
*                 2) 规则组先跑起来（它顺带把 ADC 使能）；
*                 3) 注入组最后武装。注入组只等触发，不主动转换。
*
*               ⚠️ 调用时机：必须在 bsp_inverter_init() 之后。触发源是 TIM1_TRGO，
*                  TIM1 没起的话注入组永远等不到边沿，相电流会一直是 0。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_adc_start
(
    VOID
)
{
    BSP_ADC_MODULE_T *p_mod = _bsp_adc_get_module();

    if (p_mod->handle1.Instance != ADC1)
    {
        return;
    }

    (VOID)HAL_ADCEx_Calibration_Start(&p_mod->handle1, ADC_SINGLE_ENDED);
    (VOID)HAL_ADCEx_Calibration_Start(&p_mod->handle2, ADC_SINGLE_ENDED);

    (VOID)HAL_ADC_Start_DMA(&p_mod->handle1, (U32 *)p_mod->reg1, BSP_ADC1_REG_NUM);
    (VOID)HAL_ADC_Start_DMA(&p_mod->handle2, (U32 *)p_mod->reg2, BSP_ADC2_REG_NUM);

    /* 注入组只用查询方式，不开 JEOC 中断：
     * 结果会自动锁在 JDR1~3 里，读的时候拿到的永远是最近一次触发的结果，
     * 比中断更简单，也不会和 CubeMX 生成的 ADC1_2_IRQHandler 抢同一个句柄。 */
    (VOID)HAL_ADCEx_InjectedStart(&p_mod->handle1);
}

/********************************************************************************************
* 函数名称 :   bsp_adc_get_phase_raw
* 功能描述 :   读三相电流的同步采样值（原始码值）
*
*               数据流：TIM1 数到 ARR → CH4 的 OC4REF 翻高 → TRGO 出脉冲 → ADC1 注入组
*                       依次转换 CH2/CH4/CH1 → 结果锁进 JDR1/JDR2/JDR3 → JEOC 置位
*
*               本函数先看 JEOC 有没有新值，有新值才重读三个 JDR 并清标志。
*               ⚠️ 返回值语义与 bsp_adc.h 一致：TRUE = 至少完成过一次注入转换
*                  （此后恒为 TRUE），**不是**"本次有新样本"。数据新鲜度由 JEOC 决定，
*                  本层不对外暴露；JEOC 没置位时输出参数保持上一次的值不变，可以继续用。
*
* 输入参数 :   p_u / p_v / p_w: 目标指针，不允许为 NULL
* 输出参数 :   *p_u / *p_v / *p_w: 0 ~ 4095
* 返回值   :   TRUE = 至少采到过一次；FALSE = 还没采到过（TIM1 没跑或注入组没武装）
*********************************************************************************************/
BOOL bsp_adc_get_phase_raw
(
    OUT U16     *p_u,
    OUT U16     *p_v,
    OUT U16     *p_w
)
{
    BSP_ADC_MODULE_T *p_mod = _bsp_adc_get_module();
    BOOL fresh;

    if ((p_u == NULL) || (p_v == NULL) || (p_w == NULL))
    {
        return FALSE;
    }

    /* 注入组转换完成标志：硬件置位，读 JDR 后必须手动清，否则下次判断不到新样本 */
    fresh = (__HAL_ADC_GET_FLAG(&p_mod->handle1, ADC_FLAG_JEOC) != RESET) ? TRUE : FALSE;

    if (fresh == TRUE)
    {
        p_mod->phase_u = (U16)HAL_ADCEx_InjectedGetValue(&p_mod->handle1, ADC_INJECTED_RANK_1);
        p_mod->phase_v = (U16)HAL_ADCEx_InjectedGetValue(&p_mod->handle1, ADC_INJECTED_RANK_2);
        p_mod->phase_w = (U16)HAL_ADCEx_InjectedGetValue(&p_mod->handle1, ADC_INJECTED_RANK_3);

        __HAL_ADC_CLEAR_FLAG(&p_mod->handle1, ADC_FLAG_JEOC);

        p_mod->phase_valid = TRUE;
    }

    *p_u = p_mod->phase_u;
    *p_v = p_mod->phase_v;
    *p_w = p_mod->phase_w;

    return p_mod->phase_valid;
}

/********************************************************************************************
* 函数名称 :   bsp_adc_get_regular_raw
* 功能描述 :   读规则组的原始码值
*
*               DMA 在后台循环搬运，这里只是读内存，随时可读。
*               ⚠️ 三个通道不是原子更新的：可能读到"这轮的温度 + 上轮的安全电压"。
*                  对温度、母线这种慢变量完全无所谓；要严格一致就得在 DMA 完成中断里
*                  整块拷走，本层没有这么做，是为了避免和 CubeMX 的中断向量抢句柄。
*
* 输入参数 :   adc_index:     1 = ADC1，2 = ADC2
*               channel_index: 组内下标，见 BSP_ADC1_IDX_* / BSP_ADC2_IDX_*
* 输出参数 :   无
* 返回值   :   0 ~ 4095；下标越界返回 0
*********************************************************************************************/
U16 bsp_adc_get_regular_raw
(
    IN  U8      adc_index,
    IN  U8      channel_index
)
{
    BSP_ADC_MODULE_T *p_mod = _bsp_adc_get_module();

    if (adc_index == 1U)
    {
        if (channel_index >= BSP_ADC1_REG_NUM)
        {
            return 0U;
        }
        return p_mod->reg1[channel_index];
    }

    if (adc_index == 2U)
    {
        if (channel_index >= BSP_ADC2_REG_NUM)
        {
            return 0U;
        }
        return p_mod->reg2[channel_index];
    }

    return 0U;
}

/********************************************************************************************
* 函数名称 :   bsp_adc_raw_to_mv
* 功能描述 :   原始码值 → 毫伏
*               12 位 ADC 满量程码是 4095，不是 4096；用 4096 会整体偏低 0.02%，可忽略但别用错。
* 输入参数 :   raw: 0 ~ 4095
* 输出参数 :   无
* 返回值   :   引脚电压（mV）
*********************************************************************************************/
U32 bsp_adc_raw_to_mv
(
    IN  U16     raw
)
{
    return ((U32)raw * BSP_ADC_VREF_MV) / 4095U;
}

/********************************************************************************************
* 函数名称 :   bsp_adc_phase_raw_to_ma
* 功能描述 :   相电流原始码值 → 毫安（带符号）
*
*               换算链：
*                 引脚电压   V  = raw/4095 × VREF
*                 零点偏置   V0 = VREF/2        （DRV8353 CSA 的 VREF_DIV=1b，双向模式）
*                 采样电阻压降 ΔV = (V − V0) / GAIN
*                 相电流       I  = ΔV / R_shunt
*
*               代入本板参数（VREF=3.3V, GAIN=20, R=0.5mΩ）：
*                 灵敏度 = 20 × 0.5mΩ = 10 mV/A，即 1 mV 对应 100 mA。
*                 零点 1.65 V，正负各能摆 1.65 V → 理论 ±165 A。
*                 ⚠️ 但 3.3 V 是轨到轨的上限，运放输出到不了轨，实际可用量程更小，
*                    具体看 DRV8353 CSA 的输出摆幅规格（手册 §8.5 电气特性）。
*               这个量程对 N5065 明显偏大（真正工作时相电流也就几十安），
*               如果要把分辨率用满，应该把 CSA_GAIN 调到 40 或换更大的采样电阻，
*               但那要同时改 bsp_spi 里的寄存器配置和这里的宏，两处必须一致。
*
* 输入参数 :   raw: 0 ~ 4095
* 输出参数 :   无
* 返回值   :   相电流（mA），正 = 电流由 SHx 经采样电阻流向 GND
*********************************************************************************************/
S32 bsp_adc_phase_raw_to_ma
(
    IN  U16     raw
)
{
    S32 v_mv;
    S32 delta_mv;

    v_mv     = (S32)(((U32)raw * BSP_ADC_VREF_MV) / 4095U);
    delta_mv = v_mv - (S32)(BSP_ADC_VREF_MV / 2U);

    /* I(mA) = ΔV(mV) / (GAIN × R_shunt)
     *       = ΔV × 1e6 / (GAIN × R_shunt_uΩ)
     * 用整数算，避免引入浮点。分母 (20 × 500) = 10000，结果是 ΔV × 100。 */
    return (delta_mv * 1000000) / (S32)(BSP_ADC_CSA_GAIN * BSP_ADC_SHUNT_UOHM);
}

#ifdef _cplusplus
}
#endif
