/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_adc.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* DRV8161 电流放大器增益（V/V）。由 CSAGAIN 脚上的 30 kΩ 电阻决定（数据手册典型值 20）。
 * ⚠️ 换了这个电阻就必须同步改这里，否则电流标定全错，而且错得很隐蔽（差一个固定倍数）。 */
#define BSP_ADC_DRV_CSA_GAIN                                            (20U)

/* 超容侧采样电阻（µΩ）。R_SC_SHUNT = 0.5 mΩ 6 W 1%，四端开尔文接法。 */
#define BSP_ADC_SHUNT_UOHM                                              (500U)


/************************************数据类型定义************************************************/
typedef struct _BSP_ADC_MODULE_T_
{
    ADC_HandleTypeDef       handle1;                                    // ADC1：T1/T2/T3/T4/T8/NTC_DUMP/SO/BUS
    ADC_HandleTypeDef       handle2;                                    // ADC2：T5/T6/T7/NTC_MAIN/NTC_SC/TEST_FB
    DMA_HandleTypeDef       dma1;                                       // ADC1 → DMA1_Channel1
    DMA_HandleTypeDef       dma2;                                       // ADC2 → DMA1_Channel2
    U16                     reg1[BSP_ADC1_REG_NUM];                    // 规则组 DMA 目标缓冲
    U16                     reg2[BSP_ADC2_REG_NUM];
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
* 功能描述 :   把 14 个采样脚 + 1 个比较器专用脚配成模拟输入
*
*               为什么必须是 GPIO_MODE_ANALOG：
*                 模拟模式下施密特触发器输入被断开、上下拉被切断，输入漏电流降到 nA 级。
*                 如果留成浮空输入，数字输入缓冲会吃掉一部分电荷 —— 本板的抽头分压
*                 源阻抗约 8 kΩ、母线分压约 9.4 kΩ，漏电会把读数明显拉低。
*
*               为什么要单独处理 PA7：
*                 它是 DRV_SO 的第二分支，只接 COMP2_INP，不进任何 ADC 序列。
*                 但仍然要配成模拟输入 —— 浮空脚在比较器输入端会拾噪，
*                 而比较器的输出将来要接 TIM1 Break，一个噪声尖峰就是一次误刹车。
*
*               ⚠️ PB12（NTC_DUMP）与 PA6（AD_T7）在别的复用功能里是 TIM1_BKIN。
*                  这里把它们配成模拟输入，就等于**关掉了这两条刹车通路**，
*                  和 bsp_bridge.c 里 BreakState=DISABLE 的决定是一致的。
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

    gpio_init.Mode = GPIO_MODE_ANALOG;
    gpio_init.Pull = GPIO_NOPULL;

    /* ── GPIOA：T2/T3/T4（ADC1）在 PA1/PA2/PA3；T5/T6/T7（ADC2）在 PA4/PA5/PA6 ──
     * 再加 PA7 那个比较器专用分支，一次配完。T1 在 PA0，也在 GPIOA。 */
    gpio_init.Pin = BSP_BOARD_ADC_T1_PIN | BSP_BOARD_ADC_T2_PIN | BSP_BOARD_ADC_T3_PIN |
                    BSP_BOARD_ADC_T4_PIN | BSP_BOARD_ADC_T5_PIN | BSP_BOARD_ADC_T6_PIN |
                    BSP_BOARD_ADC_T7_PIN | BSP_BOARD_ADC_SO_ALT_PIN;
    HAL_GPIO_Init(GPIOA, &gpio_init);

    /* ── GPIOB：BUS / DRV_SO / NTC_DUMP / T8 / NTC_MAIN / TEST_FB ── */
    gpio_init.Pin = BSP_BOARD_ADC_BUS_PIN | BSP_BOARD_ADC_DRV_SO_PIN |
                    BSP_BOARD_ADC_NTC_DUMP_PIN | BSP_BOARD_ADC_T8_PIN |
                    BSP_BOARD_ADC_NTC_MAIN_PIN | BSP_BOARD_ADC_TEST_FB_PIN;
    HAL_GPIO_Init(GPIOB, &gpio_init);

    /* ── GPIOC：NTC_SC 在 PC4（只有 ADC2_IN5 一个映射）── */
    gpio_init.Pin = BSP_BOARD_ADC_NTC_SC_PIN;
    HAL_GPIO_Init(GPIOC, &gpio_init);
}

/********************************************************************************************
* 函数名称 :   _bsp_adc_dma_init
* 功能描述 :   给一个 ADC 挂一个循环 DMA 通道
*
*               G4 的 DMA 请求不直接连到通道，必须先经 DMAMUX 选源：
*                 ADC1 → DMA_REQUEST_ADC1
*                 ADC2 → DMA_REQUEST_ADC2
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
* 功能描述 :   配置 ADC1（8 通道）与 ADC2（6 通道）以及各自的 DMA
*
*               几个关键取舍：
*
*               1) 时钟用 ADC_CLOCK_SYNC_PCLK_DIV4 = 170/4 = 42.5 MHz。
*                  G4 的 ADC 最高 60 MHz，42.5 MHz 留了余量；同步时钟模式不需要额外
*                  配置 RCC 的 ADC 专用时钟源，最省事也最不容易配错。
*
*               2) 两个 ADC 都是纯规则组、连续转换、软件触发。
*                  本板没有注入组，也没有和 PWM 同步的采样 —— 所有量都是慢信号，
*                  硬把它们和 PWM 对齐只会增加复杂度，不会提高可信度。
*                  （DRV_SO 例外，但它本来就走硬件比较器，见 bsp_adc.h 的说明。）
*
*               3) Overrun 选"覆盖"：读慢了就丢旧值，不要让 OVR 把 ADC 卡死。
*                  对慢信号来说，丢一个旧样本远比停转强。
*
*               4) ScanConvMode 必须 ENABLE —— 一个 ADC 要轮询 6~8 个通道，
*                  关掉的话只会反复采第一个。
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
    ADC_ChannelConfTypeDef  ch    = {0};
    BSP_ADC_MODULE_T        *p_mod = _bsp_adc_get_module();

    _bsp_adc_gpio_init();

    __HAL_RCC_ADC12_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* ⚠️ DMAMUX1 的时钟必须单独开，这一行漏了会出很典型的"静默故障"：
     *    G4 的 DMA 请求先经 DMAMUX 选源（配 DMAMUX_CxCR 的 DMAREQ_ID）。
     *    HAL_DMA_Init() 会去写这个寄存器，但 HAL **不会**帮你开 DMAMUX 的时钟
     *    （stm32g4xx_hal_dma.c 里写着 "Prior to HAL_DMA_Init the peripheral clock
     *    shall be enabled for both DMA & DMAMUX"）。
     *    时钟没开时写寄存器不报错也不生效，现象是 DMA 一次都不搬：ADC 在转、
     *    DMA 标志永远不动、缓冲区永远全是 0，而代码看起来完全正常。 */
    __HAL_RCC_DMAMUX1_CLK_ENABLE();

    /* ================= ADC1 主体（8 通道） ================= */
    p_mod->handle1.Instance                      = ADC1;
    p_mod->handle1.Init.ClockPrescaler           = ADC_CLOCK_SYNC_PCLK_DIV4;
    p_mod->handle1.Init.Resolution               = ADC_RESOLUTION_12B;
    p_mod->handle1.Init.DataAlign                = ADC_DATAALIGN_RIGHT;
    p_mod->handle1.Init.GainCompensation         = 0;
    p_mod->handle1.Init.ScanConvMode             = ADC_SCAN_ENABLE;
    p_mod->handle1.Init.EOCSelection             = ADC_EOC_SINGLE_CONV;
    p_mod->handle1.Init.LowPowerAutoWait         = DISABLE;
    p_mod->handle1.Init.ContinuousConvMode       = ENABLE;
    p_mod->handle1.Init.NbrOfConversion          = BSP_ADC1_REG_NUM;
    p_mod->handle1.Init.DiscontinuousConvMode    = DISABLE;
    p_mod->handle1.Init.ExternalTrigConv         = ADC_SOFTWARE_START;
    p_mod->handle1.Init.ExternalTrigConvEdge     = ADC_EXTERNALTRIGCONVEDGE_NONE;
    p_mod->handle1.Init.DMAContinuousRequests    = ENABLE;
    p_mod->handle1.Init.Overrun                  = ADC_OVR_DATA_OVERWRITTEN;
    p_mod->handle1.Init.OversamplingMode         = DISABLE;
    if (HAL_ADC_Init(&p_mod->handle1) != HAL_OK)
    {
        return;
    }

    /* 规则组：顺序 = 缓冲区下标，逐条对应 bsp_adc.h 里的 BSP_ADC1_IDX_* */
    ch.SamplingTime = ADC_SAMPLETIME_247CYCLES_5;
    ch.SingleDiff   = ADC_SINGLE_ENDED;
    ch.OffsetNumber = ADC_OFFSET_NONE;
    ch.Offset       = 0;

    ch.Channel = BSP_BOARD_ADC_T1_CHANNEL;                              /* PA0   AD_T1       → idx 0 */
    ch.Rank    = ADC_REGULAR_RANK_1;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_T2_CHANNEL;                              /* PA1   AD_T2       → idx 1 */
    ch.Rank    = ADC_REGULAR_RANK_2;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_T3_CHANNEL;                              /* PA2   AD_T3       → idx 2 */
    ch.Rank    = ADC_REGULAR_RANK_3;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_T4_CHANNEL;                              /* PA3   AD_T4       → idx 3 */
    ch.Rank    = ADC_REGULAR_RANK_4;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_T8_CHANNEL;                              /* PB14  SC_SUM/T8   → idx 4 */
    ch.Rank    = ADC_REGULAR_RANK_5;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_NTC_DUMP_CHANNEL;                        /* PB12  NTC_DUMP    → idx 5 */
    ch.Rank    = ADC_REGULAR_RANK_6;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_DRV_SO_CHANNEL;                          /* PB1   DRV_SO      → idx 6 */
    ch.Rank    = ADC_REGULAR_RANK_7;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_BUS_CHANNEL;                             /* PB0   BUS_ADC     → idx 7 */
    ch.Rank    = ADC_REGULAR_RANK_8;
    if (HAL_ADC_ConfigChannel(&p_mod->handle1, &ch) != HAL_OK) { return; }

    /* ================= ADC2 主体（6 通道） ================= */
    p_mod->handle2.Instance                      = ADC2;
    p_mod->handle2.Init.ClockPrescaler           = ADC_CLOCK_SYNC_PCLK_DIV4;
    p_mod->handle2.Init.Resolution               = ADC_RESOLUTION_12B;
    p_mod->handle2.Init.DataAlign                = ADC_DATAALIGN_RIGHT;
    p_mod->handle2.Init.GainCompensation         = 0;
    p_mod->handle2.Init.ScanConvMode             = ADC_SCAN_ENABLE;
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

    ch.Channel = BSP_BOARD_ADC_T5_CHANNEL;                              /* PA4   AD_T5       → idx 0 */
    ch.Rank    = ADC_REGULAR_RANK_1;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_T6_CHANNEL;                              /* PA5   AD_T6       → idx 1 */
    ch.Rank    = ADC_REGULAR_RANK_2;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_T7_CHANNEL;                              /* PA6   AD_T7       → idx 2 */
    ch.Rank    = ADC_REGULAR_RANK_3;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_NTC_MAIN_CHANNEL;                        /* PB11  NTC_MAIN   → idx 3 */
    ch.Rank    = ADC_REGULAR_RANK_4;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_NTC_SC_CHANNEL;                          /* PC4   NTC_SC     → idx 4 */
    ch.Rank    = ADC_REGULAR_RANK_5;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    ch.Channel = BSP_BOARD_ADC_TEST_FB_CHANNEL;                         /* PB15  TEST_FB    → idx 5 */
    ch.Rank    = ADC_REGULAR_RANK_6;
    if (HAL_ADC_ConfigChannel(&p_mod->handle2, &ch) != HAL_OK) { return; }

    /* ================= DMA ================= */
    (VOID)_bsp_adc_dma_init(&p_mod->handle1, &p_mod->dma1, DMA1_Channel1, DMA_REQUEST_ADC1);
    (VOID)_bsp_adc_dma_init(&p_mod->handle2, &p_mod->dma2, DMA1_Channel2, DMA_REQUEST_ADC2);
}

/********************************************************************************************
* 函数名称 :   bsp_adc_start
* 功能描述 :   校准两个 ADC，然后启动两条规则组 DMA
*
*               顺序不能乱：校准必须在 ADC 使能之前做，否则结果不准；
*               HAL_ADC_Start_DMA() 内部会把 ADC 使能，所以两次调用有先后关系。
*
*               ⚠️ 本函数不依赖 TIM1 —— 本板没有注入组、没有外部触发，
*                  所以调用时机只要在 bsp_adc_init() 之后即可，不像 foc_driver
*                  那样必须等逆变器的 TRGO 先起来。
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
}

/********************************************************************************************
* 函数名称 :   bsp_adc_get_raw
* 功能描述 :   读规则组的原始码值
*
*               DMA 在后台循环搬运，这里只是读内存，随时可读。
*               注意两个 ADC 之间不是原子更新的：一次调用拿到的 ADC1 值与 ADC2 值
*               可能来自不同时刻（差最多 50 µs）。对温度、抽头、母线都无所谓；
*               但**不要**用这个去做"Vcell8 = Vtap8 − Vtap7"以外的强一致运算。
*
* 输入参数 :   adc_index:     1 = ADC1，2 = ADC2
*               channel_index: 组内下标，见 BSP_ADC1_IDX_* / BSP_ADC2_IDX_*
* 输出参数 :   无
* 返回值   :   0 ~ 4095；下标越界返回 0
*********************************************************************************************/
U16 bsp_adc_get_raw
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
*               12 位 ADC 满量程码是 4095，不是 4096；用 4096 会整体偏低 0.02%，
*               可以忽略，但别用错。
* 输入参数 :   raw: 0 ~ 4095
* 输出参数 :   无
* 返回值   :   引脚电压（mV）
*********************************************************************************************/
U32 bsp_adc_raw_to_mv
(
    IN  U16     raw
)
{
    return ((U32)raw * BSP_BOARD_ADC_VREF_MV) / 4095U;
}

/********************************************************************************************
* 函数名称 :   bsp_adc_get_so_ma
* 功能描述 :   DRV_SO 通道 → 带符号电流（mA）
*
*               换算链：
*                 引脚电压   V  = raw/4095 × VREF
*                 零点偏置   V0 = VREF/2        （CSAREF=3.3 V，双向模式，零电流约 1.65 V）
*                 采样电阻压降 ΔV = (V − V0) / GCSA
*                 电流         I  = ΔV / R_shunt
*
*               代入本板参数（VREF=3.3 V, GCSA=20, R=0.5 mΩ）：
*                 灵敏度 = 20 × 0.5 mΩ = 10 mV/A，即 1 mV 对应 100 mA。
*                 零点 1.65 V，正负各能摆 1.65 V → 纸面 ±165 A。
*
*               ⚠️ 三个会让这个数不可信的因素，做保护逻辑时都必须算进去：
*                  1) TI 给的 SO 线性摆幅是 0.25 V ~ CSAREF−0.25 V，
*                     所以可用量程实际约 ±140 A，不是 ±165 A；
*                  2) 输入偏移 ±1.94 mV（25 °C，gain=20）等效约 ±3.88 A；
*                  3) 增益误差、shunt 公差/温漂、参考误差还会叠加。
*               结论：这个值是**测量**，不是跳闸门限。80 A 保护不要建立在它上面。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   带符号电流（mA），正 = 超容放电
*********************************************************************************************/
S32 bsp_adc_get_so_ma
(
    VOID
)
{
    U16 raw;
    S32 v_mv;
    S32 delta_mv;

    raw      = bsp_adc_get_raw(1U, BSP_ADC1_IDX_DRV_SO);
    v_mv     = (S32)(((U32)raw * BSP_BOARD_ADC_VREF_MV) / 4095U);
    delta_mv = v_mv - (S32)(BSP_BOARD_ADC_VREF_MV / 2U);

    /* I(mA) = ΔV(mV) / (GAIN × R_shunt)
     *       = ΔV × 1e6 / (GAIN × R_shunt_uΩ)
     * 本板分母 (20 × 500) = 10000 → 结果是 ΔV × 100，即 1 mV = 100 mA。 */
    return (delta_mv * 1000000) / (S32)(BSP_ADC_DRV_CSA_GAIN * BSP_ADC_SHUNT_UOHM);
}

#ifdef _cplusplus
}
#endif
