/************************************包含头文件*************************************************/
#include <string.h>
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_board.h"
#include "bsp_can.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BSP_CAN_MODULE_T_
{
    FDCAN_HandleTypeDef handle;                                         // FDCAN1 句柄
    BOOL                started;                                        // 是否已启动
}BSP_CAN_MODULE_T;


/************************************静态函数声明************************************************/
static BSP_CAN_MODULE_T * _bsp_can_get_module
(
    VOID
);


/************************************静态变量定义************************************************/
static BSP_CAN_MODULE_T s_bsp_can_module =
{
    {0},
    FALSE
};


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bsp_can_get_module
* 功能描述 :   获取 CAN 模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   CAN 模块实例指针
*********************************************************************************************/
static BSP_CAN_MODULE_T * _bsp_can_get_module
(
    VOID
)
{
    return &s_bsp_can_module;
}

/********************************************************************************************
* 函数名称 :   bsp_can_init
* 功能描述 :   初始化 FDCAN1：PA11 为 RX、PA12 为 TX（AF9），经典帧，正常模式
*              滤波器全通，接收中断挂到 FIFO0
*
*               ⚠️ 这里必须先配内核时钟源，原因见下面第一段代码的注释。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_can_init
(
    VOID
)
{
    GPIO_InitTypeDef        gpio_init  = {0};
    FDCAN_FilterTypeDef     filter     = {0};
    RCC_PeriphCLKInitTypeDef periph_clk = {0};
    BSP_CAN_MODULE_T        *p_mod     = _bsp_can_get_module();

    /* ── 0. FDCAN 内核时钟源（这一行不能省）────────────────────────────────
     * RM0440：RCC_CCIPR.FDCANSEL 复位值是 00b，含义是 **HSE**。
     * 本工程 SystemClock_Config() 只用 HSI（16 MHz / 4 × 85 / 2 = 170 MHz），
     * HSE 从来没打开，所以复位后 fdcan_ker_ck 就是 0。
     *
     * 后果特别隐蔽：FDCAN 的寄存器接口挂在 APB1 上，是活的，HAL_FDCAN_Init()
     * 照样能把 CCCR/NBTP 写进去并返回 HAL_OK；但协议引擎没有时钟，
     * 帧发不出去、也收不到，示波器量 TX 脚一动不动。
     * 现象是"初始化成功但完全不通"。
     *
     * 选 PCLK1（= 170 MHz）而不是 PLLQ：两者此处同频，但 PLLQ 是 PLL 的固定分频，
     * 不受 APB1 分频影响；选 PCLK1 只是为了和 bsp_can.h 里的位时序算式
     * （170 MHz / 20 / 17 = 500 kbit/s）保持同一基准，便于核对。 */
    periph_clk.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    periph_clk.FdcanClockSelection  = RCC_FDCANCLKSOURCE_PCLK1;

    /* 时钟配不上就别往下走：继续初始化只会得到一个"看起来正常"的死总线 */
    if (HAL_RCCEx_PeriphCLKConfig(&periph_clk) != HAL_OK)
    {
        return;
    }

    /* 时钟 */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_FDCAN_CLK_ENABLE();

    /* PA11 → FDCAN1_RX, PA12 → FDCAN1_TX, AF9
     * 注意核心板上这两个脚同时接 Type-C 的 D−/D+，用 CAN 时别同时插 USB 数据线。 */
    gpio_init.Pin       = BSP_BOARD_CAN_RX_PIN | BSP_BOARD_CAN_TX_PIN;
    gpio_init.Mode      = GPIO_MODE_AF_PP;
    gpio_init.Pull      = GPIO_NOPULL;
    gpio_init.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio_init.Alternate = BSP_BOARD_CAN_AF;
    HAL_GPIO_Init(BSP_BOARD_CAN_RX_PORT, &gpio_init);

    /* FDCAN 基础配置：500 kbit/s，经典帧 */
    p_mod->handle.Instance                  = FDCAN1;
    p_mod->handle.Init.ClockDivider         = FDCAN_CLOCK_DIV1;
    p_mod->handle.Init.FrameFormat          = FDCAN_FRAME_CLASSIC;
    p_mod->handle.Init.Mode                 = FDCAN_MODE_NORMAL;
    p_mod->handle.Init.AutoRetransmission   = ENABLE;
    p_mod->handle.Init.TransmitPause        = DISABLE;
    p_mod->handle.Init.ProtocolException    = DISABLE;

    p_mod->handle.Init.NominalPrescaler     = BSP_CAN_NOMINAL_PRESCALER;
    p_mod->handle.Init.NominalSyncJumpWidth = BSP_CAN_NOMINAL_SJW;
    p_mod->handle.Init.NominalTimeSeg1      = BSP_CAN_NOMINAL_TIME_SEG1;
    p_mod->handle.Init.NominalTimeSeg2      = BSP_CAN_NOMINAL_TIME_SEG2;

    /* 数据域参数在经典帧下不使用，仍按 HAL 要求填入合法值 */
    p_mod->handle.Init.DataPrescaler        = 1;
    p_mod->handle.Init.DataSyncJumpWidth    = 1;
    p_mod->handle.Init.DataTimeSeg1         = 1;
    p_mod->handle.Init.DataTimeSeg2         = 1;

    p_mod->handle.Init.StdFiltersNbr        = 1;
    p_mod->handle.Init.ExtFiltersNbr        = 0;
    p_mod->handle.Init.TxFifoQueueMode      = FDCAN_TX_FIFO_OPERATION;
    if (HAL_FDCAN_Init(&p_mod->handle) != HAL_OK)
    {
        return;
    }

    /* 标准帧滤波器：全通，进 FIFO0 */
    filter.IdType       = FDCAN_STANDARD_ID;
    filter.FilterIndex  = 0;
    filter.FilterType   = FDCAN_FILTER_MASK;
    filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    filter.FilterID1    = 0x00000000;
    filter.FilterID2    = 0x00000000;
    if (HAL_FDCAN_ConfigFilter(&p_mod->handle, &filter) != HAL_OK)
    {
        return;
    }

    /* 全局：不匹配的帧一律丢弃，远程帧也丢弃（本板不响应 RTR） */
    (VOID)HAL_FDCAN_ConfigGlobalFilter(&p_mod->handle,
                                       FDCAN_REJECT,
                                       FDCAN_REJECT,
                                       FDCAN_REJECT_REMOTE,
                                       FDCAN_REJECT_REMOTE);

    p_mod->started = FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_can_start
* 功能描述 :   启动 FDCAN 并打开接收中断
*
*               ⚠️ 与 foc_driver 的区别：本板收发器没有 STB 脚，所以这里**不需要**
*                  （也没有）"把收发器从待机切出来"这一步。少了一个可能忘记做的动作。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bsp_can_start
(
    VOID
)
{
    BSP_CAN_MODULE_T *p_mod = _bsp_can_get_module();

    if (p_mod->started == TRUE)
    {
        return;
    }

    if (HAL_FDCAN_Start(&p_mod->handle) != HAL_OK)
    {
        return;
    }

    /* ── NVIC ────────────────────────────────────────────────────────────────
     * 这一步不能省。本工程的 stm32g4xx_it.c 里**一个外设中断都没有**
     * （只有 Cortex 内核那几个），所以如果只调 HAL_FDCAN_ActivateNotification()，
     * 中断标志会在 FDCAN 里堆着，NVIC 那头永远是关的，RX 回调一次都不会进 ——
     * 现象是"发送正常、接收死寂"。
     *
     * 中断线怎么定的：HAL_FDCAN_ActivateNotification() 会查 ILS 寄存器的对应位，
     * 该位为 0 就把这个事件挂到中断线 0（FDCAN1_IT0_IRQn），为 1 挂到线 1。
     * ILS 复位值是 0，本工程也没改过，所以 RX FIFO0 新报文走的是 **IT0**。
     *
     * 优先级定 5：本工程没有硬实时电流环（半桥是 BL 层按周期算的，
     * 不靠中断），所以 5 只是给 HAL 的 SysTick 让路，具体数值不重要。 */
    HAL_NVIC_SetPriority(FDCAN1_IT0_IRQn, 5U, 0U);
    HAL_NVIC_EnableIRQ(FDCAN1_IT0_IRQn);

    if (HAL_FDCAN_ActivateNotification(&p_mod->handle,
                                       FDCAN_IT_RX_FIFO0_NEW_MESSAGE,
                                       0) != HAL_OK)
    {
        return;
    }

    p_mod->started = TRUE;
}

/********************************************************************************************
* 函数名称 :   bsp_can_send
* 功能描述 :   把一帧经典 CAN 报文放入发送 FIFO
* 输入参数 :   id    : 标识符
*               is_ext: TRUE 使用 29 位扩展帧
*               p_data: 数据指针，len 为 0 时可为 NULL
*               len   : 数据长度 0~8
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS: 已入队
*              TYPE_RET_PARAMETER_ERROR: 参数非法
*              TYPE_RET_FAIL: 硬件拒绝入队
*********************************************************************************************/
TYPE_RET_E bsp_can_send
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
)
{
    FDCAN_TxHeaderTypeDef   tx_header = {0};
    BSP_CAN_MODULE_T        *p_mod    = _bsp_can_get_module();
    U8                      payload[8] = {0};

    if (p_mod->started != TRUE)
    {
        return TYPE_RET_FAIL;
    }
    if (len > 8)
    {
        return TYPE_RET_PARAMETER_ERROR;
    }
    if ((len > 0) && (p_data == NULL))
    {
        return TYPE_RET_PARAMETER_ERROR;
    }

    if (len > 0)
    {
        (VOID)memcpy(payload, p_data, len);
    }

    tx_header.Identifier          = (is_ext == TRUE) ? (id & 0x1FFFFFFFU) : (id & 0x7FFU);
    tx_header.IdType              = (is_ext == TRUE) ? FDCAN_EXTENDED_ID : FDCAN_STANDARD_ID;
    tx_header.TxFrameType         = FDCAN_DATA_FRAME;
    tx_header.DataLength          = len;
    tx_header.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
    tx_header.BitRateSwitch       = FDCAN_BRS_OFF;
    tx_header.FDFormat            = FDCAN_CLASSIC_CAN;
    tx_header.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
    tx_header.MessageMarker       = 0;

    if (HAL_FDCAN_AddMessageToTxFifoQ(&p_mod->handle, &tx_header, payload) != HAL_OK)
    {
        return TYPE_RET_FAIL;
    }

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   HAL_FDCAN_RxFifo0Callback
* 功能描述 :   HAL 接收回调：把 FIFO0 中的报文取出并转交 BSP 回调分发
*              运行在中断上下文
* 输入参数 :   hfdcan     : 句柄
*               RxFifo0ITs : 中断标志
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID HAL_FDCAN_RxFifo0Callback
(
    FDCAN_HandleTypeDef *hfdcan,
    U32                 RxFifo0ITs
)
{
    FDCAN_RxHeaderTypeDef   rx_header = {0};
    U8                      data[8]   = {0};

    if ((hfdcan == NULL) || (hfdcan->Instance != FDCAN1))
    {
        return;
    }

    if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0)
    {
        return;
    }

    if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rx_header, data) != HAL_OK)
    {
        return;
    }

    /* 只上报数据帧，远程帧已在全局滤波里被拒绝 */
    if (rx_header.RxFrameType != FDCAN_DATA_FRAME)
    {
        return;
    }

    bsp_callback_can_rx(rx_header.Identifier,
                        (rx_header.IdType == FDCAN_EXTENDED_ID) ? TRUE : FALSE,
                        data,
                        (U8)rx_header.DataLength);
}

/********************************************************************************************
* 函数名称 :   FDCAN1_IT0_IRQHandler
* 功能描述 :   FDCAN1 中断线 0 的入口（覆盖启动文件里 .weak 的 Default_Handler）
*
*              为什么写在这个文件而不是 stm32g4xx_it.c：
*                句柄是 s_bsp_can_module.handle，静态的，外面拿不到。
*                把入口放在同一文件里，既不用把句柄暴露出去，也避免别人
*                在 it.c 里再写一个同名函数造成重复定义。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID FDCAN1_IT0_IRQHandler
(
    VOID
)
{
    HAL_FDCAN_IRQHandler(&s_bsp_can_module.handle);
}

#ifdef _cplusplus
}
#endif
