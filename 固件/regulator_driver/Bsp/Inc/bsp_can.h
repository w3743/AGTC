#ifndef __BSP_CAN_H__
#define __BSP_CAN_H__

#include "type.h"
#include "bsp_callback.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * FDCAN1 —— 板间总线（稳压板 / 两块 FOC 板 / 主控都在同一条 CAN 上）
 *
 * 引脚（依据 接口与采样核对.md + DS12589 Rev6 Table 12）：
 *   FDCAN1_RX  PA11  AF9   → TCAN332DR RXD（第 4 脚）
 *   FDCAN1_TX  PA12  AF9   → TCAN332DR TXD（第 1 脚）
 *
 * ⚠️ 本板**没有** CAN 收发器待机控制线。
 *    TCAN332 只有 8 个脚：TXD/GND/VCC/RXD/NC/CANL/CANH/NC，没有 STB 也没有 SHDN。
 *    所以 foc_driver 里那个 bsp_can_start() 顺手把 STB 拉低的动作，在本工程不存在；
 *    要静默总线只能 HAL_FDCAN_Stop() 停整个外设。
 *
 * ⚠️ PA11/PA12 在核心板上与 Type-C 的 D−/D+ 硬连。
 *    接 CAN 时不能同时插 USB 数据线，否则 USB 收发器会成为总线的额外负载；
 *    调试期间要么走 SWD，要么拔掉 CAN 再插 USB。
 *
 * ⚠️ 内核时钟源必须显式配置，见 bsp_can.c 里 bsp_can_init() 的第一段：
 *    RM0440 规定 RCC_CCIPR.FDCANSEL 复位值是 00b = HSE，而本工程只用 HSI（无外部晶振），
 *    不配置的话 fdcan_ker_ck = 0，HAL_FDCAN_Init() 可能照样返回 HAL_OK，
 *    但总线上一个位都发不出去 —— 这种"初始化成功但完全不通"最难查。
 ********************************************************************************************/

/************************************Macro***************************************************/
/* 位时序参数基准：FDCAN 内核时钟 = PCLK1 = 170 MHz，目标 500 kbit/s
 * 170 MHz / 500 kbit/s = 340 = Prescaler(20) × (1 + TSeg1(14) + TSeg2(2))
 * 采样点 = (1 + 14) / 17 = 88.2%（CAN 推荐 87.5%），SJW = 2
 *
 * ⚠️ 这两个工程（foc_driver / regulator_driver）用的是**同一组**参数，
 *    因为它们挂同一条总线。改一处必须同时改另一处，否则两块板通不了。
 * ⚠️ 内核时钟选的是 PCLK1，所以**改 APB1 分频会直接改掉波特率**；
 *    若哪天要动 APB1 分频，把 bsp_can.c 里的时钟源换成 PLLQ（同频但独立）即可。 */
#define BSP_CAN_NOMINAL_PRESCALER                                       (20U)
#define BSP_CAN_NOMINAL_TIME_SEG1                                       (14U)
#define BSP_CAN_NOMINAL_TIME_SEG2                                       (2U)
#define BSP_CAN_NOMINAL_SJW                                             (2U)

/* ── 本层能力边界 ──────────────────────────────────────────────────────────
 * 只做：初始化 / 启动 / 发帧 / 收帧回调分发（硬件全通过滤波，中断进 FIFO0）。
 * 不做：bus-off 与 error-passive 的检测与恢复，也不做发送重试。
 *   总线持续错误时 FDCAN 会自行进入 bus-off，本层没有周期任务可以自动拉回；
 *   而且 bsp_can_start() 里的 started 判断会拦住重复启动，恢复前需要先把
 *   started 复位。需要恢复能力时，由 BL 层读 PSR / 错误计数器并补一个
 *   "停止-重启"接口，BSP 侧再提供对应函数。 */

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_can_init
* Descriptions  :   Init FDCAN1 on PA11/PA12 (AF9), classic frame, normal mode.
*                   Sets the FDCAN kernel clock source explicitly (PCLK1).
*                   The peripheral is configured but NOT started.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_can_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_can_start
* Descriptions  :   Start the peripheral, enable the RX FIFO0 interrupt on IT0 and register
*                   the notification. There is no transceiver standby pin to release.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_can_start
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_can_send
* Descriptions  :   Queue one classic CAN frame into the TX FIFO
* Input         :
*                   id    : identifier
*                   is_ext: TRUE for 29-bit extended id
*                   p_data: payload, may be NULL when len is 0
*                   len   : payload length 0~8
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E bsp_can_send
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_CAN_H__ */
