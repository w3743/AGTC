#ifndef __BSP_CAN_H__
#define __BSP_CAN_H__

#include "type.h"
#include "bsp_callback.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * FDCAN1 —— 板间总线（四块电力板 + 主控都在这一条总线上）
 *
 * 引脚（依据 当前原理图连接.json + DS12589 Rev6 Table 12）：
 *   FDCAN1_RX  PA11  AF9   → TCAN1042 RXD（第 4 脚）
 *   FDCAN1_TX  PA12  AF9   → TCAN1042 TXD（第 1 脚）
 *   CAN_STB    PB5   GPIO  → TCAN1042 STB（第 8 脚），低=正常 / 高=待机
 *
 * ⚠️ 为什么是 PA11/PA12 而不是常见的 PB8/PB9：
 *    PB8 在 G431 上同时是 BOOT0。water_car_motor 那块板用了 PB8 做 RX，TCAN1042 的 RXD 是**推挽输出**，
 *    会把 PB8 在复位期间顶成高电平 → BOOT0 被读成 1 → 上电进 System Memory 而不是 Flash。
 *    FOC 板必须用 PA11/PA12（同一组 FDCAN1，AF9），从根上绕开这个坑。
 *
 * ⚠️ 内核时钟源必须显式配置，见 bsp_can.c 里 bsp_can_init() 的第一段：
 *    RM0440 规定 RCC_CCIPR.FDCANSEL 复位值为 00b = HSE，而本工程只用 HSI（无外部晶振），
 *    不配置的话 fdcan_ker_ck = 0，HAL_FDCAN_Init() 可能照样返回 HAL_OK，
 *    但总线上一个位都发不出去 —— 这种"初始化成功但完全不通"最难查。
 ********************************************************************************************/

/************************************Macro***************************************************/
/* 位时序参数基准：FDCAN 内核时钟 = PCLK1 = 170 MHz，目标 500 kbit/s
 * 170 MHz / 500 kbit/s = 340 = Prescaler(20) x (1 + TSeg1(14) + TSeg2(2))
 * 采样点 = (1 + 14) / 17 = 88.2%（CAN 推荐 87.5%），SJW = 2
 * 若后续修改系统时钟或波特率，只需调整这三个宏。
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

/************************************Fcunction***********************************************/
/********************************************************************************************
* Function name :   bsp_can_init
* Descriptions  :   Init FDCAN1 on PA11/PA12 (AF9), classic frame, normal mode
*                   Sets the FDCAN kernel clock source explicitly (PCLK1)
*                   Transceiver STB is left in standby until bsp_can_start()
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
* Descriptions  :   Start FDCAN and release transceiver from standby
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
* Descriptions  :   Queue a classic CAN frame into TX FIFO
* Input         :
*                   id    : 11-bit standard or 29-bit extended identifier
*                   is_ext: TRUE for extended
*                   p_data: payload buffer, may be NULL when len is 0
*                   len   : 0 ~ 8
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

#endif
