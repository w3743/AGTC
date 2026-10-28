#ifndef __DL_CAN_H__
#define __DL_CAN_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * CAN 数据层
 *
 * 分层意图：把"中断里收报文"和"业务上解析报文"彻底分开
 *
 *   FDCAN RX 中断
 *        |
 *        v
 *   bsp_callback_can_rx()        <- BSP 层回调，只做拷贝入交接槽
 *        |
 *        v
 *   dl_can_poll()                <- 主循环里取走（本层实现）
 *        |
 *        v
 *   BL 自己解析                  <- 业务语义
 *
 * 这样设计的原因：
 *   - 中断里绝不做浮点运算与协议解析，避免抖动控制节拍
 *   - 交接槽是"单格覆盖"而不是队列：CAN 帧最长约 130 µs，主循环是毫秒级，
 *     队列在这里没有实际收益，反而要处理溢出与计数；宁可丢旧帧也不要无界队列
 *   - BL 只面对"已验证的完整报文"，不需要处理半包/溢出细节
 *
 * ⚠️ 本模块的接口与 regulator_driver 工程**完全一致**（同签名、同语义），
 *    两个工程共用同一颗 G431，出问题时可以拿另一边的代码对照着看。
 ********************************************************************************************/

/************************************Macro***************************************************/
/* 报文 ID —— **全部是占位符**。
 *
 * 依据：整车 CAN 协议表还没有冻结。原理图准备文档里只写了"四块电力板 + 主控都在
 * 同一条总线上"，没有给出任何 ID 分配。这里先按"功能分段的常见习惯"取三个不冲突的值，
 * 让整条链路（采样 → 打包 → 总线）在协议定稿前就能联调通。
 * 协议表一旦定稿，只改这三个宏 + dl_can_send_status() 里的字段布局，其余代码不用动。
 *
 * 之所以不编成一张大表：编出来的东西没人会去核对，反而会被人当成"已经定好了"。 */
#define DL_CAN_ID_STATUS                                                (0x180U)   /* 本板 → 主控，周期状态 */
#define DL_CAN_ID_COMMAND                                               (0x200U)   /* 主控 → 本板，控制指令 */
#define DL_CAN_ID_FAULT                                                 (0x280U)   /* 本板 → 主控，故障事件 */

/************************************Type****************************************************/
typedef struct _DL_CAN_FRAME_T_
{
    U32     id;                                                         /* 标识符 */
    BOOL    is_ext;                                                     /* 扩展帧标志 */
    U8      len;                                                        /* 数据长度 0~8 */
    U8      data[8];                                                    /* 数据 */
}DL_CAN_FRAME_T;

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   dl_can_init
* Descriptions  :   Init CAN data layer: clear the hand-over slot and register the BSP rx hook.
*                   Does NOT start the controller - entering the bus is an upper-layer decision
*                   (bsp_can_start()).
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS
*********************************************************************************************/
extern TYPE_RET_E dl_can_init
(
    VOID
);

/********************************************************************************************
* Function name :   dl_can_poll
* Descriptions  :   Take one frame out of the hand-over slot
*                   Call from the main loop, never from an interrupt.
*                   Taking the frame clears the slot, so the next one can be written in.
* Input         :
*                   none
* Output        :
*                   p_frame: filled when a frame is available
* Return        :
*                   TRUE  : one frame taken
*                   FALSE : nothing pending
*********************************************************************************************/
extern BOOL dl_can_poll
(
    OUT DL_CAN_FRAME_T  *p_frame
);

/********************************************************************************************
* Function name :   dl_can_send
* Descriptions  :   Send one frame, thin wrapper over bsp_can_send
* Input         :
*                   id    : identifier
*                   is_ext: extended flag
*                   p_data: payload
*                   len   : 0 ~ 8
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_can_send
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
);

/********************************************************************************************
* Function name :   dl_can_send_status
* Descriptions  :   Pack and send the periodic status frame
*
*                   ⚠️ **字段布局是占位的**，等整车 CAN 协议表定稿后必须重写函数体。
*                      目前按"上电就能看出板子在干什么"编排，8 字节刚好用完：
*
*                        [0..1] 母线电压   10 mV/LSB，0 ~ 655 V
*                        [2..3] U 相电流   100 mA/LSB，带符号，±3276 A
*                        [4..5] V 相电流   同上
*                        [6]    MOS 温度   直接存摄氏度（s8），-127 ~ +127，-128 = 无效
*                        [7]    状态|霍尔  低 4 位 = DL_MOTOR_STATE_E，高 4 位 = 霍尔码
*
*                      W 相不单独占两个字节：三相电流之和恒为 0（星形、无中线），
*                      Iw = -(Iu + Iv) 就够了。省下来的字节给状态和霍尔码，
*                      现场排查相序时不用再挂调试器。
*
*                      单位选 10 mV/LSB 与 100 mA/LSB，是因为 2 字节能覆盖的范围
*                      （655 V / ±3276 A）远超本板量程（62.7 V / ±165 A），
*                      怎么改增益都不会在上面溢出。
* Input         :
*                   bus_mv      : 母线电压，mV
*                   iu_ma       : U 相电流，mA，带符号
*                   iv_ma       : V 相电流，mA，带符号
*                   temp_mos_c  : MOS 温度，degC
*                   state       : 电机状态
*                   hall_code   : 霍尔码 0 ~ 7
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E dl_can_send_status
(
    IN  U32     bus_mv,
    IN  S32     iu_ma,
    IN  S32     iv_ma,
    IN  F32     temp_mos_c,
    IN  U8      state,
    IN  U8      hall_code
);

#ifdef _cplusplus
}
#endif

#endif
