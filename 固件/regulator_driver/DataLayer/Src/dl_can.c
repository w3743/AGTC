/************************************包含头文件*************************************************/
#include <stddef.h>
#include "type.h"
#include "bsp_can.h"
#include "bsp_callback.h"
#include "dl_can.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _DL_CAN_MODULE_T_
{
    DL_CAN_FRAME_T  pending;                                            /* 中断 → 主循环的交接槽 */
    BOOL            pending_valid;
}DL_CAN_MODULE_T;


/************************************静态函数声明************************************************/
static DL_CAN_MODULE_T * _dl_can_get_module
(
    VOID
);

static VOID _dl_can_on_rx
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
);


/************************************静态变量定义************************************************/
static DL_CAN_MODULE_T s_dl_can_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _dl_can_get_module
* 功能描述 :   获取 CAN 模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static DL_CAN_MODULE_T * _dl_can_get_module
(
    VOID
)
{
    return &s_dl_can_module;
}

/********************************************************************************************
* 函数名称 :   _dl_can_on_rx
* 功能描述 :   BSP 的接收回调，**运行在中断上下文**
*
*              这里只做一件事：把报文拷进 pending 槽，然后立刻返回。
*              解析、校验、CRC 全部留给主循环（dl_can_poll 之后）。
*
*              为什么不在中断里直接解析：
*                中断里做 CRC 和范围校验，最坏情况下几十微秒，
*                会把主循环的节拍挤乱，而且一旦解析代码里有浮点或 memcpy 大块，
*                中断延迟就不可预测。本工程没有硬实时需求，但这个边界要一开始就守住。
*
*              ⚠️ pending 只有一格：连续两帧都会到达时，后一帧覆盖前一帧。
*                  CAN 的速率（500 kbit/s，每帧最多 ~130 µs）远低于主循环的
*                 处理频率（毫秒级），实际上不会丢；但这是有意的取舍 ——
*                 宁可丢旧帧，也不要在中断里做无界队列。
*
* 输入参数 :   id / is_ext / p_data / len: 见 bsp_callback.h
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _dl_can_on_rx
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
)
{
    DL_CAN_MODULE_T *p_mod = _dl_can_get_module();
    U8              i;

    if ((p_data == NULL) || (len > 8U))
    {
        return;
    }

    p_mod->pending.id    = id;
    p_mod->pending.is_ext = is_ext;
    p_mod->pending.len   = len;

    for (i = 0U; i < len; i++)
    {
        p_mod->pending.data[i] = p_data[i];
    }

    p_mod->pending_valid = TRUE;
}

/********************************************************************************************
* 函数名称 :   dl_can_init
* 功能描述 :   注册 CAN 接收回调
*
*              只注册回调，**不启动外设** —— 什么时候接入总线由上层决定
*              （bsp_can_start()）。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS
*********************************************************************************************/
TYPE_RET_E dl_can_init
(
    VOID
)
{
    DL_CAN_MODULE_T *p_mod = _dl_can_get_module();

    p_mod->pending_valid = FALSE;

    bsp_callback_can_register(_dl_can_on_rx);

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   dl_can_poll
* 功能描述 :   从交接槽里取一帧报文
*
*              调用时机：主循环（bl_task_schedule），不要在中断里调。
*              取走之后清标志，下一帧才能写进来 —— 这样即使主循环偶尔慢一点，
*              也不会读到"半新半旧"的数据。
*
* 输入参数 :   无
* 输出参数 :   p_frame: 有报文时填入
* 返回值   :   TRUE = 取到一帧，FALSE = 没有待处理报文
*********************************************************************************************/
BOOL dl_can_poll
(
    OUT DL_CAN_FRAME_T  *p_frame
)
{
    DL_CAN_MODULE_T *p_mod = _dl_can_get_module();
    U8              i;

    if ((p_frame == NULL) || (p_mod->pending_valid != TRUE))
    {
        return FALSE;
    }

    p_frame->id     = p_mod->pending.id;
    p_frame->is_ext = p_mod->pending.is_ext;
    p_frame->len    = p_mod->pending.len;

    for (i = 0U; i < p_mod->pending.len; i++)
    {
        p_frame->data[i] = p_mod->pending.data[i];
    }

    p_mod->pending_valid = FALSE;

    return TRUE;
}

/********************************************************************************************
* 函数名称 :   dl_can_send_status
* 功能描述 :   打包并发送状态帧
*
*              ⚠️ **字段布局是占位的**，等整车 CAN 协议表定稿后必须重写这个函数体。
*                  目前按"能看出来在动"编排：母线 2 字节、超容 2 字节、
*                  电流 2 字节（带符号）、温度 1 字节、1 字节保留。
*                  之所以先写一个能跑的版本，是为了让整条链路（采样→打包→总线）
*                  在协议定稿前就能联调通；布局本身随时可以换。
*
*              单位选择：电压用 10 mV/LSB、电流用 100 mA/LSB（1 字节 = 25.6 V / 25.6 A），
*              这样 2 字节能覆盖 0~655 V 和 ±3276 A，都远超本板量程，不会溢出。
*
* 输入参数 :   bus_mv / sc_mv / current_ma / temp_main_c
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E dl_can_send_status
(
    IN  U32     bus_mv,
    IN  U32     sc_mv,
    IN  S32     current_ma,
    IN  F32     temp_main_c
)
{
    U8      payload[8];
    U16     bus_raw;
    U16     sc_raw;
    S16     cur_raw;
    S8      temp_raw;

    bus_raw  = (U16)(bus_mv / 10U);                                     /* 10 mV/LSB */
    sc_raw   = (U16)(sc_mv / 10U);
    cur_raw  = (S16)(current_ma / 100);                                 /* 100 mA/LSB */

    /* 无效温度（DL_TEMP_INVALID_C = -273.15）在这里统一压成 -128，让接收端能区分 */
    if ((temp_main_c < -100.0F) || (temp_main_c > 127.0F))
    {
        temp_raw = (S8)(-128);
    }
    else
    {
        temp_raw = (S8)(temp_main_c);
    }

    payload[0] = (U8)(bus_raw >> 8);
    payload[1] = (U8)(bus_raw & 0xFFU);
    payload[2] = (U8)(sc_raw >> 8);
    payload[3] = (U8)(sc_raw & 0xFFU);
    payload[4] = (U8)((U16)cur_raw >> 8);
    payload[5] = (U8)((U16)cur_raw & 0xFFU);
    payload[6] = (U8)temp_raw;
    payload[7] = 0U;                                                    /* 保留 */

    return bsp_can_send(DL_CAN_ID_STATUS, FALSE, payload, 8U);
}

#ifdef _cplusplus
}
#endif
