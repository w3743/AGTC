/************************************包含头文件*************************************************/
#include <stddef.h>
#include "type.h"
#include "bsp_can.h"
#include "bsp_callback.h"
#include "dl_can.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 温度字段的直接存摄氏温度，S8 的 -127 ~ +127 °C 足够覆盖散热器上任何合理的读数，
 * 而 -128 留作"无效"哨兵（对应 DL_SENSE_TEMP_INVALID_C，-273.15 °C）。
 *
 * 为什么不做 +40 偏移去挤 0~255：S8 装不下 0~255。早先想过"偏移后当无符号存"，
 * 但那样 0x00 就代表 -40 °C，接收端必须知道偏移量才能解读，一旦协议表和固件
 * 对不上就是十万八千里地错。直接存摄氏度，接收端拿 (s8) 强转就是答案，没有约定可错。
 * 代价是上限 127 °C —— 那已经高于任何该出现的工作温度了。 */
#define DL_CAN_TEMP_MAX_C                                               (127.0F)
#define DL_CAN_TEMP_MIN_C                                               (-127.0F)
#define DL_CAN_TEMP_INVALID_CODE                                        ((S8)(-128))


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
*              这里只做一件事：把报文拷进交接槽，然后立刻返回。
*              解析、校验、CRC 全部留给主循环。
*
*              为什么不在中断里直接解析：
*                中断里做 CRC 和范围校验，最坏情况下几十微秒，
*                会把主循环的节拍挤乱；而且一旦解析代码里出现浮点或大块 memcpy，
*                中断延迟就不可预测。本工程没有硬实时需求，但这个边界要一开始就守住。
*
*              ⚠️ pending 只有一格：连续两帧都到达时，后一帧覆盖前一帧。
*                 这是有意的取舍 —— 宁可丢旧帧，也不要在中断里做无界队列。
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

    p_mod->pending.id     = id;
    p_mod->pending.is_ext = is_ext;
    p_mod->pending.len    = len;

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
*              （bl_init() 里调 bsp_can_start()）。
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
* 函数名称 :   dl_can_send
* 功能描述 :   发一帧，包一层 bsp_can_send
* 输入参数 :   id / is_ext / p_data / len
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E dl_can_send
(
    IN  U32         id,
    IN  BOOL        is_ext,
    IN  const U8    *p_data,
    IN  U8          len
)
{
    if (len > 8U)
    {
        return TYPE_RET_PARAMETER_ERROR;
    }

    return bsp_can_send(id, is_ext, p_data, len);
}

/********************************************************************************************
* 函数名称 :   dl_can_send_status
* 功能描述 :   打包并发送周期状态帧
*
*              ⚠️ **字段布局是占位的**，协议表定稿后必须重写函数体。
*                 布局说明见 dl_can.h 的函数注释，这里只强调三处易错点：
*
*              1) 负电流的移位：先把 S32 截成 S16，**再转成 U16** 去取字节。
*                 直接对负数做 >> 是算术右移，高位补 1，取出来的字节拼回去
*                 在接收端还是对的（补码语义一致），但可读性差、容易看错。
*                 显式 (U16) 转换把意图写清楚。
*
*              2) 电流除以 100 时用的是**截断**而不是四舍五入，2.5 A 会显示成 2.4 A。
*                 1 LSB 的误差在 ±3276 A 量程上无所谓，不值得为此加一个判断。
*
*              3) 状态与霍尔码挤在同一个字节：低 4 位状态、高 4 位霍尔码。
*                 DL_MOTOR_STATE_E 只有 5 个值、霍尔码只有 0~7，都塞得下。
*
* 输入参数 :   bus_mv / iu_ma / iv_ma / temp_mos_c / state / hall_code
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E dl_can_send_status
(
    IN  U32     bus_mv,
    IN  S32     iu_ma,
    IN  S32     iv_ma,
    IN  F32     temp_mos_c,
    IN  U8      state,
    IN  U8      hall_code
)
{
    U8      payload[8];
    U16     bus_raw;
    U16     iu_raw;
    U16     iv_raw;
    S16     iu_s16;
    S16     iv_s16;
    S8      temp_raw;

    /* 10 mV/LSB，2 字节能表示到 655.35 V。
     *
     * 钳位放在**这里**而不是调用方侧：任何调用者传一个离谱的值进来，
     * 出来的都是"655.35 V"，而不是 (U16) 截断成的随机数。
     * 把这件事放在格式定义所在的函数里，就不会有人忘。 */
    if (bus_mv > 655350U)
    {
        bus_mv = 655350U;
    }

    bus_raw = (U16)(bus_mv / 10U);

    /* 100 mA/LSB，带符号。先截成 S16 边界再转码：±3276 A 之外没有物理意义，
     * 真出现那种值说明采样已经坏了，钳位比溢出成反向大电流更安全。 */
    if (iu_ma > 3276700)
    {
        iu_ma = 3276700;
    }
    else if (iu_ma < -3276800)
    {
        iu_ma = -3276800;
    }

    if (iv_ma > 3276700)
    {
        iv_ma = 3276700;
    }
    else if (iv_ma < -3276800)
    {
        iv_ma = -3276800;
    }

    iu_s16 = (S16)(iu_ma / 100);
    iv_s16 = (S16)(iv_ma / 100);

    iu_raw = (U16)iu_s16;
    iv_raw = (U16)iv_s16;

    /* 温度：直接存摄氏度；越界统一压成 -128，让接收端能区分
     * "传感器坏了"和"真的零下"。 */
    if ((!(temp_mos_c >= DL_CAN_TEMP_MIN_C)) || (temp_mos_c > DL_CAN_TEMP_MAX_C))
    {
        temp_raw = DL_CAN_TEMP_INVALID_CODE;
    }
    else
    {
        temp_raw = (S8)temp_mos_c;
    }

    payload[0] = (U8)(bus_raw >> 8);
    payload[1] = (U8)(bus_raw & 0xFFU);
    payload[2] = (U8)(iu_raw >> 8);
    payload[3] = (U8)(iu_raw & 0xFFU);
    payload[4] = (U8)(iv_raw >> 8);
    payload[5] = (U8)(iv_raw & 0xFFU);
    payload[6] = (U8)temp_raw;
    payload[7] = (U8)(((hall_code & 0x0FU) << 4) | (state & 0x0FU));

    return bsp_can_send(DL_CAN_ID_STATUS, FALSE, payload, 8U);
}

#ifdef _cplusplus
}
#endif
