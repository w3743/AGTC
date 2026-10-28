/************************************包含头文件*************************************************/
#include "type.h"
#include "bsp.h"
#include "bsp_can.h"
#include "dl.h"
#include "bl.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 状态帧周期。100 ms = 10 Hz，对"看趋势"够用，
 * 又不会把 500 kbit/s 的总线占满（8 字节帧约 130 µs，占空比 0.13%）。 */
#define BL_STATUS_PERIOD_MS                                             (100U)


/************************************数据类型定义************************************************/
typedef struct _BL_MODULE_T_
{
    U32     last_status_tick;                                           /* 上次发状态帧的时刻 */
}BL_MODULE_T;


/************************************静态变量定义************************************************/
static BL_MODULE_T s_bl_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   bl_init
* 功能描述 :   业务层初始化
*
*              顺序：先策略层（清状态、确保断电），再 CAN 数据层（注册回调），
*              最后才真正启动 FDCAN 外设。
*
*              为什么最后才 bsp_can_start()：
*                bsp_init() 只配不启动，是因为"什么时候接入总线"是系统级决定。
*                BL 就是那个系统级 —— 但必须等回调注册好之后再启动，
*                否则第一帧到达时 dl_can 还没准备好，会被静默丢掉。
*
*              ⚠️ 本函数**不会**给功率级加电。要跑起来必须显式调
*                  bl_regulator_request_enable(TRUE)，而且还要过联锁。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E bl_init
(
    VOID
)
{
    TYPE_RET_E result = TYPE_RET_SUCCESS;

    s_bl_module.last_status_tick = 0U;

    (VOID)bl_regulator_init();

    if (dl_can_init() != TYPE_RET_SUCCESS)
    {
        result = TYPE_RET_FAIL;
    }

    /* 启动 FDCAN。起不来不算致命 —— 板子仍然可以当本地采样板用，
     * 所以只置返回值，不阻断后续流程。 */
    bsp_can_start();

    return result;
}

/********************************************************************************************
* 函数名称 :   bl_task_handler
* 功能描述 :   1 ms 周期任务
*
*              ⚠️ 调用位置由 main.c 决定：在主循环里按 bsp_get_tick() 对齐 1 ms 调用，
*                 **不放 SysTick_Handler**。
*                 原因：本函数会走到 bl_regulator_task_1ms() 里的切功率级流程，
*                 而 bsp_power_stage_enable() 用 HAL_Delay() 轮询电感电流衰减。
*                 HAL_Delay() 依赖 SysTick 计数 —— 若本函数本身跑在 SysTick 中断里，
*                 那个等待永远不会结束，直接死锁。
*
*              仍然**不能有任何阻塞调用**（除了上面那个已知的例外）：
*              不能等标志、不能浮点大运算，因为它占的是主节拍。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_task_handler
(
    VOID
)
{
    bl_regulator_task_1ms();
}

/********************************************************************************************
* 函数名称 :   bl_task_schedule
* 功能描述 :   后台任务，在 main 的 while(1) 里调用
*
*              做两件事：
*                1. 取 CAN 报文。目前**只取不解析** —— 协议表还没定，
*                   见 dl_can.h 顶部的说明。取出来是为了让链路可观测（能打断点看内容）。
*                2. 按周期发状态帧。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_task_schedule
(
    VOID
)
{
    DL_CAN_FRAME_T  frame;
    U32             now;

    /* ── 1. 取一帧待处理报文 ───────────────────────────────────────────────── */
    if (dl_can_poll(&frame) == TRUE)
    {
        /* ⚠️ 这里本该按 frame.id 分发到 bl_regulator_set_bus_ref_mv() 等接口，
         *    但整车 CAN 协议表还没冻结（连 ID 都是占位符），
         *    现在写分发逻辑等于把一套猜的协议固化进代码。
         *    所以先什么都不做，只把 frame 取走，避免交接槽一直占着。 */
        (VOID)frame;
    }

    /* ── 2. 周期状态帧 ─────────────────────────────────────────────────────── */
    now = bsp_get_tick();

    if ((now - s_bl_module.last_status_tick) >= BL_STATUS_PERIOD_MS)
    {
        s_bl_module.last_status_tick = now;

        (VOID)dl_can_send_status(bl_regulator_get_bus_mv(),
                                 bl_regulator_get_sc_mv(),
                                 bl_regulator_get_current_ma(),
                                 dl_temp_get_c(DL_TEMP_MAIN));
    }
}

#ifdef _cplusplus
}
#endif
