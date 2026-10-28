#ifndef __BL_H__
#define __BL_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 业务层（Business Layer）总入口 —— 稳压模块
 *
 * 职责：
 *   1. 决定"要做什么"：功率级启停、母线参考、泄放多少、状态迁移、故障响应
 *   2. 组织 1 ms 控制节拍里的调用顺序
 *   3. 定义对外行为（CAN 指令语义），但报文编解码交给 DL
 *
 * 边界：
 *   - **测量一律走 DL**：母线、超容、抽头、温度、电流都从 dl_* 取，
 *     BL 里不允许出现分压比、上拉阻值、CSA 增益这类数字
 *   - **执行直接调 BSP**：半桥占空比（bsp_bridge_*）、泄放（bsp_dump_*）、
 *     功率级开关（bsp_power_stage_enable）本来就是"写到硬件"的动作，
 *     塞进 DL 反而会让 DL 变成既能读又能写的四不像
 *   - 不碰寄存器，不 include stm32g4xx_hal.h
 *   - 不做测量换算，拿到的就是工程量
 *
 * 本工程只服务**稳压模块（sc-regulator，双向半桥 + 超容）**。
 * 三相无刷电机的代码在 foc_driver 工程里，两者共用同一颗 G431 核心板，
 * 但插在不同的功率板上，固件也各自独立 —— 不要把两套 BSP 混进同一个工程。
 *
 * 典型调用位置：
 *   main() 初始化：  bsp_init() -> dl_init() -> bl_init()
 *   while(1)：       bl_task_schedule() + 按 tick 对齐的 bl_task_handler()
 *
 * ⚠️ bl_task_handler() **不要**放进 SysTick_Handler。
 *    它内部切功率级时会调 bsp_power_stage_enable()，那个函数用 HAL_Delay() 轮询电感
 *    电流衰减，而 HAL_Delay() 依赖 SysTick 计数 —— 放进 SysTick 里会直接死锁。
 *    正确做法是像 main.c 那样在主循环里按 bsp_get_tick() 对齐 1 ms 调用。
 *
 * ⚠️ 当前实现的边界（很重要，别以为它已经能闭环工作）：
 *   本层的控制部分只做到"保持零电流的前馈工作点"，**没有闭环**。
 *   原理图准备文档对控制环路的原话是
 *   「软件电流/功率请求到 DRV8161 CSA、G431 ADC/COMP 与 PWM 的映射、
 *     双向零点/符号、量程、调制和环路稳定性**待联合定型**」。
 *   在环路参数没有实测依据之前，本层不编造 PI 系数。
 ********************************************************************************************/

#include "bl_regulator.h"

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bl_init
* Descriptions  :   Init the business layer. Must be called after dl_init().
*                   Leaves the power stage DE-ENERGISED: nothing is enabled just because
*                   the firmware booted.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E bl_init
(
    VOID
);

/********************************************************************************************
* Function name :   bl_task_handler
* Descriptions  :   1 ms periodic task. Call from SysTick or a 1 ms timer tick.
*                   Keep it short: no blocking, no dynamic allocation.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_task_handler
(
    VOID
);

/********************************************************************************************
* Function name :   bl_task_schedule
* Descriptions  :   Background task, call in the main while(1) loop.
*                   Handles low-priority work: CAN polling, periodic status report.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_task_schedule
(
    VOID
);

#ifdef _cplusplus
}
#endif

#endif /* __BL_H__ */
