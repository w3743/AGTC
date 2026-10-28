#ifndef __BL_H__
#define __BL_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 业务层（Business Layer）总入口 —— FOC 驱动模块
 *
 * 职责：
 *   1. 决定"要做什么"：功率级启停、电压指令、状态迁移、故障响应
 *   2. 组织 1 ms 控制节拍里的调用顺序
 *   3. 定义对外行为（CAN 指令语义），但报文编解码交给 DL
 *
 * 边界：
 *   - **测量一律走 DL**：母线、三相电流、两路温度、霍尔都从 dl_* 取，
 *     BL 里不允许出现分压比、采样电阻、CSA 增益这类数字
 *   - **执行可以直接调 BSP 或 DL 的原语**：dl_motor_set_duty() / dl_motor_enable()
 *     本身就是"写到硬件"的动作，塞进 DL 的度量接口里反而会让 DL 变成四不像
 *   - 不碰寄存器，不 include stm32g4xx_hal.h
 *   - 不做测量换算，拿到的就是工程量
 *
 * 本工程只服务**三相无刷电机（DRV8353）**。
 * 稳压模块的代码在 regulator_driver 工程里，两者共用同一颗 G431 核心板，
 * 但插在不同的功率板上、跑各自的固件 —— 不要把两套 BSP 混进同一个工程。
 *
 * 典型调用位置：
 *   main() 初始化：  bsp_init() -> dl_init() -> bl_init()
 *   while(1)：       bl_task_schedule() + 按 tick 对齐的 bl_task_handler()
 *
 * ⚠️ bl_task_handler() **不要**放进 SysTick_Handler。
 *    它内部切功率级时会走 bsp_power_stage_enable()，那个函数用 HAL_Delay() 轮询
 *    DRV8353 的故障位，而 HAL_Delay() 依赖 SysTick 计数 —— 放进 SysTick 里会直接死锁
 *    （等一个永远不会推进的 tick）。正确做法是像 main.c 那样在主循环里按
 *    bsp_get_tick() 对齐 1 ms 调用。
 *
 * ⚠️ 当前能力的真实边界（很重要，别以为它已经是一套能用的伺服）：
 *   已经能跑通：采样链路、坐标变换、SVPWM、状态机、全部联锁、CAN 状态上报。
 *   还缺的：电流环增益（默认 0 = 旁路）、角度细分（现在只有 60° 分辨率）、
 *          转速环、自启动策略、CAN 指令解析（协议表未冻结）。
 *   详见 bl_foc.h 与 bl_motor.h 顶部的说明。
 ********************************************************************************************/

#include "bl_motor.h"

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bl_init
* Descriptions  :   Init the business layer. Must be called after dl_init().
*                   Registers the CAN rx hook and joins the bus, then brings the motor
*                   strategy layer up in IDLE with the power stage de-energised.
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
* Descriptions  :   1 ms periodic task. Drives the motor state machine.
*                   Call from the main loop paced by bsp_get_tick(); do NOT call it from
*                   SysTick (see the warning above).
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
