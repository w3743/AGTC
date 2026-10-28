/************************************包含头文件*************************************************/
#include <stddef.h>
#include "type.h"
#include "bsp.h"
#include "bsp_gpio.h"
#include "bsp_inverter.h"
#include "dl_motor.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 零电压矢量对应的占空比。
 *
 * 为什么要在这里再写一遍（而不是引用 bsp.c 里的 BSP_ZERO_VECTOR_DUTY）：
 *   那个宏是 bsp.c 的私有实现，BSP 层刻意不向上暴露。DL 侧只需要知道"零点是多少"
 *   以便缓存一份和硬件一致的值（否则 get_duty() 会报出与实际 CCR 不符的数字）。
 *   两边必须保持一致 —— bsp.c 里改了这个值，这里也要改。
 *   这条耦合靠 bsp_inverter.h 的"占空比 0 = 电制动"说明和这里的注释共同守住。 */
#define DL_MOTOR_ZERO_VECTOR_DUTY                                       (0.5F)

/* 相索引，只为代码好读 */
#define DL_MOTOR_PHASE_U                                                (0U)
#define DL_MOTOR_PHASE_V                                                (1U)
#define DL_MOTOR_PHASE_W                                                (2U)
#define DL_MOTOR_PHASE_NUM                                              (3U)


/************************************数据类型定义************************************************/
typedef struct _DL_MOTOR_MODULE_T_
{
    DL_MOTOR_STATE_E    state;                                          /* 当前状态 */
    F32                 duty[DL_MOTOR_PHASE_NUM];                       /* 最近一次下发的三相占空比 */
}DL_MOTOR_MODULE_T;


/************************************静态函数声明************************************************/
static DL_MOTOR_MODULE_T * _dl_motor_get_module
(
    VOID
);

static F32 _dl_motor_clamp_duty
(
    IN  F32     value
);


/************************************静态变量定义************************************************/
static DL_MOTOR_MODULE_T s_dl_motor_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _dl_motor_get_module
* 功能描述 :   获取电机模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static DL_MOTOR_MODULE_T * _dl_motor_get_module
(
    VOID
)
{
    return &s_dl_motor_module;
}

/********************************************************************************************
* 函数名称 :   _dl_motor_clamp_duty
* 功能描述 :   把占空比钳到 [0.0, 1.0]
*
*              钳位而不是报错，是有意的：上层（调制器）算出来的占空比在过调制、
*              母线跌落、除零边界上偶尔会越界一点点。为此让整个控制链停下来不值得，
*              钳一下继续跑更实用。真正需要"报错"的是越界幅度很大那种，
*              那个由 BL 的限幅和故障判定负责，不在这里。
*
*              对 NaN 的处理：NaN 与任何数比较都是假，会原样漏过去。
*              所以先判 `!(value >= 0.0F)`，NaN 会在这里被判成"小于 0"从而归 0 ——
*              输出端落在 0（下桥常通 = 电制动），比把 NaN 写进 CCR 变成随机值可控。
*
* 输入参数 :   value: 原始占空比
* 输出参数 :   无
* 返回值   :   0.0 ~ 1.0
*********************************************************************************************/
static F32 _dl_motor_clamp_duty
(
    IN  F32     value
)
{
    if (!(value >= 0.0F))
    {
        return 0.0F;
    }

    if (value > 1.0F)
    {
        return 1.0F;
    }

    return value;
}

/********************************************************************************************
* 函数名称 :   dl_motor_init
* 功能描述 :   初始化电机数据层，把功率级留在完全关断的状态
*
*              这里做三件事，顺序有意义：
*                1. 先把缓存的三相占空比写成零点；
*                2. 调 bsp_power_stage_enable(FALSE) —— 它会拉低 ENABLE、清 MOE、
*                   并把 CCR 也摆回零点，硬件是真的断电了；
*                3. 状态记 OFF。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS
*********************************************************************************************/
TYPE_RET_E dl_motor_init
(
    VOID
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    p_mod->duty[DL_MOTOR_PHASE_U] = DL_MOTOR_ZERO_VECTOR_DUTY;
    p_mod->duty[DL_MOTOR_PHASE_V] = DL_MOTOR_ZERO_VECTOR_DUTY;
    p_mod->duty[DL_MOTOR_PHASE_W] = DL_MOTOR_ZERO_VECTOR_DUTY;

    (VOID)bsp_power_stage_enable(FALSE);

    p_mod->state = DL_MOTOR_STATE_OFF;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   dl_motor_enable
* 功能描述 :   使能 / 关断三相功率级
*
*              ── enable = TRUE ───────────────────────────────────────────────────
*              走 bsp_power_stage_enable(TRUE)，它内部是：
*                DRV FAULT 检查 → 占空比强制零矢量 → MOE=1 → ENABLE 高。
*
*              ⚠️ 注意这一步会把占空比**强制拉回零点**，之前 dl_motor_set_duty()
*                 写进去的值会被覆盖。这是 BSP 故意的，见下面"为什么不能先写占空比"。
*
*              ── enable = FALSE ──────────────────────────────────────────────────
*              ENABLE 拉低（硬件路径，最快）→ MOE=0 → 占空比回零点。
*              若当前是 FAULT 状态，**保持 FAULT 不退回 OFF** —— 故障是要人确认的，
*              不能因为调了一次"关"就当作已复位。
*
*              ── 为什么不能"先 set_duty 再 enable" ───────────────────────────────
*              这个顺序看起来更合理（先把想要的波形摆好，再开门），但它是错的：
*              功率级第一次通电的那一瞬间，如果占空比不是零矢量，三相绕组会立刻
*              承受一个阶跃电压，电流按绕组电阻 dt→0 直接冲上去。而这台电机的
*              绕组电阻只有零点几欧，48 V 下的冲击电流是几十安的量级。
*              正确做法是 enable 到零点，再由上层按斜坡把电压加上去（BL 的活）。
*              所以 dl_motor.h 里那段"先 set_duty 再 enable"的旧说明是错的，已改。
*
* 输入参数 :   enable: TRUE = 给电机加电，FALSE = 硬关断
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS: 已执行
*              TYPE_RET_FAIL   : 拒绝执行（仅 TRUE 时可能发生）
*                                · 已处于故障态
*                                · BREAK_N 已被拉低 / TIM1 刹车已锁存
*                                · DRV8353 自报故障（可能是上电瞬间 VM/VREF 未稳）
*                                ⚠️ 最后一种情况**不置 FAULT**：那是"现在还不能开"，
*                                   不是"坏了"。要不要记成故障、要不要重试，由 BL 决定。
*********************************************************************************************/
TYPE_RET_E dl_motor_enable
(
    IN  BOOL    enable
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    if (enable == FALSE)
    {
        (VOID)bsp_power_stage_enable(FALSE);

        p_mod->duty[DL_MOTOR_PHASE_U] = DL_MOTOR_ZERO_VECTOR_DUTY;
        p_mod->duty[DL_MOTOR_PHASE_V] = DL_MOTOR_ZERO_VECTOR_DUTY;
        p_mod->duty[DL_MOTOR_PHASE_W] = DL_MOTOR_ZERO_VECTOR_DUTY;

        if (p_mod->state != DL_MOTOR_STATE_FAULT)
        {
            p_mod->state = DL_MOTOR_STATE_OFF;
        }

        return TYPE_RET_SUCCESS;
    }

    /* 故障态必须先 clear_fault() 才能重新使能。见 dl_motor_clear_fault() 的说明：
     * 那里会复查当下是否真的没故障，所以不存在"连按两次就绕过"的漏洞。 */
    if (p_mod->state == DL_MOTOR_STATE_FAULT)
    {
        return TYPE_RET_FAIL;
    }

    /* 使能前先看硬件刹车：BREAK_N 还低着就开，等于把芯片往故障上推 */
    if (dl_motor_get_fault() == TRUE)
    {
        p_mod->state = DL_MOTOR_STATE_FAULT;
        return TYPE_RET_FAIL;
    }

    if (bsp_power_stage_enable(TRUE) != TYPE_RET_SUCCESS)
    {
        p_mod->state = DL_MOTOR_STATE_OFF;
        return TYPE_RET_FAIL;
    }

    /* BSP 内部把 CCR 摆到了零点，缓存跟上，保证 get_duty() 说的是真话 */
    p_mod->duty[DL_MOTOR_PHASE_U] = DL_MOTOR_ZERO_VECTOR_DUTY;
    p_mod->duty[DL_MOTOR_PHASE_V] = DL_MOTOR_ZERO_VECTOR_DUTY;
    p_mod->duty[DL_MOTOR_PHASE_W] = DL_MOTOR_ZERO_VECTOR_DUTY;

    p_mod->state = DL_MOTOR_STATE_READY;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   dl_motor_set_duty
* 功能描述 :   一次性下发三相占空比
*
*              三相必须同时写：分三次调用会让中间出现"两相新、一相旧"的瞬间，
*              那个瞬间的线电压是错的，绕组上会出现一个不属于任何调制策略的矢量。
*              本函数把三次 CCR 写入放在紧邻的几条指令里，中间不插入别的动作。
*
*              状态处理：
*                READY / RUN  → RUN（已经在输出，只是换了矢量）
*                BRAKE        → 保持 BRAKE（要退出制动得显式调 set_duty 到非零矢量）
*                OFF / FAULT  → 不改状态（写了也不会有效果，ENABLE 低着）
*
* 输入参数 :   duty_u / duty_v / duty_w: 0.0 ~ 1.0，越界自动钳位
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID dl_motor_set_duty
(
    IN  F32     duty_u,
    IN  F32     duty_v,
    IN  F32     duty_w
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();
    F32                duty[DL_MOTOR_PHASE_NUM];

    duty[DL_MOTOR_PHASE_U] = _dl_motor_clamp_duty(duty_u);
    duty[DL_MOTOR_PHASE_V] = _dl_motor_clamp_duty(duty_v);
    duty[DL_MOTOR_PHASE_W] = _dl_motor_clamp_duty(duty_w);

    p_mod->duty[DL_MOTOR_PHASE_U] = duty[DL_MOTOR_PHASE_U];
    p_mod->duty[DL_MOTOR_PHASE_V] = duty[DL_MOTOR_PHASE_V];
    p_mod->duty[DL_MOTOR_PHASE_W] = duty[DL_MOTOR_PHASE_W];

    bsp_inverter_set_duty(duty[DL_MOTOR_PHASE_U],
                          duty[DL_MOTOR_PHASE_V],
                          duty[DL_MOTOR_PHASE_W]);

    if ((p_mod->state == DL_MOTOR_STATE_READY) || (p_mod->state == DL_MOTOR_STATE_RUN))
    {
        p_mod->state = DL_MOTOR_STATE_RUN;
    }
}

/********************************************************************************************
* 函数名称 :   dl_motor_get_duty
* 功能描述 :   读回最近一次下发的某一相占空比
* 输入参数 :   phase: 0=U, 1=V, 2=W
* 输出参数 :   无
* 返回值   :   占空比 0.0 ~ 1.0；phase 越界返回 0.0
*********************************************************************************************/
F32 dl_motor_get_duty
(
    IN  U8      phase
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    if (phase >= (U8)DL_MOTOR_PHASE_NUM)
    {
        return 0.0F;
    }

    return p_mod->duty[phase];
}

/********************************************************************************************
* 函数名称 :   dl_motor_coast
* 功能描述 :   自由滑行 —— ENABLE 拉低 + MOE=0，六路栅极全关，绕组开路
*
*              和 bsp_power_stage_enable(FALSE) 是同一件事，这里只是给上层一个
*              语义清楚的名字。dl_motor_brake() 的注释里解释了它俩的区别。
*
*              ⚠️ COAST 不会等电流衰减。这不是疏忽：
*                  稳压板是双向变换器，断开前必须等电感电流归零（否则能量无处可去，
*                  会在开关上打出一个电压尖峰）；而电机绕组断电后电流经六个体二极管
*                  续流回母线，这是逆变器的正常工作方式，不需要等。
*                  真要"软一点"的停，应该先把电压斜坡降到 0 再调本函数 —— 那是 BL 的事。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID dl_motor_coast
(
    VOID
)
{
    (VOID)dl_motor_enable(FALSE);
}

/********************************************************************************************
* 函数名称 :   dl_motor_brake
* 功能描述 :   主动电制动 —— 三相占空比全 0，三个下桥常通，绕组被短接
*
*              ⚠️ 两个必须知道的后果：
*                1. 会产生制动转矩，且反电动势的能量全部在电机绕组和 MOS 上变成热
*                   （I²R）。转速越高、制动越猛，MOS 越热。
*                2. 高速时突然切零矢量，绕组电流无处可去（下桥是短路而不是开路），
*                   电流会在绕组内部环流并把母线顶高。所以**不建议在高速时用它急停**，
*                   正常停车走"斜坡降压 → coast"。
*
*              功率级没上电时调用本函数是空操作：绕组本来就断着，谈不上制动。
*              想"用制动停住一台还在转的电机"，前提是功率级已经在 RUN。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID dl_motor_brake
(
    VOID
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    if ((p_mod->state == DL_MOTOR_STATE_OFF) || (p_mod->state == DL_MOTOR_STATE_FAULT))
    {
        return;
    }

    (VOID)dl_motor_set_duty(0.0F, 0.0F, 0.0F);

    p_mod->state = DL_MOTOR_STATE_BRAKE;
}

/********************************************************************************************
* 函数名称 :   dl_motor_get_fault
* 功能描述 :   判断功率级是否处于故障
*
*              三个来源，任一为真都算：
*                1. DL 自己已经锁存了 FAULT 状态；
*                2. TIM1 的刹车标志已锁存 —— 硬件动作过，哪怕现在已经恢复也要算，
*                   因为"一闪而过的过流"正是最需要被记录的那种事件；
*                3. BREAK_N 当前是低 —— 故障仍在持续。
*              只查第 2 条会漏掉"锁存被别处清掉但故障仍在"的情况；
*              只查第 3 条会漏掉瞬时故障。两条都要。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 有故障，FALSE = 正常
*********************************************************************************************/
BOOL dl_motor_get_fault
(
    VOID
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    if (p_mod->state == DL_MOTOR_STATE_FAULT)
    {
        return TRUE;
    }

    if (bsp_inverter_break_occurred() == TRUE)
    {
        return TRUE;
    }

    /* bsp_gpio_read_break_n(): TRUE = 网络为高 = 无故障 */
    if (bsp_gpio_read_break_n() != TRUE)
    {
        return TRUE;
    }

    return FALSE;
}

/********************************************************************************************
* 函数名称 :   dl_motor_clear_fault
* 功能描述 :   清除硬件刹车锁存
*
*              清完之后**复查一遍**：如果 BREAK_N 还是低，说明故障源还在（比如 DRV8353
*              真的在报过流、或者 U3 比较器还在翻），这时候把状态退回 OFF 是在骗人 ——
*              上层会以为可以重新使能，一使能又立刻跳闸，陷入"跳闸-复位-跳闸"循环。
*              所以这里保持 FAULT。
*
*              ⚠️ 本函数不恢复输出（MOE 不会自己回来），清完必须显式 dl_motor_enable(TRUE)。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID dl_motor_clear_fault
(
    VOID
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    bsp_inverter_clear_break();

    if ((bsp_inverter_break_occurred() == TRUE) || (bsp_gpio_read_break_n() != TRUE))
    {
        p_mod->state = DL_MOTOR_STATE_FAULT;
        return;
    }

    p_mod->state = DL_MOTOR_STATE_OFF;
}

/********************************************************************************************
* 函数名称 :   dl_motor_get_state
* 功能描述 :   读电机状态，顺带刷新故障判定
*
*              ⚠️ 本函数**只改状态变量，不写任何硬件寄存器**。
*                 理由：如果 getter 里顺手关功率级，出问题时就查不出"到底是谁关的"。
*                 故障发生时桥臂本来就安全 —— TIM1 已经把 MOE 清零，六路 OISx=0，
*                 INH/INL 全低，按 6x 真值表进 Hi-Z；ENABLE 虽然还高，但那只是
*                 "栅极驱动在待命"，不是"在输出"。
*                 真正的收尾（拉低 ENABLE）由 BL 看到 FAULT 后调 dl_motor_enable(FALSE)。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   DL_MOTOR_STATE_E
*********************************************************************************************/
DL_MOTOR_STATE_E dl_motor_get_state
(
    VOID
)
{
    DL_MOTOR_MODULE_T *p_mod = _dl_motor_get_module();

    if ((p_mod->state != DL_MOTOR_STATE_FAULT) && (dl_motor_get_fault() == TRUE))
    {
        p_mod->state = DL_MOTOR_STATE_FAULT;
    }

    return p_mod->state;
}

#ifdef _cplusplus
}
#endif
