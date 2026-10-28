/************************************包含头文件*************************************************/
#include "type.h"
#include "bsp.h"
#include "bsp_bridge.h"
#include "bsp_dump.h"
#include "dl.h"
#include "bl_regulator.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************数据类型定义************************************************/
typedef struct _BL_REG_MODULE_T_
{
    BL_REG_STATE_E  state;                                              /* 当前状态 */
    U16             fault;                                              /* 锁存的故障位 */
    BOOL            run_request;                                        /* 上层请求运行 */
    U32             bus_ref_mv;                                         /* 母线参考（死区策略用） */

    U32             bus_mv;                                             /* 1 ms 缓存 */
    U32             sc_mv;
    S32             current_ma;
    F32             temp_main_c;
    F32             temp_sc_c;
    F32             temp_dump_c;
}BL_REG_MODULE_T;


/************************************静态函数声明************************************************/
static BL_REG_MODULE_T * _bl_reg_get_module
(
    VOID
);

static VOID _bl_reg_refresh_measurements
(
    IN  BL_REG_MODULE_T *p_mod
);

static U16 _bl_reg_check_interlocks
(
    IN  BL_REG_MODULE_T *p_mod
);

static F32 _bl_reg_balance_duty
(
    IN  U32     bus_mv,
    IN  U32     sc_mv
);

static VOID _bl_reg_enter_fault
(
    IN  BL_REG_MODULE_T *p_mod,
    IN  U16             fault_bits
);


/************************************静态变量定义************************************************/
static BL_REG_MODULE_T s_bl_reg_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bl_reg_get_module
* 功能描述 :   获取模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BL_REG_MODULE_T * _bl_reg_get_module
(
    VOID
)
{
    return &s_bl_reg_module;
}

/********************************************************************************************
* 函数名称 :   _bl_reg_refresh_measurements
* 功能描述 :   把 DL 的工程量刷进 1 ms 缓存
*
*              为什么缓存：同一毫秒里的联锁判断、前馈计算、状态上报要用的是**同一组**数，
*              每次都去调 dl_* 会拿到前后差几十微秒的样本，导致判断之间互相矛盾
*              （比如过压判断用的是新的，前馈用的是旧的）。
*
* 输入参数 :   p_mod: 模块
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_reg_refresh_measurements
(
    IN  BL_REG_MODULE_T *p_mod
)
{
    p_mod->bus_mv       = dl_bus_get_bus_mv();
    p_mod->sc_mv        = dl_bus_get_sc_mv();
    p_mod->current_ma   = dl_power_get_current_ma();
    p_mod->temp_main_c  = dl_temp_get_c(DL_TEMP_MAIN);
    p_mod->temp_sc_c    = dl_temp_get_c(DL_TEMP_SC);
    p_mod->temp_dump_c  = dl_temp_get_c(DL_TEMP_DUMP);
}

/********************************************************************************************
* 函数名称 :   _bl_reg_check_interlocks
* 功能描述 :   逐条检查联锁，返回故障位（0 = 全部通过）
*
*              这里的每一条都是"允不允许搬能量"，**不是器件保护**：
*                器件级保护由 DRV8161（VDS OCP、UVLO、OTSD）和 TLV6700（SC 过压）硬件负责，
*                它们不经过 MCU，也不会因为本函数写错而失效。
*                本函数管的是"软件别主动去干傻事"。
*
*              边界值取的依据见 bl_regulator.h 里各个宏的注释。这里只强调两条容易搞错的：
*
*              1) 超容高于母线时禁止搬能量。
*                 正常工作窗要求 V_SC < V_BUS。两端电压交叉时是拓扑上真正麻烦的时刻
*                 （文档原文：按经验证的停变换和隔离顺序处理，不把预置策略扩展为跨压连续控制），
*                 软件能做的只有停手，不能靠占空比去"过渡"。
*
*              2) 温度无效一律算故障，不算"没到门限"。
*                 探头掉线时 dl_temp 返回 DL_TEMP_INVALID_C = −273.15，
*                 如果当成"很冷"就会在前级已经过热时继续满载。
*                 **失效方向必须是停，不是跑。**
*
* 输入参数 :   p_mod: 模块（已刷新测量值）
* 输出参数 :   无
* 返回值   :   BL_REG_FAULT_* 位组合
*********************************************************************************************/
static U16 _bl_reg_check_interlocks
(
    IN  BL_REG_MODULE_T *p_mod
)
{
    U16 fault = BL_REG_FAULT_NONE;

    /* ── 母线窗口 ─────────────────────────────────────────────────────────── */
    if (p_mod->bus_mv > BL_REG_BUS_MAX_MV)
    {
        fault |= BL_REG_FAULT_BUS_OV;
    }
    if (p_mod->bus_mv < BL_REG_BUS_MIN_MV)
    {
        fault |= BL_REG_FAULT_BUS_UV;
    }

    /* ── 超容窗口 ─────────────────────────────────────────────────────────── */
    if (p_mod->sc_mv > BL_REG_SC_MAX_MV)
    {
        fault |= BL_REG_FAULT_SC_OV;
    }
    if (p_mod->sc_mv < BL_REG_SC_MIN_MV)
    {
        fault |= BL_REG_FAULT_SC_UV;
    }

    /* ── 拓扑约束：超容不得高于母线 ───────────────────────────────────────── */
    if (p_mod->sc_mv >= p_mod->bus_mv)
    {
        fault |= BL_REG_FAULT_SC_ABOVE_BUS;
    }

    /* ── 温度：先判有效性，再判门限 ───────────────────────────────────────── */
    if ((p_mod->temp_main_c <= DL_TEMP_INVALID_C) ||
        (p_mod->temp_sc_c   <= DL_TEMP_INVALID_C) ||
        (p_mod->temp_dump_c <= DL_TEMP_INVALID_C))
    {
        fault |= BL_REG_FAULT_TEMP_INVALID;
    }
    else
    {
        if (p_mod->temp_main_c > BL_REG_TEMP_MAIN_MAX_C)
        {
            fault |= BL_REG_FAULT_TEMP_MAIN;
        }
        if (p_mod->temp_sc_c > BL_REG_TEMP_SC_MAX_C)
        {
            fault |= BL_REG_FAULT_TEMP_SC;
        }
        if (p_mod->temp_dump_c > BL_REG_TEMP_DUMP_MAX_C)
        {
            fault |= BL_REG_FAULT_TEMP_DUMP;
        }
    }

    return fault;
}

/********************************************************************************************
* 函数名称 :   _bl_reg_balance_duty
* 功能描述 :   算"让电感电流维持不变"的占空比（前馈）
*
*              推导：开关节点平均电压 = D × V_BUS；电感另一端是 V_SC。
*                    稳态下没有直流压降，所以电流不变的充要条件是
*                        D × V_BUS = V_SC   ⟹   D = V_SC / V_BUS
*
*              这就是本层能给出的、**有物理依据**的全部内容。
*              它不做调节：如果 V_SC 变了而 D 没跟上，就会有电流；
*              当前实现每毫秒重算一次，勉强算个很慢的跟随，但**不是闭环**。
*
*              为什么返回带符号值：bsp_bridge_set_duty() 的接口是 −1~+1，
*              D=1 对应 +1、D=0 对应 −1、D=0.5 对应 0，所以 signed = 2D − 1。
*
*              边界处理：
*                · V_BUS 为 0（ADC 还没起来 / 母线真没电）时直接返回 0（中点），
*                  不去做那个除法 —— 除以 0 会得到 inf，再把 inf 钳到限幅值
*                  就等于输出 ±0.9 的满占空比，是非常危险的失败模式。
*                · 结果限到 ±BL_REG_DUTY_LIMIT，永远不贴到端点。
*
* 输入参数 :   bus_mv: 母线电压（mV）
*               sc_mv : 超容电压（mV）
* 输出参数 :   无
* 返回值   :   带符号占空比 −1.0 ~ +1.0
*********************************************************************************************/
static F32 _bl_reg_balance_duty
(
    IN  U32     bus_mv,
    IN  U32     sc_mv
)
{
    F32 d;
    F32 signed_duty;

    if (bus_mv == 0U)
    {
        return 0.0F;
    }

    d = ((F32)sc_mv) / ((F32)bus_mv);

    signed_duty = (2.0F * d) - 1.0F;

    if (signed_duty > BL_REG_DUTY_LIMIT)
    {
        signed_duty = BL_REG_DUTY_LIMIT;
    }
    if (signed_duty < -BL_REG_DUTY_LIMIT)
    {
        signed_duty = -BL_REG_DUTY_LIMIT;
    }

    return signed_duty;
}

/********************************************************************************************
* 函数名称 :   _bl_reg_enter_fault
* 功能描述 :   进入故障态：锁存故障位、切断功率级、停泄放
*
*              顺序：先断功率级（bsp_power_stage_enable(FALSE) 内部已经处理了
*              "先停 MOE → 再拉 PERMIT → 等电流衰减 → 最后断 P-MOS"），再停泄放。
*
*              ⚠️ 泄放**不**在这里继续打开。有些设计会在母线过压时靠泄放顶住，
*                 但本板的泄放由软件控制、没有独立硬件过压触发，
*                 在已经出故障（可能 MCU 状态不可信）的时候继续开泄放，
*                 风险是把电阻烧了却救不了母线。所以故障态默认全关，
*                 要不要"故障时仍允许泄放"是后续需要单独论证的策略问题。
*
* 输入参数 :   p_mod:      模块
*               fault_bits: 要锁存的故障位
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_reg_enter_fault
(
    IN  BL_REG_MODULE_T *p_mod,
    IN  U16             fault_bits
)
{
    /* 断电可能因为电流没衰减而超时，那也要继续往下走 —— 故障处理不能因为
     * 一个步骤没成功就停在半路。超时本身记成故障位。 */
    if (bsp_power_stage_enable(FALSE) != TYPE_RET_SUCCESS)
    {
        fault_bits |= BL_REG_FAULT_POWER_OFF_TIMEOUT;
    }

    bsp_dump_set_duty(0.0F);

    p_mod->fault |= fault_bits;
    p_mod->state  = BL_REG_STATE_FAULT;
}

/********************************************************************************************
* 函数名称 :   bl_regulator_init
* 功能描述 :   初始化策略层，状态停在 IDLE（功率级断电）
*
*              **不会**因为固件启动就给功率级加电 —— 那是 bl_regulator_request_enable()
*              的事，而且要过联锁。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS
*********************************************************************************************/
TYPE_RET_E bl_regulator_init
(
    VOID
)
{
    BL_REG_MODULE_T *p_mod = _bl_reg_get_module();

    p_mod->state        = BL_REG_STATE_IDLE;
    p_mod->fault        = BL_REG_FAULT_NONE;
    p_mod->run_request  = FALSE;
    p_mod->bus_ref_mv   = 0U;

    p_mod->bus_mv       = 0U;
    p_mod->sc_mv        = 0U;
    p_mod->current_ma   = 0;
    p_mod->temp_main_c  = DL_TEMP_INVALID_C;
    p_mod->temp_sc_c    = DL_TEMP_INVALID_C;
    p_mod->temp_dump_c  = DL_TEMP_INVALID_C;

    /* 确保硬件确实是关着的。bsp_init() 已经做过一次，这里再做一次是因为
     * bl_regulator_init() 可能被单独调用（比如软复位后重新初始化策略层）。 */
    (VOID)bsp_power_stage_enable(FALSE);
    bsp_dump_set_duty(0.0F);

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_regulator_task_1ms
* 功能描述 :   状态机走一步
*
*              状态迁移：
*                IDLE   --请求运行 & 联锁通过--> ARMED
*                ARMED  --联锁通过-->            ACTIVE
*                ACTIVE --联锁失败 / 请求停止--> IDLE 或 FAULT
*                FAULT  --清除故障 & 联锁恢复--> IDLE
*
*              为什么 ARMED 和 ACTIVE 要分开：
*                ARMED 是"功率级已经上电、但占空比停在前馈点"，
*                ACTIVE 才进入每毫秒更新占空比的循环。
*                分两步的意义是把"加电"和"开始调节"解耦 ——
*                加电瞬间的浪涌和调节器的第一次输出不会撞在一起，
*                出问题时也更容易判断是哪一步出的。
*                当前 ACTIVE 与 ARMED 的动作差别很小（都只是算前馈），
*                但状态语义要留出来，等闭环补上之后 ACTIVE 才有实质区别。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_regulator_task_1ms
(
    VOID
)
{
    BL_REG_MODULE_T *p_mod = _bl_reg_get_module();
    U16             fault;
    F32             duty;

    _bl_reg_refresh_measurements(p_mod);

    /* ── 故障态：只等清除，不做任何动作 ───────────────────────────────────── */
    if (p_mod->state == BL_REG_STATE_FAULT)
    {
        return;
    }

    fault = _bl_reg_check_interlocks(p_mod);

    /* 任何联锁失败都直接进故障态。这里不做"降额继续跑"的软化处理 ——
     * 降额策略需要知道每一条联锁的严重程度，而当前没有这个分级依据。 */
    if (fault != BL_REG_FAULT_NONE)
    {
        _bl_reg_enter_fault(p_mod, fault);
        return;
    }

    /* ── 联锁通过 ─────────────────────────────────────────────────────────── */
    switch (p_mod->state)
    {
        case BL_REG_STATE_IDLE:
            if (p_mod->run_request == TRUE)
            {
                if (bsp_power_stage_enable(TRUE) != TYPE_RET_SUCCESS)
                {
                    _bl_reg_enter_fault(p_mod, BL_REG_FAULT_POWER_OFF_TIMEOUT);
                    break;
                }

                /* 上电后先把占空比摆到前馈点，再让状态往前走。
                 * 注意 bsp_power_stage_enable(TRUE) 内部已经把占空比设成中点，
                 * 这里紧接着覆盖成前馈值 —— 两个动作之间只差几条指令，
                 * MOE 已经开了，所以中间那一小段确实是 50% 在跑。
                 * 对 20 µH 电感来说这段只有微秒级，电流来不及建立，可以接受；
                 * 如果以后发现上电瞬间有可测的电流尖峰，就把这行提前到 MOE 之前
                 * （需要给 bsp_power_stage_enable 加一个"指定初始占空比"的参数）。 */
                duty = _bl_reg_balance_duty(p_mod->bus_mv, p_mod->sc_mv);
                bsp_bridge_set_duty(duty);

                p_mod->state = BL_REG_STATE_ARMED;
            }
            break;

        case BL_REG_STATE_ARMED:
            p_mod->state = BL_REG_STATE_ACTIVE;
            break;

        case BL_REG_STATE_ACTIVE:
            if (p_mod->run_request != TRUE)
            {
                (VOID)bsp_power_stage_enable(FALSE);
                bsp_dump_set_duty(0.0F);
                p_mod->state = BL_REG_STATE_IDLE;
                break;
            }

            /* 当前唯一的动作：保持前馈工作点。
             * ⚠️ 这里**没有** PID、没有误差积分、没有死区策略。
             *    补闭环之前必须先有环路实测数据，理由见 bl_regulator.h 顶部。 */
            duty = _bl_reg_balance_duty(p_mod->bus_mv, p_mod->sc_mv);
            bsp_bridge_set_duty(duty);
            break;

        default:
            p_mod->state = BL_REG_STATE_IDLE;
            break;
    }
}

/********************************************************************************************
* 函数名称 :   bl_regulator_request_enable
* 功能描述 :   请求运行 / 停止
*
*              返回 FAIL 只表示"此刻有故障锁存，请求不会被受理"，
*              不代表请求被拒绝后就永远不能运行 —— 清故障后可以再请求。
*
* 输入参数 :   enable: TRUE = 请求运行，FALSE = 请求停止
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E bl_regulator_request_enable
(
    IN  BOOL    enable
)
{
    BL_REG_MODULE_T *p_mod = _bl_reg_get_module();

    p_mod->run_request = enable;

    if (enable == FALSE)
    {
        /* 停止请求立即生效，不等 1 ms 节拍：这是安全动作 */
        (VOID)bsp_power_stage_enable(FALSE);
        bsp_dump_set_duty(0.0F);

        if (p_mod->state != BL_REG_STATE_FAULT)
        {
            p_mod->state = BL_REG_STATE_IDLE;
        }

        return TYPE_RET_SUCCESS;
    }

    if (p_mod->state == BL_REG_STATE_FAULT)
    {
        return TYPE_RET_FAIL;
    }

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_regulator_set_bus_ref_mv
* 功能描述 :   设置母线参考电压（死区策略用）
*
*              ⚠️ 只是存下来，死区策略本身**尚未实现**。
*                 存它是因为状态帧要回报这个值，而且参数下发链路需要先通。
*
* 输入参数 :   ref_mv: 参考母线电压（mV）
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR
*********************************************************************************************/
TYPE_RET_E bl_regulator_set_bus_ref_mv
(
    IN  U32     ref_mv
)
{
    BL_REG_MODULE_T *p_mod = _bl_reg_get_module();

    /* 超容总压上限 19.6 V，母线最低 20.5 V，所以参考值不可能低于超容满充电压，
     * 也不可能高于 9S 电池的最高电压。范围外的值一律拒绝 ——
     * 让一个明显不合理的参考值进来，比拒绝它更危险。 */
    if ((ref_mv < BL_REG_SC_MAX_MV) || (ref_mv > BL_REG_BUS_MAX_MV))
    {
        return TYPE_RET_PARAMETER_ERROR;
    }

    p_mod->bus_ref_mv = ref_mv;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_regulator_clear_fault
* 功能描述 :   清除锁存故障
*
*              **要求此刻所有联锁都正常**才允许清除 —— 否则连续调两次就能把真故障
*              清掉，那这个锁存就没有意义了。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E bl_regulator_clear_fault
(
    VOID
)
{
    BL_REG_MODULE_T *p_mod = _bl_reg_get_module();

    _bl_reg_refresh_measurements(p_mod);

    if (_bl_reg_check_interlocks(p_mod) != BL_REG_FAULT_NONE)
    {
        return TYPE_RET_FAIL;
    }

    p_mod->fault = BL_REG_FAULT_NONE;

    if (p_mod->state == BL_REG_STATE_FAULT)
    {
        p_mod->state = BL_REG_STATE_IDLE;
    }

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_regulator_get_state / get_fault / get_*_mv / get_current_ma
* 功能描述 :   供状态上报读取的只读接口
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   见各函数
*********************************************************************************************/
BL_REG_STATE_E bl_regulator_get_state
(
    VOID
)
{
    return _bl_reg_get_module()->state;
}

U16 bl_regulator_get_fault
(
    VOID
)
{
    return _bl_reg_get_module()->fault;
}

U32 bl_regulator_get_bus_mv
(
    VOID
)
{
    return _bl_reg_get_module()->bus_mv;
}

U32 bl_regulator_get_sc_mv
(
    VOID
)
{
    return _bl_reg_get_module()->sc_mv;
}

S32 bl_regulator_get_current_ma
(
    VOID
)
{
    return _bl_reg_get_module()->current_ma;
}

#ifdef _cplusplus
}
#endif
