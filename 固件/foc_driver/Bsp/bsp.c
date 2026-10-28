/************************************包含头文件*************************************************/
#include "stm32g4xx_hal.h"
#include "type.h"
#include "bsp_gpio.h"
#include "bsp_inverter.h"
#include "bsp_adc.h"
#include "bsp_hall.h"
#include "bsp_spi.h"
#include "bsp_can.h"
#include "bsp.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 零电压矢量对应的占空比。
 * 三相都取这个值 → 每相平均电压都是 VBUS/2 → 线电压为 0 → 绕组里没有电流。
 * 取 0.5 而不是随便一个相等值，是为了让死区在上下桥之间的影响对称。 */
#define BSP_ZERO_VECTOR_DUTY                                            (0.5F)

/* 使能功率级前检查故障时，允许重试的次数。
 * 只查一次的话，上电瞬间 VREF 还没稳、VM 还没爬上来，FAULT 位大概率是 1，
 * 会误判成"芯片坏了"。这里给几次机会，每次之间靠 HAL_Delay 让电源稳定。 */
#define BSP_FAULT_CHECK_RETRY                                           (3U)
#define BSP_FAULT_CHECK_INTERVAL_MS                                     (2U)


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   bsp_get_tick
* 功能描述 :   获取系统滴答时间
*              包一层的目的是让上层（DL/BL）不用为了一个时间戳去 include stm32g4xx_hal.h，
*              保持"只有 BSP 认识 HAL"这条边界。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   自 HAL_Init() 起经过的毫秒数
*********************************************************************************************/
U32 bsp_get_tick
(
    VOID
)
{
    return HAL_GetTick();
}

/********************************************************************************************
* 函数名称 :   bsp_probe_drv8353
* 功能描述 :   验证 DRV8353 是否真的在 SPI1 上应答
*
*              做法：往 0x7 号寄存器（Driver Configuration）写 0x0001，读回来比对，
*                    再写回 0x0000 恢复复位默认值。
*
*              为什么挑 0x7（依据 SLVSDY6A §8.6.2.6 Table 19）：
*                ① 复位值是 0x0000，而且只实现了 bit0（CAL_MODE），测试图样不会有歧义；
*                ② CAL_MODE=1 只是切换"CSA_CAL_x 触发的是哪种校准"，它自己不启动校准
*                   （§8.3.4.3 写明要配合 CSA_CAL_x 才动作），所以 1→0 走一遍没有副作用；
*                ③ 0x3/0x4/0x5 带 LOCK 位，写错会把寄存器地图锁死，不能用；
*                ④ 0x0/0x1/0x2 要么只读，要么带 COAST/BRAKE/CLR_FLT 这种会动功率级的位。
*
*              为什么不能只做"读一个寄存器看看有没有值"：
*                SDO 因为开漏 + 4.7k 上拉，线断了会读到 0x7FF；
*                但 SDO 被短到地、或者芯片根本没焊（SDO 悬空被拉低）时会读到 0x000，
*                而 0x000 恰好是大部分寄存器的复位值 —— 只读的话会把"完全没通信"
*                误判成"一切正常"。只有写进去再读回来才能区分。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = SPI 链路正常；FALSE = 没应答或图样没保持住
*********************************************************************************************/
BOOL bsp_probe_drv8353
(
    VOID
)
{
    U16 value = 0;

    /* 写测试图样 */
    if (bsp_drv8353_write_reg(BSP_DRV8353_REG_CAL_MODE, 0x0001U) != TRUE)
    {
        return FALSE;
    }

    /* 读回来比对。读失败（HAL 超时）或值不对，都算链路不通 */
    if (bsp_drv8353_read_reg(BSP_DRV8353_REG_CAL_MODE, &value) != TRUE)
    {
        (VOID)bsp_drv8353_write_reg(BSP_DRV8353_REG_CAL_MODE, 0x0000U);
        return FALSE;
    }

    /* 无论成败都恢复复位默认值，避免把芯片留在非默认状态 */
    (VOID)bsp_drv8353_write_reg(BSP_DRV8353_REG_CAL_MODE, 0x0000U);

    return (value == 0x0001U) ? TRUE : FALSE;
}

/********************************************************************************************
* 函数名称 :   bsp_power_stage_enable
* 功能描述 :   通过安全时序打开 / 关闭三相功率级
*
*              ── 打开（enable = TRUE），四步，顺序不能换 ──────────────────────────
*                1. 读 DRV8353 的 FAULT 位，有故障就不开；
*                2. 三相占空比先摆到 50%（零电压矢量）；
*                3. MOE = 1，MCU 侧六路开始按 50% 互补输出；
*                4. DRV_ENABLE = 高，栅极驱动才开始跟随输入。
*
*              为什么第 2 步是 50% 而不是 0%：
*                6x PWM 模式下占空比 0 = 下桥常通 = 三相绕组被短在一起 = 电制动。
*                一使能就来个满刹车，电流会按绕组电阻直接冲上去（本电机量级几十安），
*                轻则过流保护，重则炸管。50% 时三相平均电压都是 VBUS/2，线电压为 0，
*                绕组里没有电流，电机是自由的。
*
*              为什么 ENABLE 放最后：
*                DRV_ENABLE 为低时 DRV8353 完全忽略 INHx/INLx，六路栅极硬关断。
*                所以 MCU 可以在这个"绝对安全"的窗口里把 MOE、CCR 都摆好，
*                等芯片看到的第一个开关周期就已经是平衡的零矢量。
*                反过来的顺序（先 ENABLE 后 MOE）虽然也不会炸 —— 因为本层把
*                OISx/OISxN 都配成 0，MOE=0 时六路 INH/INL 全是低，按 6x 真值表
*                进 Hi-Z —— 但它多了一段"芯片已经在听、MCU 还没说话"的窗口，
*                这段窗口里任何一点干扰都会被立刻放大成栅极动作。先摆好再开门更稳。
*
*              ── 关闭（enable = FALSE），三步 ────────────────────────────────────
*                1. DRV_ENABLE = 低：纯硬件路径，不依赖任何软件时序，最快；
*                2. MOE = 0：MCU 侧也停；
*                3. 占空比回到 50%：给下一次使能留一个干净的起点。
*
*              ⚠️ 本函数不会清 TIM1 的刹车锁存。如果 BREAK_N 被拉低过，MOE 就再也
*                 置不起来，必须先 bsp_inverter_clear_break() 再调本函数。
*                 这个"不自动恢复"是故意的，见 bsp_inverter.h 里对 AOE 的说明。
*
* 输入参数 :   enable: TRUE = 给电机加电，FALSE = 硬关断
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS: 已执行
*              TYPE_RET_FAIL   : 拒绝执行（仅 TRUE 时可能发生，DRV8353 报故障）
*********************************************************************************************/
TYPE_RET_E bsp_power_stage_enable
(
    IN  BOOL    enable
)
{
    U16         fault1  = 0;
    U16         vgs2    = 0;
    U8          retry   = 0;
    TYPE_RET_E  result  = TYPE_RET_FAIL;

    if (enable == FALSE)
    {
        /* ── 关闭：先硬杀，再收拾软件侧 ───────────────────────────────────── */
        bsp_gpio_drv_set_enable(FALSE);
        bsp_inverter_stop();
        bsp_inverter_set_duty(BSP_ZERO_VECTOR_DUTY, BSP_ZERO_VECTOR_DUTY, BSP_ZERO_VECTOR_DUTY);

        return TYPE_RET_SUCCESS;
    }

    /* ── 打开：第 1 步，先确认芯片没在报故障 ─────────────────────────────────
     * 上电后 VM 爬升、VREF 建立都需要时间，这段时间 FAULT 位本来就是 1（UVLO）。
     * 所以给几次重试，而不是一看到 FAULT 就断定芯片坏了。 */
    for (retry = 0U; retry < BSP_FAULT_CHECK_RETRY; retry++)
    {
        if (bsp_drv8353_read_fault(&fault1, &vgs2) != TRUE)
        {
            /* 连读都读不到，说明 SPI 这会儿不通，不用再等了 */
            return TYPE_RET_FAIL;
        }

        if ((fault1 & BSP_DRV8353_FLT1_FAULT) == 0U)
        {
            result = TYPE_RET_SUCCESS;
            break;
        }

        HAL_Delay(BSP_FAULT_CHECK_INTERVAL_MS);
    }

    if (result != TYPE_RET_SUCCESS)
    {
        /* 一直是故障：保持 ENABLE 低、MOE 低，什么都不动。
         * 排查方法：调 bsp_drv8353_read_fault() 把 fault1/vgs2 打出来看是哪一位 ——
         *   bit10 FAULT / bit9 VDS_OCP / bit7 UVLO / bit6 OTSD ...
         * 注意 FAULT 这一位是"上述任意故障的或"，要先看它再看具体位。 */
        return TYPE_RET_FAIL;
    }

    /* ── 第 2 步：零电压矢量 ──────────────────────────────────────────────── */
    bsp_inverter_set_duty(BSP_ZERO_VECTOR_DUTY, BSP_ZERO_VECTOR_DUTY, BSP_ZERO_VECTOR_DUTY);

    /* ── 第 3 步：MCU 侧放行（此刻 ENABLE 仍为低，芯片不理这些电平）────────── */
    bsp_inverter_start();

    /* ── 第 4 步：打开栅极驱动 ────────────────────────────────────────────── */
    bsp_gpio_drv_set_enable(TRUE);

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bsp_init
* 功能描述 :   板级支持包初始化
*
*              顺序按安全优先级排列，每一步都在为下一步创造条件，不能重排：
*
*                1. GPIO —— 最先，而且一进来就把 DRV_ENABLE 拉低、CAN_STB 拉高。
*                            这样即使后面任何一步崩了，功率级和总线都是静默的。
*                2. PWM  —— 配 TIM1 时基/通道/死区/刹车，计数器起来但 MOE=0。
*                            计数器必须在这里起跑，因为第 3 步的 ADC 注入组
*                            等的就是 TIM1 的 TRGO。
*                3. ADC  —— 配置 + 校准 + 启动。必须在上一步之后（理由同上）。
*                4. HALL —— 三个普通输入，不带偏置。
*                5. SPI  —— 配好 SPI1 才能去探测 DRV8353。
*                6. CAN  —— 放最后配；而且只配不启动，收发器继续待机。
*
*              不在这里做的事（都属于上层的决定）：
*                · 不调 bsp_can_start()            —— 什么时候接入总线是系统级的事；
*                · 不调 bsp_power_stage_enable()   —— 什么时候给电机加电更是；
*                · 不写 DRV8353 的配置寄存器       —— 栅极电流、OCP 档位、CSA 增益
*                                                     都要按实际电机和 MOS 定，BSP 层
*                                                     不猜。芯片保持自己的复位默认值
*                                                     （6x PWM、100 ns 死区、CSA 20 V/V、
*                                                      双向 VREF 分压）。
*
*              ⚠️ 关于返回值：DRV8353 探测失败时返回 TYPE_RET_FAIL，但**其余模块照常
*                 初始化完毕**，这是故意的 —— 栅极驱动坏了的板子仍然可以用来看霍尔、
*                 走 CAN 上报，别把 FAIL 理解成"这块板废了"。
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS: 全部配置完成，DRV8353 在 SPI 上有应答
*              TYPE_RET_FAIL   : DRV8353 没有应答（SPI 断路 / 芯片没焊 / 供电缺失）
*********************************************************************************************/
TYPE_RET_E bsp_init
(
    VOID
)
{
    TYPE_RET_E result = TYPE_RET_SUCCESS;

    /* ── 1. GPIO：安全态优先 ──────────────────────────────────────────────────
     * bsp_gpio_init() 内部已经把 DRV_ENABLE 拉低、CAN_STB 拉高；
     * 这里再显式写一遍，是为了让"安全态"这件事在总入口里可见，
     * 而不是藏在子模块里靠读者去翻。多写两个寄存器没有成本。 */
    bsp_gpio_init();
    bsp_gpio_drv_set_enable(FALSE);
    bsp_gpio_can_set_stb(TRUE);

    /* ── 2. PWM：配好并让计数器起跑，MOE 保持 0 ──────────────────────────── */
    bsp_inverter_init();

    /* ── 3. ADC：此刻才有 TIM1_TRGO 可用 ─────────────────────────────────── */
    bsp_adc_init();
    bsp_adc_start();

    /* ── 4. 霍尔：三个普通输入 ───────────────────────────────────────────── */
    bsp_hall_init();

    /* ── 5. SPI：DRV8353 的配置通道 ──────────────────────────────────────── */
    bsp_spi_init();

    /* 探测一下芯片在不在。注意这里**只探测不配置** —— 探测用的是寄存器 0x7 的
     * bit0（CAL_MODE），写完立刻恢复 0x0000，不会改变芯片的任何工作方式。 */
    if (bsp_probe_drv8353() != TRUE)
    {
        result = TYPE_RET_FAIL;
    }

    /* ── 6. CAN：只配置，收发器继续待机（STB 仍为高）────────────────────── */
    bsp_can_init();

    return result;
}

#ifdef _cplusplus
}
#endif
