/************************************包含头文件*************************************************/
#include <stddef.h>
#include <math.h>
#include "type.h"
#include "bsp_hall.h"
#include "dl_motor.h"
#include "dl_sense.h"
#include "bl_foc.h"


#ifdef _cplusplus
extern "C" {
#endif

/************************************宏定义****************************************************/
/* 电流环的积分步长（秒）。
 *
 * 写死 1 ms 是有前提的：bl_foc_run() 由 bl_task_handler() 按 1 ms 节拍调用。
 * 如果哪天把调用节拍改了而忘了改这里，积分项会按比例偏 —— 表现是"同样的 ki，
 * 电流环的响应变快或变慢"，很像"电机参数变了"，极难查。
 * 所以这个宏和 bl.h 里的节拍说明必须一起看。 */
#define BL_FOC_DT_S                                                     (0.001F)

/* 霍尔码 → 扇区 默认表。索引是霍尔码（bit0=A, bit1=B, bit2=C），值是扇区 0~5。
 * 0 与 7 是无效码（霍尔断线、没供电、或转子正好在两个扇区边界上）。 */
static const U8 s_bl_foc_hall_table_default[8] =
{
    BL_FOC_HALL_SECTOR_DEFAULT_INVALID,     /* code 0 = 000 : 无效 */
    0U,                                     /* code 1 = 001 : 扇区 0 */
    2U,                                     /* code 2 = 010 : 扇区 2 */
    1U,                                     /* code 3 = 011 : 扇区 1 */
    4U,                                     /* code 4 = 100 : 扇区 4 */
    5U,                                     /* code 5 = 101 : 扇区 5 */
    3U,                                     /* code 6 = 110 : 扇区 3 */
    BL_FOC_HALL_SECTOR_DEFAULT_INVALID      /* code 7 = 111 : 无效 */
};


/************************************数据类型定义************************************************/
typedef struct _BL_FOC_MODULE_T_
{
    U8                      hall_table[8];                              /* 码 → 扇区 */
    U8                      hall_code;                                  /* 最近一次霍尔码 */
    U8                      sector;                                     /* 最近一次扇区 */
    S8                      direction;                                  /* +1 / -1 */
    F32                     angle_offset_rad;                           /* 装配角偏置 */
    F32                     angle_fixed_rad;                            /* 调试用固定角 */
    BOOL                    angle_fixed_enable;                         /* 固定角使能 */
    F32                     angle_rad;                                  /* 最近一次算出的电角度 */
    BOOL                    angle_valid;                                /* 扇区是否有效 */
    /* 电流环 */
    F32                     kp_d;
    F32                     ki_d;
    F32                     kp_q;
    F32                     ki_q;
    F32                     int_d;                                      /* d 轴积分项，单位 V */
    F32                     int_q;                                      /* q 轴积分项，单位 V */
}BL_FOC_MODULE_T;


/************************************静态函数声明************************************************/
static BL_FOC_MODULE_T * _bl_foc_get_module
(
    VOID
);

static F32 _bl_foc_clamp_f32
(
    IN  F32     value,
    IN  F32     limit
);

static F32 _bl_foc_wrap_angle
(
    IN  F32     angle_rad
);

static BOOL _bl_foc_refresh_sector
(
    VOID
);

static F32 _bl_foc_sector_to_angle
(
    IN  U8      sector
);

static VOID _bl_foc_current_loop
(
    IN  F32     i_d,
    IN  F32     i_q,
    IN  F32     v_limit,
    OUT F32     *p_v_d,
    OUT F32     *p_v_q
);


/************************************静态变量定义************************************************/
static BL_FOC_MODULE_T s_bl_foc_module;


/************************************函数实现**************************************************/
/********************************************************************************************
* 函数名称 :   _bl_foc_get_module
* 功能描述 :   获取调制模块实例指针
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   模块指针
*********************************************************************************************/
static BL_FOC_MODULE_T * _bl_foc_get_module
(
    VOID
)
{
    return &s_bl_foc_module;
}

/********************************************************************************************
* 函数名称 :   _bl_foc_clamp_f32
* 功能描述 :   对称钳位到 [-limit, +limit]
*
*              用 `!(value >= -limit)` 而不是 `value < -limit`，是为了把 NaN 一起收进来：
*              NaN 与任何数比较都为假，写成 `value < -limit` 会让 NaN 原封不动漏过去，
*              一路传进 SVPWM 变成随机占空比。控制链里任何一处 NaN 都应该被就地拦下。
*
* 输入参数 :   value: 原始值；limit: 正的限制幅度
* 输出参数 :   无
* 返回值   :   -limit ~ +limit
*********************************************************************************************/
static F32 _bl_foc_clamp_f32
(
    IN  F32     value,
    IN  F32     limit
)
{
    if (!(value >= -limit))
    {
        return -limit;
    }

    if (value > limit)
    {
        return limit;
    }

    return value;
}

/********************************************************************************************
* 函数名称 :   _bl_foc_wrap_angle
* 功能描述 :   把角度折到 [0, 2π)
*
*              用 while 而不是 fmodf：角度来源决定了它最多偏出一两个 2π
*              （扇区角本身在 [0,2π)，加上 ±30° 的偏置最多差一个周期），
*              一两次减法就收敛，比调库函数快也更可预测。
*              保险起见还是用 while，万一偏置被人设成 1e6 也不会死循环之外出别的怪事。
*
* 输入参数 :   angle_rad: 任意角度
* 输出参数 :   无
* 返回值   :   [0, 2π)
*********************************************************************************************/
static F32 _bl_foc_wrap_angle
(
    IN  F32     angle_rad
)
{
    F32 angle = angle_rad;

    /* NaN 进去会在这里被拦下，回到 0 而不是在 while 里空转 */
    if (!(angle == angle))
    {
        return 0.0F;
    }

    while (angle >= BL_FOC_TWO_PI)
    {
        angle -= BL_FOC_TWO_PI;
    }

    while (angle < 0.0F)
    {
        angle += BL_FOC_TWO_PI;
    }

    return angle;
}

/********************************************************************************************
* 函数名称 :   _bl_foc_refresh_sector
* 功能描述 :   读一次霍尔，更新 hall_code / sector / angle_valid
*
*              无效码（000 / 111）时**保留上一次的有效扇区**，只把 angle_valid 置 FALSE。
*              为什么不清零：霍尔在扇区边界上会有一瞬间的抖动或全 0，
*              如果每次都把角度清成 0，那一瞬间的电压矢量会突然跳到 α 轴方向上，
*              在电机上表现为"隔一会儿抽一下"。保留上一个扇区，让角度最多滞后 60°，
*              比突然跳到 0 温和得多。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TRUE = 本次读到有效扇区
*********************************************************************************************/
static BOOL _bl_foc_refresh_sector
(
    VOID
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    U8              code  = 0U;
    U8              sector;

    /* bsp_hall_read 的返回值只表示"码是否属于六个有效态"，
     * 我们还要自己查表才能知道是哪个扇区 —— 两件事都做。 */
    (VOID)bsp_hall_read(&code);

    p_mod->hall_code = code;
    sector           = p_mod->hall_table[code & 0x07U];

    if (sector == BL_FOC_HALL_SECTOR_DEFAULT_INVALID)
    {
        p_mod->angle_valid = FALSE;
        return FALSE;
    }

    p_mod->sector      = sector;
    p_mod->angle_valid = TRUE;

    return TRUE;
}

/********************************************************************************************
* 函数名称 :   _bl_foc_sector_to_angle
* 功能描述 :   扇区 → 电角度（扇区中心）
*
*              正方向：θ = sector × 60° + 30°
*              反方向：θ = −(sector × 60° + 30°)
*
*              为什么取中心而不是边界：
*                霍尔跳变发生在扇区边界上，也就是说边界处我们**只知道**"刚好在这一刻越过"，
*                而扇区内部完全没有信息。对整个扇区取同一个角度时，取中心能让最大误差
*                均分到两侧各 ±30°；取边界的话误差会单边累积到 60°，更差。
*
*              反方向为什么是取负：三相霍尔反向安装（或者相线对调两根）时，
*              码序列会倒着走。用负角度相当于镜像整个 αβ 平面，
*              让"扇区号增大"仍然对应"电角度增大"，上层的 Park 变换就不用改。
*
* 输入参数 :   sector: 0 ~ 5
* 输出参数 :   无
* 返回值   :   电角度，rad
*********************************************************************************************/
static F32 _bl_foc_sector_to_angle
(
    IN  U8      sector
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    F32             angle;

    angle = ((F32)sector) * BL_FOC_PI_BY_3;

    if (p_mod->direction >= 0)
    {
        angle += BL_FOC_PI_BY_6;
    }
    else
    {
        angle = -(angle + BL_FOC_PI_BY_6);
    }

    return _bl_foc_wrap_angle(angle);
}

/********************************************************************************************
* 函数名称 :   _bl_foc_current_loop
* 功能描述 :   d/q 电流 PI（含积分限幅），输出 d/q 电压指令
*
*              输出 = kp·e + ∫ki·e·dt
*
*              ── 为什么没有解耦项和反电动势前馈 ─────────────────────────────────
*              完整的形式应该是
*                vd = kp·ed + ∫ + (−ω·L·iq)
*                vq = kp·eq + ∫ + (+ω·L·id + ω·ψ)
*              那两项需要 L（相电感）和 ψ（永磁磁链），都要实测。
*              没给的情况下少这两项，环路在低速基本正常，高速会因交叉耦合变差 ——
*              但至少不会因为"猜了一个 L"而把环路整成振荡。
*              补的时候只改这个函数，Clarke/Park/SVPWM 一行都不用动。
*
*              ── 积分限幅 ────────────────────────────────────────────────────────
*              积分项单独限到 ±v_limit（而不是 ±v_limit/2 之类），因为稳态时
*              输出本来就主要由积分项承担。限幅的目的只是防饱和，不是限输出。
*              输出端还有一次总限幅（在 bl_foc_run 里），两道都有必要：
*              没有积分限幅，堵转时积分会一直涨，退饱和要等很久；
*              没有输出限幅，kp 很大时一步就能给出远超 vbus 的指令。
*
* 输入参数 :   i_d / i_q: 实测 d/q 电流，A
*              v_limit  : 电压限幅，V
* 输出参数 :   p_v_d / p_v_q: d/q 电压指令，V
* 返回值   :   无
*********************************************************************************************/
static VOID _bl_foc_current_loop
(
    IN  F32     i_d,
    IN  F32     i_q,
    IN  F32     v_limit,
    OUT F32     *p_v_d,
    OUT F32     *p_v_q
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    F32             err_d;
    F32             err_q;

    /* 电流指令当前固定为 id*=0、iq*=0。
     * id*=0 是表贴式永磁电机的标准做法（不弱磁时 d 轴电流只产生热不产生转矩）。
     * iq*=0 意味着"现在不要转矩" —— 本层只负责调制，转矩指令由 bl_motor 给。
     * 等接上速度环时，把 iq* 改成从外面传进来即可。 */
    err_d = 0.0F - i_d;
    err_q = 0.0F - i_q;

    p_mod->int_d = _bl_foc_clamp_f32(p_mod->int_d + (p_mod->ki_d * err_d * BL_FOC_DT_S), v_limit);
    p_mod->int_q = _bl_foc_clamp_f32(p_mod->int_q + (p_mod->ki_q * err_q * BL_FOC_DT_S), v_limit);

    if (p_v_d != NULL)
    {
        *p_v_d = _bl_foc_clamp_f32((p_mod->kp_d * err_d) + p_mod->int_d, v_limit);
    }

    if (p_v_q != NULL)
    {
        *p_v_q = _bl_foc_clamp_f32((p_mod->kp_q * err_q) + p_mod->int_q, v_limit);
    }
}

/********************************************************************************************
* 函数名称 :   bl_foc_init
* 功能描述 :   初始化调制层
*
*              霍尔表先拷默认值，方向 +1，偏置 0，电流环增益全 0（= 开环电压模式）。
*              最后读一次霍尔，让第一个控制节拍就有扇区可用。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS
*********************************************************************************************/
TYPE_RET_E bl_foc_init
(
    VOID
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    U8              i;

    for (i = 0U; i < 8U; i++)
    {
        p_mod->hall_table[i] = s_bl_foc_hall_table_default[i];
    }

    p_mod->hall_code          = 0U;
    p_mod->sector             = 0U;
    p_mod->direction          = 1;
    p_mod->angle_offset_rad   = 0.0F;
    p_mod->angle_fixed_rad    = 0.0F;
    p_mod->angle_fixed_enable = FALSE;
    p_mod->angle_rad          = 0.0F;
    p_mod->angle_valid        = FALSE;

    p_mod->kp_d = 0.0F;
    p_mod->ki_d = 0.0F;
    p_mod->kp_q = 0.0F;
    p_mod->ki_q = 0.0F;
    p_mod->int_d = 0.0F;
    p_mod->int_q = 0.0F;

    (VOID)_bl_foc_refresh_sector();

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_foc_set_hall_table
* 功能描述 :   替换霍尔码 → 扇区表
*
*              校验规则：每一项必须是 0~5，或者 0xFF（无效）。
*              为什么不容忍别的值（比如 6）：扇区 6 不存在，越界的扇区号会让
*              sector × 60° 算出 390°，虽然 _bl_foc_wrap_angle 会把它折回来，
*              但那说明调用方的表是错的，早报比晚报好。
*
* 输入参数 :   p_table: 8 项，NULL 表示恢复默认
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR
*********************************************************************************************/
TYPE_RET_E bl_foc_set_hall_table
(
    IN  const U8    *p_table
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    U8              i;

    if (p_table == NULL)
    {
        for (i = 0U; i < 8U; i++)
        {
            p_mod->hall_table[i] = s_bl_foc_hall_table_default[i];
        }

        return TYPE_RET_SUCCESS;
    }

    for (i = 0U; i < 8U; i++)
    {
        if ((p_table[i] > 5U) && (p_table[i] != BL_FOC_HALL_SECTOR_DEFAULT_INVALID))
        {
            return TYPE_RET_PARAMETER_ERROR;
        }
    }

    /* 先整表校验再整体拷贝：边校验边写会让表停在一半新一半旧的状态，
     * 那个中间态里同一个扇区可能被两个码同时指向，角度会来回跳。 */
    for (i = 0U; i < 8U; i++)
    {
        p_mod->hall_table[i] = p_table[i];
    }

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_foc_set_direction
* 功能描述 :   设定转向约定 +1 / −1
* 输入参数 :   direction: 只接受 +1 与 −1
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR
*********************************************************************************************/
TYPE_RET_E bl_foc_set_direction
(
    IN  S8      direction
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();

    if ((direction != 1) && (direction != -1))
    {
        return TYPE_RET_PARAMETER_ERROR;
    }

    p_mod->direction = direction;

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_foc_set_angle_offset
* 功能描述 :   设定电角度偏置（吸收霍尔安装位置与 d 轴的机械误差）
* 输入参数 :   offset_rad: 电角度弧度
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_foc_set_angle_offset
(
    IN  F32     offset_rad
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();

    p_mod->angle_offset_rad = _bl_foc_wrap_angle(offset_rad);
}

/********************************************************************************************
* 函数名称 :   bl_foc_set_angle_fixed
* 功能描述 :   强制使用固定角度（调试用）
*
*              典型用法：把角度固定成 0，然后给一个很小的 v_q，转子会被拉向 α 轴方向并
*              停在那里。这时候用手去感受的是"有没有保持转矩"，而不是"能不能转" ——
*              这是验证相序和霍尔表最快的一步，比看着电机乱转有用得多。
*
* 输入参数 :   angle_rad: 电角度；enable: TRUE 启用
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_foc_set_angle_fixed
(
    IN  F32     angle_rad,
    IN  BOOL    enable
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();

    p_mod->angle_fixed_rad    = _bl_foc_wrap_angle(angle_rad);
    p_mod->angle_fixed_enable = enable;
}

/********************************************************************************************
* 函数名称 :   bl_foc_set_current_gains
* 功能描述 :   设定 d/q 电流环 PI 增益，全 0 = 旁路电流环
* 输入参数 :   kp_d / ki_d / kp_q / ki_q
* 输出参数 :   无
* 返回值   :   无
*********************************************************************************************/
VOID bl_foc_set_current_gains
(
    IN  F32     kp_d,
    IN  F32     ki_d,
    IN  F32     kp_q,
    IN  F32     ki_q
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();

    p_mod->kp_d = kp_d;
    p_mod->ki_d = ki_d;
    p_mod->kp_q = kp_q;
    p_mod->ki_q = ki_q;

    /* 换增益必须清积分：旧的积分项是按旧 ki 攒起来的，留着会让第一步的输出
     * 突然跳一下，看起来像"改参数把系统改炸了"。 */
    p_mod->int_d = 0.0F;
    p_mod->int_q = 0.0F;
}

/********************************************************************************************
* 函数名称 :   bl_foc_get_angle_rad
* 功能描述 :   读当前电角度（会先刷新一次霍尔扇区）
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   [0, 2π) 电角度
*********************************************************************************************/
F32 bl_foc_get_angle_rad
(
    VOID
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    F32             angle;

    (VOID)_bl_foc_refresh_sector();

    if (p_mod->angle_fixed_enable == TRUE)
    {
        p_mod->angle_rad = p_mod->angle_fixed_rad;
        return p_mod->angle_rad;
    }

    angle = _bl_foc_sector_to_angle(p_mod->sector);
    angle += p_mod->angle_offset_rad;
    p_mod->angle_rad = _bl_foc_wrap_angle(angle);

    return p_mod->angle_rad;
}

/********************************************************************************************
* 函数名称 :   bl_foc_get_hall_code
* 功能描述 :   读霍尔码（每次调用都重新采一次引脚）
*
*              ⚠️ 这里必须刷新，不能用缓存。原因很实际：
*                 待机时（IDLE）没有调制在跑，也就没人调 bl_foc_get_angle_rad()，
*                 缓存会一直停在上电时的那个值。而"手动慢慢转电机、看霍尔码序列对不对"
*                 恰恰是在待机状态下做的 —— 用缓存的话 CAN 上报的永远是同一个数，
*                 会让人以为霍尔坏了。
*                 代价是一次 GPIO 读（几十个周期），1 ms 一次可以忽略。
*
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   0 ~ 7
*********************************************************************************************/
U8 bl_foc_get_hall_code
(
    VOID
)
{
    (VOID)_bl_foc_refresh_sector();

    return _bl_foc_get_module()->hall_code;
}

/********************************************************************************************
* 函数名称 :   bl_foc_get_sector
* 功能描述 :   读扇区（每次调用都重新采一次霍尔）
*                  0 ~ 5，或 0xFF（霍尔码无效）
* 输入参数 :   无
* 输出参数 :   无
* 返回值   :   扇区
*********************************************************************************************/
U8 bl_foc_get_sector
(
    VOID
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();

    (VOID)_bl_foc_refresh_sector();

    if (p_mod->angle_valid != TRUE)
    {
        return BL_FOC_HALL_SECTOR_DEFAULT_INVALID;
    }

    return p_mod->sector;
}

/********************************************************************************************
* 函数名称 :   bl_foc_run
* 功能描述 :   完整调制链的一步：采样 → Clarke → Park → [电流环] → 反 Park → SVPWM → 输出
*
*              母线电压为 0 时返回 FAIL 且**不下发占空比** —— 分母是 vbus，
*              除零会得到 inf/NaN，钳位后变成 0 或 1 的极端占空比。
*              母线没电说明功率级也不该在动，保持上一次的值（很可能是零矢量）最安全。
*
*              ── 电压限幅为什么是 vbus/√3 ──────────────────────────────────────
*              SVPWM 的线性区上限是相电压峰值 vbus/√3（0.577·vbus），
*              比 SPWM 的 0.5·vbus 高 15.5%，这正是零序注入换来的。
*              超过这个值三相占空比会被钳到 0/1，输出不再跟随指令 —— 那叫过调制。
*              本层在进 SVPWM 之前先在 dq 上做圆限幅（保留矢量方向），
*              比让 SVPWM 内部逐相钳位好：逐相钳位会改变矢量方向，转矩会突变。
*
* 输入参数 :   v_d / v_q: 电压指令，V；电流环开启时被环路输出覆盖
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
TYPE_RET_E bl_foc_run
(
    IN  F32     v_d,
    IN  F32     v_q
)
{
    BL_FOC_MODULE_T *p_mod = _bl_foc_get_module();
    F32             i_u;
    F32             i_v;
    F32             i_alpha;
    F32             i_beta;
    F32             i_d;
    F32             i_q;
    F32             v_d_out;
    F32             v_q_out;
    F32             v_alpha;
    F32             v_beta;
    F32             vbus;
    F32             v_limit;
    F32             sin_t;
    F32             cos_t;
    F32             angle_rad;
    F32             magnitude;
    F32             scale;
    BL_FOC_DUTY_T   duty;

    /* ── 1. 母线电压：调制的前提 ───────────────────────────────────────────── */
    vbus = dl_sense_get_bus_v();

    if (!(vbus > 1.0F))
    {
        /* 1 V 以下当作"没有母线"。真给一个 0.5 V 的母线去算占空比，
         * 算出来的 v/vbus 会是个巨大的数，钳位之后就是满占空比 —— 危险。 */
        return TYPE_RET_FAIL;
    }

    /* ── 2. 三相电流 → αβ ─────────────────────────────────────────────────── */
    i_u = ((F32)dl_sense_get_phase_current_ma(DL_SENSE_PHASE_U)) / 1000.0F;
    i_v = ((F32)dl_sense_get_phase_current_ma(DL_SENSE_PHASE_V)) / 1000.0F;

    bl_foc_clarke(i_u, i_v, &i_alpha, &i_beta);

    /* ── 3. 电角度 ────────────────────────────────────────────────────────── */
    angle_rad = bl_foc_get_angle_rad();
    sin_t     = sinf(angle_rad);
    cos_t     = cosf(angle_rad);

    /* ── 4. αβ → dq ───────────────────────────────────────────────────────── */
    bl_foc_park(i_alpha, i_beta, sin_t, cos_t, &i_d, &i_q);

    /* ── 5. 电压指令：电流环开着就用环路的，否则用调用方给的 ───────────────── */
    v_limit = vbus * BL_FOC_ONE_BY_SQRT3;

    if ((p_mod->kp_d != 0.0F) || (p_mod->ki_d != 0.0F)
     || (p_mod->kp_q != 0.0F) || (p_mod->ki_q != 0.0F))
    {
        _bl_foc_current_loop(i_d, i_q, v_limit, &v_d_out, &v_q_out);
    }
    else
    {
        v_d_out = v_d;
        v_q_out = v_q;
    }

    /* ── 6. dq 上的圆限幅 ─────────────────────────────────────────────────── */
    magnitude = sqrtf((v_d_out * v_d_out) + (v_q_out * v_q_out));

    if (!(magnitude == magnitude))
    {
        /* NaN 已经在 _bl_foc_clamp_f32 里拦过一道，这里是第二道保险：
         * 万一 v_d/v_q 从参数里直接进来带了 NaN，也不能让它进 SVPWM。 */
        v_d_out = 0.0F;
        v_q_out = 0.0F;
    }
    else if (magnitude > v_limit)
    {
        scale   = v_limit / magnitude;
        v_d_out *= scale;
        v_q_out *= scale;
    }

    /* ── 7. dq → αβ → SVPWM → 占空比 ─────────────────────────────────────── */
    bl_foc_inv_park(v_d_out, v_q_out, sin_t, cos_t, &v_alpha, &v_beta);

    if (bl_foc_svpwm(v_alpha, v_beta, vbus, &duty) != TYPE_RET_SUCCESS)
    {
        return TYPE_RET_FAIL;
    }

    /* ── 8. 下发 ──────────────────────────────────────────────────────────── */
    dl_motor_set_duty(duty.u, duty.v, duty.w);

    return TYPE_RET_SUCCESS;
}

/********************************************************************************************
* 函数名称 :   bl_foc_clarke
* 功能描述 :   Clarke 变换（等幅值），三相 → αβ
* 输入参数 :   iu / iv: 相电流（单位任意，输出同单位）
* 输出参数 :   p_alpha / p_beta
* 返回值   :   无
*********************************************************************************************/
VOID bl_foc_clarke
(
    IN  F32     iu,
    IN  F32     iv,
    OUT F32     *p_alpha,
    OUT F32     *p_beta
)
{
    if (p_alpha != NULL)
    {
        *p_alpha = iu;
    }

    if (p_beta != NULL)
    {
        /* iβ = (iu + 2·iv) / √3 —— 由 iβ = (iv − iw)/√3 与 iw = −iu −iv 推得 */
        *p_beta = (iu + (2.0F * iv)) * BL_FOC_ONE_BY_SQRT3;
    }
}

/********************************************************************************************
* 函数名称 :   bl_foc_park
* 功能描述 :   Park 变换，αβ → dq
* 输入参数 :   alpha / beta / sin_t / cos_t
* 输出参数 :   p_d / p_q
* 返回值   :   无
*********************************************************************************************/
VOID bl_foc_park
(
    IN  F32     alpha,
    IN  F32     beta,
    IN  F32     sin_t,
    IN  F32     cos_t,
    OUT F32     *p_d,
    OUT F32     *p_q
)
{
    if (p_d != NULL)
    {
        *p_d = (alpha * cos_t) + (beta * sin_t);
    }

    if (p_q != NULL)
    {
        *p_q = (beta * cos_t) - (alpha * sin_t);
    }
}

/********************************************************************************************
* 函数名称 :   bl_foc_inv_park
* 功能描述 :   反 Park 变换，dq → αβ
* 输入参数 :   d / q / sin_t / cos_t
* 输出参数 :   p_alpha / p_beta
* 返回值   :   无
*********************************************************************************************/
VOID bl_foc_inv_park
(
    IN  F32     d,
    IN  F32     q,
    IN  F32     sin_t,
    IN  F32     cos_t,
    OUT F32     *p_alpha,
    OUT F32     *p_beta
)
{
    if (p_alpha != NULL)
    {
        *p_alpha = (d * cos_t) - (q * sin_t);
    }

    if (p_beta != NULL)
    {
        *p_beta = (d * sin_t) + (q * cos_t);
    }
}

/********************************************************************************************
* 函数名称 :   bl_foc_svpwm
* 功能描述 :   SVPWM（零序注入实现），αβ 电压 → 三相占空比
*
*              ── 零序注入的几何解释 ────────────────────────────────────────────────
*              三相桥能合成的电压矢量，在以 (va, vb, vc) 为坐标的空间里是一个正六边形。
*              纯正弦（SPWM）的轨迹是这个六边形的**内切圆**，半径只能到 vbus/2。
*              但三相绕组是星形连接、没有中线，给三相同时叠加任意相同的电压（零序），
*              线电压完全不变，绕组里看不到这个共模量。
*              所以可以把轨迹"往上推"成六边形的**外接圆**，半径 vbus/√3，多出 15.5%。
*              取 max 与 min 的中点作为共模量，正好是让轨迹贴住六边形边缘的那种推法。
*
*              ── 和教科书上"扇区 + 作用时间"的写法是什么关系 ────────────────────
*              数学上完全等价。教科书写法要先判扇区（3 个 if）、查 60° 正弦表、
*              再做两次除法求 T1/T2；本写法只要 2 次比较、1 次减法、3 次除法。
*              在控制环里少几个分支比"和图对照得上"重要。
*
* 输入参数 :   alpha / beta: αβ 电压指令，V
*              vbus        : 母线电压，V，必须 > 0
* 输出参数 :   p_duty: 三相占空比，已钳到 [0,1]
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR
*********************************************************************************************/
TYPE_RET_E bl_foc_svpwm
(
    IN  F32             alpha,
    IN  F32             beta,
    IN  F32             vbus,
    OUT BL_FOC_DUTY_T   *p_duty
)
{
    F32 v_a;
    F32 v_b;
    F32 v_c;
    F32 v_max;
    F32 v_min;
    F32 v_offset;
    F32 inv_vbus;

    if (p_duty == NULL)
    {
        return TYPE_RET_PARAMETER_ERROR;
    }

    if (!(vbus > 0.0F))
    {
        return TYPE_RET_PARAMETER_ERROR;
    }

    /* ── 1. 投影到三相轴（就是反 Clarke），得到三相相对中点的电压 ────────── */
    v_a = alpha;
    v_b = ((-0.5F) * alpha) + (BL_FOC_SQRT3_BY_2 * beta);
    v_c = ((-0.5F) * alpha) - (BL_FOC_SQRT3_BY_2 * beta);

    /* ── 2. 零序注入：减去 (max + min) / 2 ───────────────────────────────── */
    v_max = v_a;
    v_min = v_a;

    if (v_b > v_max) { v_max = v_b; }
    if (v_c > v_max) { v_max = v_c; }
    if (v_b < v_min) { v_min = v_b; }
    if (v_c < v_min) { v_min = v_c; }

    v_offset = (-0.5F) * (v_max + v_min);

    v_a += v_offset;
    v_b += v_offset;
    v_c += v_offset;

    /* ── 3. 电压 → 占空比：d = 0.5 + v / vbus ────────────────────────────── */
    inv_vbus = 1.0F / vbus;

    p_duty->u = 0.5F + (v_a * inv_vbus);
    p_duty->v = 0.5F + (v_b * inv_vbus);
    p_duty->w = 0.5F + (v_c * inv_vbus);

    /* ── 4. 逐相钳位 ───────────────────────────────────────────────────────
     * 线性区内这一步永远不会生效（零序注入保证三个值都落在 ±vbus/2 以内）。
     * 它只在过调制时起作用 —— 让输出变成六边形顶点上的"削顶"，而不是发出
     * 超出 [0,1] 的数字被 CCR 截断成奇怪的值。 */
    if (!(p_duty->u >= 0.0F)) { p_duty->u = 0.0F; }
    if (!(p_duty->v >= 0.0F)) { p_duty->v = 0.0F; }
    if (!(p_duty->w >= 0.0F)) { p_duty->w = 0.0F; }

    if (p_duty->u > 1.0F) { p_duty->u = 1.0F; }
    if (p_duty->v > 1.0F) { p_duty->v = 1.0F; }
    if (p_duty->w > 1.0F) { p_duty->w = 1.0F; }

    return TYPE_RET_SUCCESS;
}

#ifdef _cplusplus
}
#endif
