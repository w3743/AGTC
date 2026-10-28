#ifndef __BL_FOC_H__
#define __BL_FOC_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * 调制层 —— Clarke / Park / 反 Park / SVPWM + 霍尔角度
 *
 * ── 为什么这一层单独一个文件 ────────────────────────────────────────────────
 *   这几件事是**纯几何**：输入一组数，输出一组数，没有状态、不碰硬件、不含参数。
 *   把它们和状态机混在一起写，最直接的后果是"改了状态机顺手改坏了调制"，
 *   而且没法单独验证 —— 想在 PC 上算一遍 SVPWM 看看线电压对不对都做不到。
 *
 *   本文件里的 bl_foc_clarke / park / inv_park / svpwm 都是纯函数，
 *   允许在 PC 上用任意编译器单独跑（只要把 type.h 一起带上）。
 *
 * ── 坐标变换关系（αβ 静止坐标系，U 相为 α 轴）────────────────────────────────
 *   Clarke（等幅值）：
 *     iα = iu
 *     iβ = (iu + 2·iv) / √3
 *   反 Clarke（三相电流之和为 0，所以 iw = −iu −iv）：
 *     iu = iα
 *     iv = (−iα + √3·iβ) / 2
 *     iw = (−iα − √3·iβ) / 2
 *
 *   Park / 反 Park：就是旋转矩阵，正方向取"θ 增大 = 转子逆时针"，
 *   和 SVPWM 里 d 轴超前 α 轴 θ 角的约定一致。方向弄反了电机照样能转，
 *   只是转矩会变成负的、一给力矩就往反方向跑 —— 所以下面专门给了方向宏。
 *
 * ── 角度从哪来（这是本层最弱的一环，先说清楚）────────────────────────────────
 *   三相霍尔只能给出**60° 电角度的分辨率**。本层取"扇区中心角"：
 *     θ = sector × 60° + 30°（电角度）
 *   这意味着角度误差最大 ±30°。后果：
 *     · 转矩有 ±13% 量级的脉动（sin(30°) 的余弦形变），电机低速会有明显抖动；
 *     · **够用来验证相序、方向、驱动链路是否正常**；
 *     · **不够做平滑的电流矢量控制**，也不保证能自启动。
 *
 *   要往上走有两条路，都不需要动本文件的变换部分：
 *     ① 角度细分：把三路霍尔接到 TIM2（CH1/CH2/CH3，AF1，引脚已经留好），
 *        用霍尔接口模式数计数、或用输入捕获量扇区驻留时间做线性插值；
 *        换掉的只是 bl_foc_get_angle_rad() 一个函数。
 *     ② 无感：反电动势观测器 / 磁链观测器。那是另一件事，需要电流环先跑通。
 ********************************************************************************************/

/************************************Macro***************************************************/
/* √3/2 = cos(30°)，SVPWM 三相投影系数 */
#define BL_FOC_SQRT3_BY_2                                               (0.86602540378F)
/* 1/√3 */
#define BL_FOC_ONE_BY_SQRT3                                             (0.57735026919F)
/* 2π */
#define BL_FOC_TWO_PI                                                   (6.28318530718F)
/* π/6 = 30° */
#define BL_FOC_PI_BY_6                                                  (0.52359877560F)
/* π/3 = 60°，一个霍尔扇区的电角度宽度 */
#define BL_FOC_PI_BY_3                                                  (1.04719755120F)

/* 霍尔码 → 扇区 的默认表（NULL 扇区用 0xFF）。
 *
 * 依据：120° 布置的三相霍尔，转子转一圈电角度会走出 6 个有效码，
 * 标准序列是 1 → 3 → 2 → 6 → 4 → 5（反过来转就是 5 → 4 → 6 → 2 → 3 → 1）。
 * 表按"码值"索引，值是扇区号 0~5：
 *
 *   码  0    1    2    3    4    5    6    7
 *   扇区 0xFF 0    2    1    4    5    3    0xFF
 *
 * ⚠️ 这张表是**通用约定，不是这台电机的实测结果**。换电机、换霍尔安装角度、
 *    甚至同一台电机把任意两根相线对调，表都要变。
 *    现场确认方法（不用代码）：手动慢慢转电机，同时读霍尔码，
 *    记下"码按 1-3-2-6-4-5 正向出现、还是 5-4-6-2-3-1 正向出现"，
 *    前者用默认表 + 方向 +1，后者用默认表 + 方向 −1。
 *    如果连 6 个有效码都凑不齐，说明霍尔供电或接线有问题，先查硬件。
 *
 * 想换表就调 bl_foc_set_hall_table()，不用改这个宏。 */
#define BL_FOC_HALL_SECTOR_DEFAULT_INVALID                              (0xFFU)

/************************************Type****************************************************/
/* 角度来源。目前只实现了霍尔，但调用方看到的接口是"读角度"，换源时上层不用改。 */
typedef enum _BL_FOC_ANGLE_SRC_E_
{
    BL_FOC_ANGLE_SRC_HALL                                           = 0,    /* 三路霍尔，60° 分辨率 */
    BL_FOC_ANGLE_SRC_FIXED,                                                 /* 固定角度，调试用 */
    BL_FOC_ANGLE_SRC_NUM
}BL_FOC_ANGLE_SRC_E;

/* 三相输出。下标与 dl_motor_set_duty() 的参数顺序一致（U,V,W）。 */
typedef struct _BL_FOC_DUTY_T_
{
    F32     u;
    F32     v;
    F32     w;
}BL_FOC_DUTY_T;

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bl_foc_init
* Descriptions  :   Reset the modulation layer: default hall table, direction +1,
*                   zero angle offset, current-loop gains cleared (open loop).
*                   Reads the hall once so the first control tick already has a sector.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_FAIL
*********************************************************************************************/
extern TYPE_RET_E bl_foc_init
(
    VOID
);

/********************************************************************************************
* Function name :   bl_foc_set_hall_table
* Descriptions  :   Replace the hall-code -> sector table (8 entries, indexed by code).
*                   Pass NULL to restore the default.
* Input         :
*                   p_table: 8 bytes, 0~5 for a valid sector or 0xFF for "invalid code"
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR (any entry > 5 and != 0xFF)
*********************************************************************************************/
extern TYPE_RET_E bl_foc_set_hall_table
(
    IN  const U8    *p_table
);

/********************************************************************************************
* Function name :   bl_foc_set_direction
* Descriptions  :   Set the rotation direction convention used to turn a sector into an angle
*                   +1 : sector 0 -> 60 deg electrical, increasing (matches hall sequence
*                        1-3-2-6-4-5)
*                   -1 : mirrored (matches the reverse sequence)
*                   Getting this wrong does not damage anything, the motor simply pulls the
*                   other way. It is the first thing to try when "it spins but there is no
*                   torque".
* Input         :
*                   direction: +1 or -1
* Output        :
*                   none
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR
*********************************************************************************************/
extern TYPE_RET_E bl_foc_set_direction
(
    IN  S8      direction
);

/********************************************************************************************
* Function name :   bl_foc_set_angle_offset
* Descriptions  :   Electrical angle offset added on top of the sector-centre angle, in radians.
*                   This absorbs the mechanical mounting error between the hall sensors and
*                   the stator's d axis. Typical usable range is +-30 deg electrical; beyond
*                   that the sector assignment itself is probably wrong.
* Input         :
*                   offset_rad: electrical radians, wrapped into [0, 2*pi)
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_foc_set_angle_offset
(
    IN  F32     offset_rad
);

/********************************************************************************************
* Function name :   bl_foc_set_angle_fixed
* Descriptions  :   Force a fixed electrical angle. Debug only: with a fixed angle the stator
*                   field stands still, so the rotor gets pulled to one position and stays.
*                   Useful to verify the phase order (rotor should snap to the commanded angle)
*                   without any speed loop involved.
* Input         :
*                   angle_rad : electrical radians
*                   enable    : TRUE = use the fixed angle, FALSE = back to hall
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_foc_set_angle_fixed
(
    IN  F32     angle_rad,
    IN  BOOL    enable
);

/********************************************************************************************
* Function name :   bl_foc_set_current_gains
* Descriptions  :   Set the d/q current-loop PI gains.
*
*                   ⚠️ 默认全部为 0，此时电流环**完全旁路**，v_d/v_q 直接采用调用方
*                      给的电压指令（开环电压模式）。这是有意的：
*                      没有实测的 L、R，也没有示波器看过阶跃响应，任何一组系数都是编的。
*
*                   要使能电流环，至少要让 kp 非零。整定顺序建议：
*                      ① kp 从 0 开始，只给 ki=0，逐步加大到电流阶跃刚好不振荡；
*                      ② 再加 ki，从小往大调，看稳态误差消失而不出现低频摆动；
*                      ③ d 轴与 q 轴分开整，先 q 后 d（q 轴有反电动势干扰，更能暴露问题）。
*                   单位：A 的误差 → V 的输出，所以 kp 的量纲是 V/A（≈ Ω），
*                   ki 是 V/(A·s)。一个粗糙的起点是 kp ≈ 0.5·L·ω_c、ki ≈ 0.5·R·ω_c，
*                   但 ω_c 取多少取决于开关频率和采样率，最终还是要实测。
* Input         :
*                   kp_d / ki_d : d 轴增益
*                   kp_q / ki_q : q 轴增益
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_foc_set_current_gains
(
    IN  F32     kp_d,
    IN  F32     ki_d,
    IN  F32     kp_q,
    IN  F32     ki_q
);

/********************************************************************************************
* Function name :   bl_foc_get_angle_rad
* Descriptions  :   Current electrical angle used by the transforms, in [0, 2*pi)
*                   Refreshes the sector from the hall inputs first.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   electrical angle, rad
*********************************************************************************************/
extern F32 bl_foc_get_angle_rad
(
    VOID
);

/********************************************************************************************
* Function name :   bl_foc_get_hall_code
* Descriptions  :   Latest raw hall code, 0 ~ 7 (bit0=A, bit1=B, bit2=C)
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   hall code
*********************************************************************************************/
extern U8 bl_foc_get_hall_code
(
    VOID
);

/********************************************************************************************
* Function name :   bl_foc_get_sector
* Descriptions  :   Latest hall sector, 0 ~ 5, or 0xFF when the hall code is invalid
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   sector
*********************************************************************************************/
extern U8 bl_foc_get_sector
(
    VOID
);

/********************************************************************************************
* Function name :   bl_foc_run
* Descriptions  :   One step of the whole chain, ending in dl_motor_set_duty().
*
*                   read Iu/Iv -> Clarke -> Park -> [current loop] -> inverse Park
*                     -> SVPWM -> duty -> DL
*
*                   ⚠️ 调用节拍：本函数目前设计成 1 ms 调一次（和 BL 的决策节拍同频）。
*                      这意味着电流环带宽只能做到 100 Hz 量级，对"先让电机转起来、
*                      验证相序和驱动链路"够用，对真正的高性能 FOC 不够。
*                      要提速到 20 kHz，必须把它挪进 TIM1 的更新中断，那时：
*                        · 不能再用 sinf/cosf（单次上百周期），要换成 256 点查表；
*                        · Clarke/Park/SVPWM 本身没有除法以外的重活，可以直接搬；
*                        · dl_motor_set_duty 要确认不会被别处并发调用。
*
* 输入参数 :   v_d / v_q: 电压指令，伏特。电流环关闭时直接使用；打开时被环路输出覆盖。
* 输出参数 :   无
* 返回值   :   TYPE_RET_SUCCESS / TYPE_RET_FAIL（母线电压为 0 时无法调制）
*********************************************************************************************/
extern TYPE_RET_E bl_foc_run
(
    IN  F32     v_d,
    IN  F32     v_q
);

/********************************************************************************************
* Function name :   bl_foc_clarke
* Descriptions  :   Clarke transform (amplitude invariant), three phase -> alpha/beta
*                   iw is not passed in: with no neutral wire iu + iv + iw = 0.
* Input         :
*                   iu / iv : phase currents, any consistent unit
* Output        :
*                   p_alpha / p_beta : same unit as the input
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_foc_clarke
(
    IN  F32     iu,
    IN  F32     iv,
    OUT F32     *p_alpha,
    OUT F32     *p_beta
);

/********************************************************************************************
* Function name :   bl_foc_park
* Descriptions  :   Park transform, alpha/beta -> d/q
* Input         :
*                   alpha / beta : stationary frame components
*                   sin_t / cos_t: sine and cosine of the electrical angle
* Output        :
*                   p_d / p_q : rotating frame components
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_foc_park
(
    IN  F32     alpha,
    IN  F32     beta,
    IN  F32     sin_t,
    IN  F32     cos_t,
    OUT F32     *p_d,
    OUT F32     *p_q
);

/********************************************************************************************
* Function name :   bl_foc_inv_park
* Descriptions  :   Inverse Park transform, d/q -> alpha/beta
* Input         :
*                   d / q         : rotating frame components (volts)
*                   sin_t / cos_t : sine and cosine of the electrical angle
* Output        :
*                   p_alpha / p_beta : stationary frame components (volts)
* Return        :
*                   none
*********************************************************************************************/
extern VOID bl_foc_inv_park
(
    IN  F32     d,
    IN  F32     q,
    IN  F32     sin_t,
    IN  F32     cos_t,
    OUT F32     *p_alpha,
    OUT F32     *p_beta
);

/********************************************************************************************
* Function name :   bl_foc_svpwm
* Descriptions  :   Space vector PWM via min-max (zero-sequence) injection.
*
*                   Algorithm, three steps:
*                     1. project alpha/beta onto the three phase axes -> va0, vb0, vc0
*                     2. subtract the midpoint of (max, min) from all three
*                        -> the zero-sequence component that turns SPWM into SVPWM
*                     3. duty = 0.5 + v / vbus
*
*                   Why step 3 looks like that: on a half bridge with high-side duty d the
*                   average output is d*vbus; the winding neutral of a balanced three-phase
*                   load sits at vbus/2, so the phase voltage relative to neutral is
*                   (d - 0.5) * vbus. Inverting gives d = 0.5 + v/vbus.
*
*                   Min-max injection vs the classical sector formula: they produce the same
*                   output, but min-max has no sector lookup, no trigonometric table and no
*                   division by sin(60 deg - theta). Fewer branches in a control loop is
*                   worth more than matching a textbook picture.
*
*                   Linear range: after injection the phase voltage peak can reach vbus/sqrt(3)
*                   (0.577 vbus), i.e. 15 % more than SPWM's 0.5 vbus. Beyond that the three
*                   duties clip and the output stops tracking the command -- that is
*                   over-modulation, and this function does NOT compensate for it; it just
*                   clamps each duty into [0,1].
*
* Input         :
*                   alpha / beta : voltage command in the stationary frame, volts
*                   vbus         : measured bus voltage, volts (must be > 0)
* Output        :
*                   p_duty       : three duties clamped to [0, 1]
* Return        :
*                   TYPE_RET_SUCCESS / TYPE_RET_PARAMETER_ERROR (NULL pointer or vbus <= 0)
*********************************************************************************************/
extern TYPE_RET_E bl_foc_svpwm
(
    IN  F32             alpha,
    IN  F32             beta,
    IN  F32             vbus,
    OUT BL_FOC_DUTY_T   *p_duty
);

#ifdef _cplusplus
}
#endif

#endif /* __BL_FOC_H__ */
