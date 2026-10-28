#ifndef __BSP_SPI_H__
#define __BSP_SPI_H__

#include "type.h"

#ifdef _cplusplus
extern "C" {
#endif

/********************************************************************************************
 * DRV8353 配置总线 —— SPI1
 *
 * 引脚（依据 当前原理图连接.json + DS12589 Rev6 Table 12）：
 *   DRV_SPI_CS_N  PA4   SPI1_NSS   AF5  软件片选（本层当普通 GPIO 用）
 *   DRV_SPI_SCK   PA5   SPI1_SCK   AF5
 *   DRV_SPI_MISO  PA6   SPI1_MISO  AF5  ← DRV8353 SDO，开漏，板上 4.7k 上拉
 *   DRV_SPI_MOSI  PA7   SPI1_MOSI  AF5
 *
 * 帧格式（手册 §8.5.1.1，16 位）：
 *   SDI：B15 = W（0 写 / 1 读），B14~B11 = A3~A0 地址，B10~B0 = 11 位数据
 *   SDO：B15~B11 为无关位，B10~B0 = 目标寄存器当前值
 *   数据在 SCLK 下降沿被芯片采样、上升沿输出 → 对应 STM32 的 **SPI 模式 1**
 *   （CPOL = 0，CPHA = 2EDGE）。片选拉低期间 SCLK 必须为低，两次帧之间 nSCS
 *   至少拉高 400 ns。
 *
 * 寄存器地图（手册 Table 9，只列 SPI 版 DRV8353 有的）：
 *   0x0 Fault Status 1  (R)   FAULT / VDS_OCP / GDF / UVLO / OTSD / VDS_Hx / VDS_Lx
 *   0x1 VGS Status 2    (R)   SA_OC / SB_OC / SC_OC / OTW / GDUV / VGS_Hx / VGS_Lx
 *   0x2 Driver Control  (RW)  OCP_ACT DIS_GDUV DIS_GDF OTW_REP PWM_MODE 1PWM_COM 1PWM_DIR COAST BRAKE CLR_FLT
 *   0x3 Gate Drive HS   (RW)  LOCK IDRIVEP_HS IDRIVEN_HS
 *   0x4 Gate Drive LS   (RW)  CBC TDRIVE IDRIVEP_LS IDRIVEN_LS
 *   0x5 OCP Control     (RW)  TRETRY DEAD_TIME OCP_MODE OCP_DEG VDS_LVL
 *   0x6 CSA Control     (RW)  CSA_FET VREF_DIV LS_REF CSA_GAIN DIS_SEN CSA_CAL_A/B/C SEN_LVL
 *   0x7 CAL_MODE        (RW)
 *
 * 已核实的字段编码（其余字段留给 DL 层按电机实测再定，本层不猜）：
 *   PWM_MODE (0x2, b6:5)   00 = 6x PWM（复位默认，本板用这个）
 *                          01 = 3x, 10 = 1x, 11 = 独立 PWM
 *   DEAD_TIME (0x5, b9:8)  00 = 50ns, 01 = 100ns（默认）, 10 = 200ns, 11 = 400ns
 *   CSA_GAIN (0x6, b7:6)   00 = 5, 01 = 10, 10 = 20（默认）, 11 = 40 V/V
 *   VREF_DIV (0x6, b9)     0 = 单向（零点 0V）, 1 = 双向（零点 VREF/2，默认）
 ********************************************************************************************/

/* 寄存器地址 */
#define BSP_DRV8353_REG_FAULT1                                          (0x0U)
#define BSP_DRV8353_REG_VGS2                                            (0x1U)
#define BSP_DRV8353_REG_DRV_CTRL                                        (0x2U)
#define BSP_DRV8353_REG_GATE_HS                                         (0x3U)
#define BSP_DRV8353_REG_GATE_LS                                         (0x4U)
#define BSP_DRV8353_REG_OCP_CTRL                                        (0x5U)
#define BSP_DRV8353_REG_CSA_CTRL                                        (0x6U)
#define BSP_DRV8353_REG_CAL_MODE                                        (0x7U)

/* Driver Control (0x2) 位定义 */
#define BSP_DRV8353_CTRL_CLR_FLT                                        (1U << 0)   /* 写 1 清故障锁存，自清零 */
#define BSP_DRV8353_CTRL_BRAKE                                          (1U << 1)
#define BSP_DRV8353_CTRL_COAST                                          (1U << 2)
#define BSP_DRV8353_CTRL_1PWM_DIR                                       (1U << 3)
#define BSP_DRV8353_CTRL_1PWM_COM                                       (1U << 4)
#define BSP_DRV8353_CTRL_OTW_REP                                        (1U << 7)
#define BSP_DRV8353_CTRL_DIS_GDF                                        (1U << 8)
#define BSP_DRV8353_CTRL_DIS_GDUV                                       (1U << 9)
#define BSP_DRV8353_CTRL_OCP_ACT                                        (1U << 10)
#define BSP_DRV8353_CTRL_PWM_MODE_SHIFT                                 (5U)

/* Fault Status 1 (0x0) 位定义 —— 读出来按位解释 */
#define BSP_DRV8353_FLT1_FAULT                                          (1U << 10)
#define BSP_DRV8353_FLT1_VDS_OCP                                        (1U << 9)
#define BSP_DRV8353_FLT1_GDF                                            (1U << 8)
#define BSP_DRV8353_FLT1_UVLO                                           (1U << 7)
#define BSP_DRV8353_FLT1_OTSD                                           (1U << 6)
#define BSP_DRV8353_FLT1_VDS_HA                                         (1U << 5)
#define BSP_DRV8353_FLT1_VDS_LA                                         (1U << 4)
#define BSP_DRV8353_FLT1_VDS_HB                                         (1U << 3)
#define BSP_DRV8353_FLT1_VDS_LB                                         (1U << 2)
#define BSP_DRV8353_FLT1_VDS_HC                                         (1U << 1)
#define BSP_DRV8353_FLT1_VDS_LC                                         (1U << 0)

/************************************Function***********************************************/
/********************************************************************************************
* Function name :   bsp_spi_init
* Descriptions  :   Configure SPI1 (mode 1, 16-bit, software NSS) and put nSCS high.
*                   Does NOT touch the DRV8353 registers; see bsp_drv8353_clear_faults().
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   none
*********************************************************************************************/
extern VOID bsp_spi_init
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_drv8353_write_reg
* Descriptions  :   Write one 11-bit field of a DRV8353 register
* Input         :
*                   addr: 0x0 ~ 0x7
*                   data: 11-bit value
* Output        :
*                   none
* Return        :
*                   TRUE = frame transferred, FALSE = address out of range or SPI error
*********************************************************************************************/
extern BOOL bsp_drv8353_write_reg
(
    IN  U8      addr,
    IN  U16     data
);

/********************************************************************************************
* Function name :   bsp_drv8353_read_reg
* Descriptions  :   Read one 11-bit field of a DRV8353 register
* Input         :
*                   addr: 0x0 ~ 0x7
* Output        :
*                   p_data: 11-bit register value
* Return        :
*                   TRUE = read ok, FALSE = address out of range or SPI error
*********************************************************************************************/
extern BOOL bsp_drv8353_read_reg
(
    IN  U8      addr,
    OUT U16     *p_data
);

/********************************************************************************************
* Function name :   bsp_drv8353_clear_faults
* Descriptions  :   Set CLR_FLT in the Driver Control register so the latched fault bits
*                   and the nFAULT output are released. The bit self-clears.
* Input         :
*                   none
* Output        :
*                   none
* Return        :
*                   TRUE = ok
*********************************************************************************************/
extern BOOL bsp_drv8353_clear_faults
(
    VOID
);

/********************************************************************************************
* Function name :   bsp_drv8353_read_fault
* Descriptions  :   Read Fault Status 1 and VGS Status 2 in one go
* Input         :
*                   none
* Output        :
*                   p_fault1: Fault Status 1 raw value (see BSP_DRV8353_FLT1_*)
*                   p_vgs2  : VGS Status 2 raw value, 0 when the read fails
* Return        :
*                   TRUE = at least Fault Status 1 was read
*********************************************************************************************/
extern BOOL bsp_drv8353_read_fault
(
    OUT U16     *p_fault1,
    OUT U16     *p_vgs2
);

#ifdef _cplusplus
}
#endif

#endif /* __BSP_SPI_H__ */
