import { SchematicSection } from "../../../../检查/联合仿真/code_as_eda/renderer"

export const OutputAndHeaterRefs = [
  "U_ISENSE_OUT",
  "C_ISENSE_OUT_VS",
  "R_OUT_ISO_PD",
  "R_BBM_OUT_RC",
  "C_BBM_OUT_RC",
  "D_BBM_OUT_FAST",
  "R_BBM_OUT_FAST",
  "U_BBM_OUT_SCHMITT",
  "R_OUT_LED1",
  "R_OUT_LED2",
  "Q_OUT_CTL",
  "R_OUT_CMD_SER",
  "R_OUT_CMD_PD",
  "U_SW_OUT",
  "R_OUT_G1",
  "R_OUT_G2",
  "Q_OUT_1",
  "Q_OUT_2",
  "F_OUT",
  "J_BUS_CHG",
  "D_BUS_TVS",
  "R_VBUS_TOP",
  "R_VBUS_BOT",
  "C_VBUS_FILT",
  "R_LD_INH_PU",
  "Q_LD_INH_INV",
  "U_LD_PROBE_AND",
  "Q_LD_PROBE_CTL",
  "R_LD_PMOS_PU",
  "Q_LD_PROBE_SW",
  "R_LD_LIM1",
  "R_LD_LIM2",
  "D_LD_BLOCK",
  "R_HEATER_EN_PD",
  "R_HEATER_B",
  "Q_HEATER_DRV",
  "R_HEATER_PULL",
  "R_HEATER_GS_PU",
  "DZ_HEATER_GS",
  "Q_HEATER",
  "J_HEATER"
]

export function OutputAndHeater({ x, y }: { x: number; y: number }) {
  return <SchematicSection title="Output and heater" schX={x} schY={y} refs={OutputAndHeaterRefs} columns={4} />
}
