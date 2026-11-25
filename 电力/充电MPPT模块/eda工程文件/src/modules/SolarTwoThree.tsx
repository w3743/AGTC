import { SchematicSection } from "../../../../检查/联合仿真/code_as_eda/renderer"

export const SolarTwoThreeRefs = [
  "C_IPV2_VS",
  "U_DRV_PV2",
  "C_PV2_VCC",
  "D_PV2_BOOT", "R_PV2_BOOT_LIM",
  "C_PV2_BOOT",
  "L_PV2",
  "Q_PV2_LS",
  "Q_PV2_HS",
  "R_PV2_LS_G",
  "R_PV2_HS_G",
  "R_PV2_LS_PD",
  "R_PV2_HS_PD",
  "C_PV2_OUT",
  "U_ID_PV2",
  "Q_PV2_ISO",
  "J_PV3",
  "D_PV3_TVS",
  "R_PV3_SHUNT",
  "C_PV3_IN",
  "C_PV3_IN_HF",
  "R_VPV3_TOP",
  "R_VPV3_BOT",
  "C_VPV3_FILT",
  "U_IPV3",
  "C_IPV3_VS",
  "U_DRV_PV3",
  "C_PV3_VCC",
  "D_PV3_BOOT", "R_PV3_BOOT_LIM",
  "C_PV3_BOOT",
  "L_PV3",
  "Q_PV3_LS",
  "Q_PV3_HS",
  "R_PV3_LS_G",
  "R_PV3_HS_G",
  "R_PV3_LS_PD",
  "R_PV3_HS_PD",
  "C_PV3_OUT",
  "U_ID_PV3",
  "Q_PV3_ISO",
  "R_SHUNT_OUT"
]

export function SolarTwoThree({ x, y }: { x: number; y: number }) {
  return <SchematicSection title="Solar channels" schX={x} schY={y} refs={SolarTwoThreeRefs} columns={4} />
}
