import { SchematicSection } from "../../../../检查/联合仿真/code_as_eda/renderer"

export const SolarOneRefs = [
  "C_BQ_REGN",
  "C_BQ_DRV_SUP",
  "R_BQ_STAT1_PU",
  "R_BQ_STAT2_PU",
  "R_BQ_PG_PU",
  "U_BQ",
  "J_PV1",
  "D_PV1_TVS",
  "R_PV1_SHUNT",
  "C_PV1_IN",
  "C_PV1_IN_HF",
  "R_VPV1_TOP",
  "R_VPV1_BOT",
  "C_VPV1_FILT",
  "U_IPV1",
  "C_IPV1_VS",
  "U_DRV_PV1",
  "C_PV1_VCC",
  "D_PV1_BOOT", "R_PV1_BOOT_LIM",
  "C_PV1_BOOT",
  "L_PV1",
  "Q_PV1_LS",
  "Q_PV1_HS",
  "R_PV1_LS_G",
  "R_PV1_HS_G",
  "R_PV1_LS_PD",
  "R_PV1_HS_PD",
  "C_PV1_OUT",
  "U_ID_PV1",
  "Q_PV1_ISO",
  "J_PV2",
  "D_PV2_TVS",
  "R_PV2_SHUNT",
  "C_PV2_IN",
  "C_PV2_IN_HF",
  "R_VPV2_TOP",
  "R_VPV2_BOT",
  "C_VPV2_FILT",
  "U_IPV2"
]

export function SolarOne({ x, y }: { x: number; y: number }) {
  return <SchematicSection title="Charger and solar channel 1" schX={x} schY={y} refs={SolarOneRefs} columns={4} />
}
