import { SchematicSection } from "../../../../../检查/联合仿真/code_as_eda/renderer"

export const BusAndBovRefs = [
  "J_BUS_P",
  "J_BUS_N",
  "F_MAIN",
  "CBUS1",
  "CBUS2",
  "CBUS3",
  "DTVS1",
  "U7",
  "C_BUSADC",
  "C_OV_SENSE",
  "C_SAFEADC",
  "C_U7",
  "D_BUSADC_CLAMP",
  "Q_BOV_NPN",
  "Q_BOV_PNP",
  "Q_BUCK_CLAMP",
  "Q_BUCK_RELEASE",
  "R_BOV_DRIVE",
  "R_BOV_NPN_PD",
  "R_BOV_PNP_B",
  "R_BREAK_PU",
  "R_BUCK_CLAMP_BUS",
  "R_BUCK_VALID_BASE",
  "R_BUCK_VALID_PD",
  "R_BUSADC_BOT",
  "R_BUSADC_SER",
  "R_BUSADC_TOP",
  "R_DRV_EN_PD",
  "R_OVBOT",
  "R_OVTOP",
  "R_OVTOP_SER",
  "R_OV_PU",
  "R_SAFEADC_BOT",
  "R_SAFEADC_TOP",
]

export function BusAndBov({ x, y }: { x: number; y: number }) {
  return <SchematicSection title="BUS input and overvoltage interlock" schX={x} schY={y} refs={BusAndBovRefs} columns={3} />
}
