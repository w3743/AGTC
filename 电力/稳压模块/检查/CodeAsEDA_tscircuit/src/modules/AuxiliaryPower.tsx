import React from "react"
import { SchematicSection } from "../../../../../检查/联合仿真/code_as_eda/renderer"

export function AuxiliaryPower() {
  return <SchematicSection title="Isolated auxiliary supply and 12 V hold" schX={0} schY={0} columns={3} refs={[
    "C_12V_IN",
    "C_12V_IN_HF",
    "C_12V_OUT",
    "C_12V_OUT_HF",
    "C_HW_BANK",
    "C_HW_CTR",
    "D_AUX_BAT",
    "D_AUX_BUS",
    "D_HW_HOLD",
    "J_AUX_BAT",
    "J_AUX_BUS_FUSE",
    "R_AUX_MINLOAD",
    "R_AUX_MINLOAD_TRIM",
    "R_HW_POR_BOT",
    "R_HW_POR_TOP",
    "R_HW_READY_PD",
    "R_HW_READY_PU",
    "R_HW_SLEW1",
    "R_HW_SLEW2",
    "U_AUX1",
    "U_HW_POR",
  ]} />
}
