import React from 'react'
import { SchematicSection } from '../../../../../检查/联合仿真/code_as_eda/renderer'

const refs = [
  "C_AND2",
  "C_GATE1",
  "C_LPV_CONFIRM",
  "J_LOAD_POWER_VALID",
  "J_LOAD_POWER_VALID_B",
  "J_LPV_REG",
  "Q_LOAD_POWER_VALID_DRV",
  "R_LOAD_POWER_VALID_B_LED",
  "R_LOAD_POWER_VALID_GATE",
  "R_LOAD_POWER_VALID_GS",
  "R_LOAD_POWER_VALID_LED",
  "R_LOAD_POWER_VALID_REG_LED",
  "U_AND2",
  "U_GATE1",
  "U_LOAD_POWER_VALID",
  "U_LOAD_POWER_VALID_B",
  "U_LPV_CONFIRM",
]

export function Interlocks() {
  return <SchematicSection title="ISOLATED LOAD VALID INTERLOCKS" schX={110} schY={120} refs={refs} columns={5} />
}
