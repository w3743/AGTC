import React from 'react'
import { SchematicSection } from '../../../../../检查/联合仿真/code_as_eda/renderer'

const refs = [
  "C_CAN_ISOIN_1U",
  "C_CAN_PWR_IN1",
  "C_CAN_PWR_IN2",
  "C_CAN_PWR_OUT1",
  "C_CAN_PWR_OUT2",
  "C_CAN_VCC2_2",
  "D_CAN_TVS",
  "J_CAN",
  "J_CORE_P1",
  "J_CORE_P2",
  "J_EARLY_READY_IN",
  "R_ALERT_PU",
  "R_ALERT_SER",
  "R_CAN_TXD_PU",
  "R_EARLY_READY_PU",
  "R_SCL_PU",
  "R_SDA_PU",
  "U_CAN",
]

export function CoreAndCan() {
  return <SchematicSection title="CORE INTERFACE AND ISOLATED CAN" schX={275} schY={120} refs={refs} columns={5} />
}
