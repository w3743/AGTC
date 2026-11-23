import React from 'react'
import { SchematicSection } from '../../../../检查/联合仿真/code_as_eda/renderer'

const refs = [
  "C_BREG_COL",
  "C_POR",
  "C_PSU5_BST",
  "C_PSU5_EN",
  "C_PSU5_FB_FF",
  "C_PSU5_IN1",
  "C_PSU5_IN2",
  "C_PSU5_IN_HF",
  "C_PSU5_OUT1",
  "C_PSU5_OUT2",
  "C_PSU5_VCC",
  "C_PSU_IN1",
  "C_PSU_IN2",
  "C_PSU_IN3",
  "C_PSU_OUT1",
  "C_PSU_OUT2",
  "C_SEQ",
  "C_TIMER",
  "D_BREG",
  "D_PSU5_FW",
  "L_PSU5",
  "PSU5_EXT",
  "Q_BREG",
  "R_BREG_FEED",
  "R_BREG_PD",
  "R_PSU5_EN_SW",
  "R_PSU5_FB_BOT",
  "R_PSU5_FB_TOP",
  "R_TIMER_ACTIVE_PD",
  "R_TIMER_SET",
  "R_TIMER_SUPPLY",
  "SW_RUN_STORE",
  "U_POR",
  "U_SEQ",
  "U_TIMER",
]

export function Supply() {
  return <SchematicSection title="BUCK AND RUN / STORE SUPPLY" schX={-55} schY={120} refs={refs} columns={5} />
}
