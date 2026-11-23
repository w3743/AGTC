import React from "react"
import parts from "./can_parts.json"
import { ReferenceBoard, SchematicSection } from "../../../../检查/联合仿真/code_as_eda/renderer"

const interfaceRefs = [
  "C_CAN_VDDP_BULK", "C_CAN_VCC", "C_CAN_VIO", "R_CAN_TXD_PU", "U_CAN",
  "C_CAN_ISOIN_1U", "C_CAN_ISO_HF", "C_CAN_ISO_BULK", "D_CAN", "J_CAN",
]

export function IsolatedCanPreview() {
  return <ReferenceBoard title="MPPT isolated CAN interface - partial preview" name="MpptCanPreview" reference={parts}>
    <SchematicSection title="CAN interface only - external power and MCU boundaries" schX={0} schY={0}
      columns={3} refs={interfaceRefs} />
  </ReferenceBoard>
}
