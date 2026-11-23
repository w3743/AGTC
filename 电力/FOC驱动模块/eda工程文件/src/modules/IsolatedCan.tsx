import { SchematicSection } from "../../../../检查/联合仿真/code_as_eda/renderer"

export const IsolatedCanRefs = [
  "U5",
  "J_CAN",
  "D_CAN_ESD",
  "R_CAN_TERM",
  "C_U5_ISOIN_1U",
  "C_U5_ISOOUT_100N",
  "C_U5_ISOOUT_47U",
  "C_U5_VDDP_47U",
  "R_CAN_TX_PU",
  "U5_C",
  "U5_VIO_C",
]

export function IsolatedCan({ x, y }: { x: number; y: number }) {
  return <SchematicSection title="Isolated CAN" schX={x} schY={y} refs={IsolatedCanRefs} columns={3} />
}
