import { SchematicSection } from "../../../../检查/联合仿真/code_as_eda/renderer"

export const SafeAndDumpPowerRefs = [
  "U2",
  "U2_CIN",
  "U2_COUT",
  "U2_RTOP",
  "U2_RBOT",
  "U6",
  "U6_CIN",
  "U6_COUT",
  "U6_RTOP",
  "U6_RBOT",
]

export function SafeAndDumpPower({ x, y }: { x: number; y: number }) {
  return <SchematicSection title="SAFE_5V and DUMP_5V independent LDOs" schX={x} schY={y} refs={SafeAndDumpPowerRefs} columns={3} />
}
