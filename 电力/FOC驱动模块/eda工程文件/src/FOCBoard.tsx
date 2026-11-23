import reference from "./board_parts.json"
import { ReferenceBoard } from "../../../检查/联合仿真/code_as_eda/renderer"
import { BusAndBov } from "./modules/BusAndBov"
import { SafeAndDumpPower } from "./modules/SafeAndDumpPower"
import { InverterAndDriver } from "./modules/InverterAndDriver"
import { McuAndHall } from "./modules/McuAndHall"
import { IsolatedCan } from "./modules/IsolatedCan"

export function FOCBoard() {
  return <ReferenceBoard title="FOC motor drive - both sides" name="FOCBoard" reference={reference}>
    <BusAndBov x={0} y={0} />
    <SafeAndDumpPower x={90} y={0} />
    <InverterAndDriver x={180} y={0} />
    <McuAndHall x={300} y={0} />
    <IsolatedCan x={390} y={0} />
  </ReferenceBoard>
}
