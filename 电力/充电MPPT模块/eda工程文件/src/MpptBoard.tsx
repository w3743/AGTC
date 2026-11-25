import reference from "./board_parts.json"
import { ReferenceBoard } from "../../../检查/联合仿真/code_as_eda/renderer"
import { AuxSupply } from "./modules/AuxSupply"
import { ControlAndCan } from "./modules/ControlAndCan"
import { UsbProtection } from "./modules/UsbProtection"
import { PdAndInput } from "./modules/PdAndInput"
import { Charger } from "./modules/Charger"
import { SolarOne } from "./modules/SolarOne"
import { SolarTwoThree } from "./modules/SolarTwoThree"
import { OutputAndHeater } from "./modules/OutputAndHeater"

export function MpptBoard() {
  return <ReferenceBoard title="Charging MPPT board" name="MpptBoard" reference={reference}>
    <AuxSupply x={0} y={140} />
    <ControlAndCan x={160} y={140} />
    <UsbProtection x={320} y={140} />
    <PdAndInput x={480} y={140} />
    <Charger x={0} y={-120} />
    <SolarOne x={160} y={-120} />
    <SolarTwoThree x={320} y={-120} />
    <OutputAndHeater x={480} y={-120} />
  </ReferenceBoard>
}
