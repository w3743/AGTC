import React from "react"
import { ReferenceBoard } from "../../../检查/联合仿真/code_as_eda/renderer"
import reference from "./board_parts.json"
import { AuxiliaryPower } from "./modules/AuxiliaryPower"
import { DualPhasePower } from "./modules/DualPhasePower"
import { SupercapAndIsolation } from "./modules/SupercapAndIsolation"
import { DumpAndOvervoltage } from "./modules/DumpAndOvervoltage"
import { GuardsAndInterlocks } from "./modules/GuardsAndInterlocks"
import { McuAndCan } from "./modules/McuAndCan"

export function RegulatorBoard() {
  return <ReferenceBoard title="Regulator P5 Rev2" name="RegulatorBoard" reference={reference}>
    <AuxiliaryPower />
    <DualPhasePower />
    <SupercapAndIsolation />
    <DumpAndOvervoltage />
    <GuardsAndInterlocks />
    <McuAndCan />
  </ReferenceBoard>
}
