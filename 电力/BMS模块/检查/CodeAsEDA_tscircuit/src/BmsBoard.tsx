import React from 'react'
import { ReferenceBoard } from '../../../../检查/联合仿真/code_as_eda/renderer'
import reference from './board_parts.json'
import { CellMonitor } from './modules/CellMonitor'
import { PowerAndPrecharge } from './modules/PowerAndPrecharge'
import { Supply } from './modules/Supply'
import { Interlocks } from './modules/Interlocks'
import { CoreAndCan } from './modules/CoreAndCan'

export function BmsBoard() {
  return <ReferenceBoard title="9S5P BMS" name="BmsBoard" reference={reference}>
    <CellMonitor />
    <PowerAndPrecharge />
    <Supply />
    <Interlocks />
    <CoreAndCan />
  </ReferenceBoard>
}
