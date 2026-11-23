import { PowerAndProtection } from './modules/PowerAndProtection'
import { Buck5V } from './modules/Buck5V'
import { DiscreteHBridge } from './modules/DiscreteHBridge'
import { McuCoreInterface } from './modules/McuCoreInterface'
import { IsolatedCan } from './modules/IsolatedCan'

export function PumpBoard() {
  return <board name="PumpBoard" schAutoLayoutEnabled={false}>
    <schematicsheet name="Pump" displayName="Water Pump Drive Module" sheetWidth={2100} sheetHeight={600}>
      <group name="Layout" schAutoLayoutEnabled={false}>
        <PowerAndProtection x={-95} y={0} />
        <Buck5V x={-70} y={0} />
        <DiscreteHBridge x={-30} y={0} />
        <McuCoreInterface x={15} y={0} />
        <IsolatedCan x={60} y={0} />
      </group>
    </schematicsheet>
  </board>
}
