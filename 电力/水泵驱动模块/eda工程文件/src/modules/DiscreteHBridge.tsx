import { Part, SectionTitle } from '../Part'

export function DiscreteHBridge({ x, y }: { x: number; y: number }) {
  return <>
    <SectionTitle x={x} y={y + 20}>DISCRETE PUMP DRIVE AND BRAKE</SectionTitle>
    <Part refdes="C_DRV_VCC" schX={x} schY={y + 12} schRotation={0} />
    <Part refdes="U_PWM" schX={x + 9} schY={y + 12} schRotation={0} />
    <Part refdes="R_PWM_GATE_SER" schX={x + 20} schY={y + 12} schRotation={0} />
    <Part refdes="Q_PWM" schX={x + 31} schY={y + 12} schRotation={0} />
    <Part refdes="R_SLEEP" schX={x} schY={y + 4} schRotation={0} />
    <Part refdes="R_IN2_PD" schX={x + 9} schY={y + 4} schRotation={0} />
    <Part refdes="D_PUMP_FW" schX={x + 20} schY={y + 4} schRotation={0} />
    <Part refdes="J_MOTOR" schX={x + 31} schY={y + 4} schRotation={0} />
    <Part refdes="R_IN1_PD" schX={x} schY={y - 4} schRotation={0} />
    <Part refdes="R_BRAKE_BASE" schX={x + 9} schY={y - 4} schRotation={0} />
    <Part refdes="Q_BRAKE_CTL" schX={x + 20} schY={y - 4} schRotation={0} />
    <Part refdes="Q_BRAKE" schX={x + 31} schY={y - 4} schRotation={0} />
    <Part refdes="R_BRAKE_GATE_LOW" schX={x + 9} schY={y - 12} schRotation={0} />
    <Part refdes="R_BRAKE_GATE_HIGH" schX={x + 20} schY={y - 13} schRotation={0} />
    <Part refdes="C_DRV_VM1" schX={x + 31} schY={y - 12} schRotation={0} />
    <Part refdes="R_PUMP_SHUNT" schX={x + 9} schY={y - 20} schRotation={0} />
    <Part refdes="R_ADC_I" schX={x + 20} schY={y - 20} schRotation={0} />
    <Part refdes="C_ADC_I" schX={x + 31} schY={y - 20} schRotation={0} />
  </>
}
