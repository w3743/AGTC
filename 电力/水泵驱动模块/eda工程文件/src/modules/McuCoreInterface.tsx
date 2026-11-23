import { Part, SectionTitle } from '../Part'

export function McuCoreInterface({ x, y }: { x: number; y: number }) {
  return <>
    <SectionTitle x={x} y={y + 20}>MCU INTERFACE AND SENSING</SectionTitle>
    <Part refdes="J_MCU_A" schX={x + 6} schY={y + 7} schRotation={0} />
    <Part refdes="J_MCU_B" schX={x + 25} schY={y + 7} schRotation={0} />
    <Part refdes="R_BUS_TOP1" schX={x} schY={y - 10} schRotation={0} />
    <Part refdes="R_BUS_TOP2" schX={x + 9} schY={y - 10} schRotation={0} />
    <Part refdes="R_BUS_BOTTOM" schX={x + 18} schY={y - 10} schRotation={0} />
    <Part refdes="R_ADC_BUS" schX={x + 27} schY={y - 10} schRotation={0} />
    <Part refdes="C_ADC_BUS" schX={x + 36} schY={y - 10} schRotation={0} />
    <Part refdes="R_NTC_PULL" schX={x + 9} schY={y - 19} schRotation={0} />
    <Part refdes="NTC_BOARD" schX={x + 18} schY={y - 19} schRotation={0} />
    <Part refdes="C_ADC_NTC" schX={x + 27} schY={y - 19} schRotation={0} />
  </>
}
