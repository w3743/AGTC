import { Part, SectionTitle } from '../Part'

export function PowerAndProtection({ x, y }: { x: number; y: number }) {
  return <>
    <SectionTitle x={x + 8} y={y + 20}>BUS INPUT AND PROTECTION</SectionTitle>
    <Part refdes="J_BUS" schX={x} schY={y + 5} schRotation={0} />
    <Part refdes="C_VM_BULK" schX={x + 8} schY={y + 5} schRotation={0} />
    <Part refdes="D_BUS" schX={x + 16} schY={y + 5} schRotation={0} />
  </>
}
