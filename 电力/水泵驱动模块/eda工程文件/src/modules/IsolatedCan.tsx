import { Part, SectionTitle } from '../Part'

export function IsolatedCan({ x, y }: { x: number; y: number }) {
  return <>
    <SectionTitle x={x} y={y + 20}>ISOLATED CAN INTERFACE</SectionTitle>
    <Part refdes="R_CAN_TXD_PU" schX={x} schY={y + 10} schRotation={0} />
    <Part refdes="U_CAN" schX={x + 12} schY={y + 10} schRotation={0} />
    <Part refdes="D_CAN" schX={x + 25} schY={y + 10} schRotation={0} />
    <Part refdes="J_CAN" schX={x + 36} schY={y + 10} schRotation={0} />
    <Part refdes="R_TERM" schX={x + 30} schY={y + 2} schRotation={0} />
    <Part refdes="C_CAN_VCC" schX={x} schY={y - 6} schRotation={0} />
    <Part refdes="C_CAN_VDDP_BULK" schX={x + 9} schY={y - 6} schRotation={0} />
    <Part refdes="C_CAN_VIO" schX={x + 18} schY={y - 6} schRotation={0} />
    <Part refdes="C_CAN_ISO_BULK" schX={x + 27} schY={y - 6} schRotation={0} />
    <Part refdes="C_CAN_ISO_HF" schX={x + 9} schY={y - 15} schRotation={0} />
    <Part refdes="C_CAN_ISOIN_1U" schX={x + 20} schY={y - 15} schRotation={0} />
  </>
}
