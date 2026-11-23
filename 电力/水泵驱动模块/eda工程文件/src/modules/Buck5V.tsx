import { Part, SectionTitle } from '../Part'

export function Buck5V({ x, y }: { x: number; y: number }) {
  return <>
    <SectionTitle x={x + 16} y={y + 20}>5V BUCK SUPPLY</SectionTitle>
    <Part refdes="C_5V_IN" schX={x} schY={y + 5} schRotation={0} />
    <Part refdes="C_5V_IN_HF" schX={x + 7} schY={y + 5} schRotation={0} />
    <Part refdes="U_5V" schX={x + 16} schY={y + 5} schRotation={0} />
    <Part refdes="D_5V" schX={x + 24} schY={y + 5} schRotation={0} />
    <Part refdes="L_5V" schX={x + 32} schY={y + 5} schRotation={0} />
    <Part refdes="C_5V_OUT" schX={x + 24} schY={y - 3} schRotation={0} />
    <Part refdes="C_5V_OUT_HF" schX={x + 32} schY={y - 3} schRotation={0} />
    <Part refdes="R_FB_TOP" schX={x + 8} schY={y - 10} schRotation={0} />
    <Part refdes="R_FB_BOTTOM" schX={x + 16} schY={y - 10} schRotation={0} />
    <Part refdes="C_FB_FF" schX={x + 24} schY={y - 10} schRotation={0} />
  </>
}
