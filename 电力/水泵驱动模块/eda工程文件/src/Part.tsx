import React from "react"
import { landPatternFor } from "./footprints"
import { parts, type PartDefinition, type PumpReference } from "./parts"

// tscircuit selectors reserve "+" and "-". Keep the alias map narrow.
export const netAliases: Record<string, string> = { "+5V": "V5", "+3V3": "V3V3" }

type PartProps = {
  refdes: PumpReference
  schX: number
  schY: number
  schRotation?: number
}

function makeFootprint(ref: PumpReference) {
  const pads = landPatternFor(ref).pads.map(([pin, pcbX, pcbY, width, height, ccwRotation, shape, drill]) => {
    const location = { pcbX, pcbY, portHints: [`pin${pin}`] }
    if (!drill) {
      return <smtpad {...location} shape="rotated_rect" width={width} height={height} ccwRotation={ccwRotation} />
    }
    if (shape === "ellipse") {
      return <platedhole {...location} shape="circle" holeDiameter={drill} outerDiameter={width} />
    }
    return <platedhole {...location} shape="circular_hole_with_rect_pad"
      holeDiameter={drill} rectPadWidth={width} rectPadHeight={height} pcbRotation={ccwRotation} />
  })
  return React.createElement("footprint", null, ...pads)
}

const chipArrangements: Record<string, { left: number[]; right: number[]; width: number }> = {
  U_5V: { left: [1, 5, 3, 6], right: [2, 4], width: 2.5 },
  U_PWM: { left: [2, 3, 1, 4], right: [5], width: 2.5 },
  U_CAN: { left: [1, 2, 3, 4, 5, 6, 7, 8], right: [16, 15, 14, 13, 12, 11, 10, 9], width: 3.4 },
  D_CAN: { left: [1, 2], right: [3], width: 2.2 },
  J_BUS: { left: [], right: [1, 2], width: 2.4 },
  J_MOTOR: { left: [1, 2], right: [], width: 2.5 },
  J_CAN: { left: [1, 2, 3], right: [], width: 3.2 },
}

function Symbol({ refdes, schX, schY, schRotation = 0 }: PartProps) {
  const part: PartDefinition = parts[refdes]
  const common = {
    name: refdes,
    manufacturerPartNumber: part.mpn,
    supplierPartNumbers: { jlcpcb: part.lcsc ? [part.lcsc] : [] },
    footprint: makeFootprint(refdes),
    schX, schY, schRotation,
    doNotPlace: part.dnp,
  }
  switch (part.symbol) {
    case "capacitor":
      return <capacitor {...common} capacitance={part.label} polarized={part.polarized} />
    case "resistor":
      return <resistor {...common} resistance={part.label} />
    case "inductor":
      return <inductor {...common} inductance={part.label} />
    case "schottky":
      return <diode {...common} schottky pinLabels={{ pin1: "cathode", pin2: "anode" }} />
    case "tvs":
      return <diode {...common} avalanche pinLabels={{ pin1: "cathode", pin2: "anode" }} />
    case "nmos":
    case "pmos":
      return <mosfet {...common} channelType={part.symbol === "nmos" ? "n" : "p"}
        mosfetMode="enhancement" pinLabels={{ pin1: "gate", pin2: "drain", pin3: "source" }} />
    case "npn":
      return <transistor {...common} type="npn" pinLabels={{ pin1: "base", pin2: "emitter", pin3: "collector" }} />
    default: {
      const pins = Object.keys(part.pins).map(Number)
      const arrangement = chipArrangements[refdes] ?? {
        left: pins.filter((pin) => pin % 2 === 1),
        right: pins.filter((pin) => pin % 2 === 0),
        width: 2.8,
      }
      const pinLabels = Object.fromEntries(pins.map((pin) => [
        `pin${pin}`, part.pins[pin] === null ? `NC${pin}` : part.roles?.[pin] ?? String(pin),
      ]))
      const noConnect = pins.filter((pin) => part.pins[pin] === null).map((pin) => `pin${pin}`)
      const schPinStyle = Object.fromEntries(pins.map((pin) => [`pin${pin}`, { marginBottom: 0.35 }]))
      return <chip {...common} pinLabels={pinLabels} noConnect={noConnect}
        schWidth={arrangement.width} schPinStyle={schPinStyle}
        schPinArrangement={{
          leftSide: { direction: "top-to-bottom", pins: arrangement.left },
          rightSide: { direction: "top-to-bottom", pins: arrangement.right },
        }} />
    }
  }
}

export function Part(props: PartProps) {
  const { refdes, schX, schY } = props
  const traces = Object.entries(parts[refdes].pins).flatMap(([pin, net]) =>
    net === null ? [] : [
      <trace from={`.${refdes} > .pin${pin}`}
        to={`net.${netAliases[net] ?? net}`} />,
    ],
  )
  const note = notes[refdes]
  const annotation = note
    ? <schematictext text={note} schX={schX} schY={schY - 1.05} fontSize={0.3} anchor="center" />
    : null
  return React.createElement(React.Fragment, null, <Symbol {...props} />, annotation, ...traces)
}

const notes: Partial<Record<PumpReference, string>> = {
  R_TERM: "DNP by default",
  R_PUMP_SHUNT: "2 W",
  NTC_BOARD: "B3950",
}

export function SectionTitle({ x, y, children }: { x: number; y: number; children: string }) {
  return <schematictext schX={x} schY={y} fontSize={1} anchor="center" text={children} />
}
