import type { RootCircuit } from "@tscircuit/core"
import { netAliases } from "../src/Part"

type CircuitJson = ReturnType<RootCircuit["getCircuitJson"]>
type CircuitElement = CircuitJson[number]
type Label = Extract<CircuitElement, { type: "schematic_net_label" }>

const displayNames = new Map(Object.entries(netAliases).map(([official, alias]) => [alias, official]))
const directions = {
  left:   { x: -1, y: 0, anchor: "right" },
  right:  { x: 1, y: 0, anchor: "left" },
  up:     { x: 0, y: 1, anchor: "bottom" },
  down:   { x: 0, y: -1, anchor: "top" },
} as const
const stubLength = 1.25

// Keep the tscircuit topology. Lay out each named-net connection as a
// schematic port, a short wire, and a label anchored on the far end of that wire.
// The default schematic autorouter joins many same-name pins across the sheet.
export function useShortNetLabels(circuitJson: CircuitJson): CircuitJson {
  const nets = new Map(circuitJson.flatMap((element) =>
    element.type === "source_net" ? [[element.source_net_id, element]] : [],
  ))
  const ports = new Map(circuitJson.flatMap((element) =>
    element.type === "schematic_port" ? [[element.source_port_id, element]] : [],
  ))
  const sheet = circuitJson.find((element) => element.type === "schematic_sheet")
  const out = circuitJson.filter((element) =>
    element.type !== "schematic_trace" &&
    element.type !== "schematic_net_label" &&
    !(element.type === "schematic_text" && element.source_trace_id),
  ) as CircuitJson

  for (const trace of circuitJson) {
    if (trace.type !== "source_trace") continue
    const [portId] = trace.connected_source_port_ids ?? []
    const [netId] = trace.connected_source_net_ids ?? []
    const port = ports.get(portId)
    const net = nets.get(netId)
    if (!port?.facing_direction || !net ||
        trace.connected_source_port_ids?.length !== 1 ||
        trace.connected_source_net_ids?.length !== 1) {
      throw new Error(`Expected one schematic pin and one named net for ${trace.source_trace_id}`)
    }
    const direction = directions[port.facing_direction]
    const start = port.center
    const end = { x: start.x + direction.x * stubLength, y: start.y + direction.y * stubLength }
    const text = displayNames.get(net.name) ?? net.name
    const labelOffset = direction.x ? (text.length * 0.055 + 0.08) : 0.13
    const label: Label = {
      type: "schematic_net_label",
      schematic_net_label_id: `schematic_net_label_${trace.source_trace_id}`,
      source_net_id: net.source_net_id,
      source_trace_id: trace.source_trace_id,
      schematic_sheet_id: sheet?.schematic_sheet_id,
      anchor_side: direction.anchor,
      anchor_position: end,
      center: { x: end.x + direction.x * labelOffset, y: end.y + direction.y * labelOffset },
      text,
    }
    out.push({
      type: "schematic_trace",
      schematic_trace_id: `schematic_trace_${trace.source_trace_id}`,
      source_trace_id: trace.source_trace_id,
      schematic_sheet_id: sheet?.schematic_sheet_id,
      edges: [{ from: start, to: end }],
      junctions: [],
      subcircuit_connectivity_map_key: trace.subcircuit_connectivity_map_key,
    }, label)
  }
  return out
}

// The converter's default 0.18-unit annotations are too small on this
// wide sheet. Enlarge annotation text while keeping its electrical anchors.
export function enlargeSvgAnnotations(svg: string) {
  // Use plain net names with wire stubs. The stock label outlines are
  // sized for the converter's original text and obscure larger lettering.
  const withoutLabelOutlines = svg.replace(/<path class="net-label sch-net-label"[^>]*\/>/g, "")
  return withoutLabelOutlines.replace(/<text\b[^>]*>/g, (tag) => {
    const cssClass = tag.match(/\bclass="([^"]*)"/)?.[1] ?? ""
    let scale = 1
    let offsetY = 0
    if (/sch-pin-(number|label)/.test(cssClass)) scale = 1.5
    if (cssClass.includes("sch-net-label-text")) scale = 1.85
    if (cssClass.includes("sch-component-text")) {
      scale = 1.65
      offsetY = cssClass.includes("sch-component-name") ? -4 : 4
    }
    if (cssClass === "sch-text" && tag.includes("data-schematic-text-id=") &&
        /font-size="([0-5]\.|[0-5]px)/.test(tag)) scale = 1.65
    return tag
      .replace(/\bfont-size="([\d.]+)px"/, (_, size: string) =>
        `font-size="${(Number(size) * scale).toFixed(3)}px"`,
      )
      .replace(/\by="([-\d.]+)"/, (_, y: string) => `y="${Number(y) + offsetY}"`)
  })
}
