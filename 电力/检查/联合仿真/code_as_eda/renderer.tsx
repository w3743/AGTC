import React, { createContext, useContext } from "react"
import { mkdir, unlink, writeFile } from "node:fs/promises"
import { Circuit, type RootCircuit } from "@tscircuit/core"
import { MM_PER_SCHEMATIC_UNIT } from "circuit-json"
import { CircuitJsonToKicadProConverter, CircuitJsonToKicadSchConverter } from "circuit-json-to-kicad"
import { convertCircuitJsonToSchematicSvg } from "circuit-to-svg"
import { GlobalLabel, NoConnect, parseKicadSch, Pts, Stroke, TextEffects,
  TextEffectsFont, TextEffectsJustify, Wire, Xy } from "kicadts"
import { enlargeSvgAnnotations, useShortNetLabels } from "./schematic"

type Part = {
  ref: string
  mpn: string | null
  lcsc: string | null
  manufacturer: string | null
  footprint: string | null
  value: string
  dnp: boolean
  placement: string
  in_bom?: boolean
  pins: Record<string, string | null>
  pin_roles: Record<string, string>
}
type BoardData = { components: Part[] }
type BoardModel = {
  parts: Map<string, Part>
  aliases: Map<string, string>
}
const modelContext = createContext<BoardModel | null>(null)
type Section = { title: string; refs: readonly string[] }
const sectionContext = createContext<Section[] | null>(null)
type CircuitJson = ReturnType<RootCircuit["getCircuitJson"]>
const columnPitch = 27

function modelFor(data: BoardData): BoardModel {
  const parts = new Map(data.components.map((part) => [part.ref, part]))
  if (parts.size !== data.components.length) throw new Error("Duplicate component reference in TSX data")
  const nets = new Set<string>()
  for (const part of data.components) {
    for (const net of Object.values(part.pins)) if (net !== null) nets.add(net)
  }
  const aliases = new Map([...nets].sort().map((net, index) => [net, "net_" + index]))
  return { parts, aliases }
}

function physicalPins(part: Pick<Part, "pins">): string[] {
  return Object.keys(part.pins).sort((a, b) =>
    a.localeCompare(b, "en", { numeric: true }),
  )
}

function schematicPart({ part, aliases, x, y }: {
  part: Part; aliases: ReadonlyMap<string, string>; x: number; y: number
}) {
  const pins = physicalPins(part)
  const indexes = pins.map((_, index) => index + 1)
  const formattedRoles = pins.map((pin) => {
    const role = part.pin_roles[pin]
    return role && role.length <= 11 && !/^[0-9]+$/.test(role) ? role : pin
  })
  const pinLabels = Object.fromEntries(pins.map((pin, index) => {
    const label = formattedRoles[index]
    return ["pin" + (index + 1), formattedRoles.filter((value) => value === label).length > 1 ? pin : label]
  }))
  const side = {
    direction: "top-to-bottom" as const,
  }
  const common = {
    name: part.ref,
    manufacturerPartNumber: part.mpn ?? undefined,
    supplierPartNumbers: part.lcsc ? { jlcpcb: [part.lcsc] } : {},
    schX: x, schY: y, doNotPlace: part.dnp,
  }
  const twoTerminal = pins.join(",") === "1,2"
  const passiveValue = part.value || part.mpn || part.ref
  const toleranceFree = passiveValue.replace(/^(\d+)[Rr](\d+)/, "$1.$2")
  const resistorValue = toleranceFree.match(/^\s*(\d+(?:\.\d+)?)\s*(k|K|M|m)?(?:Ω|ohms?)?(?=\s|$)/)
  const capacitorValue = passiveValue.match(/^\s*(\d+(?:\.\d+)?)\s*([pnumµμ]?)F(?=\s|\/|$)/i)
  const inductorValue = passiveValue.match(/^\s*(\d+(?:\.\d+)?)\s*([pnumµμ]?)H(?=\s|\/|$)/i)
  const suffix = (value: string) => value.replace(/[µμ]/g, "u").replace("K", "k")
  let symbol
  if (twoTerminal && /^R/.test(part.ref) && !/^RN|^RL|^RY/.test(part.ref) && resistorValue) {
    symbol = <resistor {...common} resistance={resistorValue[1] + suffix(resistorValue[2] ?? "") + "ohm"} />
  } else if (twoTerminal && /^C/.test(part.ref) && !/^CN/.test(part.ref) && capacitorValue) {
    symbol = <capacitor {...common} capacitance={capacitorValue[1] + suffix(capacitorValue[2]) + "F"} />
  } else if (twoTerminal && /^L/.test(part.ref) && inductorValue) {
    symbol = <inductor {...common} inductance={inductorValue[1] + suffix(inductorValue[2]) + "H"} />
  } else {
    symbol = <chip {...common} pinLabels={pinLabels}
      noConnect={indexes.filter((index) => part.pins[pins[index - 1]] === null).map((index) => "pin" + index)}
      schWidth={pins.length <= 2 ? 3.4 : 6.6}
      schPinStyle={Object.fromEntries(indexes.map((index) => ["pin" + index, { marginBottom: 0.32 }]))}
      schPinArrangement={{
        leftSide: { ...side, pins: indexes.filter((index) => index % 2 === 1) },
        rightSide: { ...side, pins: indexes.filter((index) => index % 2 === 0) },
      }} />
  }
  const connections = pins.flatMap((pin, index) => {
    const net = part.pins[pin]
    return net === null ? [] : [
      <trace from={"." + part.ref + " > .pin" + (index + 1)} to={"net." + aliases.get(net)} />,
    ]
  })
  let annotation = null
  if (part.dnp || part.placement !== "board") {
    const notes = [part.dnp ? "DNP" : "", part.placement === "off-board" ? "OFF BOARD" :
      part.placement === "schematic-only" ? "SCHEMATIC ONLY" : ""].filter(Boolean)
    const offset = pins.length <= 2 ? 1.3 : Math.ceil(pins.length / 2) * 0.31 + 1.5
    annotation = <schematictext text={notes.join(" · ")} schX={x} schY={y - offset}
      fontSize={0.35} anchor="center" />
  }
  return React.createElement(React.Fragment, null, symbol, annotation, ...connections)
}

export function ReferenceBoard({ reference, title, name, children }: {
  reference: BoardData; title: string; name: string; children: React.ReactNode
}) {
  const model = modelFor(reference)
  return <board name={name} schAutoLayoutEnabled={false}>
    <schematicsheet name={name + "_circuit"} displayName={title} sheetWidth={6000} sheetHeight={7000}>
      <group name="Functional sections" schAutoLayoutEnabled={false}>
        <modelContext.Provider value={model}>{children}</modelContext.Provider>
      </group>
    </schematicsheet>
  </board>
}

export function SchematicSection({ title, schX, schY, columns, refs }: {
  title: string; schX: number; schY: number; columns: number; refs: readonly string[]
}) {
  const model = useContext(modelContext)
  if (!model) throw new Error("SchematicSection requires ReferenceBoard")
  useContext(sectionContext)?.push({ title, refs })
  const elements: React.ReactNode[] = [
    <schematictext schX={schX} schY={schY} text={title} fontSize={1.15} anchor="center" />,
  ]
  let top = schY - 4
  for (let index = 0; index < refs.length; index += columns) {
    const row = refs.slice(index, index + columns).map((ref) => {
      const part = model.parts.get(ref)
      if (!part) throw new Error("Unknown part " + ref + " in " + title)
      return part
    })
    const pins = Math.max(...row.map((part) => Object.keys(part.pins).length))
    const height = Math.max(8, Math.ceil(pins / 2) * 0.75 + 4)
    row.forEach((part, col) => elements.push(schematicPart({
      part, aliases: model.aliases, x: schX + col * columnPitch, y: top - height / 2,
    })))
    top -= height
  }
  return React.createElement(React.Fragment, null, ...elements)
}

function divideForKicad(elements: CircuitJson, sections: Section[], scaleFactor: number) {
  const sheetId = (index: number) => "kicad_section_" + (index + 1)
  const scale = ({ x, y }: { x: number; y: number }) => ({ x: x * scaleFactor, y: y * scaleFactor })
  const byRef = new Map(sections.flatMap((section, index) => section.refs.map((ref) => [ref, sheetId(index)])))
  const byTitle = new Map(sections.map((section, index) => [section.title, sheetId(index)]))
  const byPart = new Map(elements.flatMap((entry) => entry.type === "source_component" ?
    [[entry.source_component_id, byRef.get(entry.name)]] : []))
  const byPort = new Map(elements.flatMap((entry) => entry.type === "source_port" ?
    [[entry.source_port_id, byPart.get(entry.source_component_id)]] : []))
  const byTrace = new Map(elements.flatMap((entry) => entry.type === "source_trace" ?
    [[entry.source_trace_id, byPort.get(entry.connected_source_port_ids![0])]] : []))
  const bySymbol = new Map(elements.flatMap((entry) => entry.type === "schematic_component" ?
    [[entry.schematic_component_id, byPart.get(entry.source_component_id)]] : []))
  const output: CircuitJson = sections.map((section, index) => ({
    type: "schematic_sheet", schematic_sheet_id: sheetId(index), sheet_index: index,
    name: "section_" + (index + 1), display_name: section.title,
  }))
  for (const entry of elements) {
    switch (entry.type) {
      case "schematic_sheet": break
      case "schematic_component":
        output.push({ ...entry, schematic_sheet_id: bySymbol.get(entry.schematic_component_id),
          center: scale(entry.center),
          size: { width: entry.size.width * scaleFactor, height: entry.size.height * scaleFactor } })
        break
      case "schematic_port":
        output.push({ ...entry, schematic_sheet_id: byPort.get(entry.source_port_id),
          center: scale(entry.center),
          distance_from_component_edge: (entry.distance_from_component_edge ?? 0) * scaleFactor })
        break
      case "schematic_trace":
        output.push({ ...entry, schematic_sheet_id: byTrace.get(entry.source_trace_id!),
          edges: entry.edges.map((edge) => ({ ...edge, from: scale(edge.from), to: scale(edge.to) })) })
        break
      case "schematic_net_label":
        output.push({ ...entry, schematic_sheet_id: byTrace.get(entry.source_trace_id!),
          center: scale(entry.center), anchor_position: scale(entry.anchor_position) })
        break
      case "schematic_text": {
        const sheet = entry.schematic_component_id ? bySymbol.get(entry.schematic_component_id) : byTitle.get(entry.text)
        if (sheet) output.push({ ...entry, schematic_sheet_id: sheet, position: scale(entry.position) })
        break
      }
      default: if (!entry.type.startsWith("schematic_")) output.push(entry)
    }
  }
  return output
}

function connectKicadPins(schematic: ReturnType<typeof parseKicadSch>,
  netsByRef: Map<string, Map<string, string>>) {
  const library = new Map(schematic.libSymbols.symbols.map((symbol) => [symbol.libraryId, symbol]))
  const wires: Wire[] = []
  const labels: GlobalLabel[] = []
  const noConnects: NoConnect[] = []
  const angleInRadians = (degrees: number) => degrees * Math.PI / 180
  const rounded = (value: number) => Math.round(value * 1000) / 1000
  for (const symbol of schematic.symbols) {
    const ref = symbol.properties.find((property) => property.key === "Reference")!.value
    const lib = library.get(symbol.libraryId)!
    const at = symbol.at!
    if (symbol.mirror) throw new Error("Unexpected mirrored KiCad symbol " + ref)
    const rotation = angleInRadians(at.angle ?? 0)
    const cos = Math.cos(rotation), sin = Math.sin(rotation)
    for (const pin of [lib, ...lib.subSymbols].flatMap((unit) => unit.pins)) {
      const pos = pin.at!
      const x = rounded(at.x + cos * pos.x - sin * pos.y)
      const y = rounded(at.y - sin * pos.x - cos * pos.y)
      const net = netsByRef.get(ref)?.get(pin.numberString!)
      if (!net) {
        noConnects.push(new NoConnect({ at: [x, y], uuid: crypto.randomUUID() }))
        continue
      }
      const heading = ((pos.angle ?? 0) + (at.angle ?? 0) + 180) % 360
      const dir = angleInRadians(heading)
      const to = [rounded(x + Math.cos(dir) * 3.81), rounded(y - Math.sin(dir) * 3.81)] as const
      const wire = new Wire()
      wire.points = new Pts([new Xy(x, y), new Xy(...to)])
      const stroke = new Stroke()
      stroke.width = 0.254
      stroke.type = "default"
      wire.stroke = stroke
      wire.uuid = crypto.randomUUID()
      wires.push(wire)
      const font = new TextEffectsFont()
      font.size = { width: 1.02, height: 1.02 }
      const effects = new TextEffects({ font })
      effects.justify = new TextEffectsJustify({ horizontal: heading === 180 || heading === 270 ? "right" : "left" })
      labels.push(new GlobalLabel({
        value: net, shape: "input", at: [...to, heading], effects,
        fieldsAutoplaced: false, uuid: crypto.randomUUID(),
      }))
    }
  }
  schematic.wires = wires
  schematic.globalLabels = labels
  schematic.noConnects = noConnects
  schematic.junctions = []
}

export function buildKicadProject(elements: CircuitJson, sections: Section[],
  parts: Map<string, Pick<Part, "pins" | "dnp" | "placement" | "in_bom" | "footprint">>, name: string,
  scaleFactor = 0.22) {
  const netsByRef = new Map<string, Map<string, string>>()
  const ownerByPort = new Map(elements.flatMap((entry) => entry.type === "source_port" ?
    [[entry.source_port_id, entry]] : []))
  const refById = new Map(elements.flatMap((entry) => entry.type === "source_component" ?
    [[entry.source_component_id, entry.name]] : []))
  const labelByTrace = new Map(elements.flatMap((entry) => entry.type === "schematic_net_label" ?
    [[entry.source_trace_id, entry.text]] : []))
  for (const entry of elements) {
    if (entry.type !== "source_trace") continue
    const portId = entry.connected_source_port_ids![0]
    const port = ownerByPort.get(portId)!
    const ref = refById.get(port.source_component_id)!
    const number = physicalPins(parts.get(ref)!)[port.pin_number! - 1]
    const label = labelByTrace.get(entry.source_trace_id)
    if (!number || !label) throw new Error("Missing physical pin or net label " + entry.source_trace_id)
    if (!netsByRef.has(ref)) netsByRef.set(ref, new Map())
    netsByRef.get(ref)!.set(number, label)
  }
  if (!sections.length) throw new Error("KiCad export needs functional section metadata")
  const kicadData = divideForKicad(elements, sections, scaleFactor)
  const kicad = new CircuitJsonToKicadSchConverter(kicadData)
  kicad.runUntilFinished()
  const kicadFiles = kicad.getOutputFiles({ schematicFilename: name + ".kicad_sch" }).map((file) => {
    const schematic = parseKicadSch(file.content)
    const pinMapByLibrary = new Map<string, string[]>()
    for (const symbol of schematic.symbols) {
      const ref = symbol.properties.find((property) => property.key === "Reference")?.value
      const part = ref && parts.get(ref)
      if (!part) throw new Error("Missing/unknown symbol reference in " + file.filename)
      symbol.dnp = part.dnp
      symbol.onBoard = part.placement === "board"
      symbol.inBom = part.in_bom !== false && part.placement === "board"
      const footprint = symbol.properties.find((property) => property.key === "Footprint")
      if (footprint) footprint.value = part.footprint ?? ""
      const pins = physicalPins(part)
      const library = symbol.libraryId!
      const earlier = pinMapByLibrary.get(library)
      if (earlier && earlier.join(",") !== pins.join(",")) {
        throw new Error("KiCad library symbol has conflicting contractual pads: " + library)
      }
      pinMapByLibrary.set(library, pins)
      for (const pin of symbol.pins) pin.numberString = pins[Number(pin.numberString) - 1]
    }
    for (const library of schematic.libSymbols.symbols) {
      const pins = pinMapByLibrary.get(library.libraryId!)
      if (!pins) continue
      for (const pin of [library, ...library.subSymbols].flatMap((unit) => unit.pins)) {
        const physical = pins[Number(pin.numberString) - 1]
        if (!physical || !pin._sxNumber) throw new Error("Unmapped KiCad library pin in " + library.libraryName)
        pin._sxNumber.value = physical
      }
    }
    connectKicadPins(schematic, netsByRef)
    const content = schematic.getString().replaceAll('(project ""', '(project "' + name + '"')
    return { filename: file.filename, content }
  })
  const project = new CircuitJsonToKicadProConverter(kicadData, {
    projectName: name, schematicFilename: name + ".kicad_sch", schematicSheetPlan: kicad.schematicSheetPlan,
  })
  project.runUntilFinished()
  return [...kicadFiles, { filename: name + ".kicad_pro", content: project.getOutputString() }]
}

export async function exportBoard({ reference, board, outDir, name }: {
  reference: BoardData; board: React.ReactElement; outDir: URL; name: string
}) {
  const started = performance.now()
  const model = modelFor(reference)
  const officialNets = new Map([...model.aliases].map(([official, code]) => [code, official]))
  const circuit = new Circuit({ platform: { pcbDisabled: true } })
  const sections: Section[] = []
  circuit.add(<sectionContext.Provider value={sections}>{board}</sectionContext.Provider>)
  await circuit.renderUntilSettled()
  const renderMs = performance.now() - started
  // The base renderer checks the pre-fit bounds; the delivered sheet is fitted below.
  const content = useShortNetLabels(circuit.getCircuitJson(), officialNets)
    .filter((entry) => entry.type !== "schematic_element_outside_sheet_warning")
  const points = content.flatMap((entry) => {
    if (entry.type === "schematic_component" || entry.type === "schematic_port" ||
        entry.type === "schematic_net_label") return [entry.center]
    if (entry.type === "schematic_text") return [entry.position]
    return []
  })
  const maxX = Math.max(...points.map(({ x }) => x)), minX = Math.min(...points.map(({ x }) => x))
  const maxY = Math.max(...points.map(({ y }) => y)), minY = Math.min(...points.map(({ y }) => y))
  const width = maxX - minX + 35, height = maxY - minY + 18
  const elements = content.map((entry) => entry.type === "schematic_sheet" ? {
    ...entry, center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 },
    sheet_width: width * MM_PER_SCHEMATIC_UNIT, sheet_height: height * MM_PER_SCHEMATIC_UNIT,
  } : entry)
  const components = elements.filter((entry) => entry.type === "source_component")
  const errors = elements.filter((entry) => entry.type.includes("error"))
  if (errors.length) throw new Error(JSON.stringify(errors, null, 2))
  if (components.length !== model.parts.size || new Set(components.map((component) => component.name)).size !== components.length) {
    throw new Error("TSX reference count differs from component data or contains a duplicate placement")
  }
  const byId = new Map(components.map((component) => {
    const part = model.parts.get(component.name)
    if (!part) throw new Error("Unknown rendered reference " + component.name)
    return [component.source_component_id, part]
  }))
  const physicalByPortId = new Map(elements.flatMap((entry) => {
    if (entry.type !== "source_port") return []
    const part = byId.get(entry.source_component_id)
    if (!part) throw new Error("Rendered port has no source component: " + entry.source_port_id)
    const pin = physicalPins(part)[entry.pin_number! - 1]
    if (!pin) throw new Error("Rendered port index exceeds physical pins: " + part.ref)
    return [[entry.source_port_id, pin] as const]
  }))
  const exportJson = elements.map((entry) => {
    if (entry.type === "source_component") {
      const part = byId.get(entry.source_component_id)!
      return {
        ...entry, manufacturer: part.manufacturer, footprint_name: part.footprint,
        do_not_place: part.dnp, placement: part.placement, in_bom: part.in_bom !== false,
      }
    }
    if (entry.type === "source_port" || entry.type === "schematic_port") {
      const pin = physicalByPortId.get(entry.source_port_id)!
      return { ...entry, pin_number: /^\d+$/.test(pin) ? Number(pin) : entry.pin_number, contract_pin_number: pin }
    }
    if (entry.type === "source_net") {
      return { ...entry, name: officialNets.get(entry.name) ?? entry.name }
    }
    return entry
  })
  const svg = enlargeSvgAnnotations(convertCircuitJsonToSchematicSvg(exportJson, {
    width: Math.ceil(width * 21), height: Math.ceil(height * 21),
  }))
  const kicadFiles = buildKicadProject(elements, sections, model.parts, name)

  const header = ["Reference", "MPN", "LCSC", "Manufacturer", "Footprint", "DNP", "Placement", "In BOM", "Value"]
  const rows = components.map((component) => {
    const part = byId.get(component.source_component_id)!
    return [
      component.name, component.manufacturer_part_number ?? "", component.supplier_part_numbers?.jlcpcb?.[0] ?? "",
      part.manufacturer ?? "", part.footprint ?? "", part.dnp ? "yes" : "no", part.placement,
      part.in_bom === false ? "no" : "yes", part.value ?? "",
    ]
  })
  const quote = (value: string) => '"' + value.replaceAll('"', '""') + '"'
  const bom = [header, ...rows].map((row) => row.map(quote).join(",")).join("\n")
  const kicadDirectory = new URL("kicad/", outDir)
  await mkdir(kicadDirectory, { recursive: true })
  await Promise.all([
    writeFile(new URL("circuit.json", outDir), JSON.stringify(exportJson, null, 2) + "\n"),
    writeFile(new URL("schematic.svg", outDir), svg.replace(/[ \t]+$/gm, "").trimEnd() + "\n"),
    writeFile(new URL("bom.csv", outDir), bom + "\n"),
    ...kicadFiles.map(({ filename, content }) =>
      writeFile(new URL(filename, kicadDirectory), content.trimEnd() + "\n"),
    ),
  ])
  await unlink(new URL(name.toLowerCase() + "_circuit.kicad_sch", kicadDirectory)).catch((error) => {
    if (error.code !== "ENOENT") throw error
  })
  console.log(components.length + " components, " + renderMs.toFixed(1) + " ms render, " +
    (performance.now() - started).toFixed(1) + " ms through export.")
}
