import React, { createContext, useContext } from "react"
import { mkdir, readdir, unlink, writeFile } from "node:fs/promises"
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
  void title
  return <board name={name} schAutoLayoutEnabled={false}>
    <group name="Functional sections" schAutoLayoutEnabled={false}>
      <modelContext.Provider value={model}>{children}</modelContext.Provider>
    </group>
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

const BORDERLESS_WKS = `(kicad_wks
  (version 20231118)
  (generator "pl_editor")
  (generator_version "8.0")
  (setup
    (textsize 1.5 1.5)
    (linewidth 0.15)
    (textlinewidth 0.15)
    (left_margin 0)
    (right_margin 0)
    (top_margin 0)
    (bottom_margin 0)
  )
)
`

const KICAD_UNITS_TO_MM = 15

function prepareSingleSheetForKicad(elements: CircuitJson, scaleFactor: number) {
  const scale = ({ x, y }: { x: number; y: number }) => ({ x: x * scaleFactor, y: y * scaleFactor })
  const output: CircuitJson = []
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  const include = (x: number, y: number) => {
    minX = Math.min(minX, x); minY = Math.min(minY, y)
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y)
  }
  for (const entry of elements) {
    if (entry.type === "schematic_sheet") continue
    if (entry.type === "schematic_component") {
      const center = scale(entry.center)
      const size = { width: entry.size.width * scaleFactor, height: entry.size.height * scaleFactor }
      include(center.x - size.width / 2, center.y - size.height / 2)
      include(center.x + size.width / 2, center.y + size.height / 2)
      const next = { ...entry, center, size }
      delete next.schematic_sheet_id
      output.push(next)
    } else if (entry.type === "schematic_port") {
      const center = scale(entry.center)
      include(center.x, center.y)
      const next = {
        ...entry, center,
        distance_from_component_edge: (entry.distance_from_component_edge ?? 0) * scaleFactor,
      }
      delete next.schematic_sheet_id
      output.push(next)
    } else if (entry.type === "schematic_trace") {
      const edges = entry.edges.map((edge) => {
        const from = scale(edge.from), to = scale(edge.to)
        include(from.x, from.y); include(to.x, to.y)
        return { ...edge, from, to }
      })
      const next = { ...entry, edges }
      delete next.schematic_sheet_id
      output.push(next)
    } else if (entry.type === "schematic_net_label") {
      const center = scale(entry.center), anchor_position = scale(entry.anchor_position)
      include(center.x, center.y); include(anchor_position.x, anchor_position.y)
      const next = { ...entry, center, anchor_position }
      delete next.schematic_sheet_id
      output.push(next)
    } else if (entry.type === "schematic_text") {
      if ("source_trace_id" in entry && entry.source_trace_id) continue
      const position = scale(entry.position)
      include(position.x, position.y)
      const next = { ...entry, position }
      delete next.schematic_sheet_id
      output.push(next)
    } else if (!entry.type.startsWith("schematic_")) {
      output.push(entry)
    }
  }
  const spanX = Number.isFinite(minX) ? (maxX - minX) * KICAD_UNITS_TO_MM : 100
  const spanY = Number.isFinite(minY) ? (maxY - minY) * KICAD_UNITS_TO_MM : 100
  const width = Math.max(200, Math.ceil(spanX + 120))
  const height = Math.max(150, Math.ceil(spanY + 120))
  return {
    kicadData: output,
    paperSize: { name: "User", width, height, customSize: { width, height } },
  }
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
  const { kicadData, paperSize } = prepareSingleSheetForKicad(elements, scaleFactor)
  const kicad = new CircuitJsonToKicadSchConverter(kicadData, { paperSize })
  kicad.runUntilFinished()
  const rawFiles = kicad.getOutputFiles({ schematicFilename: name + ".kicad_sch" })
  if (rawFiles.length !== 1 || rawFiles[0].filename !== name + ".kicad_sch") {
    throw new Error("Expected a single flat KiCad schematic, got: " + rawFiles.map((file) => file.filename).join(", "))
  }
  const kicadFiles = rawFiles.map((file) => {
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
    const content = schematic.getString()
      .replace(/\(paper\s+[\d.]+\s+[\d.]+\s*\)/, `(paper "User" ${paperSize.width} ${paperSize.height})`)
      .replaceAll('(project ""', '(project "' + name + '"')
    return { filename: file.filename, content }
  })
  const project = new CircuitJsonToKicadProConverter(kicadData, {
    projectName: name, schematicFilename: name + ".kicad_sch", schematicSheetPlan: kicad.schematicSheetPlan,
  })
  project.runUntilFinished()
  const proData = JSON.parse(project.getOutputString())
  proData.schematic = { ...(proData.schematic ?? {}), page_layout_descr_file: "${KIPRJMOD}/borderless.kicad_wks" }
  proData.pcbnew = { ...(proData.pcbnew ?? {}), page_layout_descr_file: "${KIPRJMOD}/borderless.kicad_wks" }
  return [
    ...kicadFiles,
    { filename: "borderless.kicad_wks", content: BORDERLESS_WKS },
    { filename: name + ".kicad_pro", content: JSON.stringify(proData, null, 2) + "\n" },
  ]
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
    .filter((entry) => entry.type !== "schematic_element_outside_sheet_warning" && entry.type !== "schematic_sheet")
    .map((entry) => {
      if ("schematic_sheet_id" in entry) {
        const next = { ...entry }
        delete next.schematic_sheet_id
        return next
      }
      return entry
    }) as CircuitJson
  const points = content.flatMap((entry) => {
    if (entry.type === "schematic_component" || entry.type === "schematic_port" ||
        entry.type === "schematic_net_label") return [entry.center]
    if (entry.type === "schematic_text") return [entry.position]
    return []
  })
  const maxX = Math.max(...points.map(({ x }) => x)), minX = Math.min(...points.map(({ x }) => x))
  const maxY = Math.max(...points.map(({ y }) => y)), minY = Math.min(...points.map(({ y }) => y))
  const width = maxX - minX + 35, height = maxY - minY + 18
  const elements = content
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
  const existingKicad = await readdir(kicadDirectory)
  await Promise.all([
    ...existingKicad.filter((entry) => /^section_\d+\.kicad_sch$/.test(entry))
      .map((entry) => unlink(new URL(entry, kicadDirectory))),
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
