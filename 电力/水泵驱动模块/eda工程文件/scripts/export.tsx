// Build: bun install --frozen-lockfile, then bun run export.
// The repository-wide document gate recursively scans the dependency tree.
// Keep the local install outside the board directory when running that gate.
import React from "react"
import { mkdir, readdir, unlink, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { Circuit } from "@tscircuit/core"
import { convertCircuitJsonToSchematicSvg } from "circuit-to-svg"
import { buildKicadProject } from "../../../检查/联合仿真/code_as_eda/renderer"
import { PumpBoard } from "../src/PumpBoard"
import { landPatternFor } from "../src/footprints"
import { parts, type PartDefinition, type PumpReference } from "../src/parts"
import { enlargeSvgAnnotations, useShortNetLabels } from "./schematic"

const started = performance.now()
const circuit = new Circuit({ platform: { pcbDisabled: true } })
circuit.add(<PumpBoard />)
await circuit.renderUntilSettled()
const renderMs = performance.now() - started
const circuitJson = useShortNetLabels(circuit.getCircuitJson())
  .filter((element) => element.type !== "schematic_sheet")
  .map((element) => {
    if ("schematic_sheet_id" in element) {
      const next = { ...element }
      delete next.schematic_sheet_id
      return next
    }
    return element
  })
const components = circuitJson.filter((element) => element.type === "source_component")
const diagnostics = circuitJson.filter((element) => element.type.includes("error"))
if (diagnostics.length) {
  console.error(JSON.stringify(diagnostics, null, 2))
  throw new Error("Circuit render emitted diagnostics")
}

function getDefinition(name: string): PartDefinition {
  const definition = parts[name as PumpReference]
  if (!definition) throw new Error(`Unknown source component ${name}`)
  return definition
}

// The schematic-only render omits pcb_component objects. This extension records
// the same land pattern used by the TSX footprint and preserves assembly intent.
const exportJson = circuitJson.map((element) => {
  if (element.type !== "source_component") return element
  const definition = getDefinition(element.name)
  const pattern = landPatternFor(element.name as PumpReference)
  return {
    ...element,
    footprint_name: definition.footprint,
    footprint_binding: pattern.source,
    footprint_library_title: pattern.title,
    footprint_pad_numbers: pattern.pads.map(([pin]) => pin),
    do_not_place: definition.dnp ?? false,
  }
})

const exportsDirectory = join(dirname(fileURLToPath(import.meta.url)), "../exports")
const kicadDirectory = join(exportsDirectory, "kicad")
await mkdir(kicadDirectory, { recursive: true })
const existingKicad = await readdir(kicadDirectory)
const svg = enlargeSvgAnnotations(convertCircuitJsonToSchematicSvg(circuitJson, { width: 4800, height: 1380 }))
const kicadParts = new Map(Object.entries(parts).map(([ref, part]) => [ref, {
  pins: part.pins,
  dnp: part.dnp ?? false,
  placement: "board",
  in_bom: true,
  footprint: part.footprint,
}]))
const kicadFiles = buildKicadProject(circuitJson, [{
  title: "Water pump drive module",
  refs: components.map((component) => component.name),
}], kicadParts, "PumpBoard", 0.32)

const header = ["Reference", "MPN", "LCSC", "Footprint", "Footprint binding", "DNP", "Value"]
const bomRows = components.map((component) => {
  const definition = getDefinition(component.name)
  return [
    component.name, component.manufacturer_part_number ?? "",
    component.supplier_part_numbers?.jlcpcb?.[0] ?? "",
    definition.footprint, landPatternFor(component.name as PumpReference).source,
    definition.dnp ? "yes" : "no", definition.note ?? definition.label,
  ]
})
const quoteCsv = (value: string) => `"${value.replaceAll('"', '""')}"`
const bom = [header, ...bomRows].map((row) => row.map(quoteCsv).join(",")).join("\n")

await Promise.all([
  ...existingKicad.filter((entry) => /^section_\d+\.kicad_sch$/.test(entry))
    .map((entry) => unlink(join(kicadDirectory, entry))),
  writeFile(join(exportsDirectory, "circuit.json"), JSON.stringify(exportJson, null, 2) + "\n"),
  writeFile(join(exportsDirectory, "schematic.svg"), svg.replace(/[ \t]+$/gm, "").trimEnd() + "\n"),
  writeFile(join(exportsDirectory, "bom.csv"), bom + "\n"),
  ...kicadFiles.map(({ filename, content }) =>
    writeFile(join(kicadDirectory, filename), content.trimEnd() + "\n"),
  ),
])
console.log(`${components.length} components, ${renderMs.toFixed(1)} ms render, ${(performance.now() - started).toFixed(1)} ms through export.`)
