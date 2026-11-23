import React from "react"
import { exportBoard } from "../../../../检查/联合仿真/code_as_eda/renderer"
import { IsolatedCanPreview } from "../src/IsolatedCanPreview"
import parts from "../src/can_parts.json"

await exportBoard({
  board: <IsolatedCanPreview />,
  reference: parts,
  outDir: new URL("../exports/can_interface_preview/", import.meta.url),
  name: "MpptCanPreview",
})
