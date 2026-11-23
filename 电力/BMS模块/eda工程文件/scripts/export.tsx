// Install the shared Bun runtime once from the repository root:
// bun install --cwd 电力/检查/联合仿真/code_as_eda --frozen-lockfile
// In this board directory, run bun run export. The editable model is independent of the verifier reference.
import React from 'react'
import { exportBoard } from '../../../检查/联合仿真/code_as_eda/renderer'
import { BmsBoard } from '../src/BmsBoard'
import reference from '../src/board_parts.json'

await exportBoard({
  board: <BmsBoard />,
  reference,
  outDir: new URL('../exports/', import.meta.url),
  name: 'BmsBoard',
})
