import { exportBoard } from "../../../检查/联合仿真/code_as_eda/renderer"
import { MpptBoard } from "../src/MpptBoard"
import reference from "../src/board_parts.json"
await exportBoard({ board: <MpptBoard />, reference, outDir: new URL("../exports/", import.meta.url), name: "MpptBoard" })
