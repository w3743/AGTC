import { fileURLToPath } from "node:url"

const sharedModules = fileURLToPath(new URL("../../../检查/联合仿真/code_as_eda/node_modules", import.meta.url))
const processResult = Bun.spawn(["bun", "run", "scripts/export.tsx"], {
  env: { ...process.env, NODE_PATH: sharedModules },
  stdin: "inherit",
  stdout: "inherit",
  stderr: "inherit",
})
process.exit(await processResult.exited)
