import os from "node:os";
import { syncBuiltinESMExports } from "node:module";
try { os.userInfo(); } catch { os.userInfo = () => ({ username: process.env.USERNAME || "codex", homedir: process.env.USERPROFILE || process.cwd(), uid: -1, gid: -1, shell: null }); syncBuiltinESMExports(); }
