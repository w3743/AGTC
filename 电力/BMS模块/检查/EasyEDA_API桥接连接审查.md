# EasyEDA API 桥接连接审查

日期：2026-10-15。范围：按项目 `easyeda-api-skill` 检查扩展安装入口、桥接监听范围、执行接口和 CORS。未调用 EDA API，未修改原生工程或桥接脚本。

## 结果

- 已阅读项目技能文件 `.agents/skills/easyeda-api-skill/SKILL.md`，并核对其桥接脚本 `scripts/bridge-server.mjs`。
- 本地 `GET http://127.0.0.1:49620/health` 返回 `status=ok`，但 `edaConnected=false`、`edaWindowCount=0`；`GET /eda-windows` 返回空列表。因此当前没有活动的嘉立创EDA扩展窗口，不能安全地选择工程或执行 API。
- 当前 TCP 监听观测为 `127.0.0.1:49620`（PID 26496）。脚本的 `POST /execute` 接收任意 JavaScript 字符串，并转发到活动 EDA 窗口；HTTP 响应对任意来源开放 `Access-Control-Allow-Origin: *`。虽然当前监听只观测到 loopback，任意来源网页仍可能跨域访问本地接口；连接 EDA 前应保持此风险可见，不将桥接视作已审查通过的安全边界。
- 官方扩展说明：在嘉立创EDA“高级 → 扩展管理器”安装 **Run API Gateway**，启用“允许外部交互”和“显示在顶部菜单”，然后从顶部 **API Gateway → Reconnect** 重连。验收需再次确认 `/health` 与 `/eda-windows` 显示至少一个已连接窗口。官方说明见[扩展获取和使用](https://prodocs.lceda.cn/cn/api/user-guide/using-extension.html)及[Run API Gateway FAQ](https://github.com/easyeda/eext-run-api-gateway/blob/main/FAQ.md)。

## 当前结论

桥接服务运行正常，但扩展尚未连接；API 修改、网表读回和 ERC 均未在本轮执行。当前 EPRJ3 仍是原有基线，不得据此宣称最终原理图已更新或通过电气验收。桌面应用控制面在本会话不可用，因此扩展尚不能由本会话安装/启用；安装入口与验收步骤已确认。
