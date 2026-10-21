# `.eprj3` 单页生成接口准备

日期：2026-10-14  
范围：准备从现有 `.eprj2` 设计/库资料落成一个单原理图、单图页 `.eprj3` 的接口与记录方案。**本轮只查文档并更新本准备稿；没有创建或修改 EasyEDA 工程，没有生成工程文件，也没有运行测试或 ERC。**

## 结论

- 官方指南确认 `.eprj3` 是目录工程；`<工程名>.eprj3` 是项目索引文件，原理图页数据在 `sch/<原理图名>/<图页名>.esch2`。工程索引由 `createProject()` 生成，不应手工拼写或将 `.eprj2`、`.epro2` 改扩展名冒充 `.eprj3`。
- 指南第 8.2 节的可运行示例以 7 个位置参数调用 `createProject(...)`，第 7 个参数为 `EDMT_ProjectFileFormat.EPRJ3`；格式参数只在客户端半离线/全离线模式生效。旧静态 CLI `doc api` 的 `createProject` 参数展示不完整，实施时按官方完整示例传参，并检查返回 UUID、实际文件结构与单页数量。
- 官方第 9 节支持“API 建项目骨架 + 脚本写原理图页内容”的组合流程。EPRJ3 的 `.esch2` 将 `SYMBOL`、`FOOTPRINT`、`DEVICE`、`BLOB`、`SCH_PAGE` 文档嵌在同一文件；库记录优先由编辑器实际保存生成，再原样复用。
- `.eprj2` 项目文件不是 `.esch2` 原子记录流。不能把旧工程文件直接当作新格式文档库。通过编辑器/API 将源库中的器件放入目标 EPRJ3 并保存，读取目标工程生成的库文档；随后只组装 `SCH_PAGE` 业务记录。
- `doc format` 查询提供 `.esch2` 的 `DOCHEAD`、各类 `META` 和图元字段。项目索引本身不是该命名空间中的原子记录类型，因此继续由 `createProject()` 生成；`.ecfg`、`.evar` 也从 API 骨架保留，不按猜测重写。

旧版准备稿已在改写前归档并读回核验：`BMS模块/归档/历史与调试.zip` 内 `prechange/2026-10-14/BMS模块/检查/eprj3单页生成接口准备.md`，SHA256 `47483DA67370321522E5179B1737DE73D98B4C665794B7F725DCC6D8B0BB83CE`。

## 1. 由 API 建 EPRJ3 骨架

下列脚本供后续生成时使用，本轮没有执行。第 7 个实参明确指定 EPRJ3；执行前确认客户端处于半离线或全离线模式。

```javascript
const projectUuid = await eda.dmt_Project.createProject(
  'BMS Single Page',
  'bms-single-page',
  undefined, // teamUuid
  undefined, // folderUuid
  'Single-page schematic built from the existing EPRJ2 component library',
  undefined, // collaborationMode
  EDMT_ProjectFileFormat.EPRJ3
);
if (!projectUuid) throw new Error('createProject failed');

const opened = await eda.dmt_Project.openProject(projectUuid);
if (opened !== true) throw new Error('openProject failed');
await new Promise((resolve) => setTimeout(resolve, 3000));

const info = await eda.dmt_Project.getCurrentProjectInfo();
const schematic = info?.data?.[0]?.schematic;
const pages = schematic?.page ?? [];
if (pages.length !== 1) {
  throw new Error(`Expected exactly one page; got ${pages.length}`);
}
return {
  projectUuid,
  projectName: info?.data?.[0]?.name,
  schematicName: schematic?.name,
  pageUuid: pages[0]?.uuid,
  pageCount: pages.length
};
```

官方 8.2 示例的骨架文件形态为：

```text
<ProjectName>/
+-- <ProjectName>.eprj3
+-- sch/<SchematicName>/<PageName>.esch2
+-- sch/<SchematicName>/<SchematicName>.ecfg
+-- sch/<SchematicName>/<SchematicName>.evar
+-- pcb/<PCBName>.epcb2        # 示例工程可能同时创建空 PCB
`-- panel/<PanelName>.epan2    # 示例工程可能同时创建空 PANEL
```

“单页”应以 API 返回的该原理图 `page.length === 1` 为准。不要因骨架中有空 PCB/PANEL 文件，就把它们误认为新增的原理图页；也不要在页数大于 1 时自动删页。

## 2. 从 EPRJ2 源库到 EPRJ3 嵌入库

1. 保留源 `.eprj2`。在 EasyEDA 中打开源工程/库，确认所需器件可以由编辑器/API 放置；不直接复制或改写旧工程容器。
2. 建立目标 EPRJ3 骨架。将目标单页所需器件经编辑器/API 放置并保存一次，让目标工程写出对应 `SYMBOL`、`FOOTPRINT`、`DEVICE` 文档。
3. 从目标 `.esch2` 中按 `DOCHEAD` 分块读取这些库文档，保持 `DOCHEAD` 与 `META` 等记录原文不变。后续页面编排只复用这些已生成库文档，并写入 `SCH_PAGE` 中的 `COMPONENT`、`ATTR`、`WIRE`、`LINE` 等业务记录。
4. 输出顺序按编辑器顺序：所有 `SYMBOL` → `FOOTPRINT` → `DEVICE` → `BLOB` → `SCH_PAGE`。`.eprj3` 不另带独立工程库文件。

若手头只有 `.eprj2` 文件，没有从编辑器保存得到的 EPRJ3 `.esch2` 库文档，则先用编辑器/API 物化并保存器件库；不要自行猜测旧容器内部编码后直接拼装。

## 3. 格式字段与引用关系

`lceda-pro doc format --class-name <类型>` 的查询结果用于字段核对。下表列出本任务涉及的完整字段名；可选字段仍应以编辑器生成的样例值为准。

| 记录/对象 | 类型与字段 |
| --- | --- |
| `.esch2` 行外壳 | `{"type":"<type>","ticket":<number>,"id":"<16位小写hex>"}||<payload>|\n`；`id`、`ticket` 在所属文档内唯一。严格使用 LF。 |
| `DOCHEAD` payload | `TDocHead` 必需：`docType`、`uuid`、`client`；可选：`updateTime`、`version`、`user`、`ticket`、`editVersion`、`firstTicket`。`docType` 使用 `EHeaderType`，本页相关值为 `SYMBOL`、`FOOTPRINT`、`DEVICE`、`BLOB`、`SCH_PAGE`。 |
| `SCH_PAGE` 的 `META` | `TMSheet`: `schematic`、`title`、`source`、`zIndex`（均必需）。`schematic` 指向所属原理图 UUID；保留 API 骨架提供的 `source` 与默认字段。 |
| 原理图元数据 | `TMSchematic`: `board`、`title`、`source`、`zIndex`（均必需）。由项目 API 生成。 |
| 板对象元数据 | `TMBoard`: `title`、`zIndex`（均必需）。由项目 API 生成。 |
| 工程配置 | `TMConfig`: `defaultSheet`（必需，默认图页 UUID）。保留骨架中的 `.ecfg`，并确认它与唯一图页一致。 |
| 库文档 `META` | `TMSymbol`: `title`、`description`、`tags`、`source`、`docType`；`TMFootprint`: `title`、`description`、`tags`、`source`；`TMDevice`: `title`、`source`、`tags`、`images`、`attributes`；`TMBlob`: `filename`、`content`。这些内容整块从编辑器输出复用，不手工重建。 |
| 通用文档 META | `TMDocument`: `title`、`source`、`zIndex`。具体文档使用对应专用 META 类型时，以专用类型为准。 |
| `COMPONENT` | `TMSchComponent`: `partId`、`x`、`y`、`rotation`、`isMirror`、`attrs`；另包含 `TSchBase` 公共字段 `groupId`、`locked`、`zIndex`，以及可选 `yAxisDirection`。`partId` 必须对应被引用 `SYMBOL` 文档中的 `PART` 行 ID。 |
| `ATTR` | `TSchAttr`: 必需 `groupId`、`locked`、`zIndex`、`parentId`、`key`、`value`、`keyVisible`、`valueVisible`、`x`、`y`、`rotation`、`color`、`fillColor`、`fontFamily`、`fontSize`、`strikeout`、`underline`、`italic`、`fontWeight`、`align`；可选 `partId`、`yAxisDirection`、`version`。`parentId` 指向组件记录 ID；`Symbol` 与 `Device` 属性值分别指向相应库文档 UUID。 |
| `WIRE` | `TWire` 使用公共字段 `groupId`、`locked`、`zIndex`，可选 `yAxisDirection`。每条 `WIRE` 都必须有成员 `LINE`。 |
| `LINE` | `TSchLine`: `lineGroup`、`startX`、`startY`、`endX`、`endY`、`strokeColor`、`strokeStyle`、`fillColor`、`strokeWidth`、`fillStyle`；另包含 `TSchBase` 公共字段。`lineGroup` 指向所属 `WIRE` ID；线段保持正交。 |

项目索引 `<ProjectName>.eprj3` 不是 `.esch2` 的 `DOCHEAD/META` 记录，也没有可据此手写的 `doc format` 类型。由 `createProject()` 生成它；`TMSchematic`、`TMBoard`、`TMSheet`、`TMConfig` 字段只用于核对 API 骨架和引用关系。

### 记录编码辅助函数

以下仅是写入器模板，不是完整项目生成器，也没有执行。图元 payload 仍须由目标数据填充；库记录直接复制 API 保存出的原始文档块。

```python
import json
import secrets


def new_id():
    return secrets.token_hex(8)  # 16 个小写十六进制字符


def encode_record(record_type, ticket, record_id, payload):
    header = {"type": record_type, "ticket": ticket, "id": record_id}
    return (
        json.dumps(header, ensure_ascii=False, separators=(",", ":"))
        + "||"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "|\n"
    )


def write_esch2(path, symbol_docs, footprint_docs, device_docs, blob_docs, page_doc):
    # 每个 *_docs 项都是从有效 EPRJ3 .esch2 读取的完整原始文档块，含 DOCHEAD/META。
    # page_doc 是 API 骨架的 SCH_PAGE DOCHEAD/META 加新页面业务记录。
    ordered_docs = symbol_docs + footprint_docs + device_docs + blob_docs + [page_doc]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(ordered_docs))
```

## 4. 写盘前的静态检查清单

- 每行严格由 header JSON、`||`、payload JSON、`|` 和 LF 组成；每个 `DOCHEAD` 开始一个文档块。
- 文档顺序为 `SYMBOL` → `FOOTPRINT` → `DEVICE` → `BLOB` → `SCH_PAGE`；库文档的 `DOCHEAD`/`META` 成对完整，原样保留编辑器字段。
- `DOCHEAD.uuid`、库 UUID、组件/属性/导线 ID 格式正确；图元 ID 与 ticket 在所属文档内唯一。
- 每个组件 `partId` 存在于引用的符号库 `PART` 行；`Symbol`/`Device` 属性指向实际文档 UUID；每个 `ATTR.parentId` 指向现存组件。
- 每个 `WIRE` 至少关联一条 `LINE`；线段端点正交，`lineGroup` 引用现存 `WIRE`。
- 唯一 `SCH_PAGE` 的 `TMSheet.schematic` 指向实际原理图；`TMConfig.defaultSheet` 等于该页 UUID；API 读取的原理图页数恰为 1。
- 写入后再由编辑器读取核对器件/导线数量、截图与 DRC。该项属于后续正式生成验收；本轮未执行。

## 5. 官方资料与查询证据

- 本地官方指南：[EasyEDA官方CLI工程格式指南.md](EasyEDA官方CLI工程格式指南.md) 第 7.3、8.2、9.1–9.7 节：项目格式参数、项目骨架、`.esch2` 行格式、文档顺序、库文档获取方式与图元字段。
- 本轮只读查询：`lceda-pro doc format --class-name TDocHead`、`EHeaderType`、`TMBoard`、`TMSchematic`、`TMSheet`、`TMConfig`、`TMSymbol`、`TMFootprint`、`TMDevice`、`TMBlob`、`TMSchComponent`、`TSchAttr`、`TWire`、`TSchLine`、`TSchBase`、`ESchDataType`。
- 没有打开/创建项目，没有写入 `.eprj3`、`.esch2`、`.ecfg` 或 `.evar`，没有运行测试、仿真、DRC 或 ERC。
