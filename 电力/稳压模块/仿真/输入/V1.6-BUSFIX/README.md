# V1.6-BUSFIX 模型输入

## 本目录直接文件

| 文件 | 用途与边界 |
|---|---|
| [model_input.json](model_input.json) | BUSFIX 逐节保护模型的结构化运行参数和资格门输入。 |
| [source_snapshot_manifest.json](source_snapshot_manifest.json) | 标记模型所绑定的冻结输入源快照和哈希清单。 |

## 子目录

| 目录 | 内容 |
|---|---|
| `source_snapshot/` | 保存当时的设计计划、BOM、网表和控制说明副本，用于说明模型输入来源；这些副本是冻结追溯快照，不是当前源文件入口。 |
