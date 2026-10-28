# 电力项目代理维护资料

本目录存放电力项目的目录整理计划、项目级 EasyEDA API 工具和项目协作技能入口。实际设计与检查入口仍以[电力根 README](../README.md)及模块 README 为准。

## 本目录直接文件

| 文件 | 用途与边界 |
|---|---|
| [cleanup_bms_plan.json](cleanup_bms_plan.json) | BMS 目录整理计划记录。 |
| [cleanup_empty_moves.json](cleanup_empty_moves.json) | 空目录整理/移动计划记录。 |
| [cleanup_link_baseline.json](cleanup_link_baseline.json) | 目录整理前的 Markdown 链接基线记录。 |
| [cleanup_other_plan.json](cleanup_other_plan.json) | 非 BMS 模块目录整理计划记录。 |
| [cleanup_root_plan.json](cleanup_root_plan.json) | 电力根目录整理计划记录。 |
| [cleanup_shared_plan.json](cleanup_shared_plan.json) | 共用元件库目录整理计划记录。 |

这些 JSON 是维护计划或基线，不能单独证明目录整理已完成。

## 子目录

| 目录 | 用途 |
|---|---|
| [tools/](tools/README.md) | 链接检查和目录整理维护脚本；包含会改写文件/清理原件的脚本，先阅读说明和输入范围。 |
| `skills/easyeda-api-skill/` | 项目配置使用的 JLCEDA EasyEDA API 技能包；此目录包含上游技能及参考资料，详见项目 [AGENTS.md](../AGENTS.md) 的适用规则。 |
