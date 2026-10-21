# 二维策略网络连线高亮移除

日期：2026-10-20。独立改动计数：1。

只修改 `isaac_field/visualization/network_2d.html`。原实现由 `updateSelectionStyle()` 根据选中或悬停层给相关 `flow-edge` / `crossing-bridge` 添加 `.is-related`，并把其他线设为 `.is-muted`；现已删除这套状态切换及三处重复样式规则。连线一直使用基础样式，保留真实拓扑、连续曲线、箭头和 `branch-*` 分支配色。层卡片的选中/悬停框、相邻节点提示和选层详情仍工作。源文件未发现基于运行激活值的连线亮度或流光动画。

与训练源码的预期差异按不可变快照核对：

- 快照：`results/isaac_field_improvement/fresh_coupled_tracks_20261020/source_snapshot/isaac_field/visualization/network_2d.html`，SHA-256 `EAC6FC0A399151CC05C1E0469D6BA2556468DF6FF0E3479705B6A4F4D6F4BB67`
- 当前：`isaac_field/visualization/network_2d.html`，SHA-256 `01A75C2F3E48DDAB29D6ABDC71F9ACD3CCFE8E48C3B945B4A3FC8A4D91D96484`
- 快照对比仅包含上述连线/交叉弧高亮CSS与JS逻辑删除；快照文件及模型、协议、训练成绩未修改。

验证：

- `node --check results/isaac_field_viewer/edge_highlight_removed_20261020/network_2d_inline.js`：退出码 0。该 JS 是修改后 HTML 唯一内联脚本的原样提取，SHA-256 `87C763FDFC82BEE75AB0ABAFDD5342F7F8293F5E22259C9568C2367EA53E36FE`。
- `python -m pytest -q tests/test_network_2d.py`：退出码 0，14 passed。现有极端有限值用例产生 1 条 NumPy 平方溢出 RuntimeWarning。
- 搜索确认不再存在 `.flow-edge.is-related`、`.flow-edge.is-muted`、`.crossing-bridge.is-related`、`.crossing-bridge.is-muted` 或对应 JS class toggle；默认线型和分支规则仍在 HTML 中。