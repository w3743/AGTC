# 实时模型解剖与布局改进（2026-10-18）

此文档保留早期三维网络实现及当时的验证记录。用户后续要求独立二维页并删除网络三维展示，现已改为 [二维模型页与全光谱配色](../network_2d_20261018/README.md)。车辆三维页保留，本文下方的网络三维操作已不再提供。

新版地址：<http://127.0.0.1:8769/>。加载已完成的196608步checkpoint，初始暂停。旧8768进程保留，未把其他用户进程停止。新viewer输出目录为 `live_release/`，启动日志 `live_release_stdout.log`。最终核查时WSL PID2705、Windows26436/26380，后续使用须重新核实。

## 布局与来源

默认车辆与简化网络同场景并排。总览为四路32维编码 → 128维拼接 → 驾驶LSTM → 四维动作 → 实测运动；原始输入、中间卷积、MLP、价值支路、h/c与门值在详细层级查看。相机不会逐帧抢回视角，车辆走远后可主动重新并排聚焦。选中层详情、权重/字段、高级设置分组；默认固定±1色标，避免自适应范围造成跨帧颜色误读。

复用现有PyTorch forward hooks、SB3-Contrib和Viser。驾驶张量来自本次真实 `runner.action`，临时绕过传感器embedding缓存以完整采集各层，退出后恢复缓存开关并移除hooks。输入与动作在同次推理关联；执行器目标和实际车体运动在真实物理步后附加，两个时刻分别记录。没有人为波形、随机激活或用均值代替整层图。

价值支路额外执行只读前向，保持独立诊断h/c，自连续采集起累计；不声称它参与驾驶或复原了训练时完整critic时序。LSTM h/c直接读取；i/f/g/o由输入与权重重建，对照模块h/c误差通过后显示。采集会增加计算开销，冻结只冻结显示，关闭采集可停止额外前向；未测训练速度/FPS。

布局参考：[Netron](https://github.com/lutzroeder/netron)、[TensorBoard层级图](https://www.tensorflow.org/tensorboard/graphs)。[Distill](https://distill.pub/2017/feature-visualization/)用于区分激活与归因。这里显示数值和数据流，不把颜色或结构边误称为因果贡献。

## 验证

- 最终CPU：146 passed、8 skipped，见 `cpu_validation_final.log`；compileall和两个wrapper语法检查exit0。
- `native_network_probe_final.json/.log`：8个实际Newton物理步，58个采集层/输出；核验真实Viser公开API、逐层/聚焦、冻结隔离、预览不推进actor记忆与物理、reset清旧帧。该检查不是浏览器操作或任务能力评估。
- 完成后新版HTTP200。浏览器工具在获取Edge状态时因无法可靠确认Windows页面URL而停止；此后未重试UI或改用其他UI路径。新版画面尚未完成目视验收。
- 训练/环境/策略等原冻结源码19项（排除允许变化的viewer）SHA保持一致。新增展示模块集中在 `isaac_field/visualization/`，避免改变训练runner现有19项runtime文件快照契约。
- 曾在移动模块的批量替换中误改viewer和两个新测试；从原保存源码匹配段恢复，并重写新测试，完整CPU回归通过。模块搬迁后的导出路径也已修复并增加NPZ实际读取回归；误写到包内的两份本轮导出已原样移回项目results。

## Netron

正式导出：`../netron_exports/stage_04_turn90_11f6c287.pt`，同名JSON记录checkpoint及产物SHA、trace警告、3组×3步非零h/c的与SB3对比（最大误差均0）。使用CPU单步TorchScript，8个显式输入及8个输出，调用真实编码器/LSTM/MLP模块；不是只导出state_dict。回合开始由调用者清零h/c。Netron9.3.2隔离安装在项目 `.cache/netron`，未更改训练venv。

正式入口 `scripts/view_policy_netron.py` 支持导出、SHA核验复用及本地Netron服务。启动后台Netron服务的 `Start-Process` 命令被执行工具以 `blocked by policy` 拒绝，未执行；没有换方式重试这项被拒绝操作。可以在Netron中手动打开已导出的PT文件；具体命令见项目使用说明。实验性ONNX尝试保留于 `../netron_probe_20261018/`，不将未完成的ONNX推理校验作为通过结论。

## 自动执行限制记录

用户授权已明确。永久删除旧归档和启动Netron后台服务都在进程创建前被拒，返回信息只有 `blocked by policy`。本会话工具上下文为完全访问、无需审批提示，因此不能仅据此报错断定是普通沙盒Auto-review或具体规则，也未更改权限配置。与之不同，浏览器停止信息明确是无法确认URL。这三类事实分别记录，未把它们解释为用户未同意。

收尾时，复制本轮UI源码副本并写清单/批次记录的复合shell脚本也被执行工具拒绝，未执行；没有创建 `source_snapshot/` 或 `ui_followup_manifest.json`。随后只用直接文件补丁补记项目状态，不执行源码复制或重试被拒的服务/删除操作。
