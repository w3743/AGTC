# 行间通道 reset 锚点审计

本记录只审计 reset 几何，不涉及策略表现、GPU 或盲测数据。可重复算例见 `row_gap_reset_audit.py`，该脚本通过 `scripts/isaaclab_python.sh` 运行；本次实际输出及退出码见 `row_gap_reset_audit.log`。

## 结论

连续的朝向矩形可以沿构造的 0.8 m 行距通道直行，但默认 0.1 m 网格的相邻格心都被当前植物保护半径挡住。`has_grid_anchor` 因而拒绝起终点；同一结果经 `FieldTask.prepare` 触发 `InvalidEpisodePoseError`。这确认了窄通道的离散网格假阴性，不能据此断言策略运行中永远无法驶入行间。

这里的车外廓为 0.8 × 0.47 m，两排植物位于 y=4.0/4.8 m，构造的 `Scene.radius` 为 0.12 m，验证余量为 0.025 m。水平车身半宽为 0.235 m，中心线两侧余量各为 `0.4 - 0.235 - (0.12 + 0.025) = 0.02 m`，总通道净宽约 0.04 m。中心线 y=4.4 m 对应的格心 y=4.45 m，邻格心 y=4.35 m；二者的水平矩形位姿均不清。算例得到 `all_free_connected=True`、`start_goal_connected=False`。

## 口径与影响

- reset 使用 `FieldTask.prepare`、`sample_episode_start_goal` 和 `Scene.validate_connectivity`。`navigation.py` 的 H/V 直线遮罩按朝向矩形加 `Scene.radius + 0.025 m` 计算；转向区域才使用外接圆。`has_grid_anchor` 先检查连续位姿，再检查其 0.1 m 网格格心，因此会拒绝连续可行但格心不通的窄通道。默认 `Config.validate_scene=True`，随机候选会重试，显式起点会被拒绝；Isaac Lab reset 也调用相同采样器。
- 旧 `Scene.validate` 明确保留了完整外接圆膨胀（`hypot(0.8, 0.47)/2 + 0.05 ≈ 0.514 m`）用于离线完整验证。生产 reset 调用点使用 `validate_connectivity`；全仓 `.validate()` 调用搜索仅找到相应单元测试。
- 导航 clearance 用 `Scene.radius`（通常随机 0.09–0.15 m），而当前 `stem-contact-v2` 损伤几何用 `Scene.stem_radius`（围绕配置值 0.012 m）。两者不能混为实际主茎半径或损伤判据。较大的 `Scene.radius` 是当前保守导航保护区；是否应在 v2 下继续作为行间 clearance，取决于期望的行间通行标准。
- 连通性验证只参与 reset/地图筛选，不为策略提供路径图，也不拦截 episode 中的动作。故它会限制起终点采样与可验证连通性，但不能单独证明策略无法在行驶过程中进入通道。

## 最小后续验证

先用开发场景统计连续位姿通过数、`has_grid_anchor` 拒绝数和 reset 重试数，并在同一行中心线上用 `stem_radius` 检查整段车体扫掠。若目标要求行间通行，再对 clearance 半径与锚点分辨率做针对性调整，并保留转向区检查；不建议仅凭本算例全局缩小车辆外廓或圆形转向 clearance。

主要源码：`isaac_field/navigation.py`（`_free_masks`、`has_grid_anchor`、`validate_scene`）、`isaac_field/task.py`（`Scene.pose_clear`、`Scene.validate`、`sample_episode_start_goal`、`FieldTask.prepare`、损伤几何）、`isaac_field/env.py`（`IsaacFieldEnv._reset_idx`）。已有相近 CPU 用例位于 `tests/test_task.py::test_row_gap_clearance_respects_wider_envelope_and_grid_resolution`；本归档没有修改生产代码或测试。
