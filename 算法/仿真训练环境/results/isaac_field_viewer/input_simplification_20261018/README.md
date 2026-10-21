# 输入简化验收（2026-10-18）

用户确认删除定位噪声常数输入、深度仅输入一张深度图，并取消目标相关输入的100倍额外增益。本功能整体计为第十批第4项，当前4/5。

## 当前契约

- `sensor-depth-only-v3`：LiDAR `(2,400)`；深度 `(1,160,88)`；检测 `(16,10)`；状态 `(43,)`；批维由官方wrapper添加。
- 删除旧state[41]的`position_noise/size`，roll/pitch现在为41、42。定位误差的采样与位置观测保留，不把传感器模拟改成完美定位。
- 深度有效值为原轴向相机Z深度/量程（默认5m），无效值为0；不另输入mask。传感器内部mask与有效性规则、噪声和点云处理保持。有效最远深度为1，与无效0可区分。
- 目标相关输入直接进入状态线性层，取消额外放大与`state.scaled`节点。所有现行训练入口默认goal_gain=1，兼容参数仅接受1。
- 首层深度卷积为1→8，状态投影43→32。使用原有PyTorch CNN/Linear及SB3-Contrib RecurrentPPO，不增加替代编码设施。
- 不自动裁切或拼接旧权重，旧双通道/44维checkpoint明确报错。`convert_legacy_checkpoint`仅映射类路径，不能迁移这些维度。

## 预览与原权重

当前车辆三维 http://127.0.0.1:8769/ ，模型二维 http://127.0.0.1:8770/ 。WSL PID15070（后续须重新核查真实进程），主exec session14070，处于暂停状态。

加载`initial_untrained_preview.zip`，0个训练步，seed80003，仅供结构、传感器和激活连通性检查。页面持续显示“未训练 · 随机初始化”。生成脚本`create_preview.py`未调用learn；记录`preview_initialization.json`。不能把预览动作称为已学习策略，下一阶段需要新训练。

原196608步checkpoint仍在原位且SHA256为`11f6c287e8bfdbb348fb1c9d9205c2a95692f1629ef8cee4626aef3947ae506f`，与修改前一致。旧训练协议、源码快照、权重、成绩未改写；旧8768查看器未操作。

现行通用新训练入口为`python -m isaac_field train`（不传旧checkpoint），或`learning.train_curriculum(checkpoint=None)`，保存到独立新目录。历史foundation/GAE/探索实验脚本的baseline与100倍gain仅适用于旧源码快照；不能继续当作当前契约入口。此次没有启动新训练、盲测或速度/FPS基准，也不主张学习效果提升。

## 验证

- 全CPU：155 passed / 8 skipped，`cpu_validation.log`；8项原生物理集成测试未在本轮单独执行。
- 后续viewer+learning专项35 passed：`viewer_learning_followup.log`；最终UI模型步数提示专项39 passed：`final_ui_followup.log`。
- compileall与页面内嵌JS的Node语法解析通过：`static_validation_final.log`；wrapper bash语法通过。
- 实际浏览器选择深度层，显示`(1×160×88)`；整体状态卡43维，全部层API中无state.scaled，43个字段。
- 浏览器点击运行及暂停，帧1/输入0秒→帧153/输入7.55秒/反馈7.60秒，深度图、首卷积、LSTM和动作PNG均变化，无采集错误；最后帧224，输入11.10秒/反馈11.15秒，已暂停。`before_run.json`/`during_run.json`保存去除图片后的元数据和激活图SHA，`runtime_acceptance.json`保存断言结果。
- `depth_single_channel_verified.jpg`为最终实际浏览器截图，含单通道深度、状态43和未训练标记。重启间原标签曾显示连接错误；服务器HTTP200后新标签实际打开并验收，未把错误页当作完成。
- 新预览Netron TorchScript：`../netron_exports/initial_untrained_preview_d23f28c2.pt`及JSON；3组×3步与原策略分布/动作/记忆/价值比较最大误差均0，见`netron_export.log`。只执行CPU导出，没有重新尝试曾被拒的Netron后台启动。

当前运行命令：

```bash
bash scripts/isaaclab_python.sh -m isaac_field viewer --seed 80003 --episode-seed 82001 --plants 16 --flat --mode goto --reward-profile protect_plants --max-time 20 --goal-distance 1 3 --goal-angle 60 --port 8769 --checkpoint results/isaac_field_viewer/input_simplification_20261018/initial_untrained_preview.zip --output results/isaac_field_viewer/input_simplification_20261018/live_final
```

日志：`viewer_final_stdout.log`、`viewer_final_stderr.log`；最终stderr为空。`live_preview`为增加未训练常驻标记前的短暂预览记录，最终以`live_final`为准。
