# 完整二维展开布局 · 2026-10-20

用户要求保留二维展开结构，删除整体视图并整理块与块的位置。本次仅修改正式 `isaac_field/visualization/network_2d.html`，复用已有 SVG、缩放平移、实时 HTTP 接口与统一 Turbo 自适应色标，没有引入新前端库或改变策略计算。

## 布局

- 全部57个神经网络节点持续展开，删除整体/全部层模式开关。
- 雷达、深度、检测、状态四路按行排列，编码后汇入160维融合层。
- 驾驶 LSTM/动作头与价值诊断 LSTM/价值头上下分区；各自的输入状态、门、输出状态相邻放置。
- 保留68条真实父子连接，使用正交路线避开无关卡片。机械换算和实际运动不再作为网络节点。
- 点击卡片或选层后用“聚焦此层”查看100%尺寸，“适配全部”查看完整展开。窄面板同时容纳57层时文字会缩小，可放大和平移；数据刷新不改变当前视野。

## 实际验收

正式页面为 http://127.0.0.1:8770/ ，车辆为 http://127.0.0.1:8769/ 。已在正式页刷新确认57层完整展开、无整体视图切换，并载入左右履带独立速度合同的新196608步模型。

真实模型由仿真0秒运行至9.55秒后暂停：帧2→193，动作前输入时刻0→9.50秒，步后时刻9.55秒，车辆位姿变化，错误列表为空。运行前后 SVG viewBox 均为 `-45 -1011.45 5414 3670.899`。原始12个采样节点的比较见 `trained_activation_comparison.json`，不以PNG色彩变化代替原始激活证据，也不声称每层数值都必然变化。

`browser_geometry.json` 是实际浏览器的SVG几何读回：57张卡片、68条边，卡片重叠0，连线穿过无关卡片0，整体视图模式控件0。该几何读回来自临时8772预览页，正式采用HTML与预览逐字节一致。正式截图为 `official_layout_verified.png`、`official_expanded_verified.png` 和 `official_lstm_verified.png`。

正式部署后全CPU回归192 passed、10 skipped、1既有NumPy极端float32标准差溢出warning，exit0；compileall与Node内嵌JS语法检查exit0，见 `cpu_validation.log`、`static_deployment_validation.log`。左右履带物理与官方wrapper四项原生验证沿用本功能上一阶段的真实通过记录，见 `../direct_tracks_20261020/README.md`，本次纯布局没有重复运行物理测试。

## 训练与部署来源

独立训练 `results/isaac_field_improvement/fresh_direct_tracks_20261020/` 已完成196608步及free/field各4组开发配对评估，真实退出0，退出时间01:11:13（北京时间）。最终模型SHA256为 `342237b1b2cacfb08135535891f9aaf1d4eb5ef957a68d85168bfe2f43e21329`。

正式HTML采用记录为 `deployment_manifest.json`：部署记录时间01:12:18，旧SHA `149797abfe2519e9058d4cf6f580727b6ac4e8fb6ff0b47881c7ce4ced12d5a9`，新SHA `f4c1f74f107c6aa54eec95541e96bffd4d13d41409cfb81d6a5fe4565de796dc`；相对训练快照仅该HTML不同，其余27项一致，冻结快照28项均匹配协议。`Copy-Item`保留了stage文件01:03:43的LastWriteTime，不能用这个时间戳推断正式路径的部署时刻。训练完成检查最初因live HTML不同记为failed，原记录保留，部署补充解释见trial中的 `completion_deployment_addendum.json`；这些核验不等于对整个训练期间持续监测了全部live文件。

free开发集clean成功0/4→1/4，损伤与碰撞均未增加；field仍0/4→0/4、平均末距1.797→2.811m。不能据此宣称整田导航可靠或总体能力已提升。830xx保留未观察，未做训练速度或FPS基准。

IMU和电机负载反馈接入尚处于计划，须按用户要求先汇报并获准执行；本次布局没有扩充模型输入，也没有改写旧模型、实验协议、快照或成绩。
