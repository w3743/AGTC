# Isaac Field 策略网络可视化

本目录提供可选的只读策略网络观察工具。策略层采样由 viewer 面板触发；实际 actor 动作张量来自策略调用。可选 critic 诊断使用独立前向与独立循环状态，不替换训练代码、参数、观测或 actor 循环状态。该工具不属于训练必需导入。

| 文件 | 用途 |
| --- | --- |
| __init__.py | 标记该子包为仅供查看器使用的可选模块。 |
| activation_maps.py | 定义传感器、状态、检测和动作字段标签，以及激活值到颜色图像的映射。 |
| inspection.py | 只读检查 SB3/RSL-RL 单环境 actor 张量，并可单独检查 critic。 |
| network_2d.html | 独立二维页面的 HTML/SVG 容器，接收本地服务发布的帧数据并显示网络节点、连线和数值。 |
| network_2d.py | 提供 localhost HTTP 服务，保存并提供不可变策略帧，以及接收页面控制请求。 |
| network_layout.py | 按策略拓扑生成确定性布局和连线路径；帧激活值不参与布局。 |
| network_panel.py | 在 Viser 侧栏提供采集、暂停、预览、运行与导出控制，并将采样帧发布到独立二维页面。 |
| THIRD_PARTY_NOTICES.md | 记录 network viewer 中改编自 llm-viz 的部分及其上游版本信息。 |

运行和历史记录见[仿真训练环境上级说明](../../README.md)。
