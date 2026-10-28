# Native 可视化辅助

本目录包含 Native 状态查看、回放和渲染诊断代码。文件按源码实现列出；其中回放记录器使用旧 pilot checkpoint 和占位相机通道，不应视作 V2 训练结果或传感器等价证据。

| 文件 | 用途 |
| --- | --- |
| field_robot_19.usda | USD 静态车辆资产，保存 19-body 几何及显式装配/质量元数据，供本目录查看工具使用。 |
| introspect_bodies.py | 构造单个原生环境并输出模型刚体标签/结构信息。 |
| make_viewer.py | 将录制的 native rollout JSON 打包成自包含 HTML 回放页。 |
| native_live.py | 从单个 native world 读取物理状态并发布 viewer schema-v1 live-state packet；按现有查看器需要写出四个带偏移的环境。 |
| record_rollout.py | 用旧 pre-protocol LSTM pilot checkpoint 录制确定性动作回放 JSON；depth 与 detections 使用占位值以减少录制成本。 |
| test_render.py | 检查 WSL 内 MuJoCo headless EGL 渲染初始化和简单场景。 |

当前正式训练状态见 [上级迁移与验收说明](../MIGRATION_ACCEPTANCE.md)。本目录文件仅负责可视化或诊断。
