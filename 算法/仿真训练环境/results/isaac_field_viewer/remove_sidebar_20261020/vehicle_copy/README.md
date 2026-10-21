仅修改 isaac_field/viewer.py 的三处可见文案：
- 移除“Isaac Lab / Newton 任务控制面板。场景与车辆由官方 Viser 实时渲染。”静态介绍。
- 初始状态缩短为“手动控制已暂停。”
- 重置状态缩短为“官方仿真已重置。”，保留既有 viewer 回归使用的重置状态语义，移除重复点击运行提示。

验证：
- python -m py_compile isaac_field/viewer.py：exit 0。
- python -m pytest -q tests/test_isaac_field_viewer.py：25 passed，exit 0。
- 仅 CPU-only mock viewer 测试；未启动浏览器、GPU、仿真或训练。
- 本任务未新增测试，也未修改测试文件。
