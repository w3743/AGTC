# 删除辅助目录与冗余介绍

按用户截图整块删除二维页左侧目录及支路图例，清掉桌面/移动端抽屉CSS、事件与专属常量/函数，不转移到其他位置。实际SVG分区、层卡片和数据保留。初始详情标题/状态徽标去重，色标tooltip去实现术语。复审车辆页后，删除重复品牌/渲染介绍，缩短初始与重置后的按钮教学；状态逻辑和控件不变。第三方复用说明同步当前实现，MIT许可证保留。

HTML Node语法0、54次字面DOM引用均存在，见 `static_audit.json`；viewer CPU专项25 passed、py_compile0，见 `vehicle_copy/`。没有新增低影响文字测试，也没有重跑训练或物理基准。

部署前核实并停止旧viewer PID323976，新启动记录WSL PID326346/exec98232（后续须查实际进程），使用原final196608模型、暂停0秒。实际浏览器侧栏数0，导航文字不再出现；同视口画布x226→0、宽1055.333→1281.333px。60卡片71线完整，主图张量图片与全局范围逐项相同，详情选层及local±0.952539正常。网络console为空，车辆页只有第三方THREE.Clock弃用warning，无错误；品牌介绍已实际核对移除，操作控件仍存在。

截图 `network_without_sidebar.png`、`vehicle_copy/vehicle_without_intro.png`；结构化记录 `browser_acceptance.json`。当前代码相对42训练源快照的预期差异为viewer.py、network_2d.py/html及THIRD_PARTY_NOTICES.md。协议、baseline、final、训练快照及原成绩不改写。该功能及验证文档合计第十四批第4项，计数4/5。
