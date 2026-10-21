# 全局自适应色标 · 2026-10-20

用户已确认使用完整数值范围，不做百分位截断。复用现有Matplotlib Turbo全光谱LUT和activation_image，不增加着色依赖。

- 统计整帧全部已采集层的有限激活值及全部已绑定模型参数，包含整体视图隐藏层、价值诊断与执行器反馈；取最大绝对值M。全部为零或无有限值时回退±1。
- 同一不可变帧保存唯一±M范围，网络缩略图、详情激活/参数图及顶部色条均使用该范围；0保持相同颜色。
- 大值立即扩展。新sequence/sim_time才允许范围最多收缩5%；同帧重复GET、选层、切换参数/整体/明细不改变范围。回合重置/模型重新绑定清空旧范围历史。
- 前端删除手动滑块，只读展示全局自适应范围；最多6位有效数字，悬浮提供完整精度。NaN/Inf不参与统计并显示洋红，未采集仍为空态/灰。
- API `/api/frame` 默认 `scale=auto`，显式numeric scale返回400，避免客户端使用不同的手动范围。

## 实际验证

CPU全量176 passed、9 skipped、1 warning，exit0。warning来自极端float32-max测试触发现有NumPy标准差square溢出；JSON标准差安全返回null，有限数值的全局色标仍完整覆盖。staged专项12 passed、1 warning。见cpu_validation.log、staged_test.log。

实际浏览器加载本次最终196608步模型，HTTP200：初始范围±2.146289825439453，切换参数范围不变；车辆运行时扩展到±21.128650665283203；暂停重复GET范围稳定。零值跨state与fusion图RGB均为[164,252,60]。重置后时间归0、恢复初始范围，无旧极值残留。最终整体12张卡片无重叠、无手动色标滑块。证据见runtime_acceptance.json、layout_acceptance.json、frame_*.json、global_adaptive_verified.png。PNG变化也可能来自统一范围变化，不能仅据PNG变化断言原始激活值变化。

## 训练与部署边界

fresh_depth_radar_20261019_v2训练和配对开发评估先正常完成196608步、真实exit0，再部署本次两个viewer源文件。deployment_manifest.json保留部署前后SHA和最终模型SHA；原训练协议、27个源码快照、模型与成绩没有被改写。最终模型SHA为fbb2e39f741bfa812b192c9b9320f711bb82187e46dc64fc5461b17a6296cc77。两组各4开发种子初态配对均有效，free/field评估均no_improvement；不是架构优越性或可靠导航结论，830xx未观察。未做训练速度/FPS基准。

当前8769车辆页、8770独立二维页加载该最终权重。viewer WSL PID9383、exec46342是部署时记录，后续需核实实际进程。运行命令见使用说明.md；完整批次整理见results/project_history/cleanup_and_validation_20261020_batch11.md。
