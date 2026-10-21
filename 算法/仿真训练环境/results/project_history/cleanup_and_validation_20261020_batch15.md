# 第十五批整理与验证（2026-10-20）

## 计数与范围

两个聊天共享同一工作目录。训练方法聊天先完成4项独立功能：训练诊断、能力门槛/旧任务采样、安全选模、独立对照入口。本聊天完成第5项：可选的成对被动前导向轮，包含资产、环境/材料、CLI布线与原生验收。收口后先完成本次整理，再接续车身稳定目标和全场random_rough方案的发布验收；两者的并行预备实现也纳入本次源稳定性及回归检查，没有按文件拆分计数。

## 整理内容

- 划清共享文件写入权；保留对方新增的奖励分项、终止步快照、学习诊断和GAE接口，没有按旧快照恢复共享环境代码。
- 导向轮/稳定性参数通过Isaac配置系统时统一为plain dict，环境恢复公开的类型化规格；避免configclass递归改写frozen dataclass。可选配置关闭时保持旧资产哈希及默认接触metadata结构，不添加空guide字段。
- 导向轮不匹配驱动轮执行器正则；不增加motor、力矩或同侧约束。三维继续由官方Viser显示实际USD，没有另写车辆mesh或控制器。
- rough配对使用真实Scene seed和field USD哈希；缺失地形、初态或非有限姿态统计拒绝比较。没有用FakeAdapter预填字段代替实际native接口验证。
- 接入记录与命令同步到使用说明、ISAAC_LAB和本轮诊断README。整理前3份主文档原字节保存在 `cleanup_20261020_batch15/docs_before/`，SHA清单同目录。
- 本轮代理对continuation入口的未完成修改已回退为原child快照字节；没有改历史快照、模型、协议或原始成绩。实验失败尝试和真实退出保留。临时诊断只在独立results目录，不扩展生产框架。

## 验证

- 最终整理CPU：336 passed、17 skipped、8 warnings，pytest实际exit0；17个跳过属于显式门控的native检查，不称它们在CPU运行。
- compileall实际exit0；项目 `scripts/` 内3个shell脚本bash-n均0；本轮viewer/native wrapper的bash-n、台阶probe的py_compile均0。
- `isaac_field/`、`scripts/`、`tests/` 源与测试共79项，运行前后SHA逐项相同；两份清单SHA均 `2193961c06ccad4a2075d1ee40a7f755218b1b2d40d650ba0133c22ca1f77d7b`。
- outer wrapper与WSL实际exit0，证据在 `cleanup_20261020_batch15/outer_20261020T081643Z/` 和 `run_20261020T081643Z/`。
- 本轮CUDA原生分开运行：新增guide/stability专项2 passed、38上游warnings、exit0；原有env integration 14 passed、55上游warnings、exit0。见 `results/isaac_field_diagnosis/front_guide_integration_20261020/summary.json` 和各attempt原件。这是16项物理/接口回归，不是学习效果评估。

## 保留的失败与限制

第一次native因configclass给frozen dataclass赋值失败；配置改为字典后，首次关节名匹配验收又失败。最终按Newton正式per-world关节/DOF映射验收，不把Newton全局ID与MuJoCo模板ID混比。测试CUDA tensor转换顺序也已修正。失败日志与exit未覆盖。

整理验证第一次外层命令因PowerShell到WSL变量传递失败，runner未启动，真实exit2保留在 `outer_20261020T081540Z/`；改为直接执行LF outer脚本后才得到上述实际exit0。

前导向轮仍是离散轮代理，不能代表连续履带斜段牵引或实车最大越障高度。固定台阶探针、手动查看器和后续准备协议有各自独立记录；整理通过不等于新策略已经训练或车辆能力提升。未做FPS/加速倍率基准、未观察830xx，原生定时任务保持PAUSED。

2026-10-20后续归档路径提示：本批自由前导向轮的6个旧证据/准备目录已原字节移至归档/front_guide_free_rotation_20261020/，原路径映射、SHA与实际核验见该目录archive_manifest_before.json和archive_verification.json。本批原始验证结果不变，后续联动轮不是本批自由轮。
