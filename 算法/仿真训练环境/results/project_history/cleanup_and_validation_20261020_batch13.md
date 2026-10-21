# 第十三批整理与验证（2026-10-20）

本批第5项已完成：修复同侧代理轮缺少机械联动的建模缺项，按预设收敛标准采用1ms物理积分，并统一当前drive契约为 coupled_pi_v7。连同此前布局、输入精简、native限幅和PI执行环四项，本批达到5项；本次整理后计数0/5，进入第十四批。

## 整理范围

- 正式实现只有一个 `isaac_field/newton_drive.py`：公开Newton world hook、同侧mimic/equality、一次模型检查与canonical物理metadata。诊断staging及失败产物留在results供复现，生产不导入结果目录，不复制vendor或建立替代物理框架。
- env、fresh训练预检、当前fixtures及native回归使用一致的v7接口；保留旧v6/v5/v4拒绝边界和旧metadata重评能力。输入、动作、神经网络、奖励、课程及原力矩预算不变。
- 重写当前使用说明与项目状态，移出反复叠加的过期“当前”描述；原三份文档原字节复制至 `batch13_before_coupled_tracks_20261020/` 并核SHA。AGENTS保留协作、五项计数、任务边界、实车事实、已纠正误判及证据索引。
- 没有发现本轮新增生产helper内需另保留的重复运行路径；诊断脚本/原始实验记录/权重/协议不删除、不覆写。生产reset未改变；新增测试遵循官方完整reset生命周期，未用底层sim.step绕过环境检查。

## 验证结果

| 检查 | 最终结果 | 证据 |
|---|---|---|
| 全CPU | 244 passed、14 skipped、8 warnings，WSL/pytest实际0 | `steering_contact_repair_20261020/cpu_validation/attempt1/` |
| compileall | exit0 | 同目录compileall.log/validation_exit.json |
| 39个保存bash语法 | 全部exit0，历史脚本未为通过而修改 | 同目录bash_syntax.log |
| 当前runner/fixture专项 | 55 passed，真实exit0 | `cpu_v7_migration/pytest_final.log` |
| 全native | 14 passed、55 warnings，pytest/LF bash/WSL实际0 | `production_v7_validation/v7_native_full_suite_attempt1.*` |
| 1ms同设置八动作配对 | 各64000物理/1280policy，真实exit0；候选8/8机械/运动/制动门槛通过 | `staged_response/same_side_mimic_motion_1ms_v1/paired_comparison.*` |
| 源码/历史保留 | 当前42训练源已记录；旧PI protocol/baseline/final SHA匹配；三文档归档SHA匹配 | `cleanup_and_validation_20261020_batch13/source_audit.json` |

CPU warnings为既有极端float32统计溢出与TorchScript tracing/deprecation；native warnings来自upstream/既有接口。测试失败尝试保留：首次写入脚本部分成功后断言计数错误；新native首次漏完整reset观测初始化；focused pytest通过后CRLF wrapper退出1。最终全14套件采用LF包装，真实各层退出0。

## 结论边界与后续

增大mu并未改善转向；未采用mu扫描、KF10000或plane替换作为调参修复。当前同侧联动在同1ms下提升双向转向响应，保留原每轮2Nm预算；1ms对.5ms仅该固定动作yaw按原门槛收敛，不声称所有物理量或实车参数已标定。制动反向速度积分约.0615m仍存在；固定动作通过不是导航成绩。

现在允许独立prepare/launch `fresh_coupled_tracks_20261020`，新初始化196608步、源码协议冻结、正式0baseline/检查点与同物理同初态开发配对。训练前不得宣称已经启动，真实退出前不宣称训练完成。不观察830xx、不做速度/FPS基准；旧试验报告不跨物理混算。
