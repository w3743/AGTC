# 联动前导向轮、稳定性与全场rough整理记录（2026-10-20）

本轮完成导向轮同侧传动、车身稳定奖励/统计、全场随机rough独立入口三项功能；第十六批计数3/5，按用户要求提前整理。此前第十五批训练方法4项+自由前导向轮第5项已先完成整理，没有越过五项规则。本轮整理不是新的功能计数。

## 已收口的逻辑

`FrontGuideSpec → USD几何/质量 → cfg dict → 环境只读规格 → Newton构建期world hook → drive metadata → fresh冻结合同`。

左右导向轮各跟随同侧轮0，角速度比R_main/R_guide，当前45mm/45mm=1∶1。每world12负重轮关系+2导向轮关系，14个原驱动执行器及各2Nm限幅保持；导向轮无独立DriveAPI、actuator或PID。新增轮的质量从车体扣除，实际导入30kg。自由轮仅诊断，None保留旧资产/hash/合同。

车身稳定目标只用仿真车体真值作奖励/统计，不加观测维度。真实body角速度由root world角速度旋转得到，超过35°软阈值的总倾角及车体X/Y摆动受罚；原1.2rad硬终止不变。输出姿态峰值、超阈时长、角速度RMS、分开的终止原因，均为策略步统计。

fresh新profile为100×100m、plants0、rocks0、幅值预算.35m、波长4–10m、60s、4env/seed69009、新随机196608 transition。全场随机起点/朝向/目标，每run固定一份地形。粗糙开发配对要求实际USD hash/path/bytes、地形seed和初态相同，姿态信息缺失/非有限拒绝。未更改网络输入、LSTM尺寸或四维动作。

## 验证

| 检查 | 真实结果与路径 |
|---|---|
| 新导向轮/稳定性native | attempt8：3 passed/39 warnings，实际exit0 |
| 原有环境native | attempt9：14 passed/55 warnings，实际exit0 |
| 完整CPU回归 | 349 passed/18 skipped/8 warnings，实际exit0 |
| 编译/静态 | compileall、3项目bash-n和7项额外检查实际0 |
| 源码稳定性 | 80份源码/测试SHA前后相同 |
| 新rough准备 | prepare/preflight/外层均实际0，45份live+snapshot逐项相同 |

native详见 [联动报告](../../isaac_field_diagnosis/front_guide_coupled_20261020/README.md)。平地直行/反向pivot下guide对同侧主轮位置残差最坏4.96e-5rad，速度p95 .01286/max .01617rad/s，均通过原门槛；partial reset只清对应world统计。所有失败尝试保留，未放宽门槛。

CPU/raw静态见 `run_20261020T084433Z/`、`outer_20261020T084433Z/`，汇总 [read_only_audit_attempt2.json](read_only_audit_attempt2.json)。第一轮outer因汇总器误写哈希文件名exit1，内层验证已0；第二轮已实际执行完成并为0，收到停止重复指令后只读汇总，没有伪称仅重做汇总。原始失败保留。

## 台阶配对的实际结论

[pair_attempt3](../../isaac_field_diagnosis/front_guide_coupled_step_probe_20261020/pair_attempt3/README.md) 在同源码、同平地USD、同初态/固定动作/物理设置下，顺序运行free和coupled。两run、CPU analyzer和launcher实际exit均0，均200个有限样本、10秒仿真；动作[.05,.05,0,0]，60mm高、800mm长的固定台阶。

两模式都在187/200策略步快照记录到导向轮与台阶shape pair，联动模式真实14条等式、导向轮自身actuator输出0、最大驱动轮effort .9334Nm≤2。台阶工况guide速度关系最大残差.0921rad/s，应与平地专项数值分开看。

**没有越过台阶，也没有表现出该固定动作下的翻越提升**：free/coupled车体root最大x约2.2538/2.2535m，台阶前缘2.6m、后缘3.4m；全车保守后缘也未清空台阶。它证明接触和传动约束进入求解，不证明持续承载、接触力方向、最大越障高度或策略学习能力。attempt1/2的probe审计错误（关节命名、误过滤eq_objtype）均在物理动作前失败，保留，不当作物理失败。修正仅限诊断probe，没有改仿真动作、摩擦、步长或生产源码。

## 车辆查看器

新手动页面 [8781](http://127.0.0.1:8781/) 使用当前联动几何、零植物/岩石、粗糙地形、60秒、body-stability，无checkpoint；8782接口与二维服务正常。实际启动Python PID333204，命令和初始metadata见 `results/isaac_field_viewer/front_guides_coupled_rough_20261020/`；PID只属于本次验收时点。

GUI已检查轮组实际显示及相机视角；手动[.03,.03,0,0]运行至5.45秒，页面车辆由约(15.87,1.32)移动到(16.53,1.62)，显示速度约.13m/s、偏航约.42rad，损伤/碰撞0。随后暂停、手动指令归零并重置回0秒；最终保留暂停页面。显示数值有限，浏览器console error为0，保留1条上游THREE.Clock弃用warn。没有模型，因此不声称已验证新策略激活。

证据包括 `viewer_replay/viewer_initial_state.json`、`paused_after_motion.ax.txt`、`reset_final.ax.txt`、`console.json`、`final_api_frame.json`、`actual_process_argv.txt` 与 `vehicle_guides_side.png`。API无策略状态下sequence/state为空属正常，没有用颜色代替raw激活验证。旧8769/8770未替换。

## 整理与归档

主文档使用说明/ISAAC_LAB/AGENTS已合并当前接口、任务、命令和状态，移去堆叠的历史“正在运行/当前”等描述；原字节与SHA保存在 [docs_before/manifest.json](docs_before/manifest.json)。训练方法功能和必需协作/验收规则保留，方法旧v2协议已标明源码过期需另prepare。

6个自由轮阶段旧目录已用原生PowerShell移动至 `归档/front_guide_free_rotation_20261020/`，目标绝对路径先核验在workspace内；130文件、2554465字节，逐文件SHA/大小全部相同，实际exit0。包括旧free资产/接入验证/台阶探针、旧rough prepare及失败preflight、未启动的旧viewer包装器。见 [归档清单](../../../归档/front_guide_free_rotation_20261020/archive_manifest_before.json) 和 [移动后核验](../../../归档/front_guide_free_rotation_20261020/archive_verification.json)。原始记录内的旧路径保持原字节，通过清单映射，不改写历史protocol/成绩。

当前联动代码、对照、失败attempt和最新验证保留在活动results目录；资产缓存、external、共享运行时、旧正式模型与协议/快照/成绩未移动。只读核查父/续训协议与两个final四个冻结SHA均符合旧证据。

## 新训练的实际状态

新目录 `results/isaac_field_improvement/random_global_rough_coupled_guides_20261020/` 只有protocol、run_status和45源snapshot；正式0步baseline尚未创建，状态prepared/0步，**没有启动训练**。protocol SHA：

`5de27ea7b7e1714b8822ec8fe11f0b53e6f08d194577b253d376913a00405980`

真实准备/预检见 `results/isaac_field_improvement/random_global_rough_coupled_guides_20261020_validation/prepare_preflight_attempt1/preflight_audit.json`。旧free trial的失败预检已归档，没有修写它。若以后生产源改变，必须重开独立trial，不能改本协议。原生automation保持用户要求的PAUSED。

机械联动与本轮整理已完成；可靠崎岖导航、连续履带的倾角翻越能力与新的训练效果仍待后续验证。

最终只读收口审计实际exit0：45冻结源/协议仍匹配；6旧目录原位已无、130归档文件SHA/字节全部匹配；3份docs_before与manifest一致；4份活动Markdown的17个本地链接目标存在。证据位于 results/isaac_field_improvement/random_global_rough_coupled_guides_20261020_validation/final_read_only_audit.json 及同名.exit_code。该审计没有再运行测试或仿真。
