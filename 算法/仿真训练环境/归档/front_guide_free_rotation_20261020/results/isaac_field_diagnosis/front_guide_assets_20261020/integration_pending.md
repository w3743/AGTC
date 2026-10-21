# 接入交接状态（2026-10-20）

本记录只描述未完成工作，不是完成验收或训练启动记录。原生定时任务继续 PAUSED。

## 已实现、已有 CPU 证据

- `isaac_field/assets.py`：可选 `FrontGuideSpec`、独立哈希目录、两只被动轮与总质量/COM/惯量补偿。资产专项 9 passed、真实退出 0。
- `scripts/train_isaac_foundation_fresh.py`：独立 `random_rough` 方案，全场独立随机起点/目标、plants0、rock_density0、60s、4env、seed69009、单阶段196608步、新初始化。Python `prepare` 默认旧方案，CLI 默认新方案。没有执行正式 prepare/launch。fresh+continue 专项 31 passed，记录位于 `results/isaac_field_improvement/random_global_rough_20261020_validation/`。
- `isaac_field/body_stability.py`：35°试验软阈值之上总倾角惩罚、body X/Y角速度平方时间积分；None 时原奖励不变。该阈值不是安全角标定。
- `isaac_field/learning.py`：可选车身 z/pitch/roll、真实三维角速度轨迹字段，以及评估行的稳定性汇总和独立终止条件字段。`tests/test_body_stability.py` 加 `tests/test_body_stability_recording.py` 联合 16 passed、真实 CPU 退出0，记录在 `results/isaac_field_diagnosis/body_stability_20261020/trajectory_recording/`。
- `scripts/continue_isaac_foundation.py` 是本轮代理早先未完成修改的回退：逐字节恢复原 child source snapshot，SHA256 `feb34732fec95d18eda67b851b870e3543ffedd34cde163987a8ae2c129136ef`。历史快照本身未改。

## 已落地但未完成的环境部分

本轮代理已改 `env.py` 的可选 guide/stability cfg 字段、资产创建、复位支撑点、root world 角速度转 body 保存、task稳定性统计初始化；`newton_drive.py` 增加可选 passive_front_guide 接触材料识别。环境段暂停写入，尚未声称通过原生测试。

另一个聊天“评估并优化项目训练方法”（thread `01a15836-cb4e-7682-b370-7e10e965a13d`）正在改同一目录的 `env.py/task.py/learning.py/policy.py`。只读聊天记录已核实新增 `diagnostic_infos`、`reward_components` 来自该聊天，必须保留。未经人类授权没有向那个聊天发送消息。正在等待协调授权或对方结束。

## 必须补齐再运行新方案

1. 核对当前重叠修改来源及写入范围后，补 `make_env(body_stability_config=None)` 到 FieldEnvCfg 的传递。
2. `_get_dones` 在现有奖励诊断逻辑上添加稳定性采样/累积/惩罚及拆分边界、roll、pitch终止信息，避免双扣奖励或丢失终止前诊断。
3. `drive_config` 使用带guide的资产hash，启用时追加 guide/stability合同；None时兼容旧规范。接触metadata的空guide字段不应无条件出现。
4. `tests/test_front_guide_native.py` 只是待运行测试，Newton和MuJoCo的joint/DOF索引必须分别可靠映射，不可混比。执行真实双world质量、被动effort、材料、reset、有限性和统计局部reset验证。
5. 实际几何/接触测试后才能描述翻越能力；单个自由轮不等于连续履带斜段。旧模型/协议/成绩不能改贴新几何标签。
6. 最终接口/专项/原生验证通过后，才可冻结新源码、生成正式协议和0步baseline。没有启动新的训练、没有恢复定时跟进。

这些功能尚未整体接入验收，不提前宣布完成。第十五批原计数0/5，根代理在实际收口后统一计数并同步AGENTS；没有为此修改并发聊天也可能正在编辑的主状态文档。
