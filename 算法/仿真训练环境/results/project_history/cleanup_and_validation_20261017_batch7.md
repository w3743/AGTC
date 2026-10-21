# 第七批整理与验证（2026-10-17）

本批五项已整理：GAE lambda 单变量 runner；`stem-contact-v2` 损伤规则及旧 v1 元数据兼容；60 mm履带宽与安全外廓；复用现有 `train_curriculum` 的 fresh-foundation 入口；用户确认470 mm总外宽后将履带中心距定为410 mm。文档已同步当前几何和实验状态。车身仍为300 mm宽（不含履带）；旧250 mm几何训练因外宽更新而有意中断，exec15841 exit2/SIGINT、最后检查点73,728步、无最终held-out评估。新410 mm训练尚未启动。模型、快照、协议、原始结果和中断记录均保留。

CPU验证在Isaac Lab Ubuntu-24.04 WSL runtime完成：`bash scripts/isaaclab_python.sh -m pytest -q tests`，exit0，**101 passed、8 skipped**。跳过项为未设置 `RUN_ISAAC_PHYSICS=1` 的原生物理案例。`bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests` 和 `bash -n scripts/install_isaaclab.sh scripts/isaaclab_python.sh` 均exit0。完整终端记录见同名 `.log`。

410 mm中心距/470 mm外廓下的8项原生集成测试已由root单独运行并exit0（8 passed，49 upstream warnings），没有在整理中重复启动。原始日志：`results/isaac_field_diagnosis/stem_contact_v2_20261017/native_width470_integration.log`。±0.2纯差速2 s的展开yaw为−0.42649/+0.40545 rad，最大中心漂移0.0832/0.1052 m；这是轻量代理的接触/转向回归，不是实车角速度标定或导航能力结论。

无训练、GPU重跑、速度或FPS基准；没有改运行时代码。规则备份为 `results/project_history/AGENTS_before_20261017_batch7_cleanup.md`。整理后批次计数归零。