# 第八批整理与验证（2026-10-17）

本次只整理恢复说明和 viewer 状态，没有启动训练、Viewer 或原生物理测试。代码复查没有发现可安全删除的接口、命名或注释：prepare/launch、父级与子级快照校验，以及 viewer 的 attach/run、客户端检测和 owner-thread 刷新分别承担不同职责。仅将恢复 manifest 中的 viewer 漂移理由从特定的暂停循环修复改为通用的 `viewer-only UI/idle updates`，并同步断言；漂移仍只允许 `isaac_field/viewer.py`，旧新 SHA 均记录。

Root 同步整理了 `AGENTS.md`，并将整理前全文保存在 `results/project_history/AGENTS_before_20261017_batch8_cleanup.md`；本次未另行修改该文件。

`使用说明.md` 和 `ISAAC_LAB.md` 已同步恢复 prepare/launch 命令、8192 步权重和 Adam 续训预算及其边界。说明明确阶段计划为 near 剩余24,576步、steer30 32,768步、steer60/turn90 各65,536步，最终全局计数196,608；环境 RNG、LSTM 隐状态和未完成 rollout 不会从 checkpoint 恢复。Viewer 说明了实时三维 Viser 为默认入口、运行时目标最高20 Hz、暂停时事件触发刷新与0.25/0.5秒轮询限制，并记录目前无可用8768预览。CUDA/Warp健康探针三项通过且exit 0，但没有运行Isaac仿真，不能据此称CUDA999根因已解决。

完整CPU验证在 Ubuntu-24.04 WSL 项目环境完成。未设置 `RUN_ISAAC_PHYSICS`，原生8项按预期跳过：

- `bash scripts/isaaclab_python.sh -m pytest -q tests`：**exit 0**，117 passed、8 skipped。
- `bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests`：**exit 0**。
- `bash -n scripts/install_isaaclab.sh scripts/isaaclab_python.sh`：**exit 0**。

完整输出见同目录 `cleanup_and_validation_20261017_batch8.log`。没有改写父训练日志、协议、checkpoint 或其他实验结果；也没有进行速度/FPS测试。
