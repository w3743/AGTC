# 2026-10-20 第十四批整理与验证

本批围绕第十四批已完成的四项网络 UI 修改及第五项独立续训入口进行整理。没有改动冻结的训练实现、旧实验协议/模型/成绩，也没有准备或启动训练。

修正了 `tests/test_network_layout.py` 中过时的侧栏断言：测试改为确认 `nav-open`、`nav-drawer-toggle` 和 `region-sidebar` 均已移除，并保留真实曲线、箭头和分支颜色检查。两份运行文档现列出从已验证 final 继续训练的 `--max-time 60` 命令，说明续训保留 policy 权重与 Adam 状态、重置环境/RNG/LSTM/rollout 状态，并明确 60 秒会改变 `episode_time_ratio`；父 baseline 与新 final 需按同一 60 秒配置重评，不能与历史 20 秒成绩混算。

全 CPU 验证通过：pytest 为 **255 passed、14 skipped、8 warnings**，真实退出码 0；`compileall` 退出码 0；限定范围内 **43 个 Bash 脚本**语法检查退出码 0；当前 `network_2d.html` 的 1 个内联脚本经 Node 语法检查通过，退出码 0。Native 测试按项目约定跳过，未启用 GPU。详细命令输出与退出码保存在 [attempt2 证据目录](cleanup_and_validation_20261020_batch14/)。

首轮 pytest 暴露了测试断言已过时：`test_network_layout.py:137` 仍要求侧栏抽屉存在，实际页面已移除侧栏。该轮输出为 **1 failed、254 passed、14 skipped、8 warnings**；其包装错误地把退出码记录为 0，不能作为通过依据。原始失败日志及错误的尾部标记均保留在 `full_cpu_pytest.log`。更新断言后，attempt2 全套 pytest 真实退出码为 0，见 `pytest_attempt2.exit_code` 和 `validation_attempt2.exit.json`。

哈希检查也保留了包装问题与恢复证据。attempt2 原始 WSL `sha256sum -c` 因输入清单使用 Windows 反斜杠路径而真实退出码为 1；原始清单和失败日志未覆盖。独立 PowerShell 校验后，对路径分隔符规范化，并只将获准修改的 `tests/test_network_layout.py` 更新为实际哈希，再做独立复核，真实退出码为 0：68 项清单中 **67 项与整理前相同**，唯一差异是上述测试修正，没有意外变化。68 项是本次本地代码/测试哈希范围，不代表训练快照源文件数量；它也不等同于 43 个 Bash 脚本语法检查。

证据索引：

- 首轮失败及捕获错误：`cleanup_and_validation_20261020_batch14/full_cpu_pytest.log`
- attempt2 pytest 与真实退出码：`full_cpu_pytest_attempt2.log`、`pytest_attempt2.exit_code`
- compileall、Bash、Node：`compileall.exit_code`、`bash_syntax.count`、`bash_syntax.exit_code`、`node_syntax.json`
- 原始哈希路径错误与规范化后的独立核验：`source_hashes_after.log`、`hash_recovery_context.json`、`hash_recovery.log`、`hash_recovery.exit_code`
- 完整 attempt2 状态：`validation_attempt2.exit.json`；其外层 wrapper 因原哈希路径问题退出 1，子检查各自真实退出码如上记录。
